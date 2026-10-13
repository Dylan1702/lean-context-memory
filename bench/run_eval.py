"""
bench/run_eval.py — Structured 3-way evaluation: 12 priming turns + 4 recall questions.

Usage:
    python bench/run_eval.py [--modes lean fifo unmanaged] [--pause] [--no-offline-guard]

Design
------
Facts are planted at priming turns 2, 4, 6, 8 (1-based).  The remaining 8
priming turns are fillers.  After the 12 priming turns, 4 recall questions are
asked (one per fact), each ending with "Answer in one short sentence."

Validation rule (enforced at startup):
    The last planted-fact turn must be at least RAM_BUFFER_CAP+1 turns before
    the first recall question.  With 12 priming turns and the last fact at turn
    8, there are 4 gap turns (12-8=4 > RAM_BUFFER_CAP=3), so lean mode cannot
    trivially answer from the RAM buffer alone — it must retrieve from SQLite.

Every recall question is scored with:
    - Containment: expected_value.lower() in reply.lower()
    - ROUGE-L recall: fraction of reference tokens appearing in the reply

Output folder: bench/output/run_YYYYMMDD_HHMMSS/
    script_used.txt    — this script's source
    transcript.jsonl   — one line per turn (mode, turn, user, reply, metrics)
    summary.csv        — one row per mode
SHA-256 of transcript.jsonl is printed at the end.

--pause waits for Enter between modes.

NONCE
Every prompt is prefixed "[req N]" to prevent cross-turn KV-cache hits from
artefacting the timing measurements (see nodes.py module docstring for the
full caching finding).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Import config first — sets posthog/urllib3/backoff loggers to CRITICAL
# so ChromaDB telemetry warnings never appear during offline runs.
import lean_context.config as _cfg  # noqa: F401

from lean_context import storage
from lean_context.nodes import RAM_BUFFER_CAP, extract_kv
from lean_context.config import MODEL, OLLAMA_OPTIONS, NUM_CTX, print_startup_banner
from lean_context.graph import build_graph
from langgraph.checkpoint.memory import MemorySaver
from bench.run_benchmark import _call_ollama, _SYSTEM_PROMPT, _next_nonce
from bench.rouge import rouge_l_recall, _find_evidence

# ---------------------------------------------------------------------------
# Script layout: 12 priming turns + 4 recall questions
# Facts at priming turns 2, 4, 6, 8 (1-based)
# ---------------------------------------------------------------------------

PRIMING_FACTS: dict[int, tuple[str, str, str]] = {
    # turn: (user_message, expected_key, expected_value)
    2:  ("my project target is to achieve 95% recall accuracy",
         "project_target", "95% recall accuracy"),
    4:  ("my dataset is MNIST handwritten digits",
         "dataset",        "MNIST handwritten digits"),
    6:  ("set learning_rate to 0.001",
         "learning_rate",  "0.001"),
    8:  ("my preferred batch size is 64",
         "batch_size",     "64"),
}

_PRIMING_FILLERS = [
    "What is gradient descent?",
    "Explain the transformer architecture.",
    "How does attention work?",
    "What is backpropagation?",
    "Describe convolutional neural networks.",
    "What is overfitting?",
    "How do you prevent overfitting?",
    "What is regularisation?",
]

RECALL_QUESTIONS: list[tuple[str, str, str]] = [
    # (question_text, expected_key, expected_value)
    ("What is my project target or goal for recall accuracy? Answer in one short sentence.",
     "project_target", "95% recall accuracy"),
    ("What dataset am I using? Answer in one short sentence.",
     "dataset",        "MNIST handwritten digits"),
    ("What learning rate did I set? Answer in one short sentence.",
     "learning_rate",  "0.001"),
    ("What is my preferred batch size? Answer in one short sentence.",
     "batch_size",     "64"),
]

N_PRIMING = 12


def _build_priming_script() -> list[str]:
    filler = iter(_PRIMING_FILLERS)
    turns = []
    for i in range(1, N_PRIMING + 1):
        if i in PRIMING_FACTS:
            turns.append(PRIMING_FACTS[i][0])
        else:
            try:
                turns.append(next(filler))
            except StopIteration:
                turns.append(f"Tell me about deep learning topic {i}.")
    return turns


def _validate_script() -> None:
    """Ensure last fact turn is at least RAM_BUFFER_CAP+1 turns before first recall Q."""
    last_fact_turn = max(PRIMING_FACTS.keys())
    gap = N_PRIMING - last_fact_turn
    required = RAM_BUFFER_CAP + 1
    if gap < required:
        print(
            f"[ERROR] Validation failed: last fact is at priming turn {last_fact_turn}, "
            f"first recall Q is at turn {N_PRIMING + 1}. "
            f"Gap = {gap} turns, required >= {required} (RAM_BUFFER_CAP+1={required}).\n"
            f"Adjust PRIMING_FACTS or N_PRIMING."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Per-mode evaluators
# ---------------------------------------------------------------------------

class _TurnRecord:
    __slots__ = ("mode","turn","user","reply","prompt_tokens","completion_tokens",
                 "model_ms","memory_ms","total_ms","server_total_ms","created_at",
                 "is_recall","fact_key","expected","found","rouge_l")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return {s: getattr(self, s, None) for s in self.__slots__}


def _run_lean(priming: list[str], tmp_dir: Path) -> list[_TurnRecord]:
    storage.SQLITE_DB_PATH = str(tmp_dir / "eval_lean.db")
    storage.CHROMA_PERSIST_DIR = str(tmp_dir / "chroma_eval_lean")
    storage._chroma_client = None
    storage._chroma_collection = None

    from lean_context.nodes import reset_metrics, snapshot_metrics

    reset_metrics()
    app = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "eval-lean"}}
    records: list[_TurnRecord] = []
    first = True

    def _invoke(user_input: str, turn_idx: int, is_recall: bool,
                fact_key: str = "", expected: str = "") -> _TurnRecord:
        nonlocal first
        payload = (
            {"user_input": user_input, "ram_buffer": [],
             "extracted_facts": {}, "retrieved_facts": {},
             "lean_prompt": "", "final_output": ""}
            if first
            else {"user_input": user_input}
        )
        first = False

        before = snapshot_metrics()
        t_wall = time.perf_counter()
        result = app.invoke(payload, config=config)
        total_ms = (time.perf_counter() - t_wall) * 1000
        after = snapshot_metrics()

        reply = result.get("final_output", "")
        found = expected.lower() in reply.lower() if expected else None
        score = rouge_l_recall(reply, expected) if expected else None

        return _TurnRecord(
            mode="lean", turn=turn_idx, user=user_input, reply=reply,
            prompt_tokens=after["prompt_tokens"] - before["prompt_tokens"],
            completion_tokens=after["completion_tokens"] - before["completion_tokens"],
            model_ms=after["model_ms"] - before["model_ms"],
            memory_ms=after["memory_ms"] - before["memory_ms"],
            total_ms=total_ms,
            server_total_ms=after["server_total_ms"] - before["server_total_ms"],
            created_at=after.get("created_at"),
            is_recall=is_recall, fact_key=fact_key,
            expected=expected, found=found, rouge_l=score,
        )

    for i, user_input in enumerate(priming, 1):
        rec = _invoke(user_input, i, is_recall=False)
        _print_priming_turn("lean", i, N_PRIMING, rec)
        records.append(rec)

    for qi, (question, key, expected) in enumerate(RECALL_QUESTIONS, 1):
        turn_idx = N_PRIMING + qi
        rec = _invoke(question, turn_idx, is_recall=True,
                      fact_key=key, expected=expected)
        _print_recall_turn("lean", qi, rec)
        records.append(rec)

    return records


def _run_fifo(priming: list[str]) -> list[_TurnRecord]:
    ram_buffer: list[dict] = []
    records: list[_TurnRecord] = []

    def _invoke(user_input: str, turn_idx: int, is_recall: bool,
                fact_key: str = "", expected: str = "") -> _TurnRecord:
        nonce = _next_nonce()
        parts = [nonce, f"[SYSTEM]\n{_SYSTEM_PROMPT}"]
        if ram_buffer:
            hist = "\n".join(
                f"User: {t['user']}\nAssistant: {t['assistant']}"
                for t in ram_buffer[-RAM_BUFFER_CAP:]
            )
            parts.append(f"[RECENT CONVERSATION]\n{hist}")
        parts.append(f"[CURRENT QUERY]\n{user_input}")
        prompt = "\n\n".join(parts)

        t_wall = time.perf_counter()
        reply, meta = _call_ollama(prompt)
        total_ms = (time.perf_counter() - t_wall) * 1000

        ram_buffer.append({"user": user_input, "assistant": reply})
        if len(ram_buffer) > RAM_BUFFER_CAP:
            ram_buffer[:] = ram_buffer[-RAM_BUFFER_CAP:]

        found = expected.lower() in reply.lower() if expected else None
        score = rouge_l_recall(reply, expected) if expected else None

        return _TurnRecord(
            mode="fifo", turn=turn_idx, user=user_input, reply=reply,
            prompt_tokens=meta["prompt_tokens"],
            completion_tokens=meta["completion_tokens"],
            model_ms=meta["model_ms"], memory_ms=0.0, total_ms=total_ms,
            server_total_ms=meta["server_total_ms"],
            created_at=meta.get("created_at"),
            is_recall=is_recall, fact_key=fact_key,
            expected=expected, found=found, rouge_l=score,
        )

    for i, user_input in enumerate(priming, 1):
        rec = _invoke(user_input, i, is_recall=False)
        _print_priming_turn("fifo", i, N_PRIMING, rec)
        records.append(rec)

    for qi, (question, key, expected) in enumerate(RECALL_QUESTIONS, 1):
        rec = _invoke(question, N_PRIMING + qi, is_recall=True,
                      fact_key=key, expected=expected)
        _print_recall_turn("fifo", qi, rec)
        records.append(rec)

    return records


def _run_unmanaged(priming: list[str]) -> list[_TurnRecord]:
    full_history: list[dict] = []
    records: list[_TurnRecord] = []

    def _invoke(user_input: str, turn_idx: int, is_recall: bool,
                fact_key: str = "", expected: str = "") -> _TurnRecord:
        nonce = _next_nonce()
        parts = [nonce, f"[SYSTEM]\n{_SYSTEM_PROMPT}"]
        if full_history:
            hist = "\n".join(
                f"User: {t['user']}\nAssistant: {t['assistant']}"
                for t in full_history
            )
            parts.append(f"[CONVERSATION HISTORY]\n{hist}")
        parts.append(f"[CURRENT QUERY]\n{user_input}")
        prompt = "\n\n".join(parts)

        t_wall = time.perf_counter()
        reply, meta = _call_ollama(prompt)
        total_ms = (time.perf_counter() - t_wall) * 1000

        full_history.append({"user": user_input, "assistant": reply})

        found = expected.lower() in reply.lower() if expected else None
        score = rouge_l_recall(reply, expected) if expected else None

        return _TurnRecord(
            mode="unmanaged", turn=turn_idx, user=user_input, reply=reply,
            prompt_tokens=meta["prompt_tokens"],
            completion_tokens=meta["completion_tokens"],
            model_ms=meta["model_ms"], memory_ms=0.0, total_ms=total_ms,
            server_total_ms=meta["server_total_ms"],
            created_at=meta.get("created_at"),
            is_recall=is_recall, fact_key=fact_key,
            expected=expected, found=found, rouge_l=score,
        )

    for i, user_input in enumerate(priming, 1):
        rec = _invoke(user_input, i, is_recall=False)
        _print_priming_turn("unmanaged", i, N_PRIMING, rec)
        records.append(rec)

    for qi, (question, key, expected) in enumerate(RECALL_QUESTIONS, 1):
        rec = _invoke(question, N_PRIMING + qi, is_recall=True,
                      fact_key=key, expected=expected)
        _print_recall_turn("unmanaged", qi, rec)
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_priming_turn(mode: str, i: int, total: int, rec: _TurnRecord) -> None:
    fact_marker = "[FACT] " if i in PRIMING_FACTS else "       "
    print(
        f"  [{mode:>10}] {fact_marker}turn {i:2d}/{total}  "
        f"in={rec.prompt_tokens} out={rec.completion_tokens} | "
        f"memory={rec.memory_ms:.0f} ms | model={rec.model_ms:.0f} ms | "
        f"total={rec.total_ms:.0f} ms"
    )


def _print_recall_turn(mode: str, qi: int, rec: _TurnRecord) -> None:
    found_str = "YES ✓" if rec.found else "NO  ✗"
    evidence = _find_evidence(rec.reply, rec.expected or "")
    print(f"\n  [{mode:>10}] recall Q{qi}/{len(RECALL_QUESTIONS)}")
    print(f"    You:  {rec.user}")
    print(f"    Bot:  {rec.reply[:120]}")
    print(
        f"    Expected: {rec.expected!r} | Found: {found_str} | "
        f"ROUGE-L recall: {rec.rouge_l:.2f}"
    )
    if evidence:
        print(f"    evidence: \"{evidence}\"")
    print(
        f"    tokens in={rec.prompt_tokens} out={rec.completion_tokens} | "
        f"model={rec.model_ms:.0f} ms | total={rec.total_ms:.0f} ms"
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(all_records: dict[str, list[_TurnRecord]]) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    header = (
        f"{'Mode':<12} {'recall':>6} {'ROUGE-L':>7} "
        f"{'avg_in':>6} {'max_in':>6} "
        f"{'avg_mem_ms':>10} {'avg_total_s':>11} {'calls/turn':>10}"
    )
    print(header)
    print("-" * 72)
    for mode, records in all_records.items():
        priming = [r for r in records if not r.is_recall]
        recall  = [r for r in records if r.is_recall]
        if not recall:
            continue
        n_hit    = sum(1 for r in recall if r.found)
        avg_rl   = sum(r.rouge_l or 0 for r in recall) / len(recall)
        in_toks  = [r.prompt_tokens for r in priming if r.prompt_tokens]
        avg_in   = sum(in_toks) / len(in_toks) if in_toks else 0
        max_in   = max(in_toks) if in_toks else 0
        mem_vals = [r.memory_ms for r in priming if r.memory_ms]
        avg_mem  = sum(mem_vals) / len(mem_vals) if mem_vals else 0
        tot_vals = [r.total_ms for r in priming]
        avg_tot  = sum(tot_vals) / len(tot_vals) / 1000 if tot_vals else 0
        calls_t  = (
            sum(1 for r in priming if r.completion_tokens and r.completion_tokens > 0)
            / len(priming)
        )
        print(
            f"{mode:<12} {n_hit}/{len(recall):>3}   {avg_rl:>6.2f} "
            f"{avg_in:>6.0f} {max_in:>6.0f} "
            f"{avg_mem:>10.0f} {avg_tot:>11.1f} {calls_t:>10.2f}"
        )
    print("=" * 72)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_transcript(
    all_records: dict[str, list[_TurnRecord]],
    out_dir: Path,
) -> Path:
    path = out_dir / "transcript.jsonl"
    with open(path, "w") as f:
        for records in all_records.values():
            for rec in records:
                f.write(json.dumps(rec.to_dict()) + "\n")
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_summary_csv(all_records: dict[str, list[_TurnRecord]], out_dir: Path) -> None:
    path = out_dir / "summary.csv"
    rows = []
    for mode, records in all_records.items():
        priming = [r for r in records if not r.is_recall]
        recall  = [r for r in records if r.is_recall]
        n_hit = sum(1 for r in recall if r.found)
        avg_rl = sum(r.rouge_l or 0 for r in recall) / len(recall) if recall else 0
        in_toks = [r.prompt_tokens for r in priming if r.prompt_tokens]
        avg_in = sum(in_toks) / len(in_toks) if in_toks else 0
        max_in = max(in_toks) if in_toks else 0
        mem_vals = [r.memory_ms for r in priming if r.memory_ms]
        avg_mem = sum(mem_vals) / len(mem_vals) if mem_vals else 0
        tot_vals = [r.total_ms for r in priming]
        avg_tot = sum(tot_vals) / len(tot_vals) / 1000 if tot_vals else 0
        calls_t = (
            sum(1 for r in priming if r.completion_tokens and r.completion_tokens > 0)
            / len(priming) if priming else 0
        )
        rows.append({
            "mode": mode,
            "recall_n": n_hit, "recall_d": len(recall),
            "avg_rouge_l": round(avg_rl, 3),
            "avg_prompt_tokens": round(avg_in, 1),
            "max_prompt_tokens": max_in,
            "avg_memory_ms": round(avg_mem, 1),
            "avg_total_s": round(avg_tot, 2),
            "calls_per_turn": round(calls_t, 2),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → summary.csv: {path}")


def _save_script(out_dir: Path) -> None:
    import shutil
    shutil.copy(__file__, out_dir / "script_used.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Lean-Context structured evaluation")
    parser.add_argument(
        "--modes", nargs="+",
        choices=["lean", "fifo", "unmanaged"],
        default=["lean", "fifo", "unmanaged"],
    )
    parser.add_argument("--pause", action="store_true",
                        help="Wait for Enter between modes")
    parser.add_argument("--no-offline-guard", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=_ROOT / "bench" / "output",
    )
    args = parser.parse_args()

    if not args.no_offline_guard:
        from lean_context.offline_guard import enable as _eg
        _eg()

    import ollama as _ol
    try:
        _ol.list()
    except Exception:
        print("[ERROR] Ollama not reachable — run: ollama serve")
        sys.exit(1)

    # Validate script layout
    _validate_script()

    priming = _build_priming_script()
    tmp_dir = args.output_dir / "tmp_storage"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped output folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print_startup_banner([
        f"  eval modes   : {', '.join(args.modes)}",
        f"  priming turns: {N_PRIMING}  recall Qs: {len(RECALL_QUESTIONS)}",
        f"  output       : {out_dir}",
    ])

    # Print full script
    print("\n── EVALUATION SCRIPT ─────────────────────────────────────────────")
    for i, msg in enumerate(priming, 1):
        marker = " [FACT]" if i in PRIMING_FACTS else ""
        print(f"  priming {i:2d}{marker}: {msg}")
    for qi, (q, key, _) in enumerate(RECALL_QUESTIONS, 1):
        print(f"  recall  {qi:2d}: {q}")
    print("──────────────────────────────────────────────────────────────────\n")

    all_records: dict[str, list[_TurnRecord]] = {}

    for mode in args.modes:
        print(f"\n{'='*62}")
        print(f"  MODE: {mode.upper()}")
        print(f"{'='*62}")

        if mode == "lean":
            all_records["lean"] = _run_lean(priming, tmp_dir)
        elif mode == "fifo":
            all_records["fifo"] = _run_fifo(priming)
        else:
            all_records["unmanaged"] = _run_unmanaged(priming)

        if args.pause and mode != args.modes[-1]:
            input("\n  [Press Enter to continue to next mode…]")

    _print_summary(all_records)

    _save_script(out_dir)
    transcript_path = _save_transcript(all_records, out_dir)
    _save_summary_csv(all_records, out_dir)

    digest = _sha256(transcript_path)
    print(f"\n  transcript.jsonl SHA-256: {digest}")
    print(f"  → transcript: {transcript_path}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
