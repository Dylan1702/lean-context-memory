"""
bench/run_benchmark.py — 3-way benchmark: unmanaged vs fifo vs lean (Requirement 9).

Usage:
    python bench/run_benchmark.py --mode lean
    python bench/run_benchmark.py --mode fifo
    python bench/run_benchmark.py --mode unmanaged
    python bench/run_benchmark.py --mode all

Modes (Requirement 9.4):
  lean       — full Lean-Context pipeline (SQLite + ChromaDB + RAM cap)
  fifo       — RAM sliding window only, N=3, no SQLite/ChromaDB (Baseline B)
  unmanaged  — full history appended to every prompt, unbounded (Baseline A)

Per-turn metrics recorded:
  total_gen_ms — wall-clock time for the full ollama.generate() call (prefill + decode).
                 NOT true TTFT (no streaming).
  rss_mb       — process RSS at end of turn (MB)

Turn counts:
  lean / fifo  : 15 turns
  unmanaged    : 30 turns (needs more turns to show prompt-growth slope)

NONCE
  Every prompt in fifo/unmanaged is prefixed with "[req N]" — the same nonce
  scheme used by assemble_context_node in lean mode — so cross-turn KV-cache
  hits cannot flatten the generation-time measurements.

  Caching finding: prompt_eval_count is always the FULL token count regardless
  of cache hit, so the nonce is not needed for metric accuracy but is good
  practice to prevent subtle cross-turn timing artefacts.

Output:
  bench/output/benchmark_results.csv
  bench/output/total_gen_chart.png
  bench/output/rss_chart.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import psutil
import ollama
from ollama import ResponseError

# HF offline flags must be set before any sentence-transformers import
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lean_context import storage
from lean_context.nodes import RAM_BUFFER_CAP
from lean_context.graph import build_graph
from lean_context.config import MODEL, OLLAMA_OPTIONS, NUM_CTX, print_startup_banner
from langgraph.checkpoint.memory import MemorySaver

# System prompt — must match nodes.py _SYSTEM_PROMPT exactly.
_SYSTEM_PROMPT = "You are a helpful, concise assistant."

# ---------------------------------------------------------------------------
# Scripted conversation
# Facts at turns 2, 6, 10, 14.  lean/fifo=15 turns; unmanaged=30 turns.
# ---------------------------------------------------------------------------

PLANTED_FACTS = {
    2:  ("my project target is to achieve 95% recall accuracy",
         "project_target", "95% recall accuracy"),
    6:  ("my dataset is MNIST handwritten digits",
         "dataset",        "MNIST handwritten digits"),
    10: ("set learning_rate to 0.001",
         "learning_rate",  "0.001"),
    14: ("my preferred batch size is 64",
         "batch_size",     "64"),
}

_FILLER_QUESTIONS = [
    "What is gradient descent?",
    "Explain the transformer architecture.",
    "How does attention work?",
    "What is backpropagation?",
    "Describe convolutional neural networks.",
    "What is overfitting?",
    "How do you prevent overfitting?",
    "What is regularisation?",
    "Explain dropout.",
    "What is batch normalisation?",
    "How does Adam optimiser work?",
    "What is transfer learning?",
    "Explain BERT.",
    "What is a language model?",
    "Describe tokenisation.",
    "What is word embedding?",
    "Explain cosine similarity.",
    "What is precision vs recall?",
    "How is F1 score calculated?",
    "What is a confusion matrix?",
    "Explain cross-entropy loss.",
    "What is the vanishing gradient problem?",
    "How do LSTMs work?",
    "What is a GAN?",
    "Describe reinforcement learning.",
    "What is Q-learning?",
    "Explain policy gradient.",
    "What is a reward function?",
    "How does RAG work?",
    "What is vector similarity search?",
]


def build_scripted_turns(n: int = 15) -> list[str]:
    """Return *n* turns with facts planted at fixed 1-based positions."""
    filler_iter = iter(_FILLER_QUESTIONS)
    turns: list[str] = []
    for i in range(1, n + 1):
        if i in PLANTED_FACTS:
            turns.append(PLANTED_FACTS[i][0])
        else:
            try:
                turns.append(next(filler_iter))
            except StopIteration:
                turns.append(f"Tell me about deep learning topic {i}.")
    return turns


# ---------------------------------------------------------------------------
# Nonce counter (same scheme as nodes.py so all modes are treated equally)
# ---------------------------------------------------------------------------
_bench_nonce = 0


def _next_nonce() -> str:
    global _bench_nonce
    _bench_nonce += 1
    return f"[req {_bench_nonce}]"


# ---------------------------------------------------------------------------
# Ollama helper — token cap + timeout + retry
# Returns (text, meta_dict) so callers can log per-turn metrics.
# Existing callers that do  response, _ = _call_ollama(...)  continue to work.
# ---------------------------------------------------------------------------

_OLLAMA_TIMEOUT = 60
_OLLAMA_RETRIES = 2

_META_EMPTY: dict = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "model_ms": 0.0,
    "server_total_ms": 0.0,
    "created_at": None,
}


def _call_ollama(prompt: str) -> tuple[str, dict]:
    """
    Call ollama.generate with the shared options, a per-call timeout, and up to
    _OLLAMA_RETRIES attempts.

    Returns (response_text, meta) where meta contains:
        prompt_tokens     int
        completion_tokens int
        model_ms          float   wall-clock ms for this call
        server_total_ms   float   total_duration / 1e6 (0 if missing)
        created_at        str|None

    On persistent failure returns ('[error]', _META_EMPTY) so the benchmark
    continues rather than crashing.
    """
    import httpx

    last_exc: Exception | None = None
    t0 = time.perf_counter()

    for attempt in range(1, _OLLAMA_RETRIES + 1):
        try:
            client = ollama.Client(timeout=_OLLAMA_TIMEOUT)
            resp = client.generate(model=MODEL, prompt=prompt, options=OLLAMA_OPTIONS)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            meta = {
                "prompt_tokens":     int(resp.get("prompt_eval_count") or 0),
                "completion_tokens": int(resp.get("eval_count") or 0),
                "model_ms":          elapsed_ms,
                "server_total_ms":   float((resp.get("total_duration") or 0)) / 1e6,
                "created_at":        resp.get("created_at"),
            }
            # Warn if prompt is near the context limit
            if meta["prompt_tokens"] > 0.9 * NUM_CTX:
                print(
                    f"  [WARNING] prompt_tokens={meta['prompt_tokens']} > 90% of "
                    f"NUM_CTX={NUM_CTX} — prompt may be truncated"
                )
            return resp["response"].strip(), meta
        except (httpx.ConnectError, httpx.ReadTimeout,
                httpx.RemoteProtocolError, ResponseError,
                ConnectionError, OSError) as exc:
            last_exc = exc
            if attempt < _OLLAMA_RETRIES:
                backoff = 2 ** attempt
                print(
                    f"    [warn] Ollama error attempt {attempt}/{_OLLAMA_RETRIES} "
                    f"({type(exc).__name__}: {str(exc)[:60]}). Retrying in {backoff}s…"
                )
                time.sleep(backoff)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"    [error] Ollama failed after {_OLLAMA_RETRIES} attempts: {str(last_exc)[:80]}")
    meta = dict(_META_EMPTY)
    meta["model_ms"] = elapsed_ms
    return "[error]", meta


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _print_turn(mode: str, i: int, total: int, elapsed_ms: float) -> None:
    print(f"  [{mode:>10}]  turn {i:2d}/{total}  {elapsed_ms:7.0f} ms  RSS {_rss_mb():.0f} MB")


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def run_lean(turns: list[str], tmp_dir: Path) -> list[dict]:
    """Full Lean-Context pipeline — isolated storage in tmp_dir."""
    storage.SQLITE_DB_PATH = str(tmp_dir / "bench_lean.db")
    storage.CHROMA_PERSIST_DIR = str(tmp_dir / "chroma_lean")
    storage._chroma_client = None
    storage._chroma_collection = None

    app = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "bench-lean"}}
    first = True
    records = []

    for i, user_input in enumerate(turns, start=1):
        payload = (
            {"user_input": user_input, "ram_buffer": [],
             "extracted_facts": {}, "retrieved_facts": {},
             "lean_prompt": "", "final_output": ""}
            if first
            else {"user_input": user_input}
        )
        first = False

        t0 = time.perf_counter()
        result = app.invoke(payload, config=config)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        _print_turn("lean", i, len(turns), elapsed_ms)
        records.append({
            "mode": "lean", "turn": i,
            "total_gen_ms": round(elapsed_ms, 2),
            "rss_mb": round(_rss_mb(), 2),
        })

    return records


def run_fifo(turns: list[str]) -> list[dict]:
    """Baseline B: RAM-only sliding window, N=3, no SQLite/ChromaDB."""
    ram_buffer: list[dict] = []
    records = []

    for i, user_input in enumerate(turns, start=1):
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

        t0 = time.perf_counter()
        response, _ = _call_ollama(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        ram_buffer.append({"user": user_input, "assistant": response})
        if len(ram_buffer) > RAM_BUFFER_CAP:
            ram_buffer = ram_buffer[-RAM_BUFFER_CAP:]

        _print_turn("fifo", i, len(turns), elapsed_ms)
        records.append({
            "mode": "fifo", "turn": i,
            "total_gen_ms": round(elapsed_ms, 2),
            "rss_mb": round(_rss_mb(), 2),
        })

    return records


def run_unmanaged(turns: list[str]) -> list[dict]:
    """Baseline A: full unbounded history appended to every prompt."""
    full_history: list[dict] = []
    records = []

    for i, user_input in enumerate(turns, start=1):
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

        t0 = time.perf_counter()
        response, _ = _call_ollama(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        full_history.append({"user": user_input, "assistant": response})

        _print_turn("unmanaged", i, len(turns), elapsed_ms)
        records.append({
            "mode": "unmanaged", "turn": i,
            "total_gen_ms": round(elapsed_ms, 2),
            "rss_mb": round(_rss_mb(), 2),
        })

    return records


# ---------------------------------------------------------------------------
# CSV + chart output
# ---------------------------------------------------------------------------

def save_csv(records: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    fieldnames = ["mode", "turn", "total_gen_ms", "rss_mb"]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(records)
    print(f"  → CSV appended: {csv_path}")
    return csv_path


def save_charts(csv_path: Path, output_dir: Path) -> None:
    import csv as csv_mod
    import matplotlib.pyplot as plt
    from collections import defaultdict

    data: dict[str, dict[str, list]] = defaultdict(
        lambda: {"turn": [], "total_gen_ms": [], "rss_mb": []}
    )
    with open(csv_path) as f:
        for row in csv_mod.DictReader(f):
            mode = row["mode"]
            data[mode]["turn"].append(int(row["turn"]))
            data[mode]["total_gen_ms"].append(float(row["total_gen_ms"]))
            data[mode]["rss_mb"].append(float(row["rss_mb"]))

    colors = {"lean": "tab:blue", "fifo": "tab:orange", "unmanaged": "tab:red"}

    for metric, ylabel, title, filename in [
        ("total_gen_ms",
         "Total generation time (ms)\n[prefill + decode; not true TTFT]",
         "Generation Time vs Turn — 3-way comparison",
         "total_gen_chart.png"),
        ("rss_mb", "RSS (MB)", "Process RSS vs Turn — 3-way comparison", "rss_chart.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for mode, vals in data.items():
            ax.plot(vals["turn"], vals[metric], label=mode, color=colors.get(mode))
        ax.set_xlabel("Turn")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Chart: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Lean-Context 3-way benchmark")
    parser.add_argument(
        "--mode", choices=["lean", "fifo", "unmanaged", "all"], default="all")
    parser.add_argument(
        "--turns", type=int, default=None,
        help="Override turn count for all modes")
    parser.add_argument(
        "--output-dir", type=Path, default=_ROOT / "bench" / "output")
    args = parser.parse_args()

    _DEFAULT_TURNS = {"lean": 15, "fifo": 15, "unmanaged": 30}

    output_dir: Path = args.output_dir
    tmp_dir = output_dir / "tmp_storage"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        ollama.list()
    except Exception:
        print("\n[ERROR] Cannot connect to Ollama.\n  → Run: ollama serve\n")
        sys.exit(1)

    print_startup_banner([f"  mode         : {args.mode}"])

    modes_to_run = (
        ["lean", "fifo", "unmanaged"] if args.mode == "all" else [args.mode]
    )

    csv_path = output_dir / "benchmark_results.csv"
    if csv_path.exists() and args.mode == "all":
        csv_path.unlink()

    for mode in modes_to_run:
        n = args.turns if args.turns is not None else _DEFAULT_TURNS[mode]
        turns = build_scripted_turns(n)
        print(f"\n{'='*52}")
        print(f"  Mode: {mode}   ({n} turns, cap {OLLAMA_OPTIONS['num_predict']} tokens/resp)")
        print(f"{'='*52}")
        if mode == "lean":
            records = run_lean(turns, tmp_dir)
        elif mode == "fifo":
            records = run_fifo(turns)
        else:
            records = run_unmanaged(turns)
        save_csv(records, output_dir)

        avg = sum(r["total_gen_ms"] for r in records) / len(records)
        print(f"  avg total_gen: {avg:.0f} ms  over {len(records)} turns")

    if len(modes_to_run) > 1:
        print("\nGenerating charts…")
        save_charts(output_dir / "benchmark_results.csv", output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
