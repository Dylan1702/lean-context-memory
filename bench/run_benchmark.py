"""
bench/run_benchmark.py — 3-way benchmark: unmanaged vs fifo vs lean (Requirement 9).

Usage:
    python bench/run_benchmark.py --mode lean
    python bench/run_benchmark.py --mode fifo
    python bench/run_benchmark.py --mode unmanaged
    python bench/run_benchmark.py --mode all        # runs all three, combined CSV + charts

Modes (Requirement 9.4):
  lean       — full Lean-Context pipeline (SQLite + ChromaDB + RAM cap)
  fifo       — RAM sliding window only, N=3 turns, no SQLite/ChromaDB (Baseline B)
  unmanaged  — full history appended to every prompt, unbounded (Baseline A)

Per-turn metrics recorded:
  ttft_ms  — wall-clock time for the full ollama.generate() call (ms)
  total_ms — same value (synchronous local inference, no streaming split)
  rss_mb   — process RSS at end of turn (MB)

Output:
  bench/output/benchmark_results.csv   — combined CSV with 'mode' column
  bench/output/ttft_chart.png          — TTFT vs turn, all modes overlaid
  bench/output/rss_chart.png           — RSS vs turn, all modes overlaid
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

# Force HuggingFace offline before any storage / embedding imports so the
# sentence-transformers model loader never attempts a network round-trip to
# check for updates even when model weights are already cached locally.
# Without this flag, every embed() call can issue a slow HTTP HEAD request
# to hf.co, which was the root cause of ~3.6 min/turn benchmark times.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lean_context import storage
from lean_context.nodes import RAM_BUFFER_CAP
from lean_context.graph import build_graph
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# Scripted conversation — 15 turns with 4 facts planted at 2, 6, 10, 14
# (Reduced from 50 to keep total benchmark runtime manageable on CPU.)
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
]


def build_scripted_turns(n: int = 15) -> list[str]:
    """
    Return a list of *n* user messages with facts planted at fixed 1-based turns.
    """
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
# Ollama helper — token cap + timeout + retry
# ---------------------------------------------------------------------------

_OLLAMA_OPTIONS = {"num_predict": 80}    # cap response length; keeps per-turn time bounded
_OLLAMA_TIMEOUT = 60                     # seconds before a single call is considered hung
_OLLAMA_RETRIES = 2                      # attempts per turn before giving up and continuing


def _call_ollama(prompt: str) -> tuple[str, float]:
    """
    Call ollama.generate with a response-length cap, a per-call timeout, and
    up to _OLLAMA_RETRIES attempts.

    Returns (response_text, elapsed_ms).
    On persistent failure logs the error and returns ('[error]', elapsed_ms)
    so the benchmark continues rather than crashing the whole run.
    """
    import httpx

    last_exc: Exception | None = None
    t0 = time.perf_counter()

    for attempt in range(1, _OLLAMA_RETRIES + 1):
        try:
            client = ollama.Client(timeout=_OLLAMA_TIMEOUT)
            resp = client.generate(
                model="llama3.2:3b",
                prompt=prompt,
                options=_OLLAMA_OPTIONS,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return resp["response"].strip(), elapsed_ms
        except (httpx.ConnectError, httpx.ReadTimeout,
                httpx.RemoteProtocolError, ResponseError,
                ConnectionError, OSError) as exc:
            last_exc = exc
            if attempt < _OLLAMA_RETRIES:
                backoff = 2 ** attempt          # 2s, then 4s
                print(
                    f"    [warn] Ollama error on attempt {attempt}/{_OLLAMA_RETRIES} "
                    f"({type(exc).__name__}: {exc!s:.60}). "
                    f"Retrying in {backoff}s…"
                )
                time.sleep(backoff)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"    [error] Ollama failed after {_OLLAMA_RETRIES} attempts: {last_exc!s:.80}")
    return "[error]", elapsed_ms


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _print_turn(mode: str, i: int, total: int, elapsed_ms: float) -> None:
    rss = _rss_mb()
    print(f"  [{mode:>10}]  turn {i:2d}/{total}  {elapsed_ms:7.0f} ms  RSS {rss:.0f} MB")


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
            {
                "user_input": user_input, "ram_buffer": [],
                "extracted_facts": {}, "retrieved_facts": {},
                "lean_prompt": "", "final_output": "",
            }
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
            "ttft_ms": round(elapsed_ms, 2),
            "total_ms": round(elapsed_ms, 2),
            "rss_mb": round(_rss_mb(), 2),
        })

    return records


def run_fifo(turns: list[str]) -> list[dict]:
    """Baseline B: RAM-only sliding window, N=3, no SQLite/ChromaDB."""
    ram_buffer: list[dict] = []
    records = []

    for i, user_input in enumerate(turns, start=1):
        parts = ["[SYSTEM]\nYou are a helpful assistant."]
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
            "ttft_ms": round(elapsed_ms, 2),
            "total_ms": round(elapsed_ms, 2),
            "rss_mb": round(_rss_mb(), 2),
        })

    return records


def run_unmanaged(turns: list[str]) -> list[dict]:
    """Baseline A: full unbounded history appended to every prompt."""
    full_history: list[dict] = []
    records = []

    for i, user_input in enumerate(turns, start=1):
        parts = ["[SYSTEM]\nYou are a helpful assistant."]
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
            "ttft_ms": round(elapsed_ms, 2),
            "total_ms": round(elapsed_ms, 2),
            "rss_mb": round(_rss_mb(), 2),
        })

    return records


# ---------------------------------------------------------------------------
# CSV + chart output
# ---------------------------------------------------------------------------

def save_csv(records: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    fieldnames = ["mode", "turn", "ttft_ms", "total_ms", "rss_mb"]
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
        lambda: {"turn": [], "ttft_ms": [], "rss_mb": []}
    )
    with open(csv_path) as f:
        for row in csv_mod.DictReader(f):
            mode = row["mode"]
            data[mode]["turn"].append(int(row["turn"]))
            data[mode]["ttft_ms"].append(float(row["ttft_ms"]))
            data[mode]["rss_mb"].append(float(row["rss_mb"]))

    colors = {"lean": "tab:blue", "fifo": "tab:orange", "unmanaged": "tab:red"}

    for metric, ylabel, title, filename in [
        ("ttft_ms", "TTFT (ms)", "Time-to-First-Token vs Turn — 3-way comparison", "ttft_chart.png"),
        ("rss_mb",  "RSS (MB)",  "Process RSS vs Turn — 3-way comparison",         "rss_chart.png"),
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
        "--mode",
        choices=["lean", "fifo", "unmanaged", "all"],
        default="all",
        help="Which pipeline mode(s) to benchmark (default: all)",
    )
    parser.add_argument(
        "--turns", type=int, default=15,
        help="Number of scripted turns (default: 15)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=_ROOT / "bench" / "output",
        help="Directory for CSV and chart output",
    )
    args = parser.parse_args()

    turns = build_scripted_turns(args.turns)
    output_dir: Path = args.output_dir
    tmp_dir = output_dir / "tmp_storage"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Verify Ollama is reachable before starting
    try:
        ollama.list()
    except Exception:
        print(
            "\n[ERROR] Cannot connect to Ollama at localhost:11434.\n"
            "  → Run: ollama serve\n"
            "  → Then: ollama pull llama3.2:3b\n"
        )
        sys.exit(1)

    modes_to_run = (
        ["lean", "fifo", "unmanaged"] if args.mode == "all" else [args.mode]
    )

    # Clear existing CSV so charts are clean when running all modes
    csv_path = output_dir / "benchmark_results.csv"
    if csv_path.exists() and args.mode == "all":
        csv_path.unlink()

    for mode in modes_to_run:
        print(f"\n{'='*52}")
        print(f"  Mode: {mode}   ({len(turns)} turns, cap {_OLLAMA_OPTIONS['num_predict']} tokens)")
        print(f"{'='*52}")
        if mode == "lean":
            records = run_lean(turns, tmp_dir)
        elif mode == "fifo":
            records = run_fifo(turns)
        else:
            records = run_unmanaged(turns)
        save_csv(records, output_dir)

        avg = sum(r["ttft_ms"] for r in records) / len(records)
        print(f"  avg TTFT: {avg:.0f} ms  over {len(records)} turns")

    if len(modes_to_run) > 1:
        print("\nGenerating charts…")
        save_charts(output_dir / "benchmark_results.csv", output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
