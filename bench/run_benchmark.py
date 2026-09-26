"""
bench/run_benchmark.py — 3-way benchmark: unmanaged vs fifo vs lean (Requirement 9).

Usage:
    python bench/run_benchmark.py --mode lean
    python bench/run_benchmark.py --mode fifo
    python bench/run_benchmark.py --mode unmanaged
    python bench/run_benchmark.py --mode all        # runs all three, produces combined CSV + chart

Modes (Requirement 9.4):
  lean       — full Lean-Context pipeline (SQLite + ChromaDB + RAM cap)
  fifo       — RAM sliding window only, N=3 turns, no SQLite/ChromaDB
  unmanaged  — full history appended to every prompt, unbounded (Baseline A)

For each mode, runs the same scripted 50-turn conversation and records per turn:
  ttft_ms          — time-to-first-token (total generate() call time, ms)
  total_ms         — same as ttft_ms for local synchronous Ollama
  rss_mb           — process resident set size (MB) at end of turn

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
from typing import Literal

import psutil
import ollama

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lean_context import storage
from lean_context.nodes import (
    assemble_context_node,
    local_inference_node,
    OllamaUnavailableError,
    RAM_BUFFER_CAP,
    GraphState,
)
from lean_context.graph import build_graph
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# Scripted 50-turn conversation (same for all modes)
# ---------------------------------------------------------------------------

# Facts planted at specific turns for the recall test (bench/recall_test.py reuses this)
PLANTED_FACTS = {
    2:  ("my project target is to achieve 95% recall accuracy", "project_target", "95% recall accuracy"),
    10: ("my dataset is MNIST handwritten digits",              "dataset",        "MNIST handwritten digits"),
    25: ("set learning_rate to 0.001",                         "learning_rate",  "0.001"),
    35: ("my preferred batch size is 64",                      "batch_size",     "64"),
}

# Generic filler turns (indices not in PLANTED_FACTS)
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
    "Explain HNSW indexing.",
    "What is quantisation?",
    "Describe model pruning.",
    "What is knowledge distillation?",
    "How does Ollama work?",
    "What is LangGraph?",
    "Explain state machines.",
    "What is a checkpointer?",
    "What is SQLite WAL mode?",
    "Describe ChromaDB.",
    "What are embeddings?",
    "Explain sentence transformers.",
    "What is cosine distance?",
    "How does ChromaDB store vectors?",
    "What is a persistent client?",
    "Explain metadata filtering.",
]


def build_scripted_turns(n: int = 50) -> list[str]:
    """
    Return a list of *n* user messages with facts planted at fixed turns.
    Turn indices are 1-based to match the design doc / task description.
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
                turns.append(f"Tell me more about deep learning topic {i}.")
    return turns


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _call_ollama(prompt: str) -> tuple[str, float]:
    """Call ollama.generate and return (response_text, elapsed_ms)."""
    t0 = time.perf_counter()
    resp = ollama.generate(model="llama3.2:3b", prompt=prompt)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return resp["response"].strip(), elapsed_ms


def run_lean(turns: list[str], tmp_dir: Path) -> list[dict]:
    """Full Lean-Context pipeline. Uses isolated storage in tmp_dir."""
    # Point storage at isolated paths for this benchmark run
    storage.SQLITE_DB_PATH = str(tmp_dir / "bench_lean.db")
    storage.CHROMA_PERSIST_DIR = str(tmp_dir / "chroma_lean")
    storage._chroma_client = None
    storage._chroma_collection = None

    app = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "bench-lean"}}
    first = True
    records = []

    for i, user_input in enumerate(turns, start=1):
        if first:
            payload = {
                "user_input": user_input, "ram_buffer": [],
                "extracted_facts": {}, "retrieved_facts": {},
                "lean_prompt": "", "final_output": "",
            }
            first = False
        else:
            payload = {"user_input": user_input}

        t0 = time.perf_counter()
        result = app.invoke(payload, config=config)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        records.append({
            "mode": "lean", "turn": i,
            "ttft_ms": round(elapsed_ms, 2),
            "total_ms": round(elapsed_ms, 2),
            "rss_mb": round(_rss_mb(), 2),
        })
        print(f"  [lean]  turn {i:2d}/{len(turns)}  {elapsed_ms:6.0f}ms  RSS {_rss_mb():.0f}MB")

    return records


def run_fifo(turns: list[str]) -> list[dict]:
    """
    Baseline B: RAM-only sliding window, N=3, no SQLite/ChromaDB.
    Isolates whether Tier 2/3 persistence is actually necessary (Req 9.4c).
    """
    ram_buffer: list[dict] = []
    records = []

    for i, user_input in enumerate(turns, start=1):
        # Build minimal prompt: system + last 3 RAM turns + current query
        parts = [
            "[SYSTEM]\nYou are a helpful assistant.",
        ]
        if ram_buffer:
            hist = "\n".join(
                f"User: {t['user']}\nAssistant: {t['assistant']}"
                for t in ram_buffer[-RAM_BUFFER_CAP:]
            )
            parts.append(f"[RECENT CONVERSATION]\n{hist}")
        parts.append(f"[CURRENT QUERY]\n{user_input}")
        prompt = "\n\n".join(parts)

        t0 = time.perf_counter()
        try:
            response, _ = _call_ollama(prompt)
        except Exception:
            response = "[error]"
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # FIFO buffer
        ram_buffer.append({"user": user_input, "assistant": response})
        if len(ram_buffer) > RAM_BUFFER_CAP:
            ram_buffer = ram_buffer[-RAM_BUFFER_CAP:]

        records.append({
            "mode": "fifo", "turn": i,
            "ttft_ms": round(elapsed_ms, 2),
            "total_ms": round(elapsed_ms, 2),
            "rss_mb": round(_rss_mb(), 2),
        })
        print(f"  [fifo]  turn {i:2d}/{len(turns)}  {elapsed_ms:6.0f}ms  RSS {_rss_mb():.0f}MB")

    return records


def run_unmanaged(turns: list[str]) -> list[dict]:
    """
    Baseline A: full history appended to every prompt, unbounded (Req 9.4b).
    """
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
        try:
            response, _ = _call_ollama(prompt)
        except Exception:
            response = "[error]"
        elapsed_ms = (time.perf_counter() - t0) * 1000

        full_history.append({"user": user_input, "assistant": response})

        records.append({
            "mode": "unmanaged", "turn": i,
            "ttft_ms": round(elapsed_ms, 2),
            "total_ms": round(elapsed_ms, 2),
            "rss_mb": round(_rss_mb(), 2),
        })
        print(f"  [unmanaged]  turn {i:2d}/{len(turns)}  {elapsed_ms:6.0f}ms  RSS {_rss_mb():.0f}MB")

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

    data: dict[str, dict[str, list]] = defaultdict(lambda: {"turn": [], "ttft_ms": [], "rss_mb": []})
    with open(csv_path) as f:
        for row in csv_mod.DictReader(f):
            mode = row["mode"]
            data[mode]["turn"].append(int(row["turn"]))
            data[mode]["ttft_ms"].append(float(row["ttft_ms"]))
            data[mode]["rss_mb"].append(float(row["rss_mb"]))

    colors = {"lean": "tab:blue", "fifo": "tab:orange", "unmanaged": "tab:red"}

    # TTFT chart
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode, vals in data.items():
        ax.plot(vals["turn"], vals["ttft_ms"], label=mode, color=colors.get(mode))
    ax.set_xlabel("Turn")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time-to-First-Token vs Turn — 3-way comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ttft_path = output_dir / "ttft_chart.png"
    fig.savefig(ttft_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → TTFT chart: {ttft_path}")

    # RSS chart
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode, vals in data.items():
        ax.plot(vals["turn"], vals["rss_mb"], label=mode, color=colors.get(mode))
    ax.set_xlabel("Turn")
    ax.set_ylabel("RSS (MB)")
    ax.set_title("Process RSS vs Turn — 3-way comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    rss_path = output_dir / "rss_chart.png"
    fig.savefig(rss_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → RSS chart: {rss_path}")


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
        "--turns", type=int, default=50,
        help="Number of scripted turns (default: 50)",
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

    modes_to_run = ["lean", "fifo", "unmanaged"] if args.mode == "all" else [args.mode]

    # Clear existing CSV so charts are clean
    csv_path = output_dir / "benchmark_results.csv"
    if csv_path.exists() and args.mode == "all":
        csv_path.unlink()

    all_records: list[dict] = []
    for mode in modes_to_run:
        print(f"\n{'='*50}")
        print(f"Running mode: {mode}  ({args.turns} turns)")
        print(f"{'='*50}")
        if mode == "lean":
            records = run_lean(turns, tmp_dir)
        elif mode == "fifo":
            records = run_fifo(turns)
        else:
            records = run_unmanaged(turns)
        all_records.extend(records)
        save_csv(records, output_dir)

    if len(modes_to_run) > 1 or args.mode == "all":
        print("\nGenerating charts…")
        save_charts(output_dir / "benchmark_results.csv", output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
