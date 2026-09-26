"""
bench/recall_test.py — 3-way recall accuracy test (Requirement 9.3).

Plants 4 distinct facts at turns 2, 10, 25, 35 of a 50-turn conversation,
then asks one recall question per fact at turn 50, across all three modes.

Scores exact-match recall per fact per mode and prints a table, e.g.:

  ┌─────────────────┬──────┬──────┬───────────┐
  │ Fact            │ lean │ fifo │ unmanaged │
  ├─────────────────┼──────┼──────┼───────────┤
  │ project_target  │  ✓   │  ✗   │     ✓     │
  │ dataset         │  ✓   │  ✗   │     ✓     │
  │ learning_rate   │  ✓   │  ✗   │     ✓     │
  │ batch_size      │  ✓   │  ✗   │     ✓     │
  ├─────────────────┼──────┼──────┼───────────┤
  │ TOTAL           │ 4/4  │ 0/4  │    2/4    │
  └─────────────────┴──────┴──────┴───────────┘

Requirement 9.3: score is a fraction per mode, not a single pass/fail.

Usage:
    python bench/recall_test.py
    python bench/recall_test.py --turns 50 --output-dir bench/output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

import ollama

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lean_context import storage
from lean_context.nodes import RAM_BUFFER_CAP, GraphState
from lean_context.graph import build_graph
from langgraph.checkpoint.memory import MemorySaver
from bench.run_benchmark import (
    build_scripted_turns,
    PLANTED_FACTS,
    run_fifo,
    run_unmanaged,
    _call_ollama,
)

# ---------------------------------------------------------------------------
# Recall questions — one per planted fact
# ---------------------------------------------------------------------------

RECALL_QUESTIONS: dict[str, tuple[str, str]] = {
    # key: (question_to_ask, expected_value_substring)
    "project_target": (
        "What is my project target or goal for recall accuracy?",
        "95",
    ),
    "dataset": (
        "What dataset am I using?",
        "mnist",
    ),
    "learning_rate": (
        "What learning rate did I set?",
        "0.001",
    ),
    "batch_size": (
        "What is my preferred batch size?",
        "64",
    ),
}


# ---------------------------------------------------------------------------
# Recall evaluator
# ---------------------------------------------------------------------------

def check_recall(response: str, expected_substring: str) -> bool:
    """Case-insensitive substring match."""
    return expected_substring.lower() in response.lower()


# ---------------------------------------------------------------------------
# Per-mode recall runners
# ---------------------------------------------------------------------------

def recall_lean(turns: list[str], tmp_dir: Path) -> dict[str, bool]:
    """Run 50 turns with the full Lean pipeline, then ask recall questions."""
    storage.SQLITE_DB_PATH = str(tmp_dir / "recall_lean.db")
    storage.CHROMA_PERSIST_DIR = str(tmp_dir / "chroma_recall_lean")
    storage._chroma_client = None
    storage._chroma_collection = None

    app = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "recall-lean"}}

    # Run all 50 turns
    first = True
    for user_input in turns:
        if first:
            payload = {
                "user_input": user_input, "ram_buffer": [],
                "extracted_facts": {}, "retrieved_facts": {},
                "lean_prompt": "", "final_output": "",
            }
            first = False
        else:
            payload = {"user_input": user_input}
        app.invoke(payload, config=config)

    # Ask recall questions
    scores: dict[str, bool] = {}
    for fact_key, (question, expected) in RECALL_QUESTIONS.items():
        result = app.invoke({"user_input": question}, config=config)
        response = result.get("final_output", "")
        scores[fact_key] = check_recall(response, expected)
        print(f"  [lean] {fact_key}: '{response[:80]}…' → {'✓' if scores[fact_key] else '✗'}")

    return scores


def recall_fifo(turns: list[str]) -> dict[str, bool]:
    """Run 50 turns with FIFO-only, then ask recall questions at turn 51+."""
    ram_buffer: list[dict] = []

    def _run_turn(user_input: str) -> str:
        parts = ["[SYSTEM]\nYou are a helpful assistant."]
        if ram_buffer:
            hist = "\n".join(
                f"User: {t['user']}\nAssistant: {t['assistant']}"
                for t in ram_buffer[-RAM_BUFFER_CAP:]
            )
            parts.append(f"[RECENT CONVERSATION]\n{hist}")
        parts.append(f"[CURRENT QUERY]\n{user_input}")
        prompt = "\n\n".join(parts)
        try:
            response, _ = _call_ollama(prompt)
        except Exception:
            response = "[error]"
        ram_buffer.append({"user": user_input, "assistant": response})
        if len(ram_buffer) > RAM_BUFFER_CAP:
            ram_buffer.__setitem__(slice(None), ram_buffer[-RAM_BUFFER_CAP:])
        return response

    for user_input in turns:
        _run_turn(user_input)

    scores: dict[str, bool] = {}
    for fact_key, (question, expected) in RECALL_QUESTIONS.items():
        response = _run_turn(question)
        scores[fact_key] = check_recall(response, expected)
        print(f"  [fifo] {fact_key}: '{response[:80]}…' → {'✓' if scores[fact_key] else '✗'}")

    return scores


def recall_unmanaged(turns: list[str]) -> dict[str, bool]:
    """Run 50 turns with full unbounded history, then ask recall questions."""
    full_history: list[dict] = []

    def _run_turn(user_input: str) -> str:
        parts = ["[SYSTEM]\nYou are a helpful assistant."]
        if full_history:
            hist = "\n".join(
                f"User: {t['user']}\nAssistant: {t['assistant']}"
                for t in full_history
            )
            parts.append(f"[CONVERSATION HISTORY]\n{hist}")
        parts.append(f"[CURRENT QUERY]\n{user_input}")
        prompt = "\n\n".join(parts)
        try:
            response, _ = _call_ollama(prompt)
        except Exception:
            response = "[error]"
        full_history.append({"user": user_input, "assistant": response})
        return response

    for user_input in turns:
        _run_turn(user_input)

    scores: dict[str, bool] = {}
    for fact_key, (question, expected) in RECALL_QUESTIONS.items():
        response = _run_turn(question)
        scores[fact_key] = check_recall(response, expected)
        print(f"  [unmanaged] {fact_key}: '{response[:80]}…' → {'✓' if scores[fact_key] else '✗'}")

    return scores


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_table(results: dict[str, dict[str, bool]]) -> None:
    """Print a readable recall results table and return scores per mode."""
    fact_keys = list(RECALL_QUESTIONS.keys())
    modes = list(results.keys())

    col_w = max(len(k) for k in fact_keys) + 2
    mode_w = 12

    sep = "─" * (col_w + len(modes) * (mode_w + 1) + 2)
    header = f"{'Fact':<{col_w}}" + "".join(f"{m:^{mode_w}}" for m in modes)

    print(f"\n{'='*len(sep)}")
    print("RECALL ACCURACY RESULTS")
    print(f"{'='*len(sep)}")
    print(header)
    print(sep)

    for fact_key in fact_keys:
        row = f"{fact_key:<{col_w}}"
        for mode in modes:
            hit = results[mode].get(fact_key, False)
            row += f"{'✓':^{mode_w}}" if hit else f"{'✗':^{mode_w}}"
        print(row)

    print(sep)
    total_row = f"{'TOTAL':<{col_w}}"
    for mode in modes:
        hits = sum(1 for k in fact_keys if results[mode].get(k, False))
        total_row += f"{hits}/{len(fact_keys)}".center(mode_w)
    print(total_row)
    print(sep)
    print()


def save_results(results: dict[str, dict[str, bool]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "recall_results.json"
    # Build serialisable summary
    summary = {}
    for mode, scores in results.items():
        hits = sum(1 for v in scores.values() if v)
        summary[mode] = {
            "scores": scores,
            "fraction": f"{hits}/{len(scores)}",
            "pct": round(hits / len(scores) * 100, 1) if scores else 0,
        }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → Recall results saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Lean-Context recall accuracy test")
    parser.add_argument("--turns", type=int, default=50)
    parser.add_argument(
        "--output-dir", type=Path,
        default=_ROOT / "bench" / "output",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["lean", "fifo", "unmanaged"],
        default=["lean", "fifo", "unmanaged"],
        help="Which modes to test (default: all three)",
    )
    args = parser.parse_args()

    # Verify Ollama
    try:
        ollama.list()
    except Exception:
        print(
            "\n[ERROR] Cannot connect to Ollama at localhost:11434.\n"
            "  → Run: ollama serve\n"
            "  → Then: ollama pull llama3.2:3b\n"
        )
        sys.exit(1)

    turns = build_scripted_turns(args.turns)
    tmp_dir = args.output_dir / "tmp_storage"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, bool]] = {}

    for mode in args.modes:
        print(f"\n{'='*50}")
        print(f"Recall test — mode: {mode}")
        print(f"{'='*50}")
        if mode == "lean":
            results["lean"] = recall_lean(turns, tmp_dir)
        elif mode == "fifo":
            results["fifo"] = recall_fifo(turns)
        elif mode == "unmanaged":
            results["unmanaged"] = recall_unmanaged(turns)

    print_table(results)
    save_results(results, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
