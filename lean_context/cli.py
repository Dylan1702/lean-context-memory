"""
lean_context/cli.py — Interactive REPL for the Lean-Context chatbot.

Usage:
    python -m lean_context.cli [--fresh] [--no-offline-guard]

Commands:
    quit / exit        — exit the REPL
    /facts             — list all stored SQLite facts
    /clear             — reset the conversation thread
    /recall list       — list all user-stated facts tracked this session
    /recall <key>      — ask the model about a specific fact and score the reply
    --fresh            — delete memory_store.db, chroma_data/ and checkpoint.db
                         before starting (clean slate)
    --no-offline-guard — disable the socket guard (allow external connections)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Offline env vars must be set before any network-capable import
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Import config first — sets posthog/urllib3/backoff loggers to CRITICAL
# so ChromaDB telemetry warnings never appear during offline runs.
import lean_context.config as _cfg  # noqa: F401

logging.basicConfig(level=logging.WARNING)

_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--fresh", action="store_true",
                   help="Delete all stored data before starting")
    p.add_argument("--no-offline-guard", action="store_true",
                   help="Disable the socket guard")
    p.add_argument("-h", "--help", action="help")
    return p.parse_args()


def _fresh_start() -> None:
    """Delete memory_store.db, chroma_data/ and checkpoint.db."""
    import shutil
    from lean_context import storage as _st
    from lean_context.graph import CHECKPOINT_DB_PATH

    targets = [
        Path(_st.SQLITE_DB_PATH),
        Path(_st.CHROMA_PERSIST_DIR),
        Path(CHECKPOINT_DB_PATH),
    ]
    for t in targets:
        if t.exists():
            if t.is_dir():
                shutil.rmtree(t)
            else:
                t.unlink()
            print(f"  [fresh] deleted: {t}")
        else:
            print(f"  [fresh] not found (skipped): {t}")


def _print_facts() -> None:
    from lean_context import storage
    facts = storage.get_all_facts()
    if not facts:
        print("  (no facts stored yet)")
    else:
        for k, v in facts.items():
            print(f"  {k}: {v}")


def _handle_recall(
    cmd: str,
    session_facts: dict[str, str],
    app,
    config: dict,
) -> None:
    """
    /recall list         — print facts the user has stated this session.
    /recall <key>        — ask the model "What is my <key>?" and score.

    session_facts is a dict {normalised_key: raw_value} built from extract_kv
    on each user turn.  It is ground-truth (never fed to the model).
    """
    from lean_context.nodes import extract_kv
    from bench.rouge import rouge_l_recall

    parts = cmd.strip().split(None, 1)
    sub = parts[1].strip() if len(parts) > 1 else ""

    if sub == "list" or sub == "":
        if not session_facts:
            print("  (no facts recorded this session — state some with 'my X is Y')")
        else:
            print("  [Session facts]")
            for k, v in session_facts.items():
                print(f"    {k}: {v}")
        return

    # /recall <key> — normalise key (replace spaces with underscore)
    key_norm = sub.lower().replace(" ", "_")
    if key_norm not in session_facts:
        # Try a fuzzy prefix match
        matches = [k for k in session_facts if k.startswith(key_norm)]
        if len(matches) == 1:
            key_norm = matches[0]
        else:
            print(f"  [recall] Key '{sub}' not found in session facts.")
            print(f"  Known keys: {list(session_facts.keys()) or '(none)'}")
            return

    expected = session_facts[key_norm]
    # Ask the model — format as natural question
    key_display = key_norm.replace("_", " ")
    question = f"What is my {key_display}? Answer in one short sentence."
    print(f"  [recall] Asking: {question!r}")

    t0 = time.perf_counter()
    result = app.invoke({"user_input": question}, config=config)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    reply = result.get("final_output", "")
    found = expected.lower() in reply.lower()
    score = rouge_l_recall(reply, expected)
    indicator = "YES ✓" if found else "NO  ✗"

    print(f"  Bot: {reply}")
    print(
        f"  Expected: {expected!r} | Found: {indicator} | "
        f"ROUGE-L recall: {score:.2f} | {elapsed_ms:.0f} ms"
    )


def run_repl(fresh: bool = False, no_offline_guard: bool = False) -> None:
    """Main REPL loop."""
    if not no_offline_guard:
        from lean_context.offline_guard import enable as _enable_guard
        _enable_guard()

    if fresh:
        print("[--fresh] Clearing stored data…")
        _fresh_start()

    from lean_context.config import print_startup_banner, NUM_CTX
    from lean_context.graph import get_default_app
    from lean_context.nodes import OllamaUnavailableError, snapshot_metrics, extract_kv

    print_startup_banner([
        "  commands     : /facts  /recall  /clear  quit",
        "  flag         : --fresh  --no-offline-guard",
    ])

    try:
        app = get_default_app()
    except Exception as exc:
        print(f"[ERROR] Failed to initialise pipeline: {exc}")
        sys.exit(1)

    _THREAD_ID = "lean-context-default"
    thread_id = _THREAD_ID
    config = {"configurable": {"thread_id": thread_id}}
    _first_turn = True
    _turn_number = 0
    _session_facts: dict[str, str] = {}   # ground-truth tracker — never fed to model

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        low = user_input.lower()

        if low in ("quit", "exit"):
            print("Bye!")
            break

        if low == "/facts":
            print("[Stored facts]")
            _print_facts()
            continue

        if low == "/clear":
            thread_id = f"lean-context-{id(user_input)}"
            config = {"configurable": {"thread_id": thread_id}}
            _first_turn = True
            _turn_number = 0
            _session_facts.clear()
            print("[Conversation thread cleared]")
            continue

        if low.startswith("/recall"):
            _handle_recall(user_input, _session_facts, app, config)
            continue

        # Build payload
        if _first_turn:
            payload = {
                "user_input": user_input,
                "ram_buffer": [],
                "extracted_facts": {},
                "retrieved_facts": {},
                "lean_prompt": "",
                "final_output": "",
            }
            _first_turn = False
        else:
            payload = {"user_input": user_input}

        _turn_number += 1
        t_wall = time.perf_counter()

        try:
            # Snapshot metrics before to compute deltas
            before = snapshot_metrics()
            result = app.invoke(payload, config=config)
            elapsed_wall_ms = (time.perf_counter() - t_wall) * 1000
            after = snapshot_metrics()

            reply = result.get("final_output", "")
            print(f"Bot: {reply}")

            # Track facts for /recall
            kv = extract_kv(user_input)
            if kv:
                _session_facts[kv[0]] = kv[1]

            # Per-turn metrics line
            d_prompt  = after["prompt_tokens"]     - before["prompt_tokens"]
            d_out     = after["completion_tokens"]  - before["completion_tokens"]
            d_mem_ms  = after["memory_ms"]          - before["memory_ms"]
            d_model_ms = after["model_ms"]          - before["model_ms"]
            d_calls   = after["llm_calls"]          - before["llm_calls"]

            ctx_pct = f"{d_prompt/NUM_CTX*100:.0f}%" if d_prompt else "—"

            print(
                f"  [turn {_turn_number}] "
                f"tokens in={d_prompt} out={d_out} | "
                f"memory={d_mem_ms:.0f} ms | model={d_model_ms:.0f} ms | "
                f"total={elapsed_wall_ms:.0f} ms | "
                f"calls={d_calls} | ctx={ctx_pct}"
            )

            # Warn if prompt is near context limit
            if d_prompt and d_prompt > 0.9 * NUM_CTX:
                print(
                    f"  [WARNING] prompt tokens ({d_prompt}) > 90% of "
                    f"NUM_CTX ({NUM_CTX}) — prompt may be truncated"
                )

        except OllamaUnavailableError:
            print(
                "\n[ERROR] Local model unavailable.\n"
                "  → Make sure Ollama is running: ollama serve\n"
                "  → Make sure the model is pulled: ollama pull llama3.2:3b\n"
            )
        except Exception as exc:
            print(f"[ERROR] Unexpected error: {exc}")


def main() -> None:
    args = _parse_args()
    run_repl(fresh=args.fresh, no_offline_guard=args.no_offline_guard)


if __name__ == "__main__":
    main()
