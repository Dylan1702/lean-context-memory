"""
lean_context/offline_check.py — Smoke-test the full lean pipeline under the offline guard.

Run with:
    python -m lean_context.offline_check

Sends three turns through the compiled graph (hi → fact → question), checks the
offline guard is active throughout, and prints OFFLINE CHECK PASSED or the
blocked host name and a traceback.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Offline env vars must be set BEFORE any import that could touch the network
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lean_context.offline_guard import enable as _enable_guard, OfflineViolation

# Enable the guard before importing anything else
_enable_guard()

from lean_context import storage
from lean_context.graph import build_graph
from langgraph.checkpoint.memory import MemorySaver


def main() -> None:
    print("lean_context/offline_check — verifying fully-offline pipeline…\n")

    tmp = tempfile.mkdtemp(prefix="offline_check_")
    storage.SQLITE_DB_PATH = str(Path(tmp) / "check.db")
    storage.CHROMA_PERSIST_DIR = str(Path(tmp) / "chroma_check")
    storage._chroma_client = None
    storage._chroma_collection = None

    try:
        app = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "offline-check"}}

        turns = [
            ("hi",                    "evict check  — no LLM, no storage"),
            ("my name is CheckBot",   "persist check — SQLite write"),
            ("What is my name?",      "active check  — retrieval + LLM"),
        ]

        first = True
        for user_input, label in turns:
            payload = (
                {"user_input": user_input, "ram_buffer": [],
                 "extracted_facts": {}, "retrieved_facts": {},
                 "lean_prompt": "", "final_output": ""}
                if first
                else {"user_input": user_input}
            )
            first = False
            result = app.invoke(payload, config=config)
            out = result.get("final_output", "")[:60]
            print(f"  [{label}]")
            print(f"    → {out!r}")

        storage._chroma_client = None
        storage._chroma_collection = None
        print("\n✓  OFFLINE CHECK PASSED — no non-loopback connections were attempted.")
        sys.exit(0)

    except OfflineViolation as exc:
        print(f"\n✗  OFFLINE CHECK FAILED — blocked host: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n✗  OFFLINE CHECK FAILED — unexpected error: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
