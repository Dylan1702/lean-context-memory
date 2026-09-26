"""
cli.py — Minimal interactive REPL for the Lean-Context chatbot (Requirement 10.2).

Usage:
    python -m lean_context.cli
    # or
    python lean_context/cli.py

Each user turn is routed through the compiled LangGraph pipeline.
OllamaUnavailableError is caught here and shown as a clear message
rather than a crash or hang (Requirement 7.2, design.md §Error Handling).
"""

from __future__ import annotations

import sys
import logging

logging.basicConfig(level=logging.WARNING)  # keep output clean in interactive use

from lean_context.graph import get_default_app
from lean_context.nodes import OllamaUnavailableError

_BANNER = """
╔══════════════════════════════════════════════════╗
║   Lean-Context Memory  •  Local AI Chatbot       ║
║   Model: llama3.2:3b via Ollama (localhost)      ║
║   Type  'quit' or 'exit' to stop                 ║
║   Type  '/facts' to show stored facts            ║
║   Type  '/clear' to reset conversation thread    ║
╚══════════════════════════════════════════════════╝
"""

_THREAD_ID = "lean-context-default"


def _print_facts() -> None:
    from lean_context import storage
    facts = storage.get_all_facts()
    if not facts:
        print("  (no facts stored yet)")
    else:
        for k, v in facts.items():
            print(f"  {k}: {v}")


def run_repl() -> None:
    """
    Main REPL loop.  Builds the graph once, then invokes it per turn.
    Requirement 10.2: simple CLI for interactive use.
    """
    print(_BANNER)

    try:
        app = get_default_app()
    except Exception as exc:
        print(f"[ERROR] Failed to initialise pipeline: {exc}")
        sys.exit(1)

    thread_id = _THREAD_ID
    config = {"configurable": {"thread_id": thread_id}}

    # Seed state on first turn — checkpointer carries it forward afterwards
    _first_turn = True

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # Built-in slash commands (not routed through the graph)
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if user_input.lower() == "/facts":
            print("[Stored facts]")
            _print_facts()
            continue
        if user_input.lower() == "/clear":
            thread_id = f"lean-context-{id(user_input)}"
            config = {"configurable": {"thread_id": thread_id}}
            _first_turn = True
            print("[Conversation thread cleared]")
            continue

        # Build invocation payload
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

        try:
            result = app.invoke(payload, config=config)
            print(f"Bot: {result.get('final_output', '')}")
        except OllamaUnavailableError as exc:
            # Clear, actionable error message — not a raw stack trace (Req 7.2)
            print(
                "\n[ERROR] Local model unavailable.\n"
                "  → Make sure Ollama is running: ollama serve\n"
                "  → Make sure the model is pulled: ollama pull llama3.2:3b\n"
            )
        except Exception as exc:
            print(f"[ERROR] Unexpected error: {exc}")


if __name__ == "__main__":
    run_repl()
