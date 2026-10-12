"""
bench/cli_unmanaged.py — Interactive REPL running the unmanaged baseline.

Full conversation history appended to every prompt; unbounded growth.
Use this to demonstrate the prompt-growth problem that Lean-Context solves.

Usage:
    python bench/cli_unmanaged.py [--no-offline-guard]

Commands:
    quit / exit        — exit
    /recall list       — list facts stated this session (harness only)
    /recall <key>      — ask the model about a fact and score the reply
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lean_context.config import MODEL, OLLAMA_OPTIONS, NUM_CTX, print_startup_banner
from lean_context.nodes import extract_kv
from bench.run_benchmark import _call_ollama, _SYSTEM_PROMPT, _next_nonce
from bench.rouge import rouge_l_recall


def _handle_recall_unmanaged(
    cmd: str,
    session_facts: dict[str, str],
    full_history: list[dict],
) -> None:
    parts = cmd.strip().split(None, 1)
    sub = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("list", ""):
        if not session_facts:
            print("  (no facts recorded this session)")
        else:
            print("  [Session facts]")
            for k, v in session_facts.items():
                print(f"    {k}: {v}")
        return

    key_norm = sub.lower().replace(" ", "_")
    if key_norm not in session_facts:
        matches = [k for k in session_facts if k.startswith(key_norm)]
        if len(matches) == 1:
            key_norm = matches[0]
        else:
            print(f"  Key '{sub}' not in session facts: {list(session_facts)}")
            return

    expected = session_facts[key_norm]
    key_display = key_norm.replace("_", " ")
    question = f"What is my {key_display}? Answer in one short sentence."
    print(f"  [recall] Asking: {question!r}")

    nonce = _next_nonce()
    parts_p = [nonce, f"[SYSTEM]\n{_SYSTEM_PROMPT}"]
    if full_history:
        hist = "\n".join(
            f"User: {t['user']}\nAssistant: {t['assistant']}"
            for t in full_history
        )
        parts_p.append(f"[CONVERSATION HISTORY]\n{hist}")
    parts_p.append(f"[CURRENT QUERY]\n{question}")
    prompt = "\n\n".join(parts_p)

    t0 = time.perf_counter()
    reply, meta = _call_ollama(prompt)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    found = expected.lower() in reply.lower()
    score = rouge_l_recall(reply, expected)
    print(f"  Bot: {reply}")
    print(
        f"  Expected: {expected!r} | Found: {'YES ✓' if found else 'NO  ✗'} | "
        f"ROUGE-L: {score:.2f} | {elapsed_ms:.0f} ms"
    )


def run_repl(no_offline_guard: bool = False) -> None:
    if not no_offline_guard:
        from lean_context.offline_guard import enable as _eg
        _eg()

    import ollama as _ol
    try:
        _ol.list()
    except Exception:
        print("[ERROR] Ollama not reachable — run: ollama serve")
        sys.exit(1)

    print_startup_banner([
        "  mode         : unmanaged (full history, unbounded)",
        "  commands     : /recall  quit",
    ])

    full_history: list[dict] = []
    session_facts: dict[str, str] = {}
    turn_number = 0

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
        if low.startswith("/recall"):
            _handle_recall_unmanaged(user_input, session_facts, full_history)
            continue

        turn_number += 1
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
        elapsed_wall_ms = (time.perf_counter() - t_wall) * 1000

        full_history.append({"user": user_input, "assistant": reply})

        kv = extract_kv(user_input)
        if kv:
            session_facts[kv[0]] = kv[1]

        print(f"Bot: {reply}")
        ctx_pct = (
            f"{meta['prompt_tokens']/NUM_CTX*100:.0f}%"
            if meta["prompt_tokens"] else "—"
        )
        n_hist = len(full_history)
        print(
            f"  [turn {turn_number}] "
            f"tokens in={meta['prompt_tokens']} out={meta['completion_tokens']} | "
            f"model={meta['model_ms']:.0f} ms | total={elapsed_wall_ms:.0f} ms | "
            f"ctx={ctx_pct} | history={n_hist} turns"
        )
        if meta["prompt_tokens"] and meta["prompt_tokens"] > 0.9 * NUM_CTX:
            print(f"  [WARNING] prompt tokens > 90% of NUM_CTX={NUM_CTX}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-offline-guard", action="store_true")
    args = p.parse_args()
    run_repl(no_offline_guard=args.no_offline_guard)


if __name__ == "__main__":
    main()
