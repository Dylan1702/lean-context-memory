"""
lean_context/config.py — Shared model settings used by every Ollama call site.

Import this before any ollama call so all three modes (lean / fifo / unmanaged)
use identical options and startup banners are consistent.
"""

from __future__ import annotations

import logging

# Silence ChromaDB telemetry and urllib3 retry noise for offline runs.
# These must be set before any chromadb / posthog import, which is
# guaranteed because config.py is imported first by every entry-point.
logging.getLogger("posthog").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("backoff").setLevel(logging.CRITICAL)

MODEL: str = "llama3.2:3b"
NUM_CTX: int = 8192       # context window passed to Ollama options
NUM_PREDICT: int = 80     # max tokens per reply

# Options dict ready to splat into every ollama generate call.
OLLAMA_OPTIONS: dict = {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}


def get_model_info() -> dict:
    """
    Query `ollama show <MODEL>` and return a dict with at least:
        max_context_length : int | None   (from modelinfo, if present)
        parameter_size     : str | None
        raw                : dict         (full show response)

    Returns gracefully if Ollama is unreachable — callers print "unknown".
    """
    try:
        import ollama as _ol
        info = _ol.show(MODEL)
        # ollama-python ≥0.4 returns an object with .modelinfo / .details attrs
        # Older versions return a plain dict.
        if hasattr(info, "model_info"):
            minfo = info.model_info or {}
        elif isinstance(info, dict):
            minfo = info.get("model_info") or info.get("modelinfo") or {}
        else:
            minfo = {}

        # Context length lives under various keys depending on architecture
        ctx = (
            minfo.get("llama.context_length")
            or minfo.get("context_length")
            or minfo.get("llm.context_length")
        )

        if hasattr(info, "details") and info.details:
            details = info.details
            psize = getattr(details, "parameter_size", None) or (
                details.get("parameter_size") if isinstance(details, dict) else None
            )
        elif isinstance(info, dict):
            psize = (info.get("details") or {}).get("parameter_size")
        else:
            psize = None

        return {"max_context_length": ctx, "parameter_size": psize, "raw": info}
    except Exception as exc:
        return {"max_context_length": None, "parameter_size": None, "raw": None, "error": str(exc)}


def print_startup_banner(extra_lines: list[str] | None = None) -> None:
    """
    Print the standard startup banner that every CLI / run_eval.py shows.
    Reads the model's declared maximum context length from ollama show.
    If prompt_eval_count > 0.9 * NUM_CTX on a given turn callers should
    print a WARNING themselves — this function only prints the static info.
    """
    info = get_model_info()
    max_ctx = info.get("max_context_length")
    max_ctx_str = str(max_ctx) if max_ctx else "unknown"
    psize = info.get("parameter_size") or "unknown"

    lines = [
        f"  model        : {MODEL}",
        f"  num_ctx      : {NUM_CTX}  (context window sent to Ollama)",
        f"  num_predict  : {NUM_PREDICT}  (max tokens per reply)",
        f"  model max ctx: {max_ctx_str}",
        f"  param size   : {psize}",
    ]
    if extra_lines:
        lines += extra_lines

    width = max(len(l) for l in lines) + 4
    bar = "═" * width
    print(f"\n╔{bar}╗")
    for l in lines:
        print(f"║  {l:<{width-2}}║")
    print(f"╚{bar}╝\n")
