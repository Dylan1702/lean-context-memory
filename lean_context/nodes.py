"""
nodes.py — LangGraph node implementations for the Lean-Context pipeline.

5-state memory lifecycle (design.md §The 5-State Memory Lifecycle):
  1. KEEP     — append incoming user+assistant turn to active RAM buffer
  2. COMPRESS — evict oldest RAM turn when buffer > 3 (FIFO)
  3. ARCHIVE  — write facts/narrative to SQLite or ChromaDB
  4. RETRIEVE — query SQLite + ChromaDB for context relevant to current query
  5. FORGET   — drop social filler before it reaches any storage tier

Node → lifecycle mapping (per design.md table):
  evict_node           → FORGET  (state 5)
  persist_node         → ARCHIVE (state 3)
  active_ram_node      → KEEP (append) + RETRIEVE (query both stores)
  assemble_context_node → no lifecycle state; pure prompt builder
  local_inference_node  → KEEP (final append) + COMPRESS (buffer trim)

METRICS
  Module-level counters accumulated across the lifetime of the process:
    METRICS["llm_calls"]           int   — successful Ollama calls
    METRICS["prompt_tokens"]       int   — sum of prompt_eval_count
    METRICS["completion_tokens"]   int   — sum of eval_count
    METRICS["model_ms"]            float — total ms inside Ollama
    METRICS["memory_ms"]           float — total ms in router+persist+active_ram+assemble
    METRICS["server_total_ms"]     float — sum of total_duration / 1e6
    METRICS["created_at"]          str|None — ISO timestamp of last call

  Use reset_metrics() to zero all counters, snapshot_metrics() to read a copy.

NONCE
  Every prompt assembled by assemble_context_node is prefixed with a unique
  per-turn request ID "[req N]\\n" so Ollama's KV-cache cannot carry over a
  cached prefix from a previous turn with the same text.  The nonce does not
  affect the semantic content visible to the model.

  Caching finding (measured on this machine, see git history):
    - Identical prompt sent twice: call 1 ≈74s, call 2 ≈10s.
    - prompt_eval_count is always the FULL count regardless of cache hit —
      so caching does NOT hide the true prompt size in the metric.
    - We add the nonce anyway to prevent accidental cross-turn cache hits
      from flattening generation-time measurements in the benchmark.
"""

from __future__ import annotations

import re
import time
import threading
import logging
from typing import TypedDict

import ollama as _ollama_client  # module-level import keeps mock target stable

from lean_context import storage
from lean_context.config import MODEL, OLLAMA_OPTIONS, NUM_CTX

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph state schema (Requirement 8.1, design.md §7)
# ---------------------------------------------------------------------------

RAM_BUFFER_CAP = 3  # Requirement 2.1


class GraphState(TypedDict):
    """
    Shared state propagated through every node in the LangGraph StateGraph.

    Fields mirror design.md §Graph State exactly:
      user_input      : raw text from the user this turn
      routing_flag    : "evict" | "persist" | "active" — set by cpu_router_edge
      ram_buffer      : list of {"user": str, "assistant": str} dicts (Tier 1)
      extracted_facts : {key: val} dict extracted by persist_node this turn
      retrieved_facts : {key: val} dict + narrative list assembled for context
      lean_prompt     : the assembled minimal prompt sent to the LLM
      final_output    : the assistant response returned to the user

    Note on ram_buffer: we use a plain list (no reducer) so nodes that return
    a new ram_buffer value replace the field entirely.  The inference node is
    the only node that mutates it — it returns the complete updated list.
    """

    user_input: str
    routing_flag: str
    ram_buffer: list[dict]    # plain replace semantics — inference node owns writes
    extracted_facts: dict
    retrieved_facts: dict
    lean_prompt: str
    final_output: str


# ---------------------------------------------------------------------------
# Module-level METRICS (thread-safe via lock)
# ---------------------------------------------------------------------------

_metrics_lock = threading.Lock()

METRICS: dict = {
    "llm_calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "model_ms": 0.0,
    "memory_ms": 0.0,
    "server_total_ms": 0.0,
    "created_at": None,
}


def reset_metrics() -> None:
    """Zero all METRICS counters.  Thread-safe."""
    with _metrics_lock:
        METRICS["llm_calls"] = 0
        METRICS["prompt_tokens"] = 0
        METRICS["completion_tokens"] = 0
        METRICS["model_ms"] = 0.0
        METRICS["memory_ms"] = 0.0
        METRICS["server_total_ms"] = 0.0
        METRICS["created_at"] = None


def snapshot_metrics() -> dict:
    """Return a shallow copy of METRICS.  Thread-safe."""
    with _metrics_lock:
        return dict(METRICS)


def _update_metrics_from_response(resp: dict, model_ms: float) -> dict:
    """
    Extract per-call metrics from an Ollama response dict and accumulate into
    the module-level METRICS.  Returns a per-call dict of the same shape for
    use by callers that want per-turn values.

    Missing fields are treated as 0 / None (tests pass plain dicts that omit
    server fields).
    """
    prompt_tok  = int(resp.get("prompt_eval_count") or 0)
    compl_tok   = int(resp.get("eval_count") or 0)
    # total_duration is in nanoseconds
    server_ms   = float((resp.get("total_duration") or 0)) / 1e6
    created_at  = resp.get("created_at") or None

    per_call = {
        "prompt_tokens":    prompt_tok,
        "completion_tokens": compl_tok,
        "model_ms":         model_ms,
        "server_total_ms":  server_ms,
        "created_at":       created_at,
    }

    with _metrics_lock:
        METRICS["llm_calls"]         += 1
        METRICS["prompt_tokens"]     += prompt_tok
        METRICS["completion_tokens"] += compl_tok
        METRICS["model_ms"]          += model_ms
        METRICS["server_total_ms"]   += server_ms
        if created_at:
            METRICS["created_at"] = created_at

    # Warn if prompt is close to context-window limit
    if prompt_tok > 0 and prompt_tok > 0.9 * NUM_CTX:
        logger.warning(
            "WARNING: prompt_eval_count=%d > 90%% of NUM_CTX=%d — prompt may be truncated",
            prompt_tok, NUM_CTX,
        )

    return per_call


# ---------------------------------------------------------------------------
# Per-turn nonce counter
# ---------------------------------------------------------------------------
_nonce_counter = 0
_nonce_lock = threading.Lock()


def _next_nonce() -> str:
    global _nonce_counter
    with _nonce_lock:
        _nonce_counter += 1
        return f"[req {_nonce_counter}]"


# ---------------------------------------------------------------------------
# Extraction regex (shared by persist_node and tests)
# ---------------------------------------------------------------------------

_EXTRACTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("kv", re.compile(r"\bmy\s+((?:\w+\s+){0,2}?\w+)\s+is\s+(.+)", re.I)),
    ("kv", re.compile(r"\bset\s+(\w[\w_]*)\s+to\s+(.+)", re.I)),
    ("kv", re.compile(r"^(\w[\w_\s]*?)\s*:\s+(.+)$", re.M)),
    ("kv", re.compile(r"\bremember\s+that\s+(?:my\s+)?(\w[\w_\s]*?)\s+is\s+(.+)", re.I)),
]


def extract_kv(text: str) -> tuple[str, str] | None:
    """
    Attempt structured key-value extraction from *text*.
    Returns (key, value) on match, or None if no pattern matches.
    Used by persist_node (ARCHIVE lifecycle state 3).
    """
    for _kind, pattern in _EXTRACTION_PATTERNS:
        m = pattern.search(text)
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            val = m.group(2).strip().rstrip(".")
            if key and val:
                return key, val
    return None


# ---------------------------------------------------------------------------
# Evict node — lifecycle state 5: FORGET (Req 5.1, 5.2)
# ---------------------------------------------------------------------------

_EVICT_ACKNOWLEDGMENTS = [
    "Got it!",
    "Sure thing.",
    "Understood.",
    "OK!",
    "Noted.",
]
_ack_index = 0


def evict_node(state: GraphState) -> dict:
    """
    Lifecycle state 5 — FORGET.

    Returns a canned acknowledgment. Never touches RAM buffer, SQLite,
    or ChromaDB (Requirement 5.1). Does NOT call the LLM (Requirement 5.2).
    """
    global _ack_index
    ack = _EVICT_ACKNOWLEDGMENTS[_ack_index % len(_EVICT_ACKNOWLEDGMENTS)]
    _ack_index += 1
    logger.debug("evict_node: FORGET — returning canned ack")
    return {
        "routing_flag": "evict",
        "final_output": ack,
    }


# ---------------------------------------------------------------------------
# Persist node — lifecycle state 3: ARCHIVE (Req 3.1, 4.1)
# ---------------------------------------------------------------------------

def persist_node(state: GraphState) -> dict:
    """
    Lifecycle state 3 — ARCHIVE.

    1. Attempt structured key-value extraction.
       Match → write to SQLite via storage.set_fact  (Tier 2, Req 3.1).
    2. No structured match but content is substantive →
       write as narrative to ChromaDB                (Tier 3, Req 4.1).
    3. Falls through; active_ram_node runs next for conversational reply.
    """
    _t0 = time.perf_counter()
    text = state["user_input"]
    ram = state.get("ram_buffer", [])
    turn_index = len(ram)

    extracted: dict = {}
    kv = extract_kv(text)

    if kv:
        key, val = kv
        storage.set_fact(key, val)
        extracted = {key: val}
        logger.debug("persist_node: ARCHIVE SQLite — %s = %s", key, val)
    else:
        words = text.split()
        is_substantive = len(words) >= 6 or bool(re.search(r"[.!?]", text))
        if is_substantive:
            storage.add_narrative(text, turn_index=turn_index)
            logger.debug("persist_node: ARCHIVE ChromaDB narrative, turn=%d", turn_index)

    with _metrics_lock:
        METRICS["memory_ms"] += (time.perf_counter() - _t0) * 1000

    return {"extracted_facts": extracted}


# ---------------------------------------------------------------------------
# Active RAM node — lifecycle states 1 (KEEP) + 4 (RETRIEVE)
# (Req 2.1, 2.2, 3.3, 4.2)
# ---------------------------------------------------------------------------

def active_ram_node(state: GraphState) -> dict:
    """
    Lifecycle state 1 — KEEP (reads current RAM buffer).
    Lifecycle state 4 — RETRIEVE (queries SQLite + ChromaDB).
    """
    _t0 = time.perf_counter()
    user_input = state["user_input"]
    ram = state.get("ram_buffer", [])

    facts: dict = storage.get_all_facts()
    logger.debug("active_ram_node: RETRIEVE SQLite — %d facts", len(facts))

    narrative_chunks: list[str] = storage.query_narrative(user_input, n_results=3)
    logger.debug("active_ram_node: RETRIEVE ChromaDB — %d chunks", len(narrative_chunks))

    capped_ram = ram[-RAM_BUFFER_CAP:] if len(ram) > RAM_BUFFER_CAP else ram

    retrieved_facts = {
        "facts": facts,
        "narrative": narrative_chunks,
        "capped_ram": capped_ram,
    }

    with _metrics_lock:
        METRICS["memory_ms"] += (time.perf_counter() - _t0) * 1000

    return {"retrieved_facts": retrieved_facts}


# ---------------------------------------------------------------------------
# Context assembly node (Req 6.1, 6.2)
# Prompt template: P = [req N] + S_system + F_retrieved + M(t-N..t) + query
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = "You are a helpful, concise assistant."


def assemble_context_node(state: GraphState) -> dict:
    """
    Build the minimal prompt: nonce + S_system + F_retrieved + M(t-N..t) + query.

    The nonce "[req N]" is prepended so Ollama's KV-cache cannot carry over a
    cached prefix from a previous identical prompt.  See module docstring for
    the caching finding that motivated this choice.

    Prompt layout:
      [req N]          ← unique per-turn nonce (cache-buster)
      [SYSTEM]         ← fixed system prompt
      [FACTS]          ← SQLite key-value facts (omitted if empty)
      [RELEVANT CONTEXT] ← ChromaDB narrative chunks (omitted if empty)
      [RECENT CONVERSATION] ← last ≤3 RAM turns (omitted if empty)
      [CURRENT QUERY]  ← user_input this turn
    """
    _t0 = time.perf_counter()
    user_input = state["user_input"]
    retrieved = state.get("retrieved_facts", {})

    facts: dict = retrieved.get("facts", {})
    narrative: list[str] = retrieved.get("narrative", [])
    capped_ram: list[dict] = retrieved.get("capped_ram", [])
    capped_ram = capped_ram[-RAM_BUFFER_CAP:]

    nonce = _next_nonce()
    parts: list[str] = [nonce, f"[SYSTEM]\n{_SYSTEM_PROMPT}"]

    if facts:
        parts.append("[FACTS]\n" + "\n".join(f"{k}: {v}" for k, v in facts.items()))

    if narrative:
        parts.append("[RELEVANT CONTEXT]\n" + "\n".join(f"- {c}" for c in narrative))

    if capped_ram:
        lines = []
        for turn in capped_ram:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        parts.append("[RECENT CONVERSATION]\n" + "\n".join(lines))

    parts.append(f"[CURRENT QUERY]\n{user_input}")

    lean_prompt = "\n\n".join(parts)
    logger.debug(
        "assemble_context_node: nonce=%s prompt=%d chars ram=%d",
        nonce, len(lean_prompt), len(capped_ram),
    )

    with _metrics_lock:
        METRICS["memory_ms"] += (time.perf_counter() - _t0) * 1000

    return {"lean_prompt": lean_prompt}


# ---------------------------------------------------------------------------
# Local inference node — lifecycle states 1 (KEEP) + 2 (COMPRESS)
# (Req 7.1, 7.2, 7.3, 2.2)
# ---------------------------------------------------------------------------

class OllamaUnavailableError(Exception):
    """Raised when the local Ollama service cannot be reached (Req 7.2)."""


_INFERENCE_TIMEOUT_S = 60
_INFERENCE_RETRIES = 2


def _generate_with_timeout(prompt: str) -> tuple[str, dict]:
    """
    Call Ollama with a hard timeout and retry.

    Returns (response_text, per_call_metrics_dict).
    Raises OllamaUnavailableError after all retries are exhausted.

    Keeps calling the module-level `_ollama_client.generate` (not a Client
    instance) so existing tests that patch that name continue to work.
    The call runs in a worker thread so we can apply a hard timeout.
    """
    import concurrent.futures

    last_exc: Exception | None = None

    for attempt in range(1, _INFERENCE_RETRIES + 1):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        t0 = time.perf_counter()
        try:
            future = pool.submit(
                _ollama_client.generate,
                model=MODEL,
                prompt=prompt,
                options=OLLAMA_OPTIONS,
            )
            response = future.result(timeout=_INFERENCE_TIMEOUT_S)
            model_ms = (time.perf_counter() - t0) * 1000
            text = response["response"].strip()
            per_call = _update_metrics_from_response(response, model_ms)
            return text, per_call
        except Exception as exc:
            last_exc = exc
            if attempt < _INFERENCE_RETRIES:
                backoff = 2 ** attempt
                logger.warning(
                    "local_inference_node: attempt %d/%d failed (%s). Retrying in %ds…",
                    attempt, _INFERENCE_RETRIES, type(exc).__name__, backoff,
                )
                time.sleep(backoff)
        finally:
            pool.shutdown(wait=False)

    raise OllamaUnavailableError(
        "Local model unavailable — is `ollama serve` running?"
    ) from last_exc


def local_inference_node(state: GraphState) -> dict:
    """
    Lifecycle state 1 — KEEP  (append new turn to RAM buffer after generation).
    Lifecycle state 2 — COMPRESS (FIFO evict oldest turn when buffer > cap).
    """
    lean_prompt = state.get("lean_prompt", "")
    user_input = state.get("user_input", "")
    ram = list(state.get("ram_buffer", []))

    assistant_text, _per_call = _generate_with_timeout(lean_prompt)

    # KEEP
    ram.append({"user": user_input, "assistant": assistant_text})

    # COMPRESS
    if len(ram) > RAM_BUFFER_CAP:
        ram = ram[-RAM_BUFFER_CAP:]
        logger.debug("local_inference_node: COMPRESS — RAM trimmed to %d turns", RAM_BUFFER_CAP)

    logger.debug("local_inference_node: KEEP — RAM buffer now %d turns", len(ram))

    return {
        "final_output": assistant_text,
        "ram_buffer": ram,
    }
