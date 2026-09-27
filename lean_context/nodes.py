"""
nodes.py — LangGraph node implementations for the Lean-Context pipeline.

5-state memory lifecycle (design.md §The 5-State Memory Lifecycle):
  1. KEEP     — append incoming user+assistant turn to active RAM buffer
  2. COMPRESS — evict oldest RAM turn when buffer > 3 (FIFO)
  3. ARCHIVE  — write facts/narrative to SQLite or ChromaDB
  4. RETRIEVE — query SQLite + ChromaDB for context relevant to current query
  5. FORGET   — drop social filler before it reaches any storage tier

Node → lifecycle mapping (per design.md table):
  evict_node          → FORGET  (state 5)
  persist_node        → ARCHIVE (state 3)
  active_ram_node     → KEEP (append) + RETRIEVE (query both stores)
  assemble_context    → no lifecycle state; pure prompt builder
  local_inference_node → KEEP (final append) + COMPRESS (buffer trim)
"""

from __future__ import annotations

import re
import logging
from typing import TypedDict

import ollama as _ollama_client  # module-level import keeps mock target stable

from lean_context import storage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task 5 — Graph state schema (Requirement 8.1, design.md §7)
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
# Extraction regex (shared by persist_node and tests)
# ---------------------------------------------------------------------------

# Patterns tried in order; first match wins
_EXTRACTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    # "my <key> is <value>"
    ("kv", re.compile(r"\bmy\s+(\w+)\s+is\s+(.+)", re.I)),
    # "set <key> to <value>"
    ("kv", re.compile(r"\bset\s+(\w[\w_]*)\s+to\s+(.+)", re.I)),
    # "key: value"
    ("kv", re.compile(r"^(\w[\w_\s]*?)\s*:\s+(.+)$", re.M)),
    # "remember that <key> is <value>"
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
# Task 6 — Evict node: lifecycle state 5 — FORGET (Requirements 5.1, 5.2)
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
# Task 7 — Persist node: lifecycle state 3 — ARCHIVE (Requirements 3.1, 4.1)
# ---------------------------------------------------------------------------

def persist_node(state: GraphState) -> dict:
    """
    Lifecycle state 3 — ARCHIVE.

    1. Attempt structured key-value extraction.
       Match → write to SQLite via storage.set_fact  (Tier 2, Req 3.1).
    2. No structured match but content is substantive →
       write as narrative to ChromaDB                (Tier 3, Req 4.1).
    3. Falls through; active_ram_node runs next for conversational reply.

    'turn_index' is approximated from RAM buffer length so chunks carry a
    meaningful position even before the buffer is updated this turn.
    """
    text = state["user_input"]
    ram = state.get("ram_buffer", [])
    turn_index = len(ram)

    extracted: dict = {}
    kv = extract_kv(text)

    if kv:
        # ARCHIVE — structured fact to SQLite
        key, val = kv
        storage.set_fact(key, val)
        extracted = {key: val}
        logger.debug("persist_node: ARCHIVE SQLite — %s = %s", key, val)
    else:
        # No structured match — check if content is substantive
        # Heuristic: at least 6 words or contains a sentence-ending punctuation
        words = text.split()
        is_substantive = len(words) >= 6 or bool(re.search(r"[.!?]", text))
        if is_substantive:
            storage.add_narrative(text, turn_index=turn_index)
            logger.debug("persist_node: ARCHIVE ChromaDB narrative, turn=%d", turn_index)
        else:
            logger.debug("persist_node: content not substantive enough for storage")

    return {"extracted_facts": extracted}


# ---------------------------------------------------------------------------
# Task 8 — Active RAM node: lifecycle states 1 (KEEP) + 4 (RETRIEVE)
# (Requirements 2.1, 2.2, 3.3, 4.2)
# ---------------------------------------------------------------------------

def active_ram_node(state: GraphState) -> dict:
    """
    Lifecycle state 1 — KEEP (reads current RAM buffer).
    Lifecycle state 4 — RETRIEVE (queries SQLite + ChromaDB).

    Steps:
      1. Read the existing RAM buffer from state (KEEP — will be appended to
         after LLM inference in local_inference_node).
      2. Query SQLite for all stored facts (Req 3.3).
      3. Query ChromaDB for top-3 narrative chunks relevant to user_input (Req 4.2).
      4. Stage everything into retrieved_facts for use by assemble_context_node.

    The RAM buffer is NOT modified here — the append (KEEP) happens in
    local_inference_node after the assistant reply is generated, so we don't
    corrupt state on persist-routed turns that fall through here.
    """
    user_input = state["user_input"]
    ram = state.get("ram_buffer", [])

    # RETRIEVE — Tier 2: all SQLite facts
    facts: dict = storage.get_all_facts()
    logger.debug("active_ram_node: RETRIEVE SQLite — %d facts", len(facts))

    # RETRIEVE — Tier 3: top-3 ChromaDB narrative chunks
    narrative_chunks: list[str] = storage.query_narrative(user_input, n_results=3)
    logger.debug(
        "active_ram_node: RETRIEVE ChromaDB — %d chunks", len(narrative_chunks)
    )

    # Cap RAM to last RAM_BUFFER_CAP turns for context assembly
    capped_ram = ram[-RAM_BUFFER_CAP:] if len(ram) > RAM_BUFFER_CAP else ram

    retrieved_facts = {
        "facts": facts,
        "narrative": narrative_chunks,
        "capped_ram": capped_ram,
    }

    return {"retrieved_facts": retrieved_facts}


# ---------------------------------------------------------------------------
# Task 9 — Context assembly node (Requirements 6.1, 6.2)
# Prompt template: P = S_system + F_retrieved + M(t-N..t) + current query
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "You have access to stored facts and recent conversation history below. "
    "Answer the user's question using that context where relevant."
)


def assemble_context_node(state: GraphState) -> dict:
    """
    Build the minimal prompt: S_system + F_retrieved + M(t-N..t) + current query.

    Prompt token count is bounded by a constant independent of total turns t
    because:
      - System prompt is fixed.
      - Retrieved facts come from the (small, bounded) SQLite table.
      - Narrative chunks are capped at n_results=3.
      - RAM history is capped to last RAM_BUFFER_CAP (3) turns.
    (Requirement 6.1, 6.2)

    Assembled Prompt Layout (design.md §Context Assembly Node):
      [SYSTEM]
      <system instructions>

      [FACTS]
      key: value
      ...

      [RELEVANT CONTEXT]
      - chunk text
      ...

      [RECENT CONVERSATION]
      User: ...
      Assistant: ...
      ...

      [CURRENT QUERY]
      <user_input>
    """
    user_input = state["user_input"]
    retrieved = state.get("retrieved_facts", {})

    facts: dict = retrieved.get("facts", {})
    narrative: list[str] = retrieved.get("narrative", [])
    capped_ram: list[dict] = retrieved.get("capped_ram", [])

    # Enforce cap defensively even if active_ram_node was bypassed
    capped_ram = capped_ram[-RAM_BUFFER_CAP:]

    parts: list[str] = []

    # --- System ---
    parts.append(f"[SYSTEM]\n{_SYSTEM_PROMPT}")

    # --- Facts ---
    if facts:
        facts_lines = "\n".join(f"{k}: {v}" for k, v in facts.items())
        parts.append(f"[FACTS]\n{facts_lines}")

    # --- Narrative ---
    if narrative:
        narrative_lines = "\n".join(f"- {chunk}" for chunk in narrative)
        parts.append(f"[RELEVANT CONTEXT]\n{narrative_lines}")

    # --- Recent RAM turns ---
    if capped_ram:
        history_lines = []
        for turn in capped_ram:
            history_lines.append(f"User: {turn['user']}")
            history_lines.append(f"Assistant: {turn['assistant']}")
        parts.append(f"[RECENT CONVERSATION]\n" + "\n".join(history_lines))

    # --- Current query ---
    parts.append(f"[CURRENT QUERY]\n{user_input}")

    lean_prompt = "\n\n".join(parts)
    logger.debug(
        "assemble_context_node: prompt length=%d chars, ram_turns=%d",
        len(lean_prompt),
        len(capped_ram),
    )
    return {"lean_prompt": lean_prompt}


# ---------------------------------------------------------------------------
# Task 10 — Local inference node: lifecycle states 1 (KEEP) + 2 (COMPRESS)
# (Requirements 7.1, 7.2, 7.3, 2.2)
# ---------------------------------------------------------------------------

class OllamaUnavailableError(Exception):
    """Raised when the local Ollama service cannot be reached (Req 7.2)."""


def local_inference_node(state: GraphState) -> dict:
    """
    Lifecycle state 1 — KEEP  (append new turn to RAM buffer after generation).
    Lifecycle state 2 — COMPRESS (FIFO evict oldest turn when buffer > cap).

    Calls ollama.generate(model='llama3.2:3b', prompt=lean_prompt).
    On connection failure raises OllamaUnavailableError — caught at CLI boundary
    so the user sees a clear message, not a raw stack trace (Req 7.2).
    On success appends {user, assistant} to RAM buffer and trims to cap (Req 7.3).
    """
    lean_prompt = state.get("lean_prompt", "")
    user_input = state.get("user_input", "")
    ram = list(state.get("ram_buffer", []))

    try:
        response = _ollama_client.generate(
            model="llama3.2:3b",
            prompt=lean_prompt,
            options={"num_predict": 80},    # cap response length to keep per-turn time bounded
        )
        assistant_text: str = response["response"].strip()
    except Exception as exc:
        # Catches connection errors, missing model, etc.
        raise OllamaUnavailableError(
            "Local model unavailable — is `ollama serve` running?"
        ) from exc

    # KEEP — append new turn (lifecycle state 1)
    new_turn = {"user": user_input, "assistant": assistant_text}
    ram.append(new_turn)

    # COMPRESS — FIFO evict oldest turn if buffer exceeds cap (lifecycle state 2)
    if len(ram) > RAM_BUFFER_CAP:
        ram = ram[-RAM_BUFFER_CAP:]
        logger.debug("local_inference_node: COMPRESS — trimmed RAM to %d turns", RAM_BUFFER_CAP)

    logger.debug("local_inference_node: KEEP — RAM buffer now %d turns", len(ram))

    return {
        "final_output": assistant_text,
        "ram_buffer": ram,
    }
