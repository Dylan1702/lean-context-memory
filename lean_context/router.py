"""
router.py — Zero-LLM input router (Requirement 1).

cpu_router_edge classifies every incoming user message as one of:
  - "evict"   → social filler; FORGET lifecycle state (Req 1.2, 5.1)
  - "persist" → explicit fact/assignment; ARCHIVE lifecycle state (Req 1.3)
  - "active"  → substantive query; KEEP + RETRIEVE lifecycle states (Req 1.4)

Pure function: no I/O, no LLM call, no side effects.
Classification must complete in well under 5 ms on commodity CPU (Req 1.5).
"""

import re
import time
from typing import Literal

# ---------------------------------------------------------------------------
# Evict patterns — social filler / zero-entropy turns (Req 1.2)
# ---------------------------------------------------------------------------

# Stop-list: single words that carry no information
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "hi", "hey", "hello", "hiya", "howdy",
        "bye", "goodbye", "cya", "ttyl",
        "ok", "okay", "k", "kk",
        "yes", "yeah", "yep", "yup", "sure", "alright",
        "no", "nope", "nah",
        "thanks", "thank", "thx", "ty",
        "please", "pls", "plz",
        "great", "cool", "nice", "good", "awesome", "wonderful",
        "wow", "oh", "ah", "hmm", "hm", "uh", "um",
        "lol", "haha", "hehe", "lmao",
        "np", "welcome",
    }
)

# Phrases that are always evict regardless of length
_EVICT_PHRASES: tuple[re.Pattern, ...] = (
    re.compile(r"^\s*(good\s+)?(morning|afternoon|evening|night)\s*[!.?]*\s*$", re.I),
    re.compile(r"^\s*thank(s|\s+you)\s*[!.?]*\s*$", re.I),
    re.compile(r"^\s*(you['']re\s+)?welcome\s*[!.?]*\s*$", re.I),
    re.compile(r"^\s*(that['']?s?\s+)?(great|perfect|awesome|cool|nice|wonderful)\s*[!.?]*\s*$", re.I),
    re.compile(r"^\s*(got\s+it|makes?\s+sense|i\s+see|understood)\s*[!.?]*\s*$", re.I),
    re.compile(r"^\s*(ok|okay|k)\s*[!.?]*\s*$", re.I),
)

# ---------------------------------------------------------------------------
# Persist patterns — explicit fact / assignment (Req 1.3)
# ---------------------------------------------------------------------------

_PERSIST_PATTERNS: tuple[re.Pattern, ...] = (
    # "remember that ..."
    re.compile(r"\bremember\s+that\b", re.I),
    # "my <noun> is <value>"  /  "my name is Alice"
    re.compile(r"\bmy\s+\w+\s+is\b", re.I),
    # "set <key> to <value>"
    re.compile(r"\bset\s+\w[\w\s]*\s+to\b", re.I),
    # "key: value"  (colon-assignment)
    re.compile(r"\b\w[\w\s]*:\s+\S", re.I),
    # "store / save / note / keep / log that ..."
    re.compile(r"\b(store|save|note|keep|log)\s+(that|this|the)\b", re.I),
    # "I am / I'm a <noun>"  e.g. "I am a researcher"
    re.compile(r"\bi'?m\s+(?:a\s+|an\s+)?\w+", re.I),
    # "my <fact-noun> is/was/will be <value>"
    # Requires a verb after the keyword so bare queries ("tell me about my project")
    # don't match.
    re.compile(
        r"\bmy\s+(preference|goal|target|objective|project|dataset|parameter)"
        r"\s+(?:is|was|will\s+be|=)\b",
        re.I,
    ),
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cpu_router_edge(text: str) -> Literal["evict", "persist", "active"]:
    """
    Classify *text* as 'evict', 'persist', or 'active' using only
    regex/lexical rules — no model call, no I/O (Requirement 1.1).

    Order of checks (cheapest first per design.md §Router):
      1. Evict: stop-word match or phrase match  → FORGET
      2. Persist: assignment-pattern match       → ARCHIVE
      3. Active: everything else                 → KEEP + RETRIEVE
    """
    stripped = text.strip()

    # ------------------------------------------------------------------
    # 1. Evict — social filler
    # ------------------------------------------------------------------
    if _is_evict(stripped):
        return "evict"

    # ------------------------------------------------------------------
    # 2. Persist — explicit fact assignment
    # ------------------------------------------------------------------
    if _is_persist(stripped):
        return "persist"

    # ------------------------------------------------------------------
    # 3. Active — substantive query (default)
    # ------------------------------------------------------------------
    return "active"


def _is_evict(text: str) -> bool:
    """Return True if *text* is social filler with no information value."""
    if not text:
        return True  # empty string → nothing to remember

    # Punctuation-only (no alphabetic content) → treat as empty/filler
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return True

    # Word-count ≤ 2 and every word is in the stop-list
    if len(words) <= 2 and all(w in _STOP_WORDS for w in words):
        return True

    # Phrase-level patterns
    for pattern in _EVICT_PHRASES:
        if pattern.match(text):
            return True

    return False


def _is_persist(text: str) -> bool:
    """Return True if *text* contains an explicit fact/assignment pattern."""
    for pattern in _PERSIST_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Timing utility (used in tests to verify <5ms budget)
# ---------------------------------------------------------------------------

def timed_route(text: str) -> tuple[Literal["evict", "persist", "active"], float]:
    """Return (route, elapsed_ms). Used only in tests/benchmarks."""
    t0 = time.perf_counter()
    result = cpu_router_edge(text)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms
