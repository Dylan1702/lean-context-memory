"""
tests/test_router.py

Table-driven unit tests for cpu_router_edge (Requirement 1).
Covers: evict/persist/active classification, edge cases, and the 5 ms timing budget.
"""

import pytest
from lean_context.router import cpu_router_edge, timed_route

# ---------------------------------------------------------------------------
# Parametrised classification table
# Format: (input_text, expected_route, description)
# ---------------------------------------------------------------------------

EVICT_CASES = [
    # --- stop-list single words ---
    ("hi", "evict", "single stop-word 'hi'"),
    ("Hey", "evict", "stop-word case-insensitive"),
    ("ok", "evict", "stop-word 'ok'"),
    ("thanks", "evict", "stop-word 'thanks'"),
    ("lol", "evict", "stop-word 'lol'"),
    # --- stop-list two-word combos ---
    ("ok thanks", "evict", "two stop-words"),
    ("yes please", "evict", "two stop-words 2"),
    # --- phrase patterns ---
    ("Good morning!", "evict", "good morning phrase"),
    ("Good evening", "evict", "good evening phrase"),
    ("Thank you!", "evict", "thank you phrase"),
    ("You're welcome", "evict", "you're welcome phrase"),
    ("That's great!", "evict", "that's great phrase"),
    ("Got it", "evict", "got it phrase"),
    ("Makes sense", "evict", "makes sense phrase"),
    ("I see", "evict", "i see phrase"),
    # --- edge cases ---
    ("", "evict", "empty string → evict"),
    ("   ", "evict", "whitespace-only → evict (stripped to empty)"),
    ("OK.", "evict", "ok with punctuation"),
    ("ok!", "evict", "ok with exclamation"),
]

PERSIST_CASES = [
    ("My name is Alice", "persist", "my X is Y pattern"),
    ("my project is lean-context", "persist", "my project keyword"),
    ("my goal is to reduce latency", "persist", "my goal keyword"),
    ("set temperature to 0.7", "persist", "set X to Y pattern"),
    ("Set threshold to 10ms", "persist", "set X to Y, capitalised"),
    ("remember that my deadline is Friday", "persist", "remember that pattern"),
    ("dataset: ImageNet", "persist", "colon-assignment pattern"),
    ("learning_rate: 0.001", "persist", "underscore key colon pattern"),
    ("store that the batch size is 32", "persist", "store that pattern"),
    ("save this for later: API key is xyz", "persist", "save this pattern"),
    ("I'm a researcher in NLP", "persist", "I'm a <noun> pattern"),
    ("my dataset is MNIST", "persist", "my dataset keyword"),
    ("my parameter is 42", "persist", "my parameter keyword"),
    ("note that the target is 95% accuracy", "persist", "note that pattern"),
    ("my project is lean-context memory", "persist", "my project IS pattern (verb present)"),
    ("my dataset is CIFAR-10", "persist", "my dataset IS pattern"),
]

ACTIVE_CASES = [
    ("What is the capital of France?", "active", "factual question"),
    ("How does attention mechanism work?", "active", "technical question"),
    ("Explain backpropagation to me", "active", "explanation request"),
    ("What did I say my name was?", "active", "recall question (no assignment)"),
    ("Summarise our conversation so far", "active", "summarise request"),
    ("Can you help me debug this code?", "active", "debug request"),
    ("Tell me about my project", "active", "query about stored info — no verb after 'project'"),
    ("What is 2 + 2?", "active", "simple arithmetic"),
    ("Write a haiku about memory", "active", "creative request"),
    ("hello there how are you doing today", "active", "long greeting is active (>2 words with non-stop content)"),
]


@pytest.mark.parametrize("text,expected,desc", EVICT_CASES)
def test_evict_classification(text, expected, desc):
    assert cpu_router_edge(text) == expected, f"FAILED: {desc!r} → expected {expected}"


@pytest.mark.parametrize("text,expected,desc", PERSIST_CASES)
def test_persist_classification(text, expected, desc):
    assert cpu_router_edge(text) == expected, f"FAILED: {desc!r} → expected {expected}"


@pytest.mark.parametrize("text,expected,desc", ACTIVE_CASES)
def test_active_classification(text, expected, desc):
    assert cpu_router_edge(text) == expected, f"FAILED: {desc!r} → expected {expected}"


# ---------------------------------------------------------------------------
# Edge cases: punctuation-only, mixed-case, unicode-adjacent
# ---------------------------------------------------------------------------

def test_punctuation_only():
    """A string of only punctuation has no words → treated as empty → evict."""
    assert cpu_router_edge("!!!") == "evict"

def test_mixed_case_persist():
    assert cpu_router_edge("MY NAME IS BOB") == "persist"

def test_mixed_case_evict():
    assert cpu_router_edge("OK") == "evict"

def test_leading_trailing_whitespace():
    assert cpu_router_edge("  hi  ") == "evict"
    assert cpu_router_edge("  my name is Dave  ") == "persist"

def test_persist_takes_priority_over_short_text():
    """Short text with a persist pattern should NOT be evicted."""
    assert cpu_router_edge("set x to 1") == "persist"


# ---------------------------------------------------------------------------
# Timing budget: each classification must complete in < 5 ms (Req 1.5)
# We test across a representative set of inputs and take the worst case.
# ---------------------------------------------------------------------------

_TIMING_INPUTS = [
    "hi",
    "Good morning!",
    "my name is Alice and I want to remember my project deadline is Friday",
    "Explain how transformers work in natural language processing",
    "set learning_rate to 0.0001",
    "",
    "What is the meaning of life, the universe, and everything?",
]

@pytest.mark.parametrize("text", _TIMING_INPUTS)
def test_classification_under_5ms(text):
    """Requirement 1.5: classification must complete in < 5 ms."""
    _, elapsed_ms = timed_route(text)
    assert elapsed_ms < 5.0, (
        f"Router took {elapsed_ms:.3f} ms for {repr(text)[:40]!s} — exceeds 5 ms budget"
    )
