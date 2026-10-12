"""tests/test_metrics.py — Tests for the METRICS system in nodes.py."""

import pytest
from unittest.mock import patch

from lean_context.nodes import (
    METRICS, reset_metrics, snapshot_metrics,
    _update_metrics_from_response, _generate_with_timeout,
    OllamaUnavailableError,
)


def test_reset_metrics_zeros_all_counters():
    METRICS["llm_calls"] = 99
    METRICS["prompt_tokens"] = 500
    reset_metrics()
    assert METRICS["llm_calls"] == 0
    assert METRICS["prompt_tokens"] == 0
    assert METRICS["completion_tokens"] == 0
    assert METRICS["model_ms"] == 0.0
    assert METRICS["memory_ms"] == 0.0
    assert METRICS["server_total_ms"] == 0.0
    assert METRICS["created_at"] is None


def test_snapshot_returns_copy():
    reset_metrics()
    snap = snapshot_metrics()
    snap["llm_calls"] = 999          # mutating snapshot must not affect METRICS
    assert METRICS["llm_calls"] == 0


def test_update_metrics_from_plain_dict():
    """Tests must pass plain dicts (no server fields) — missing fields treated as 0."""
    reset_metrics()
    # Plain dict as returned by mock in tests
    resp = {"response": "hello"}
    per_call = _update_metrics_from_response(resp, model_ms=1500.0)
    assert per_call["prompt_tokens"] == 0
    assert per_call["completion_tokens"] == 0
    assert per_call["model_ms"] == 1500.0
    assert per_call["server_total_ms"] == 0.0
    assert per_call["created_at"] is None
    assert METRICS["llm_calls"] == 1
    assert METRICS["model_ms"] == 1500.0


def test_update_metrics_from_full_response():
    reset_metrics()
    resp = {
        "response": "some text",
        "prompt_eval_count": 42,
        "eval_count": 15,
        "total_duration": 5_000_000_000,   # 5 seconds in ns
        "created_at": "2026-01-01T00:00:00Z",
    }
    per_call = _update_metrics_from_response(resp, model_ms=5000.0)
    assert per_call["prompt_tokens"] == 42
    assert per_call["completion_tokens"] == 15
    assert abs(per_call["server_total_ms"] - 5000.0) < 1.0
    assert per_call["created_at"] == "2026-01-01T00:00:00Z"
    assert METRICS["prompt_tokens"] == 42
    assert METRICS["completion_tokens"] == 15
    assert METRICS["created_at"] == "2026-01-01T00:00:00Z"


def test_metrics_accumulate_across_calls():
    reset_metrics()
    _update_metrics_from_response({"response": "a", "prompt_eval_count": 10, "eval_count": 5}, 100.0)
    _update_metrics_from_response({"response": "b", "prompt_eval_count": 20, "eval_count": 8}, 200.0)
    assert METRICS["llm_calls"] == 2
    assert METRICS["prompt_tokens"] == 30
    assert METRICS["completion_tokens"] == 13
    assert abs(METRICS["model_ms"] - 300.0) < 1.0


def test_generate_with_timeout_accumulates_metrics():
    reset_metrics()
    with patch(
        "lean_context.nodes._ollama_client.generate",
        return_value={"response": "answer", "prompt_eval_count": 50, "eval_count": 10},
    ):
        text, per_call = _generate_with_timeout("test prompt")
    assert text == "answer"
    assert METRICS["llm_calls"] == 1
    assert METRICS["prompt_tokens"] == 50
    assert per_call["prompt_tokens"] == 50


def test_generate_with_timeout_raises_on_failure():
    reset_metrics()
    with patch(
        "lean_context.nodes._ollama_client.generate",
        side_effect=ConnectionError("refused"),
    ):
        with pytest.raises(OllamaUnavailableError):
            _generate_with_timeout("prompt")
    # llm_calls should NOT be incremented on failure
    assert METRICS["llm_calls"] == 0
