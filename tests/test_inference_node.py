"""
tests/test_inference_node.py — Task 10

Unit tests for local_inference_node with Ollama mocked.
Covers: KEEP (RAM append), COMPRESS (FIFO eviction), OllamaUnavailableError.
"""

import pytest
from unittest.mock import patch, MagicMock

from lean_context.nodes import (
    local_inference_node,
    OllamaUnavailableError,
    RAM_BUFFER_CAP,
    GraphState,
)


def _make_state(user_input="What is 2+2?", lean_prompt="prompt text",
                ram=None) -> GraphState:
    return {
        "user_input": user_input,
        "routing_flag": "active",
        "ram_buffer": ram or [],
        "extracted_facts": {},
        "retrieved_facts": {},
        "lean_prompt": lean_prompt,
        "final_output": "",
    }


def _mock_ollama_response(text="Four."):
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: text if key == "response" else None
    return mock


class TestLocalInferenceNode:

    def test_returns_assistant_response(self):
        with patch("lean_context.nodes._ollama_client.generate", return_value={"response": "Four."}):
            result = local_inference_node(_make_state())
        assert result["final_output"] == "Four."

    def test_keep_appends_turn_to_ram(self):
        """KEEP lifecycle state 1: new turn appended to RAM buffer."""
        with patch("lean_context.nodes._ollama_client.generate", return_value={"response": "42"}):
            result = local_inference_node(_make_state(user_input="What is 6x7?"))
        assert len(result["ram_buffer"]) == 1
        assert result["ram_buffer"][0]["user"] == "What is 6x7?"
        assert result["ram_buffer"][0]["assistant"] == "42"

    def test_compress_evicts_oldest_when_over_cap(self):
        """COMPRESS lifecycle state 2: oldest turn evicted when buffer > RAM_BUFFER_CAP."""
        existing = [{"user": f"u{i}", "assistant": f"a{i}"} for i in range(RAM_BUFFER_CAP)]
        with patch("lean_context.nodes._ollama_client.generate", return_value={"response": "new answer"}):
            result = local_inference_node(_make_state(ram=existing))
        ram = result["ram_buffer"]
        assert len(ram) == RAM_BUFFER_CAP, f"RAM should be capped at {RAM_BUFFER_CAP}, got {len(ram)}"
        # Oldest turn (u0) must be gone
        assert ram[0]["user"] != "u0", "Oldest turn was not evicted"
        # Newest turn must be last
        assert ram[-1]["assistant"] == "new answer"

    def test_compress_keeps_exactly_cap_turns(self):
        """Buffer with exactly cap turns + new turn → still cap turns after."""
        existing = [{"user": f"q{i}", "assistant": f"r{i}"} for i in range(RAM_BUFFER_CAP)]
        with patch("lean_context.nodes._ollama_client.generate", return_value={"response": "resp"}):
            result = local_inference_node(_make_state(ram=existing))
        assert len(result["ram_buffer"]) == RAM_BUFFER_CAP

    def test_ollama_unavailable_raises_typed_error(self):
        """Req 7.2: connection failure → OllamaUnavailableError, not raw exception."""
        with patch("lean_context.nodes._ollama_client.generate", side_effect=Exception("connection refused")):
            with pytest.raises(OllamaUnavailableError):
                local_inference_node(_make_state())

    def test_response_stripped_of_whitespace(self):
        with patch("lean_context.nodes._ollama_client.generate", return_value={"response": "  answer \n"}):
            result = local_inference_node(_make_state())
        assert result["final_output"] == "answer"

    def test_empty_initial_ram_starts_at_one_after_turn(self):
        with patch("lean_context.nodes._ollama_client.generate", return_value={"response": "hi"}):
            result = local_inference_node(_make_state(ram=[]))
        assert len(result["ram_buffer"]) == 1
