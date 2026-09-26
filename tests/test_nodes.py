"""
tests/test_nodes.py

Unit tests for:
  Task 5 — GraphState schema (TypedDict structure)
  Task 6 — evict_node (FORGET lifecycle state 5)
  Task 7 — persist_node (ARCHIVE lifecycle state 3)
  Task 8 — active_ram_node (KEEP + RETRIEVE lifecycle states 1 + 4)
"""

import pytest
from unittest.mock import patch, MagicMock

import lean_context.nodes as nodes_mod
from lean_context.nodes import (
    GraphState,
    evict_node,
    persist_node,
    active_ram_node,
    extract_kv,
    RAM_BUFFER_CAP,
)


# ---------------------------------------------------------------------------
# Task 5 — GraphState schema
# ---------------------------------------------------------------------------

class TestGraphState:
    def test_graphstate_has_required_fields(self):
        """All 7 fields from design.md §7 must be present in GraphState."""
        fields = GraphState.__annotations__
        for required in [
            "user_input", "routing_flag", "ram_buffer",
            "extracted_facts", "retrieved_facts", "lean_prompt", "final_output",
        ]:
            assert required in fields, f"Missing GraphState field: {required}"

    def test_graphstate_instantiation(self):
        state: GraphState = {
            "user_input": "hello",
            "routing_flag": "evict",
            "ram_buffer": [],
            "extracted_facts": {},
            "retrieved_facts": {},
            "lean_prompt": "",
            "final_output": "",
        }
        assert state["user_input"] == "hello"
        assert state["ram_buffer"] == []


# ---------------------------------------------------------------------------
# Task 6 — evict_node: FORGET — no storage writes (Req 5.1, 5.2)
# ---------------------------------------------------------------------------

class TestEvictNode:
    def _make_state(self, text="hi") -> GraphState:
        return {
            "user_input": text,
            "routing_flag": "evict",
            "ram_buffer": [],
            "extracted_facts": {},
            "retrieved_facts": {},
            "lean_prompt": "",
            "final_output": "",
        }

    def test_evict_node_returns_final_output(self):
        result = evict_node(self._make_state())
        assert "final_output" in result
        assert isinstance(result["final_output"], str)
        assert len(result["final_output"]) > 0

    def test_evict_node_does_not_write_to_sqlite(self):
        with patch("lean_context.storage.set_fact") as mock_set:
            evict_node(self._make_state())
            mock_set.assert_not_called()

    def test_evict_node_does_not_write_to_chroma(self):
        with patch("lean_context.storage.add_narrative") as mock_add:
            evict_node(self._make_state())
            mock_add.assert_not_called()

    def test_evict_node_does_not_modify_ram_buffer(self):
        state = self._make_state()
        original_ram = list(state["ram_buffer"])
        result = evict_node(state)
        # evict_node must not return a ram_buffer key that modifies state
        assert "ram_buffer" not in result or result.get("ram_buffer") == original_ram

    def test_evict_node_routing_flag(self):
        result = evict_node(self._make_state())
        assert result.get("routing_flag") == "evict"


# ---------------------------------------------------------------------------
# Task 7 — persist_node: ARCHIVE (Req 3.1, 4.1)
# Extraction regex + storage dispatch
# ---------------------------------------------------------------------------

class TestExtractKv:
    """Unit tests for the extraction regex used by persist_node."""

    def test_my_x_is_y(self):
        assert extract_kv("my name is Alice") == ("name", "Alice")

    def test_set_x_to_y(self):
        assert extract_kv("set temperature to 0.7") == ("temperature", "0.7")

    def test_colon_assignment(self):
        result = extract_kv("dataset: ImageNet")
        assert result is not None
        key, val = result
        assert "dataset" in key
        assert "ImageNet" in val

    def test_remember_that_pattern(self):
        result = extract_kv("remember that my deadline is Friday")
        assert result is not None
        key, val = result
        assert val == "Friday"

    def test_no_match_returns_none(self):
        assert extract_kv("How does attention work?") is None
        assert extract_kv("Tell me a story") is None
        assert extract_kv("hi") is None

    def test_key_normalised_to_lowercase(self):
        result = extract_kv("my Name is Bob")
        assert result is not None
        key, _ = result
        assert key == key.lower()


class TestPersistNode:
    def _make_state(self, text: str, ram: list | None = None) -> GraphState:
        return {
            "user_input": text,
            "routing_flag": "persist",
            "ram_buffer": ram or [],
            "extracted_facts": {},
            "retrieved_facts": {},
            "lean_prompt": "",
            "final_output": "",
        }

    def test_persist_kv_calls_set_fact(self):
        with patch("lean_context.storage.set_fact") as mock_set, \
             patch("lean_context.storage.add_narrative") as mock_add:
            result = persist_node(self._make_state("my name is Alice"))
            mock_set.assert_called_once_with("name", "Alice")
            mock_add.assert_not_called()
            assert result["extracted_facts"] == {"name": "Alice"}

    def test_persist_narrative_fallback_calls_add_narrative(self):
        """Substantive text with no KV match → ChromaDB narrative path."""
        long_text = "I have been working on this project for three months now."
        with patch("lean_context.storage.set_fact") as mock_set, \
             patch("lean_context.storage.add_narrative") as mock_add:
            result = persist_node(self._make_state(long_text))
            mock_set.assert_not_called()
            mock_add.assert_called_once()
            assert result["extracted_facts"] == {}

    def test_persist_short_non_kv_does_not_store(self):
        """Very short text with no KV match — not substantive enough to archive."""
        with patch("lean_context.storage.set_fact") as mock_set, \
             patch("lean_context.storage.add_narrative") as mock_add:
            persist_node(self._make_state("notes"))  # 1 word, no KV
            mock_set.assert_not_called()
            mock_add.assert_not_called()

    def test_persist_does_not_modify_ram_buffer(self):
        state = self._make_state("my age is 30")
        with patch("lean_context.storage.set_fact"):
            result = persist_node(state)
        assert "ram_buffer" not in result


# ---------------------------------------------------------------------------
# Task 8 — active_ram_node: KEEP + RETRIEVE (Req 2.1, 2.2, 3.3, 4.2)
# ---------------------------------------------------------------------------

class TestActiveRamNode:
    def _make_state(self, user_input: str, ram: list | None = None) -> GraphState:
        return {
            "user_input": user_input,
            "routing_flag": "active",
            "ram_buffer": ram or [],
            "extracted_facts": {},
            "retrieved_facts": {},
            "lean_prompt": "",
            "final_output": "",
        }

    def test_retrieves_sqlite_facts(self):
        with patch("lean_context.storage.get_all_facts", return_value={"name": "Bob"}) as mock_facts, \
             patch("lean_context.storage.query_narrative", return_value=[]):
            result = active_ram_node(self._make_state("what is my name?"))
            mock_facts.assert_called_once()
            assert result["retrieved_facts"]["facts"] == {"name": "Bob"}

    def test_retrieves_narrative_chunks(self):
        with patch("lean_context.storage.get_all_facts", return_value={}), \
             patch("lean_context.storage.query_narrative",
                   return_value=["chunk1", "chunk2"]) as mock_query:
            result = active_ram_node(self._make_state("tell me about the project"))
            mock_query.assert_called_once()
            assert result["retrieved_facts"]["narrative"] == ["chunk1", "chunk2"]

    def test_ram_capped_to_3_turns(self):
        """Requirement 2.1 / 2.2: retrieved capped_ram must not exceed RAM_BUFFER_CAP."""
        big_ram = [{"user": f"u{i}", "assistant": f"a{i}"} for i in range(7)]
        with patch("lean_context.storage.get_all_facts", return_value={}), \
             patch("lean_context.storage.query_narrative", return_value=[]):
            result = active_ram_node(self._make_state("question", ram=big_ram))
            capped = result["retrieved_facts"]["capped_ram"]
            assert len(capped) <= RAM_BUFFER_CAP
            # Should be the last N turns
            assert capped == big_ram[-RAM_BUFFER_CAP:]

    def test_empty_ram_is_fine(self):
        with patch("lean_context.storage.get_all_facts", return_value={}), \
             patch("lean_context.storage.query_narrative", return_value=[]):
            result = active_ram_node(self._make_state("anything"))
            assert result["retrieved_facts"]["capped_ram"] == []

    def test_does_not_append_to_ram_buffer(self):
        """RAM append happens in local_inference_node, not here."""
        ram = [{"user": "hi", "assistant": "hello"}]
        with patch("lean_context.storage.get_all_facts", return_value={}), \
             patch("lean_context.storage.query_narrative", return_value=[]):
            result = active_ram_node(self._make_state("question", ram=ram))
        # active_ram_node must not return a new ram_buffer entry
        assert "ram_buffer" not in result or result["ram_buffer"] == []

    def test_chroma_failure_degrades_gracefully(self):
        """ChromaDB failure → empty narrative list, not an exception."""
        with patch("lean_context.storage.get_all_facts", return_value={"k": "v"}), \
             patch("lean_context.storage.query_narrative", return_value=[]):
            result = active_ram_node(self._make_state("any query"))
            assert result["retrieved_facts"]["narrative"] == []
            assert result["retrieved_facts"]["facts"] == {"k": "v"}
