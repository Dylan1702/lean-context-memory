"""
tests/test_context_assembly.py — Task 9

Unit tests for assemble_context_node.
Key assertion: prompt length does NOT grow when RAM history > 3 turns (Req 6.2).
"""

import pytest
from lean_context.nodes import assemble_context_node, RAM_BUFFER_CAP, GraphState


def _make_state(user_input="Tell me about the project",
                facts=None, narrative=None, capped_ram=None) -> GraphState:
    return {
        "user_input": user_input,
        "routing_flag": "active",
        "ram_buffer": capped_ram or [],
        "extracted_facts": {},
        "retrieved_facts": {
            "facts": facts or {},
            "narrative": narrative or [],
            "capped_ram": capped_ram or [],
        },
        "lean_prompt": "",
        "final_output": "",
    }


class TestAssembleContextNode:

    def test_prompt_contains_system_section(self):
        result = assemble_context_node(_make_state())
        assert "[SYSTEM]" in result["lean_prompt"]

    def test_prompt_contains_current_query(self):
        result = assemble_context_node(_make_state(user_input="What is my name?"))
        assert "[CURRENT QUERY]" in result["lean_prompt"]
        assert "What is my name?" in result["lean_prompt"]

    def test_facts_section_included_when_present(self):
        result = assemble_context_node(_make_state(facts={"name": "Alice", "project": "Lean"}))
        assert "[FACTS]" in result["lean_prompt"]
        assert "name: Alice" in result["lean_prompt"]
        assert "project: Lean" in result["lean_prompt"]

    def test_facts_section_absent_when_empty(self):
        result = assemble_context_node(_make_state(facts={}))
        assert "[FACTS]" not in result["lean_prompt"]

    def test_narrative_section_included_when_present(self):
        result = assemble_context_node(_make_state(narrative=["I am working on NLP."]))
        assert "[RELEVANT CONTEXT]" in result["lean_prompt"]
        assert "I am working on NLP." in result["lean_prompt"]

    def test_recent_conversation_section_included(self):
        ram = [{"user": "hi", "assistant": "hello"}]
        result = assemble_context_node(_make_state(capped_ram=ram))
        assert "[RECENT CONVERSATION]" in result["lean_prompt"]
        assert "User: hi" in result["lean_prompt"]
        assert "Assistant: hello" in result["lean_prompt"]

    def test_prompt_length_bounded_regardless_of_history(self):
        """
        Requirement 6.2: prompt size must be O(1) w.r.t. total conversation turns.

        We build two states:
          - state_3:  exactly 3 RAM turns (at cap)
          - state_7:  7 RAM turns fed in (should be capped to last 3 internally)

        The assembled prompt for state_7 must be NO LONGER than state_3's.
        """
        turns_3 = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(3)]
        turns_7 = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(7)]

        prompt_3 = assemble_context_node(_make_state(capped_ram=turns_3))["lean_prompt"]
        # Simulate passing 7 turns — assemble_context_node must cap internally
        state_7 = _make_state()
        state_7["retrieved_facts"]["capped_ram"] = turns_7  # bypass the normal cap
        prompt_7 = assemble_context_node(state_7)["lean_prompt"]

        assert len(prompt_7) <= len(prompt_3) + 10, (
            f"Prompt grew with more turns: {len(prompt_3)} → {len(prompt_7)}"
        )

    def test_ram_capped_to_3_in_assembled_prompt(self):
        """Even if capped_ram has more than 3 entries, prompt only shows last 3."""
        big_ram = [{"user": f"u{i}", "assistant": f"a{i}"} for i in range(6)]
        state = _make_state()
        state["retrieved_facts"]["capped_ram"] = big_ram
        result = assemble_context_node(state)
        prompt = result["lean_prompt"]
        # Only last 3 turns (u3, u4, u5) should appear, not u0, u1, u2
        assert "User: u5" in prompt
        assert "User: u4" in prompt
        assert "User: u3" in prompt
        assert "User: u0" not in prompt
        assert "User: u1" not in prompt
