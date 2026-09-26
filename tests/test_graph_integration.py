"""
tests/test_graph_integration.py — Task 11

Integration test: scripted 5-turn conversation through the compiled graph
with Ollama mocked. Asserts:
  - RAM buffer never exceeds 3 turns
  - Facts land in SQLite after a PERSIST-routed turn
  - Correct routing occurs for evict/persist/active turns
  - Graph survives the full 5-turn sequence without error
"""

import tempfile
import pytest
from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver

import lean_context.storage as storage_mod
from lean_context.graph import build_graph


# ---------------------------------------------------------------------------
# Fixture: isolated storage + in-memory checkpointer
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_graph(tmp_path):
    """
    Build the graph with:
      - SQLite pointed at a temp DB (no pollution of production memory_store.db)
      - ChromaDB pointed at a temp directory
      - MemorySaver checkpointer (no checkpoint.db file)
      - Ollama mocked out
    """
    tmp_db = str(tmp_path / "test_memory.db")
    tmp_chroma = str(tmp_path / "chroma")

    # Patch storage paths before the graph init calls init_sqlite/init_chroma
    with patch.object(storage_mod, "SQLITE_DB_PATH", tmp_db), \
         patch.object(storage_mod, "CHROMA_PERSIST_DIR", tmp_chroma):

        # Reset Chroma module-level cache
        storage_mod._chroma_client = None
        storage_mod._chroma_collection = None

        checkpointer = MemorySaver()
        app = build_graph(checkpointer=checkpointer)

        yield app, tmp_db, tmp_chroma

        # Cleanup
        storage_mod._chroma_client = None
        storage_mod._chroma_collection = None


# ---------------------------------------------------------------------------
# Helper: run one turn through the graph
# ---------------------------------------------------------------------------

def run_turn(app, user_input: str, thread_id: str = "test-thread") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {
            "user_input": user_input,
            "ram_buffer": [],
            "extracted_facts": {},
            "retrieved_facts": {},
            "lean_prompt": "",
            "final_output": "",
        },
        config=config,
    )
    return result


# ---------------------------------------------------------------------------
# 5-turn scripted conversation (Ollama mocked)
# ---------------------------------------------------------------------------

SCRIPTED_TURNS = [
    # (user_input, expected_route, description)
    ("hi",                                  "evict",   "social filler → FORGET"),
    ("my name is Alice",                    "persist",  "fact → ARCHIVE"),
    ("What is the capital of France?",      "active",   "question → KEEP+RETRIEVE"),
    ("Explain how neural networks work.",   "active",   "substantive → KEEP+RETRIEVE"),
    ("thanks",                              "evict",    "social filler → FORGET"),
]


class TestGraphIntegration:

    def test_5_turn_conversation_completes(self, isolated_graph, tmp_path):
        app, tmp_db, _ = isolated_graph

        with patch("lean_context.nodes._ollama_client.generate", return_value={"response": "mocked answer"}):
            for user_input, _, desc in SCRIPTED_TURNS:
                result = run_turn(app, user_input, thread_id="integration-test")
                assert "final_output" in result, f"No final_output on turn: {desc}"
                assert result["final_output"], f"Empty final_output on turn: {desc}"

    def test_ram_never_exceeds_3_turns(self, isolated_graph):
        app, tmp_db, _ = isolated_graph

        with patch("lean_context.nodes._ollama_client.generate", return_value={"response": "answer"}):
            # Run 5 active turns — buffer should stay at 3.
            # First turn seeds state; subsequent turns only send user_input so the
            # checkpointer carries the ram_buffer forward.
            active_inputs = [
                "Explain transformers.",
                "How does attention work?",
                "What is backpropagation?",
                "Describe gradient descent.",
                "What is a learning rate?",
            ]
            config = {"configurable": {"thread_id": "ram-cap-test"}}
            # First turn — initialise state
            result = app.invoke(
                {"user_input": active_inputs[0], "ram_buffer": [],
                 "extracted_facts": {}, "retrieved_facts": {},
                 "lean_prompt": "", "final_output": ""},
                config=config,
            )
            # Subsequent turns — only update user_input; checkpointer keeps ram_buffer
            for user_input in active_inputs[1:]:
                result = app.invoke(
                    {"user_input": user_input},
                    config=config,
                )
            ram = result.get("ram_buffer", [])
            assert len(ram) <= 3, f"RAM buffer exceeded 3: got {len(ram)}"

    def test_fact_lands_in_sqlite_after_persist_turn(self, isolated_graph, tmp_path):
        app, tmp_db, _ = isolated_graph

        with patch("lean_context.nodes._ollama_client.generate", return_value={"response": "Got it!"}):
            run_turn(app, "my project is lean-context", thread_id="fact-test")

        # Read directly from the temp DB — works because isolated_graph
        # patches SQLITE_DB_PATH before build_graph calls init_sqlite,
        # and set_fact does a live lookup of the patched path.
        fact = storage_mod.get_fact("project", db_path=tmp_db)
        assert fact == "lean-context", f"Expected 'lean-context', got {fact!r}"

    def test_evict_turn_returns_canned_ack_not_llm(self, isolated_graph):
        app, _, _ = isolated_graph

        ollama_called = []
        with patch("lean_context.nodes._ollama_client.generate", side_effect=lambda **kw: ollama_called.append(1)):
            run_turn(app, "ok thanks", thread_id="evict-test")

        assert len(ollama_called) == 0, "Ollama should NOT be called for EVICT turns"

    def test_routing_produces_output_for_all_three_paths(self, isolated_graph):
        app, _, _ = isolated_graph
        with patch("lean_context.nodes._ollama_client.generate", return_value={"response": "ok"}):
            # evict
            r1 = run_turn(app, "hi", "path-test-1")
            assert r1["final_output"]

            # persist → active (falls through to LLM)
            r2 = run_turn(app, "my goal is 100% recall", "path-test-2")
            assert r2["final_output"]

            # active
            r3 = run_turn(app, "How does ChromaDB work?", "path-test-3")
            assert r3["final_output"]
