"""
graph.py — LangGraph StateGraph assembly for the Lean-Context pipeline.

Architecture (design.md §Architecture):

    User Turn → cpu_router_edge (regex, <5ms)
                  ├─ evict   → evict_node → END
                  ├─ persist → persist_node → active_ram_node → assemble_context → llm_infer → END
                  └─ active  →               active_ram_node → assemble_context → llm_infer → END

Requirements: 8.1 (distinct nodes), 8.2 (conditional entry point), 8.3 (SqliteSaver checkpoint).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from lean_context.nodes import (
    GraphState,
    evict_node,
    persist_node,
    active_ram_node,
    assemble_context_node,
    local_inference_node,
)
from lean_context.router import cpu_router_edge
from lean_context import storage

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DB_PATH = str(_BASE_DIR / "checkpoint.db")

# ---------------------------------------------------------------------------
# Router wrapper — converts user_input in state to a routing decision
# ---------------------------------------------------------------------------

def _route(state: GraphState) -> Literal["evict", "persist", "active"]:
    """
    Pure routing function used as the conditional entry point.
    Reads state["user_input"] and delegates to cpu_router_edge (no LLM, Req 1.1).
    """
    return cpu_router_edge(state["user_input"])


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None) -> StateGraph:
    """
    Build and compile the Lean-Context StateGraph.

    Parameters
    ----------
    checkpointer : optional langgraph checkpointer
        If None, uses SqliteSaver pointed at checkpoint.db (Req 8.3).
        Pass a MemorySaver or mock in tests to avoid file I/O.

    Returns
    -------
    Compiled LangGraph app ready to invoke.
    """
    # Initialise storage layers
    storage.init_sqlite()
    storage.init_chroma()

    builder = StateGraph(GraphState)

    # --- Add nodes (Requirement 8.1) ---
    builder.add_node("evict_node",       evict_node)
    builder.add_node("persist_node",     persist_node)
    builder.add_node("active_ram_node",  active_ram_node)
    builder.add_node("assemble_context", assemble_context_node)
    builder.add_node("llm_infer",        local_inference_node)

    # --- Conditional entry point keyed off router output (Requirement 8.2) ---
    builder.set_conditional_entry_point(
        _route,
        {
            "evict":   "evict_node",
            "persist": "persist_node",
            "active":  "active_ram_node",
        },
    )

    # --- Edges ---
    # evict path: short-circuit, no LLM
    builder.add_edge("evict_node", END)

    # persist path: archive → retrieve/assemble → infer
    builder.add_edge("persist_node",    "active_ram_node")
    builder.add_edge("active_ram_node", "assemble_context")
    builder.add_edge("assemble_context","llm_infer")
    builder.add_edge("llm_infer",        END)

    # --- Checkpointing (Requirement 8.3) ---
    if checkpointer is None:
        # Open a persistent connection for the lifetime of the app.
        # SqliteSaver.from_conn_string() is a context manager — calling it
        # without `with` yields a _GeneratorContextManager, not a SqliteSaver.
        # Instead, open the connection directly so it stays alive for the REPL.
        conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    app = builder.compile(checkpointer=checkpointer)
    logger.debug("Graph compiled with checkpointer: %s", type(checkpointer).__name__)
    return app


def get_default_app():
    """Return a compiled graph using the default SqliteSaver checkpoint DB."""
    return build_graph()
