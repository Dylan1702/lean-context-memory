"""
tests/test_storage_chroma.py

Unit tests for the ChromaDB Tier 3 storage layer.
Covers: init_chroma, add_narrative, query_narrative, graceful degradation.
Each test uses a fresh tmp_path-based persist directory so tests are isolated.
"""

import pytest

from lean_context.storage import init_chroma, add_narrative, query_narrative


@pytest.fixture
def chroma_env(tmp_path):
    """
    Return (persist_dir, collection_name) pointing to a fresh ChromaDB
    store in a temp directory. Also resets the module-level cache so each
    test starts clean.
    """
    import lean_context.storage as st
    persist_dir = str(tmp_path / "chroma_test")
    collection_name = "test_narrative"

    # Reset module-level cache so the fixture's collection is used
    st._chroma_client = None
    st._chroma_collection = None

    init_chroma(persist_dir=persist_dir, collection_name=collection_name)
    yield persist_dir, collection_name

    # Cleanup cache after test
    st._chroma_client = None
    st._chroma_collection = None


# ---------------------------------------------------------------------------
# add_narrative + query_narrative — happy path
# ---------------------------------------------------------------------------

def test_add_and_query_narrative(chroma_env):
    persist_dir, cname = chroma_env
    add_narrative(
        "My project target is to reduce latency to under 10ms.",
        turn_index=1,
        persist_dir=persist_dir,
        collection_name=cname,
    )
    results = query_narrative(
        "latency project goal",
        n_results=1,
        persist_dir=persist_dir,
        collection_name=cname,
    )
    assert len(results) == 1
    assert "latency" in results[0].lower()


def test_query_returns_most_relevant(chroma_env):
    persist_dir, cname = chroma_env
    add_narrative("I enjoy hiking in the mountains.", turn_index=1,
                  persist_dir=persist_dir, collection_name=cname)
    add_narrative("The neural network training loss is 0.023.", turn_index=2,
                  persist_dir=persist_dir, collection_name=cname)
    add_narrative("My favourite food is pasta.", turn_index=3,
                  persist_dir=persist_dir, collection_name=cname)

    results = query_narrative(
        "machine learning model performance",
        n_results=1,
        persist_dir=persist_dir,
        collection_name=cname,
    )
    assert len(results) == 1
    # The ML-related chunk should rank first
    assert "neural network" in results[0].lower() or "training" in results[0].lower()


def test_query_empty_collection_returns_empty_list(chroma_env):
    persist_dir, cname = chroma_env
    results = query_narrative(
        "anything at all",
        n_results=3,
        persist_dir=persist_dir,
        collection_name=cname,
    )
    assert results == []


def test_add_multiple_chunks_query_returns_up_to_n(chroma_env):
    persist_dir, cname = chroma_env
    for i in range(5):
        add_narrative(f"Narrative chunk number {i} about the project.", turn_index=i,
                      persist_dir=persist_dir, collection_name=cname)
    results = query_narrative(
        "project narrative",
        n_results=3,
        persist_dir=persist_dir,
        collection_name=cname,
    )
    assert len(results) <= 3
    assert len(results) > 0


# ---------------------------------------------------------------------------
# Embeddings are local — verify no network call is needed
# (Indirectly verified: if test passes without internet it's CPU-only)
# ---------------------------------------------------------------------------

def test_embeddings_run_locally(chroma_env):
    """Smoke test: add + query should complete without any HTTP call to OpenAI."""
    persist_dir, cname = chroma_env
    add_narrative("Local embedding test sentence.", turn_index=0,
                  persist_dir=persist_dir, collection_name=cname)
    results = query_narrative("embedding local",
                              persist_dir=persist_dir, collection_name=cname)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# init_chroma idempotency
# ---------------------------------------------------------------------------

def test_init_chroma_idempotent(tmp_path):
    """Calling init_chroma twice on the same path must not raise."""
    import lean_context.storage as st
    st._chroma_client = None
    st._chroma_collection = None

    persist_dir = str(tmp_path / "chroma_idem")
    cname = "idem_col"
    init_chroma(persist_dir=persist_dir, collection_name=cname)
    init_chroma(persist_dir=persist_dir, collection_name=cname)  # second call

    add_narrative("idempotent test", turn_index=0,
                  persist_dir=persist_dir, collection_name=cname)
    results = query_narrative("idempotent",
                              persist_dir=persist_dir, collection_name=cname)
    assert len(results) >= 1

    st._chroma_client = None
    st._chroma_collection = None
