"""
storage.py — Tier 2 (SQLite) and Tier 3 (ChromaDB) storage layers.

Tier 2: structured key-value facts in SQLite (memory_store.db)
Tier 3: narrative/semantic chunks in a local ChromaDB collection

Design note on default path resolution:
  All public functions accept an optional db_path / persist_dir argument.
  When None is passed (the default), the function looks up the module-level
  SQLITE_DB_PATH / CHROMA_PERSIST_DIR constant at *call time* — this makes
  the paths trivially patchable in tests without needing to rewrite function
  default values.
"""

import sqlite3
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Silence ChromaDB telemetry before the library is imported so PostHog /
# urllib3 retry messages never appear during offline runs.
import os as _os
_os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
_os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "")

import chromadb
import chromadb.config as _chroma_config
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — patch these in tests to redirect storage to temp dirs
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent.parent
SQLITE_DB_PATH: str = str(_BASE_DIR / "memory_store.db")
CHROMA_PERSIST_DIR: str = str(_BASE_DIR / "chroma_data")
CHROMA_COLLECTION_NAME: str = "lean_context_narrative"

# ---------------------------------------------------------------------------
# Tier 2 — SQLite (structured facts)
# ---------------------------------------------------------------------------

def _get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode enabled."""
    path = db_path if db_path is not None else SQLITE_DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_sqlite(db_path: Optional[str] = None) -> None:
    """
    Create memory_store.db with WAL mode and the facts table.

    Schema:
        facts(key TEXT PRIMARY KEY, val TEXT NOT NULL, updated_at TEXT)
    """
    path = db_path if db_path is not None else SQLITE_DB_PATH
    conn = _get_connection(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                key        TEXT PRIMARY KEY,
                val        TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key);")
        conn.commit()
        logger.debug("SQLite initialised at %s", path)
    finally:
        conn.close()


def set_fact(key: str, val: str, db_path: Optional[str] = None) -> None:
    """
    Write a key-value fact to SQLite.

    Uses INSERT OR REPLACE so a repeated key overwrites the previous value
    (Requirement 3.5). Retries once on OperationalError (e.g. locked file)
    with a short backoff (design.md Error Handling).
    """
    path = db_path if db_path is not None else SQLITE_DB_PATH
    updated_at = datetime.now(timezone.utc).isoformat()
    sql = "INSERT OR REPLACE INTO facts (key, val, updated_at) VALUES (?, ?, ?);"

    for attempt in range(2):
        try:
            conn = _get_connection(path)
            try:
                conn.execute(sql, (key, val, updated_at))
                conn.commit()
                logger.debug("set_fact: %s = %s", key, val)
                return
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            if attempt == 0:
                logger.warning("SQLite write failed (%s); retrying in 50ms…", exc)
                time.sleep(0.05)
            else:
                logger.error("SQLite write failed after retry: %s", exc)
                # Degrade gracefully — don't crash the conversation


def get_fact(key: str, db_path: Optional[str] = None) -> Optional[str]:
    """Return the value for *key*, or None if it doesn't exist."""
    path = db_path if db_path is not None else SQLITE_DB_PATH
    conn = _get_connection(path)
    try:
        cursor = conn.execute("SELECT val FROM facts WHERE key = ?;", (key,))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_all_facts(db_path: Optional[str] = None) -> dict[str, str]:
    """
    Return every stored fact as a plain dict {key: val}.

    Full table scan is intentional — the facts table is small at this scale
    and an index on key is present for individual lookups (design.md §4).
    Requirement 3.3: all facts included in every ACTIVE context assembly.
    """
    path = db_path if db_path is not None else SQLITE_DB_PATH
    conn = _get_connection(path)
    try:
        cursor = conn.execute("SELECT key, val FROM facts;")
        return {row[0]: row[1] for row in cursor.fetchall()}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tier 3 — ChromaDB (narrative / semantic chunks)
# ---------------------------------------------------------------------------

_chroma_client: Optional[chromadb.ClientAPI] = None
_chroma_collection = None


def init_chroma(
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
):
    """
    Initialise a persistent local ChromaDB client and return the collection.

    Embeddings are generated with sentence-transformers (all-MiniLM-L6-v2)
    entirely on-CPU — no network calls (Requirement 4.3).

    HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE are set to "1" before loading the
    embedding function so that HuggingFace Hub never attempts a network round-trip
    to check for model updates on every call.  Without this, each embed() call
    can trigger a slow (or blocking) HTTP HEAD request to hf.co even when the
    model weights are already cached locally — which was the root cause of the
    ~3.6 min/turn benchmark slowness.
    """
    import os as _os
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    global _chroma_client, _chroma_collection

    _persist_dir = persist_dir if persist_dir is not None else CHROMA_PERSIST_DIR
    _collection_name = collection_name if collection_name is not None else CHROMA_COLLECTION_NAME

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        # local_files_only=True is passed via the SentenceTransformer kwargs
        # so the underlying model loader also never goes to the network.
        local_files_only=True,
    )

    _chroma_client = chromadb.PersistentClient(
        path=_persist_dir,
        settings=_chroma_config.Settings(anonymized_telemetry=False, allow_reset=False),
    )
    _chroma_collection = _chroma_client.get_or_create_collection(
        name=_collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug(
        "ChromaDB initialised at %s, collection=%s", _persist_dir, _collection_name
    )
    return _chroma_collection


def _get_collection():
    """Return the cached collection, initialising lazily if needed."""
    global _chroma_collection
    if _chroma_collection is None:
        init_chroma()
    return _chroma_collection


def add_narrative(
    text: str,
    turn_index: int,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> None:
    """
    Store a narrative chunk in ChromaDB.

    Metadata records turn_index and an ISO-8601 timestamp so chunks can be
    traced back to conversation position (design.md Data Models §ChromaDB).
    Lifecycle state 3 — ARCHIVE.
    """
    collection = _get_collection()
    doc_id = f"turn_{turn_index}_{int(time.time() * 1000)}"
    metadata = {
        "turn_index": turn_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        collection.add(documents=[text], metadatas=[metadata], ids=[doc_id])
        logger.debug("add_narrative: stored chunk id=%s", doc_id)
    except Exception as exc:
        # Degrade gracefully (design.md Error Handling — ChromaDB failure)
        logger.error("ChromaDB add failed: %s", exc)


def query_narrative(
    query: str,
    n_results: int = 3,
    max_distance: float = 0.75,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> list[str]:
    """
    Query ChromaDB for narrative chunks most semantically similar to *query*.

    Only returns chunks whose cosine distance is below *max_distance*.
    This prevents irrelevant stored narrative from being injected into the
    prompt when the current query is on an entirely different topic — which
    would cause the model to over-anchor on unrelated context and refuse to
    answer the actual question.

    Empirical distance ranges (all-MiniLM-L6-v2, cosine space):
      clearly related   ≈ 0.35–0.55
      loosely related   ≈ 0.55–0.75
      clearly unrelated ≈ 0.80–1.00

    Default threshold 0.75 keeps genuinely relevant chunks and drops noise.
    Returns at most *n_results* document strings, or [] if none pass the
    threshold (same as the existing empty-collection case).
    On any ChromaDB failure degrades gracefully to [] (design.md Error Handling).
    Lifecycle state 4 — RETRIEVE.
    """
    collection = _get_collection()
    try:
        count = collection.count()
        if count == 0:
            return []
        actual_n = min(n_results, count)
        results = collection.query(
            query_texts=[query],
            n_results=actual_n,
            include=["documents", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        # Filter by distance threshold — only return chunks close enough to be relevant
        filtered = [
            doc for doc, dist in zip(docs, distances)
            if doc and dist <= max_distance
        ]
        logger.debug(
            "query_narrative: %d/%d chunks passed distance threshold %.2f",
            len(filtered), len(docs), max_distance,
        )
        return filtered
    except Exception as exc:
        logger.error("ChromaDB query failed (degrading gracefully): %s", exc)
        return []
