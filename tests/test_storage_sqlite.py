"""
tests/test_storage_sqlite.py

Unit tests for the SQLite Tier 2 storage layer.
Covers: init, insert, overwrite (INSERT OR REPLACE), read, get_all_facts.
Each test uses a temporary file-based DB so tests are fully isolated.
"""

import os
import tempfile
import pytest

from lean_context.storage import init_sqlite, set_fact, get_fact, get_all_facts


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a fresh, initialised SQLite DB for each test."""
    db = str(tmp_path / "test_memory.db")
    init_sqlite(db_path=db)
    return db


# ---------------------------------------------------------------------------
# Basic insert + read
# ---------------------------------------------------------------------------

def test_set_and_get_fact(tmp_db):
    set_fact("name", "Alice", db_path=tmp_db)
    assert get_fact("name", db_path=tmp_db) == "Alice"


def test_get_missing_fact_returns_none(tmp_db):
    assert get_fact("nonexistent", db_path=tmp_db) is None


# ---------------------------------------------------------------------------
# Overwrite / INSERT OR REPLACE (Requirement 3.5)
# ---------------------------------------------------------------------------

def test_overwrite_same_key(tmp_db):
    set_fact("project", "Alpha", db_path=tmp_db)
    set_fact("project", "Beta", db_path=tmp_db)
    assert get_fact("project", db_path=tmp_db) == "Beta"


def test_overwrite_does_not_duplicate(tmp_db):
    set_fact("city", "London", db_path=tmp_db)
    set_fact("city", "Paris", db_path=tmp_db)
    all_facts = get_all_facts(db_path=tmp_db)
    # There must be exactly one entry for "city"
    assert list(all_facts.keys()).count("city") == 1
    assert all_facts["city"] == "Paris"


# ---------------------------------------------------------------------------
# get_all_facts
# ---------------------------------------------------------------------------

def test_get_all_facts_empty(tmp_db):
    assert get_all_facts(db_path=tmp_db) == {}


def test_get_all_facts_multiple(tmp_db):
    set_fact("name", "Bob", db_path=tmp_db)
    set_fact("language", "Python", db_path=tmp_db)
    set_fact("version", "3.11", db_path=tmp_db)
    facts = get_all_facts(db_path=tmp_db)
    assert facts == {"name": "Bob", "language": "Python", "version": "3.11"}


# ---------------------------------------------------------------------------
# Idempotent init (calling init_sqlite twice should not raise)
# ---------------------------------------------------------------------------

def test_init_sqlite_idempotent(tmp_path):
    db = str(tmp_path / "idempotent.db")
    init_sqlite(db_path=db)
    init_sqlite(db_path=db)  # second call must not raise
    set_fact("x", "1", db_path=db)
    assert get_fact("x", db_path=db) == "1"
