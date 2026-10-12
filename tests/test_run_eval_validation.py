"""tests/test_run_eval_validation.py — Tests for run_eval.py script validation."""

import pytest
import sys
from unittest.mock import patch


def test_validate_script_passes_with_valid_layout():
    """Default layout: last fact at turn 8, N_PRIMING=12, gap=4 > RAM_BUFFER_CAP+1=4."""
    from bench.run_eval import _validate_script
    # Should not raise or call sys.exit
    _validate_script()


def test_validate_script_fails_when_gap_too_small(monkeypatch):
    """Monkeypatch PRIMING_FACTS to put last fact too close to recall questions."""
    import bench.run_eval as re_mod

    original_facts = re_mod.PRIMING_FACTS
    original_n = re_mod.N_PRIMING

    # Last fact at turn 11 with N_PRIMING=12 → gap=1, required=4
    monkeypatch.setattr(re_mod, "PRIMING_FACTS", {
        2: ("fact A", "key_a", "val_a"),
        11: ("fact B", "key_b", "val_b"),
    })
    monkeypatch.setattr(re_mod, "N_PRIMING", 12)

    with pytest.raises(SystemExit):
        re_mod._validate_script()


def test_build_priming_script_length():
    from bench.run_eval import _build_priming_script, N_PRIMING
    turns = _build_priming_script()
    assert len(turns) == N_PRIMING


def test_build_priming_script_facts_at_correct_positions():
    from bench.run_eval import _build_priming_script, PRIMING_FACTS
    turns = _build_priming_script()
    for turn_idx, (msg, _, _) in PRIMING_FACTS.items():
        assert turns[turn_idx - 1] == msg, (
            f"Expected fact at position {turn_idx} but got {turns[turn_idx-1]!r}"
        )


def test_recall_questions_end_with_one_short_sentence():
    from bench.run_eval import RECALL_QUESTIONS
    for q, _, _ in RECALL_QUESTIONS:
        assert "Answer in one short sentence" in q, (
            f"Recall question missing instruction: {q!r}"
        )


def test_n_recall_questions_equals_n_facts():
    from bench.run_eval import RECALL_QUESTIONS, PRIMING_FACTS
    assert len(RECALL_QUESTIONS) == len(PRIMING_FACTS)
