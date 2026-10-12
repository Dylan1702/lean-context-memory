"""tests/test_verify_results.py — Tests for bench/verify_results.py."""

import json
import hashlib
import pytest
from pathlib import Path


def _write_transcript(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "transcript.jsonl"
    with open(p, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return p


def test_sha256_matches_file(tmp_path):
    from bench.verify_results import _sha256
    content = b"hello world\n"
    path = tmp_path / "test.bin"
    path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert _sha256(path) == expected


def test_score_record_adds_found_and_rouge(tmp_path):
    from bench.verify_results import _score_record
    rec = {
        "mode": "lean", "turn": 13, "user": "q", "reply": "The learning rate is 0.001.",
        "expected": "0.001", "is_recall": True,
    }
    scored = _score_record(rec)
    assert scored["found"] is True
    assert scored["rouge_l"] == 1.0


def test_score_record_no_expected_unchanged(tmp_path):
    from bench.verify_results import _score_record
    rec = {"mode": "lean", "turn": 1, "user": "hi", "reply": "ok", "expected": "", "is_recall": False}
    scored = _score_record(rec)
    # No change when expected is empty
    assert scored == rec


def test_verify_runs_on_valid_transcript(tmp_path, capsys):
    from bench.verify_results import verify

    records = [
        {"mode": "lean", "turn": 1, "user": "gradient descent?",
         "reply": "An optimization algorithm.", "expected": "",
         "is_recall": False, "prompt_tokens": 50, "completion_tokens": 10,
         "model_ms": 5000.0, "memory_ms": 10.0, "total_ms": 5010.0,
         "server_total_ms": 5000.0, "created_at": None,
         "fact_key": None, "found": None, "rouge_l": None},
        {"mode": "lean", "turn": 13, "user": "What is my learning rate?",
         "reply": "Your learning rate is 0.001.",
         "expected": "0.001", "is_recall": True,
         "prompt_tokens": 55, "completion_tokens": 8,
         "model_ms": 4500.0, "memory_ms": 12.0, "total_ms": 4512.0,
         "server_total_ms": 4500.0, "created_at": None,
         "fact_key": "learning_rate", "found": True, "rouge_l": 1.0},
    ]
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    _write_transcript(run_dir, records)

    verify(run_dir)
    out = capsys.readouterr().out
    assert "SHA-256" in out
    assert "lean" in out
    assert "1/" in out   # e.g. "1/  1" or "1/1" depending on formatting


def test_verify_missing_transcript_exits(tmp_path):
    from bench.verify_results import verify
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(SystemExit):
        verify(empty_dir)
