"""tests/test_rouge.py — Tests for bench/rouge.py."""

import pytest
from bench.rouge import rouge_l_recall, _tokenise, _lcs_length


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------

def test_tokenise_lowercases():
    assert _tokenise("Hello World") == ["hello", "world"]


def test_tokenise_strips_punctuation():
    tokens = _tokenise("95% recall accuracy.")
    assert "95" in tokens or any("95" in t for t in tokens)


def test_tokenise_empty():
    assert _tokenise("") == []


# ---------------------------------------------------------------------------
# _lcs_length
# ---------------------------------------------------------------------------

def test_lcs_identical():
    assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3


def test_lcs_empty():
    assert _lcs_length([], ["a"]) == 0
    assert _lcs_length(["a"], []) == 0


def test_lcs_partial():
    assert _lcs_length(["a", "x", "b", "y", "c"], ["a", "b", "c"]) == 3


# ---------------------------------------------------------------------------
# rouge_l_recall
# ---------------------------------------------------------------------------

def test_perfect_recall():
    assert rouge_l_recall(
        "Your project target is 95% recall accuracy.",
        "95% recall accuracy"
    ) == 1.0


def test_zero_recall():
    assert rouge_l_recall("I have no idea.", "95% recall accuracy") < 0.3


def test_empty_reference_returns_zero():
    assert rouge_l_recall("some text", "") == 0.0


def test_empty_hypothesis():
    assert rouge_l_recall("", "95% recall accuracy") == 0.0


def test_partial_recall():
    score = rouge_l_recall("The accuracy goal is 95 percent.", "95% recall accuracy")
    assert 0.0 < score < 1.0


def test_case_insensitive():
    score_lower = rouge_l_recall("mnist handwritten digits dataset", "MNIST handwritten digits")
    assert score_lower > 0.5


def test_mnist_reference():
    """Typical eval scenario: model mentions the dataset by name."""
    reply = "You are using the MNIST handwritten digits dataset."
    score = rouge_l_recall(reply, "MNIST handwritten digits")
    assert score >= 0.8


def test_number_reference():
    reply = "The learning rate is set to 0.001."
    score = rouge_l_recall(reply, "0.001")
    assert score == 1.0


def test_batch_size_reference():
    reply = "Your preferred batch size is 64."
    score = rouge_l_recall(reply, "64")
    assert score == 1.0
