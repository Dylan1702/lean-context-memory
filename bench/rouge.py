"""
bench/rouge.py — Minimal ROUGE-L recall implementation (no external deps).

Only ROUGE-L *recall* is implemented — the metric used in run_eval.py and
verify_results.py to score how much of the reference fact appears in the
model's reply.

    rouge_l_recall(hypothesis, reference) -> float in [0.0, 1.0]

Where:
  reference  = the planted fact value (e.g. "95% recall accuracy")
  hypothesis = the model's reply

Uses the standard LCS-based formulation:
  R_lcs = LCS(hyp_tokens, ref_tokens) / len(ref_tokens)

Tokenisation: lowercase, split on whitespace and punctuation (simple but
sufficient for short fact strings).

Example:
    >>> rouge_l_recall("Your project target is 95% recall accuracy.", "95% recall accuracy")
    1.0
    >>> rouge_l_recall("I don't know.", "95% recall accuracy")
    0.0
"""

from __future__ import annotations

import re


def _tokenise(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+(?:[%\.][0-9]+)?", text.lower())


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Return the length of the longest common subsequence of *a* and *b*."""
    if not a or not b:
        return 0
    # Use O(min(len_a, len_b)) space DP
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, 1):
            if tok_a == tok_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[len(b)]


def rouge_l_recall(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L recall: what fraction of the reference tokens appear
    (in order) in the hypothesis.

    Returns a float in [0.0, 1.0].  Returns 0.0 if *reference* is empty.
    """
    ref_tokens = _tokenise(reference)
    hyp_tokens = _tokenise(hypothesis)
    if not ref_tokens:
        return 0.0
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    return lcs / len(ref_tokens)


def _find_evidence(hypothesis: str, reference: str, window: int = 8) -> str:
    """
    Return the shortest substring of *hypothesis* that contains the most
    consecutive reference tokens (up to *window* words around the best match).
    Used for the 'evidence: "..."' display in run_eval.py.
    """
    ref_tokens = _tokenise(reference)
    hyp_words = hypothesis.split()
    if not ref_tokens or not hyp_words:
        return ""

    best_start, best_len = 0, 0
    for i, word in enumerate(hyp_words):
        if _tokenise(word) and _tokenise(word)[0] == ref_tokens[0]:
            run = 1
            j = i + 1
            k = 1
            while k < len(ref_tokens) and j < len(hyp_words):
                if _tokenise(hyp_words[j]) and _tokenise(hyp_words[j])[0] == ref_tokens[k]:
                    run += 1
                    k += 1
                j += 1
            if run > best_len:
                best_len = run
                best_start = i

    start = max(0, best_start - 1)
    end = min(len(hyp_words), best_start + best_len + window // 2)
    snippet = " ".join(hyp_words[start:end])
    return snippet[:120]
