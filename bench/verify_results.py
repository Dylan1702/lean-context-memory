"""
bench/verify_results.py — Independently re-score recall from transcript.jsonl.

Usage:
    python bench/verify_results.py bench/output/run_YYYYMMDD_HHMMSS/

Reads transcript.jsonl, re-applies containment + ROUGE-L scoring to every
recall turn, prints the same summary table as run_eval.py, and prints the
SHA-256 of the transcript file so you can verify it hasn't been modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from bench.rouge import rouge_l_recall


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _score_record(rec: dict) -> dict:
    """Re-compute found and rouge_l from raw reply + expected fields."""
    expected = rec.get("expected") or ""
    reply = rec.get("reply") or ""
    if not expected:
        return rec
    found = expected.lower() in reply.lower()
    score = rouge_l_recall(reply, expected)
    return {**rec, "found": found, "rouge_l": score}


def verify(run_dir: Path) -> None:
    transcript = run_dir / "transcript.jsonl"
    if not transcript.exists():
        print(f"[ERROR] transcript.jsonl not found in {run_dir}")
        sys.exit(1)

    digest = _sha256(transcript)
    print(f"\ntranscript.jsonl SHA-256: {digest}")
    print(f"File: {transcript}\n")

    by_mode: dict[str, list[dict]] = defaultdict(list)
    with open(transcript) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec = _score_record(rec)
            by_mode[rec["mode"]].append(rec)

    print("=" * 72)
    print("VERIFICATION TABLE  (independently re-scored)")
    print("=" * 72)
    header = (
        f"{'Mode':<12} {'recall':>6} {'ROUGE-L':>7} "
        f"{'avg_in':>6} {'max_in':>6} "
        f"{'avg_mem_ms':>10} {'avg_total_s':>11}"
    )
    print(header)
    print("-" * 72)

    for mode, records in sorted(by_mode.items()):
        priming = [r for r in records if not r.get("is_recall")]
        recall  = [r for r in records if r.get("is_recall")]
        if not recall:
            print(f"{mode:<12} (no recall turns)")
            continue
        n_hit   = sum(1 for r in recall if r.get("found"))
        avg_rl  = sum(r.get("rouge_l") or 0 for r in recall) / len(recall)
        in_toks = [r["prompt_tokens"] for r in priming if r.get("prompt_tokens")]
        avg_in  = sum(in_toks) / len(in_toks) if in_toks else 0
        max_in  = max(in_toks) if in_toks else 0
        mem_v   = [r.get("memory_ms") or 0 for r in priming]
        avg_mem = sum(mem_v) / len(mem_v) if mem_v else 0
        tot_v   = [r.get("total_ms") or 0 for r in priming]
        avg_tot = sum(tot_v) / len(tot_v) / 1000 if tot_v else 0
        print(
            f"{mode:<12} {n_hit}/{len(recall):>3}   {avg_rl:>6.2f} "
            f"{avg_in:>6.0f} {max_in:>6.0f} "
            f"{avg_mem:>10.0f} {avg_tot:>11.1f}"
        )

        # Per-fact recall detail
        for r in recall:
            found_str = "✓" if r.get("found") else "✗"
            print(
                f"  {found_str}  Q: {r.get('user','')[:60]:60}  "
                f"ROUGE-L: {r.get('rouge_l',0):.2f}  "
                f"expected: {r.get('expected','')!r}"
            )

    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score recall from transcript.jsonl")
    parser.add_argument("run_dir", type=Path,
                        help="Path to a run_YYYYMMDD_HHMMSS/ output directory")
    args = parser.parse_args()
    verify(args.run_dir)


if __name__ == "__main__":
    main()
