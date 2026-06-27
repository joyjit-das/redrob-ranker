#!/usr/bin/env python3
"""
Single-command reproduction entrypoint.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Reads the candidate pool (.jsonl or .jsonl.gz), ranks all candidates against
the released JD, and writes a spec-compliant top-100 CSV
(candidate_id, rank, score, reasoning). CPU-only, no network, < 5 min on 100k.
"""
import argparse
import csv
import gzip
import json
import time

from backend.ranker import rank_candidates


def load_candidates(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description="Redrob candidate ranker")
    ap.add_argument("--candidates", required=True, help="candidates.jsonl[.gz]")
    ap.add_argument("--out", default="submission.csv", help="output CSV path")
    ap.add_argument("--top_k", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    candidates = list(load_candidates(args.candidates))
    print(f"[rank] loaded {len(candidates)} candidates in {time.time()-t0:.1f}s")

    results = rank_candidates(candidates, top_k=args.top_k)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            wtr.writerow([r["candidate_id"], r["rank"], f'{r["score"]:.6f}', r["reasoning"]])

    print(f"[rank] wrote {len(results)} rows to {args.out} "
          f"in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
