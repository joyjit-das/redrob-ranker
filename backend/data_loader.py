#!/usr/bin/env python3
"""
data_loader.py

Loads the candidate pool, cleans/normalises it, generates ranking features via
the ranker, and stores everything the API/dashboard needs into a local SQLite
database (no synthetic data, nothing hardcoded).

Three tables:
  * ranking   — the top-N ranked candidates (id, rank, score, reasoning,
                component breakdown JSON, behavioural modifier, penalties).
  * candidate — full profile JSON for those ranked candidates (for detail view).
  * pool_stat — pre-aggregated analytics over the FULL 100k pool for the
                dashboard charts (title mix, score histogram, geography, etc.).

Run:  python -m backend.data_loader --candidates ./candidates.jsonl
"""
import argparse
import gzip
import json
import os
import sqlite3
from collections import Counter

from .ranker import rank_candidates
from . import config as C

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "redrob.db")
STORE_TOP_N = 500  # detailed records to keep for the UI


def load_candidates(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean(cand):
    """Light normalisation: trim strings, coerce missing numerics to sane defaults."""
    p = cand.setdefault("profile", {})
    p["years_of_experience"] = float(p.get("years_of_experience") or 0)
    for k in ("current_title", "current_company", "location", "country"):
        if isinstance(p.get(k), str):
            p[k] = p[k].strip()
    cand.setdefault("career_history", [])
    cand.setdefault("skills", [])
    cand.setdefault("redrob_signals", {})
    return cand


def build_pool_stats(candidates, ranked_index):
    """Aggregate analytics over the entire pool for the dashboard."""
    titles = Counter()
    countries = Counter()
    yoe_buckets = Counter()
    resp_buckets = Counter()
    honeypots = ranked_index["honeypots_total"]
    for c in candidates:
        p = c["profile"]
        titles[p.get("current_title", "Unknown")] += 1
        countries[p.get("country", "Unknown")] += 1
        y = p.get("years_of_experience", 0)
        yoe_buckets[f"{int(y // 2) * 2}-{int(y // 2) * 2 + 2}"] += 1
        r = c["redrob_signals"].get("recruiter_response_rate", 0) or 0
        resp_buckets[f"{int(r * 10) * 10}-{int(r * 10) * 10 + 10}%"] += 1
    return {
        "total_candidates": len(candidates),
        "title_distribution": dict(titles.most_common(15)),
        "country_distribution": dict(countries.most_common(10)),
        "yoe_distribution": dict(sorted(yoe_buckets.items())),
        "response_distribution": dict(sorted(resp_buckets.items())),
        "honeypots_detected": honeypots,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    args = ap.parse_args()

    print("[loader] reading candidates ...")
    candidates = [clean(c) for c in load_candidates(args.candidates)]
    print(f"[loader] {len(candidates)} candidates loaded")

    results = rank_candidates(candidates, top_k=STORE_TOP_N)

    # count honeypots across the whole pool (for the dashboard)
    from .features import extract_features
    honeypots_total = sum(1 for c in candidates if extract_features(c)["is_honeypot"])
    stats = build_pool_stats(candidates, {"honeypots_total": honeypots_total})

    ranked_ids = {r["candidate_id"] for r in results}
    detail = {c["candidate_id"]: c for c in candidates if c["candidate_id"] in ranked_ids}

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""CREATE TABLE ranking (candidate_id TEXT PRIMARY KEY, rank INTEGER,
                   score REAL, reasoning TEXT, components TEXT, behavioural REAL,
                   penalties TEXT)""")
    cur.execute("CREATE TABLE candidate (candidate_id TEXT PRIMARY KEY, profile_json TEXT)")
    cur.execute("CREATE TABLE pool_stat (key TEXT PRIMARY KEY, value TEXT)")

    for r in results:
        cur.execute("INSERT INTO ranking VALUES (?,?,?,?,?,?,?)",
                    (r["candidate_id"], r["rank"], r["score"], r["reasoning"],
                     json.dumps(r["components"]), r["behavioural"],
                     json.dumps(r["penalties"])))
    for cid, c in detail.items():
        cur.execute("INSERT INTO candidate VALUES (?,?)", (cid, json.dumps(c)))
    cur.execute("INSERT INTO pool_stat VALUES (?,?)", ("stats", json.dumps(stats)))
    cur.execute("INSERT INTO pool_stat VALUES (?,?)",
                ("jd", json.dumps({"title": C.JD_TITLE, "text": C.JD_TEXT})))
    con.commit()
    con.close()
    print(f"[loader] wrote SQLite db -> {DB_PATH}")
    print(f"[loader] honeypots detected in pool: {honeypots_total}")


if __name__ == "__main__":
    main()
