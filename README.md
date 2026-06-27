# Redrob — Intelligent Candidate Ranker

Ranks 100,000 candidates against the *Senior AI Engineer — Founding Team* job
description and produces a spec-compliant top-100 shortlist with per-candidate
reasoning.

---

## Project Overview

The hard part of this challenge is not similarity search — it is **not being
fooled**. The dataset is adversarial:

- **The `skills` array is uniformly random noise** (~12,000 occurrences per
  skill across the pool). Matching AI keywords in skills is the explicit trap;
  the provided `sample_submission.csv` even ranks an *HR Manager with 9 AI
  skills* at #1 as bait.
- It contains **keyword stuffers**, **plain-language strong candidates** who
  never use buzzwords, and **~80 honeypots** with subtly impossible profiles.
  A submission with >10% honeypots in the top 100 is disqualified.

So the ranker reads **career-history descriptions, titles, and summaries** —
the evidence a recruiter would actually trust — and applies the JD's explicit
disqualifiers. It never lets the skills list drive the semantic score.

## Architecture

```
candidates.jsonl
      │
      ▼
features.py        ← reads each profile into interpretable components
      │
      ▼
ranker.py          ← encodes evidence text, fuses scores, applies penalties
      │
      ▼
rank.py            ← CLI entrypoint: pool in → submission.csv out
```

## Repository Structure

```
redrob-ranker/
├── backend/
│   ├── __init__.py
│   ├── config.py       ← JD text, vocabularies, fusion weights
│   ├── features.py     ← feature extraction from raw profiles
│   └── ranker.py       ← scoring, penalty, honeypot, reasoning engine
├── rank.py             ← reproduction entrypoint
├── requirements.txt
├── data/
│   └── submission.csv  ← ranked top-100 output
└── submission_metadata.yaml
```

## Dataset

- 100,000 candidate profiles in `candidates.jsonl` (~465 MB; **not committed** —
  place it in the repo root). Compressed `.jsonl.gz` is also accepted.
- Each profile: `profile`, `career_history[]`, `education[]`, `skills[]`, and 23
  `redrob_signals` (engagement/behavioural). See `candidate_schema.json` in the
  challenge bundle.
- Title mix is dominated by ~12 non-tech roles (Business Analyst, HR Manager,
  …) as noise; genuine fits (ML / AI / NLP / Search / Recommendation engineers)
  are rare by design — exactly as the JD warns.

## Features

`backend/features.py` reads each profile into interpretable components in `[0,1]`:

- **semantic** — cosine of the candidate's *evidence text* against a distilled
  JD query (skills array deliberately excluded so stuffing can't inflate it).
- **career_evidence** — weighted hits for retrieval / ranking / recsys, ML, NLP,
  ranking-evaluation (NDCG/MRR/MAP/A-B), and production-at-scale signals.
- **title** — role prior; strong ML/IR titles score high, the 12 noise titles
  near-zero, medium titles lifted only by real evidence.
- **experience** — closeness to the 5–9 band (ideal 6–8).
- **skill_depth** — AI-core skills, **gated by career evidence** (anti-stuffer).
- **location** — Pune/Noida prime; Hyderabad/Mumbai/Delhi-NCR strong; India +
  relocation considered; outside-India down-weighted (no visa sponsorship).
- **behavioural multiplier** — availability from the 23 signals (recency,
  recruiter response rate, open-to-work, notice period, interview completion).

**Integrity / honeypot detection:** profiles with multiple impossible
attributes (career tenure far exceeding total experience, a single role longer
than the whole career, or several advanced/expert skills with zero usage time)
are flagged and floored so they cannot enter the top 100.

## Ranking Model

A transparent weighted fusion, chosen for interview-defensibility over a black
box:

```
base = 0.30·semantic + 0.26·career_evidence + 0.16·title
     + 0.12·experience + 0.08·skill_depth + 0.05·location + 0.03·nice_to_have

score = base × Π(penalties) × behavioural        # honeypots → floor
```

Penalties (multiplicative, <1) implement the JD's explicit "do NOT want" list:
consulting-only careers, pure-research-without-production, CV/speech/robotics
without NLP/IR, keyword stuffing, and title-chasing.

**Semantic encoder.** Default is TF-IDF (1–2 grams) + Truncated SVD (LSA) —
pure scikit-learn, no downloads, so the repo always reproduces on CPU. An
optional `sentence-transformers` encoder (e.g. `BAAI/bge-small-en-v1.5`) plugs
into the same interface when installed and `RANKER_USE_ST=1` is set; the code
falls back gracefully if it is absent.

**Compute budget.** Ranking all 100k candidates completes in ~2 minutes on a
single CPU core (well within the 5-minute / 16 GB / CPU-only / no-network
limit).

## Evaluation Metrics

Submissions are scored against a hidden ground truth with:

```
composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10
```

The design targets these directly: heavy weight on top-10 quality (precise
semantic + evidence + behavioural fit), honeypot flooring to protect precision,
and honest, rank-consistent reasoning for the Stage-4 manual review.

Validate the output with the bundle's checker:

```bash
python validate_submission.py submission.csv      # -> "Submission is valid."
```

## Installation

```bash
pip install -r requirements.txt
```

## Running

**Produce the submission CSV:**
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

**Optional — deeper embeddings via sentence-transformers:**
```bash
pip install sentence-transformers torch
RANKER_USE_ST=1 python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

To rank against a different role, edit `JD_QUERY` / `JD_TEXT` in
`backend/config.py` and re-run.

---

*AI tools were used as part of development (see `submission_metadata.yaml`); the
ranking pipeline runs fully offline and sends no data to any external API.*
