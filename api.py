"""
FastAPI backend for the Redrob ranker dashboard.

Endpoints
  GET  /api/jd                  -> the job description being ranked against
  GET  /api/stats               -> pre-aggregated pool analytics (dashboard)
  GET  /api/ranking?limit&q     -> ranked shortlist (optionally text-filtered)
  GET  /api/candidate/{id}      -> full profile + score breakdown for one candidate
  POST /api/rank_sample         -> rank a small uploaded sample end-to-end
                                   (powers the sandbox/demo requirement)

Reads the SQLite db produced by data_loader.py. CORS open for the React dev
server. No auth (intentionally — adds no ML value).
"""
import json
import os
import sqlite3

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .ranker import rank_candidates
from . import config as C

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "redrob.db")
app = FastAPI(title="Redrob Intelligent Candidate Ranker")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _con():
    if not os.path.exists(DB_PATH):
        raise HTTPException(503, "Database not built. Run data_loader.py first.")
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


@app.get("/api/jd")
def get_jd():
    con = _con()
    row = con.execute("SELECT value FROM pool_stat WHERE key='jd'").fetchone()
    con.close()
    return json.loads(row["value"]) if row else {"title": C.JD_TITLE, "text": C.JD_TEXT}


@app.get("/api/stats")
def get_stats():
    con = _con()
    row = con.execute("SELECT value FROM pool_stat WHERE key='stats'").fetchone()
    con.close()
    if not row:
        raise HTTPException(404, "No stats")
    return json.loads(row["value"])


@app.get("/api/ranking")
def get_ranking(limit: int = 100, q: str = ""):
    con = _con()
    rows = con.execute(
        "SELECT r.*, c.profile_json FROM ranking r "
        "LEFT JOIN candidate c ON c.candidate_id=r.candidate_id "
        "ORDER BY r.rank ASC"
    ).fetchall()
    con.close()
    out = []
    for row in rows:
        prof = json.loads(row["profile_json"])["profile"] if row["profile_json"] else {}
        item = {
            "candidate_id": row["candidate_id"],
            "rank": row["rank"],
            "score": row["score"],
            "reasoning": row["reasoning"],
            "components": json.loads(row["components"]),
            "behavioural": row["behavioural"],
            "penalties": json.loads(row["penalties"]),
            "title": prof.get("current_title"),
            "company": prof.get("current_company"),
            "location": prof.get("location"),
            "years_of_experience": prof.get("years_of_experience"),
        }
        if q:
            blob = json.dumps(item).lower()
            if q.lower() not in blob:
                continue
        out.append(item)
        if len(out) >= limit:
            break
    return {"count": len(out), "results": out}


@app.get("/api/candidate/{cid}")
def get_candidate(cid: str):
    con = _con()
    row = con.execute(
        "SELECT r.*, c.profile_json FROM ranking r "
        "LEFT JOIN candidate c ON c.candidate_id=r.candidate_id "
        "WHERE r.candidate_id=?", (cid,)
    ).fetchone()
    con.close()
    if not row:
        raise HTTPException(404, "Candidate not found in ranking")
    return {
        "candidate_id": row["candidate_id"],
        "rank": row["rank"],
        "score": row["score"],
        "reasoning": row["reasoning"],
        "components": json.loads(row["components"]),
        "behavioural": row["behavioural"],
        "penalties": json.loads(row["penalties"]),
        "profile": json.loads(row["profile_json"]) if row["profile_json"] else None,
    }


class SampleRequest(BaseModel):
    candidates: list
    top_k: int = 20


@app.post("/api/rank_sample")
def rank_sample(req: SampleRequest):
    """Rank a small uploaded sample end-to-end — the sandbox sanity check."""
    if not req.candidates:
        raise HTTPException(400, "No candidates provided")
    if len(req.candidates) > 200:
        raise HTTPException(400, "Sample too large (<=200).")
    results = rank_candidates(req.candidates, top_k=min(req.top_k, len(req.candidates)),
                              verbose=False)
    return {"count": len(results), "results": results}


@app.get("/")
def root():
    return {"service": "Redrob Intelligent Candidate Ranker", "docs": "/docs"}
