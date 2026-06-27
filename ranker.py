"""
The ranking engine.

Pipeline:
  1. Read every candidate into interpretable features (features.py).
  2. Build a semantic representation of each candidate's *evidence text* and of
     the distilled JD query, then score cosine similarity.
        - Default encoder: TF-IDF + Truncated SVD (LSA). Pure scikit-learn, no
          downloads, runs in seconds on CPU — so the repo always reproduces.
        - Optional encoder: sentence-transformers (BGE/MiniLM) when installed
          (set RANKER_USE_ST=1). Same interface, deeper semantics.
  3. Fuse the semantic score with the engineered components using config
     weights, apply penalty multipliers and the behavioural modifier, and force
     honeypots to a floor.
  4. Sort, take the top-k, write specific reasoning per candidate.

Designed for the challenge's compute budget: 100k candidates rank in well under
5 minutes on a single CPU core, in < 16 GB RAM.
"""
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from . import config as C
from .features import extract_features


# --------------------------- semantic encoders -----------------------------
class LSAEncoder:
    """TF-IDF (1-2 grams) -> L2-normalised Truncated SVD. Fast, dependency-light."""

    def __init__(self, n_components=200, max_features=40000):
        self.vec = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2),
            min_df=3, sublinear_tf=True, stop_words="english",
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.tfidf_matrix = None

    def fit_transform_corpus(self, texts):
        X = self.vec.fit_transform(texts)
        self.tfidf_matrix = normalize(X)
        n_comp = min(self.svd.n_components, X.shape[1] - 1)
        self.svd.n_components = max(2, n_comp)
        Z = self.svd.fit_transform(X)
        return normalize(Z)

    def transform_query(self, text):
        x = self.vec.transform([text])
        x_tfidf = normalize(x)
        z = normalize(self.svd.transform(x))
        return x_tfidf, z


class STEncoder:
    """Optional sentence-transformers encoder (only if installed + enabled)."""

    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.emb = None

    def fit_transform_corpus(self, texts):
        self.emb = self.model.encode(
            texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True
        )
        return self.emb

    def transform_query(self, text):
        q = self.model.encode([text], normalize_embeddings=True)
        return None, q


def _build_encoder():
    if os.environ.get("RANKER_USE_ST") == "1":
        try:
            return STEncoder(), "sentence-transformers"
        except Exception as e:  # graceful fallback
            print(f"[ranker] sentence-transformers unavailable ({e}); using LSA.")
    return LSAEncoder(), "tfidf+lsa"


# ------------------------------- main ranker --------------------------------
def rank_candidates(candidates, top_k=100, verbose=True):
    """`candidates`: iterable of raw candidate dicts. Returns list of result dicts."""
    feats, texts = [], []
    for cand in candidates:
        f = extract_features(cand)
        feats.append(f)
        texts.append(f["evidence_text"] or " ")
    n = len(feats)
    if verbose:
        print(f"[ranker] extracted features for {n} candidates")

    encoder, kind = _build_encoder()
    if verbose:
        print(f"[ranker] semantic encoder: {kind}")
    corpus_vecs = encoder.fit_transform_corpus(texts)
    q_tfidf, q_lsa = encoder.transform_query(C.JD_QUERY)

    # semantic similarity = blend of LSA cosine and (when available) raw TF-IDF
    sem_lsa = corpus_vecs @ q_lsa[0]
    if isinstance(encoder, LSAEncoder):
        sem_tfidf = (encoder.tfidf_matrix @ q_tfidf.T).toarray().ravel()
        semantic = 0.55 * sem_lsa + 0.45 * sem_tfidf
    else:
        semantic = sem_lsa
    # squash to [0,1] across the pool
    lo, hi = float(semantic.min()), float(semantic.max())
    semantic = (semantic - lo) / (hi - lo + 1e-9)

    w = C.WEIGHTS
    scores = np.zeros(n, dtype=np.float64)
    for i, f in enumerate(feats):
        comp = f["components"]
        comp["semantic"] = float(semantic[i])
        base = (
            w["semantic"] * comp["semantic"]
            + w["career_evidence"] * comp["career_evidence"]
            + w["title"] * comp["title"]
            + w["experience"] * comp["experience"]
            + w["skill_depth"] * comp["skill_depth"]
            + w["location"] * comp["location"]
            + w["nice_to_have"] * comp["nice_to_have"]
        )
        score = base * f["penalty_factor"] * f["behavioural"]
        if f["is_honeypot"]:
            score = C.HONEYPOT_FLOOR
        scores[i] = score

    order = np.argsort(-scores, kind="stable")
    # deterministic tie-break by candidate_id ascending (validator requires it)
    order = sorted(order, key=lambda i: (-scores[i], feats[i]["candidate_id"]))

    results = []
    for rank_pos, idx in enumerate(order[:top_k], start=1):
        f = feats[idx]
        results.append({
            "candidate_id": f["candidate_id"],
            "rank": rank_pos,
            "score": round(float(scores[idx]), 6),
            "reasoning": build_reasoning(f, rank_pos),
            "components": {k: round(float(v), 4) for k, v in f["components"].items()},
            "behavioural": round(float(f["behavioural"]), 4),
            "penalties": f["facts"]["penalties"],
        })
    # enforce strict non-increasing score for the validator (ties allowed)
    for i in range(1, len(results)):
        if results[i]["score"] > results[i - 1]["score"]:
            results[i]["score"] = results[i - 1]["score"]
    return results


# ----------------------------- reasoning -----------------------------------
def build_reasoning(f, rank_pos):
    """Write 1-2 specific sentences from real facts. Varied, honest, no hallucination."""
    x = f["facts"]
    title = x["title"] or "Candidate"
    yoe = x["yoe"]
    yoe_s = f"{yoe:g} yrs" if isinstance(yoe, (int, float)) else "exp n/a"

    # lead clause: strongest true positive
    lead_bits = [f"{title}, {yoe_s}"]
    if x["ir_evidence"] >= 2:
        lead_bits.append("clear retrieval/ranking/recsys evidence in career history")
    elif x["ir_evidence"] == 1:
        lead_bits.append("some retrieval/ranking signal in past roles")
    elif x["ml_evidence"] >= 2:
        lead_bits.append("applied ML background")
    if x["eval_evidence"] >= 1:
        lead_bits.append("ranking-evaluation experience")
    if x["product_evidence"] >= 1:
        lead_bits.append("production/at-scale work")
    lead = "; ".join(lead_bits) + "."

    # location + availability
    loc_bits = []
    if x["location"]:
        loc_bits.append(str(x["location"]))
    if x["country"] and x["country"].lower() != "india":
        loc_bits.append("outside India" + (", open to relocate" if x["relocate"] else ", no relocation flag"))
    rr = x["response_rate"]
    if isinstance(rr, (int, float)):
        loc_bits.append(f"recruiter response {rr:.0%}")
    di = x["days_inactive"]
    if isinstance(di, (int, float)) and di > 120:
        loc_bits.append(f"inactive ~{int(di)}d")
    avail = "; ".join(loc_bits)

    # honest concerns
    concerns = []
    pen = x["penalties"]
    if "consulting_only" in pen:
        concerns.append("entirely services/consulting career")
    if "research_only" in pen:
        concerns.append("research-leaning, thin production signal")
    if "cv_speech_only" in pen:
        concerns.append("CV/speech focus rather than NLP/IR")
    if "keyword_stuffer" in pen:
        concerns.append("AI skills listed but unsupported by history")
    if "title_chaser" in pen:
        concerns.append("short average tenure")
    if "wrapper_only" in pen:
        concerns.append("LLM-wrapper-only AI exposure")
    if x["integrity_flags"]:
        concerns.append("profile integrity concerns")
    if x["ir_evidence"] == 0 and x["ml_evidence"] == 0 and not concerns:
        concerns.append("limited direct ML/IR evidence")

    out = lead
    if avail:
        out += f" {avail}."
    if concerns and rank_pos > 3:
        out += " Concern: " + ", ".join(concerns[:2]) + "."
    elif concerns:
        out += " Note: " + concerns[0] + "."
    return out[:260]
