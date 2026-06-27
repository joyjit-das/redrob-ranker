"""
Feature engineering — the part that "reads" a candidate.

`extract_features(candidate)` turns a raw profile into:
  * an `evidence_text` string (titles + descriptions + summary) used for
    semantic vectorisation. NB: the random `skills` list is *excluded* here on
    purpose, so keyword-stuffing cannot inflate semantic similarity.
  * component scores in [0,1]: title, career_evidence, experience, skill_depth,
    location, nice_to_have.
  * penalty multipliers (consulting/research/cv/stuffer/title-chaser/wrapper).
  * a behavioural availability multiplier from the 23 Redrob signals.
  * integrity checks that flag honeypots (subtly impossible profiles).
  * `facts`: concrete values used to write specific, non-templated reasoning.

All thresholds come from config.py so the logic stays auditable.
"""
from datetime import date
from . import config as C


# ----------------------------- small helpers -------------------------------
def _count_hits(text, vocab):
    """Number of distinct vocab phrases present in text."""
    return sum(1 for term in vocab if term in text)


def _any(text, vocab):
    return any(term in text for term in vocab)


def _days_between(d_from, d_to):
    try:
        a = date.fromisoformat(d_from)
        b = date.fromisoformat(d_to)
        return (b - a).days
    except Exception:
        return None


def _norm(x, lo, hi):
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


# ------------------------------- main entry --------------------------------
def extract_features(cand):
    p = cand.get("profile", {})
    sig = cand.get("redrob_signals", {})
    history = cand.get("career_history", []) or []
    skills = cand.get("skills", []) or []

    title = (p.get("current_title") or "").lower()
    summary = (p.get("summary") or "").lower()
    headline = (p.get("headline") or "").lower()
    yoe = float(p.get("years_of_experience") or 0)

    # Evidence text: titles + role descriptions + summary + headline.
    # Descriptions are repeated once via inclusion of both title and description;
    # the skills array is intentionally NOT included.
    parts = [headline, summary]
    for h in history:
        parts.append((h.get("title") or "").lower())
        parts.append((h.get("description") or "").lower())
    evidence_text = " . ".join(parts)
    padded = " " + evidence_text + " "  # so " ml " / " rag " match at edges

    # ---- career evidence ---------------------------------------------------
    ir = _count_hits(padded, C.CORE_IR_EVIDENCE)
    ml = _count_hits(padded, C.ML_EVIDENCE)
    prod = _count_hits(padded, C.PRODUCT_EVIDENCE)
    ev = _count_hits(padded, C.EVAL_EVIDENCE)
    # IR evidence is worth the most; eval + product mindset matter; ML is base.
    career_evidence = _norm(
        2.2 * min(ir, 4) + 1.0 * min(ml, 4) + 1.3 * min(ev, 3) + 0.8 * min(prod, 4),
        0, 2.2 * 4 + 1.0 * 4 + 1.3 * 3 + 0.8 * 4,
    )

    # ---- title prior -------------------------------------------------------
    if _any(title, C.TITLE_STRONG):
        title_score = 1.0
    elif _any(title, C.TITLE_MEDIUM):
        # medium titles are lifted by real evidence, capped without it
        title_score = 0.45 + 0.45 * career_evidence
    elif _any(title, C.TITLE_ANTI):
        title_score = 0.02
    elif _any(title, C.TITLE_WEAK):
        title_score = 0.18 + 0.25 * career_evidence
    else:
        title_score = 0.30 + 0.30 * career_evidence

    # ---- experience fit ----------------------------------------------------
    if C.EXP_IDEAL_LOW <= yoe <= C.EXP_IDEAL_HIGH:
        exp_score = 1.0
    elif C.EXP_OK_LOW <= yoe <= C.EXP_OK_HIGH:
        exp_score = 0.88
    else:
        # gaussian-ish falloff around the centre of the ideal band
        centre = (C.EXP_IDEAL_LOW + C.EXP_IDEAL_HIGH) / 2.0
        exp_score = max(0.12, 1.0 - (abs(yoe - centre) / 6.0))

    # ---- skill depth (GATED by evidence — anti-stuffer) --------------------
    skill_names = [(s.get("name") or "").lower() for s in skills]
    ai_skill_count = sum(1 for n in skill_names if n in C.AI_CORE_SKILLS)
    advanced_ai = sum(
        1 for s in skills
        if (s.get("name") or "").lower() in C.AI_CORE_SKILLS
        and s.get("proficiency") in ("advanced", "expert")
    )
    raw_skill_depth = _norm(min(advanced_ai, 5) + 0.3 * min(ai_skill_count, 6), 0, 5 + 0.3 * 6)
    # Skills only count if the career history backs them up.
    evidence_gate = min(1.0, 0.15 + 0.85 * career_evidence) if (ir + ml) > 0 else 0.15
    skill_depth = raw_skill_depth * evidence_gate

    # ---- location ----------------------------------------------------------
    loc = (p.get("location") or "").lower()
    country = (p.get("country") or "").lower()
    relocate = bool(sig.get("willing_to_relocate"))
    if _any(loc, C.LOCATION_PRIME):
        location_score = 1.0
    elif _any(loc, C.LOCATION_GOOD):
        location_score = 0.85
    elif country == "india":
        location_score = 0.72 if relocate else 0.62
    else:
        location_score = 0.42 if relocate else 0.32

    # ---- nice-to-have bonuses ---------------------------------------------
    nice = 0.0
    if _any(padded, ["lora", "qlora", "peft", "fine-tun", "finetun"]):
        nice += 0.35
    if _any(padded, ["learning to rank", "learning-to-rank", "xgboost", "gradient boost"]):
        nice += 0.30
    if _any(padded, ["hr-tech", "hr tech", "recruit", "talent", "marketplace", "hiring"]):
        nice += 0.20
    if _any(padded, ["distributed", "large-scale inference", "inference optimization"]):
        nice += 0.15
    gh = float(sig.get("github_activity_score") or -1)
    if gh > 40:
        nice += 0.20
    nice_to_have = min(1.0, nice)

    # ---- penalties ---------------------------------------------------------
    penalties = {}
    companies = [(h.get("company") or "").lower() for h in history] + \
                [(p.get("current_company") or "").lower()]
    is_services = [any(s in c for s in C.SERVICES_COMPANIES) for c in companies if c]
    if is_services and all(is_services):
        penalties["consulting_only"] = C.PENALTY["consulting_only"]

    has_production = (prod > 0) or _any(padded, ["product company", "startup", "saas"])
    if _any(padded, C.RESEARCH_FLAGS) and not has_production:
        penalties["research_only"] = C.PENALTY["research_only"]

    neg = _count_hits(padded, C.NEGATIVE_DOMAIN)
    if neg >= 2 and ir == 0 and not _any(padded, ["nlp", "natural language", "retrieval"]):
        penalties["cv_speech_only"] = C.PENALTY["cv_speech_only"]

    # keyword stuffer: many AI skills, no career evidence, weak/anti title
    weak_or_anti = _any(title, C.TITLE_ANTI) or _any(title, C.TITLE_WEAK) or \
        not (_any(title, C.TITLE_STRONG) or _any(title, C.TITLE_MEDIUM))
    if ai_skill_count >= 4 and (ir + ml) == 0 and weak_or_anti:
        penalties["keyword_stuffer"] = C.PENALTY["keyword_stuffer"]

    # title chaser: 4+ roles, average tenure < 18 months
    tenures = [h.get("duration_months") for h in history if h.get("duration_months")]
    if len(history) >= 4 and tenures and (sum(tenures) / len(tenures)) < 18:
        penalties["title_chaser"] = C.PENALTY["title_chaser"]

    # thin wrapper only: langchain/openai-wrapper present, low yoe, weak evidence
    if _any(padded, C.WRAPPER_FLAGS) and ir == 0 and ml <= 1 and yoe < 4:
        penalties["wrapper_only"] = C.PENALTY["wrapper_only"]

    penalty_factor = 1.0
    for v in penalties.values():
        penalty_factor *= v

    # ---- integrity / honeypot ---------------------------------------------
    # We only trust signals the challenge documents as truly impossible:
    #  (a) total career tenure greatly exceeding stated experience,
    #  (b) a single role longer than the whole career,
    #  (c) several advanced/expert skills with zero months of usage.
    # Skill durations and date spans are independently-generated noise in this
    # dataset, so we deliberately do NOT flag minor mismatches there.
    integrity_flags = []
    total_tenure = sum(t for t in tenures) if tenures else 0
    if yoe > 0 and total_tenure > yoe * 12 + 30:
        integrity_flags.append("career tenure far exceeds total experience")
    for h in history:
        d = h.get("duration_months") or 0
        if yoe > 0 and d > yoe * 12 + 18:
            integrity_flags.append("single role longer than entire career")
            break
    zero_dur_expert = sum(
        1 for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and (s.get("duration_months") or 0) == 0
    )
    if zero_dur_expert >= 4:
        integrity_flags.append("multiple expert skills with zero usage time")
    is_honeypot = len(integrity_flags) >= C.HONEYPOT_FLAG_THRESHOLD
    # one lone impossibility: demote (suspicious) but don't hard-exclude
    if len(integrity_flags) == 1:
        penalties["integrity_soft"] = 0.6
        penalty_factor *= 0.6

    # ---- behavioural availability multiplier ------------------------------
    avail = _behavioural_multiplier(sig)

    facts = _collect_facts(p, sig, history, ir, ml, ev, prod, ai_skill_count,
                           penalties, integrity_flags)

    return {
        "candidate_id": cand.get("candidate_id"),
        "evidence_text": evidence_text,
        "components": {
            "semantic": 0.0,  # filled in by the ranker (needs the corpus)
            "career_evidence": career_evidence,
            "title": title_score,
            "experience": exp_score,
            "skill_depth": skill_depth,
            "location": location_score,
            "nice_to_have": nice_to_have,
        },
        "penalty_factor": penalty_factor,
        "penalties": penalties,
        "behavioural": avail,
        "is_honeypot": is_honeypot,
        "integrity_flags": integrity_flags,
        "facts": facts,
    }


def _behavioural_multiplier(sig):
    """Map the 23 platform signals to a modifier in roughly [0.55, 1.08].

    A perfect-on-paper candidate who is inactive and unresponsive is, for
    hiring purposes, not actually available — exactly as the JD warns.
    """
    last_active = sig.get("last_active_date") or C.REFERENCE_DATE
    days = _days_between(last_active, C.REFERENCE_DATE)
    if days is None:
        days = 180
    # recency: <=30d -> 1.0, ~180d+ -> 0.6
    recency = max(0.6, 1.0 - max(0, days - 30) / 375.0)

    resp = float(sig.get("recruiter_response_rate") or 0)
    resp_f = 0.72 + 0.28 * max(0.0, min(1.0, resp))  # 0->0.72, 1->1.0

    otw = 1.0 if sig.get("open_to_work_flag") else 0.9

    notice = sig.get("notice_period_days")
    if notice is None:
        notice_f = 0.95
    elif notice <= 30:
        notice_f = 1.0
    else:
        notice_f = max(0.85, 1.0 - (notice - 30) / 600.0)

    icr = sig.get("interview_completion_rate")
    icr_f = 0.95 + 0.05 * (icr if isinstance(icr, (int, float)) and icr >= 0 else 0.5)

    completeness = float(sig.get("profile_completeness_score") or 50) / 100.0
    comp_f = 0.92 + 0.08 * completeness

    saved = sig.get("saved_by_recruiters_30d") or 0
    saved_f = 1.0 + min(0.04, saved / 250.0)  # tiny demand bonus

    mult = recency * resp_f * otw * notice_f * icr_f * comp_f * saved_f
    return max(0.55, min(1.08, mult))


def _collect_facts(p, sig, history, ir, ml, ev, prod, ai_skill_count,
                   penalties, integrity_flags):
    """Concrete, true facts for reasoning generation (no hallucination)."""
    days = _days_between(sig.get("last_active_date") or C.REFERENCE_DATE, C.REFERENCE_DATE)
    return {
        "title": p.get("current_title"),
        "yoe": p.get("years_of_experience"),
        "company": p.get("current_company"),
        "location": p.get("location"),
        "country": p.get("country"),
        "ir_evidence": ir,
        "ml_evidence": ml,
        "eval_evidence": ev,
        "product_evidence": prod,
        "ai_skill_count": ai_skill_count,
        "response_rate": sig.get("recruiter_response_rate"),
        "days_inactive": days,
        "notice_days": sig.get("notice_period_days"),
        "open_to_work": sig.get("open_to_work_flag"),
        "relocate": sig.get("willing_to_relocate"),
        "github": sig.get("github_activity_score"),
        "n_roles": len(history),
        "penalties": list(penalties.keys()),
        "integrity_flags": integrity_flags,
    }
