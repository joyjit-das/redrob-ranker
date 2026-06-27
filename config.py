"""
Central configuration for the Redrob Intelligent Candidate Ranker.

Everything that encodes *domain knowledge* about the job lives here so the
rest of the code stays generic:

  - JD_TEXT / JD_QUERY : the role, plus a distilled "ideal candidate" query
                         used for semantic matching (we match against what the
                         JD *means*, not its verbose prose).
  - Vocabularies       : term lists used by the feature extractor to read
                         career history the way a recruiter would.
  - WEIGHTS            : the fusion weights for the final score.

These are intentionally readable and tunable. The whole point of the challenge
is that the intelligence is in *how we read a profile*, not in a black box.
"""

# Reference "today" for recency features. The dataset's most-recent
# last_active_date is 2026-05-27; we anchor recency to that so the ranking is
# reproducible regardless of when the code is run.
REFERENCE_DATE = "2026-05-27"

# ---------------------------------------------------------------------------
# Job description
# ---------------------------------------------------------------------------
JD_TITLE = "Senior AI Engineer — Founding Team @ Redrob AI"

# A distilled, meaning-focused query for semantic retrieval. We deliberately
# describe the *ideal candidate* rather than pasting the full JD, so cosine
# similarity rewards genuine retrieval/ranking/recommendation experience
# instead of the JD's boilerplate.
JD_QUERY = (
    "senior applied machine learning engineer who has built and shipped "
    "embeddings based retrieval, semantic search, ranking and recommendation "
    "systems to real users in production at a product company. experience with "
    "vector search, hybrid retrieval, reranking, learning to rank, and rigorous "
    "ranking evaluation using ndcg mrr map and a/b testing. strong python. "
    "nlp and information retrieval background. five to nine years of experience, "
    "ideally six to eight, mostly applied ml at product companies rather than "
    "pure research or services consulting. based in or willing to relocate to "
    "pune noida hyderabad mumbai or delhi ncr india."
)

# Full JD text — surfaced in the UI and used as additional semantic context.
JD_TEXT = (
    "Senior AI Engineer on the founding team of Redrob AI, a Series A AI-native "
    "talent intelligence platform (Pune/Noida, hybrid, 5-9 yrs). Own the "
    "intelligence layer: the ranking, retrieval and matching systems behind "
    "candidate-JD search. Must have production experience with embeddings-based "
    "retrieval (sentence-transformers / BGE / E5 / FAISS), vector or hybrid "
    "search infrastructure, strong Python, and rigorous ranking evaluation "
    "(NDCG, MRR, MAP, A/B testing). Nice to have: LLM fine-tuning (LoRA/QLoRA), "
    "learning-to-rank, HR-tech, distributed systems, open-source. Wants a "
    "shipper over a researcher. Explicitly NOT wanted: pure-research-only "
    "backgrounds, recent LangChain-wrapping-OpenAI as the only AI experience, "
    "title-chasers, framework tutorialists, entirely-consulting careers, "
    "computer-vision/speech/robotics specialists without NLP/IR. Located in or "
    "willing to relocate to Noida/Pune (Hyderabad, Mumbai, Delhi NCR welcome)."
)

# ---------------------------------------------------------------------------
# Vocabularies  (lower-cased; matched against career descriptions + summary)
# ---------------------------------------------------------------------------

# The strongest possible evidence: actually building retrieval/ranking/search.
CORE_IR_EVIDENCE = [
    "information retrieval", "retrieval", "embedding", "vector search",
    "vector database", "semantic search", "rerank", "re-rank", "learning to rank",
    "learning-to-rank", "ranking", "recommendation system", "recommender",
    "recommendation engine", "personalization", "personalisation",
    "search relevance", "search engine", "bm25", "nearest neighbor",
    "nearest neighbour", "approximate nearest", " ann ", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
    "sentence-transformers", "sentence transformers", "retrieval augmented",
    "retrieval-augmented", " rag ", "matching engine", "candidate matching",
]

# General ML / NLP — supports fit but is weaker than IR evidence.
ML_EVIDENCE = [
    "machine learning", " ml ", "deep learning", "nlp", "natural language",
    "language model", " llm", "fine-tun", "finetun", "lora", "qlora", "peft",
    "transformer", "pytorch", "tensorflow", "hugging face", "huggingface",
    "model training", "model inference", "mlops", "xgboost", "gradient boost",
    "feature engineering", "classification model", "embeddings model",
]

# Signals that the work was real production at scale (product mindset).
PRODUCT_EVIDENCE = [
    "in production", "to production", "real users", "at scale", "millions of",
    "low latency", "latency", "throughput", "deployed", "shipped", "a/b test",
    "ab test", "experimentation", "served", "high traffic", "qps",
]

# Rigorous evaluation thinking — explicitly demanded by the JD.
EVAL_EVIDENCE = [
    "ndcg", " mrr", " map", "mean average precision", "precision@", "recall@",
    "offline evaluation", "offline metric", "online metric", "evaluation framework",
    "ranking metric", "relevance judgment", "relevance label", "a/b test",
]

# Domains the JD explicitly does NOT want when they dominate without NLP/IR.
NEGATIVE_DOMAIN = [
    "computer vision", "image classification", "object detection",
    "image segmentation", "speech recognition", "speech-to-text", "robotics",
    "autonomous vehicle", "self-driving", "opencv", "point cloud", "slam",
    "lidar", "video analytics", "ocr pipeline",
]

# Research-only red flags (penalized only when no production/product signal).
RESEARCH_FLAGS = [
    "research scientist", "research-only", "academic research", "phd thesis",
    "research lab", "publications in", "published papers", "research intern",
    "postdoc", "post-doc", "purely research",
]

# Framework-tutorialist / thin-LLM-wrapper red flags.
WRAPPER_FLAGS = [
    "langchain", "llamaindex", "prompt engineering", "gpt wrapper",
    "chatgpt wrapper", "openai api",
]

# Indian IT-services / consulting firms (entirely-services careers penalized).
SERVICES_COMPANIES = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mindtree", "ltimindtree", "lti",
    "mphasis", "l&t infotech", "hexaware", "dxc", "genpact", "ntt data",
    "syntel", "mastek", "birlasoft", "zensar", "cybage", "igate", "polaris",
]

# Current-title buckets. Score is the role-fit prior; career evidence can lift
# medium/weak titles, but anti-titles are near-zero by design (a "Marketing
# Manager" with AI skills is the canonical trap in this dataset).
TITLE_STRONG = [
    "ml engineer", "machine learning engineer", "ai research engineer",
    "ai engineer", "applied scientist", "data scientist", "nlp engineer",
    "search engineer", "recommendation systems engineer", "relevance engineer",
    "research engineer", "ai specialist", "(ml)",
]
TITLE_MEDIUM = [
    "data engineer", "senior data engineer", "analytics engineer",
    "backend engineer", "software engineer", "senior software engineer",
    "full stack developer", "platform engineer",
]
TITLE_WEAK = [
    "frontend engineer", "mobile developer", ".net developer", "java developer",
    "qa engineer", "devops engineer", "cloud engineer", "developer",
]
TITLE_ANTI = [
    "hr manager", "accountant", "sales executive", "marketing manager",
    "operations manager", "civil engineer", "mechanical engineer",
    "customer support", "content writer", "graphic designer",
    "business analyst", "project manager",
]

# Skills considered "AI-core" — used ONLY for the gated skill-depth feature and
# for keyword-stuffer detection. We never let the raw skills list drive the
# semantic score, because in this dataset skills are uniformly random noise.
AI_CORE_SKILLS = [
    "nlp", "fine-tuning llms", "llm", "rag", "embeddings", "vector search",
    "information retrieval", "learning to rank", "recommendation systems",
    "semantic search", "pytorch", "tensorflow", "transformers", "hugging face",
    "sentence transformers", "faiss", "elasticsearch", "reranking",
    "machine learning", "deep learning", "mlops",
]

# Location buckets (matched against profile.location, case-insensitive).
LOCATION_PRIME = ["pune", "noida"]
LOCATION_GOOD = ["hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
                 "bangalore", "bengaluru", "ncr", "chennai", "gurugram"]

# ---------------------------------------------------------------------------
# Fusion weights (sum of positive components ~= 1.0 before modifiers)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "semantic":        0.30,  # cosine vs JD query (evidence text only)
    "career_evidence": 0.26,  # IR/ML/eval/product term evidence in history
    "title":           0.16,  # current-title role prior
    "experience":      0.12,  # closeness to the 5-9 (ideal 6-8) band
    "skill_depth":     0.08,  # AI-core skills, GATED by real evidence
    "location":        0.05,  # Pune/Noida/India relocation fit
    "nice_to_have":    0.03,  # LoRA/LTR/HR-tech/OSS/distributed bonuses
}

# Experience band
EXP_IDEAL_LOW, EXP_IDEAL_HIGH = 6.0, 8.0
EXP_OK_LOW, EXP_OK_HIGH = 5.0, 9.0

# Penalty multipliers (applied to the base fit; <1 hurts)
PENALTY = {
    "consulting_only": 0.55,   # entire career at services firms
    "research_only":   0.45,   # pure research, no production
    "cv_speech_only":  0.45,   # CV/speech/robotics, no NLP/IR
    "keyword_stuffer": 0.18,   # many AI skills, zero career evidence
    "title_chaser":    0.82,   # job-hops every <18 months across 4+ roles
    "wrapper_only":    0.70,   # thin LangChain/OpenAI-wrapper as only AI work
}

# Honeypot: if the integrity score crosses this, the candidate is forced to a
# floor score so they can never enter the top-100 (Stage-3 DQ guard).
HONEYPOT_FLOOR = 1e-6
HONEYPOT_FLAG_THRESHOLD = 2  # number of hard inconsistencies to flag
