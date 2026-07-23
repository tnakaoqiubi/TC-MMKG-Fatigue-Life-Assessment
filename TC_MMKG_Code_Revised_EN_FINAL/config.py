# -*- coding: utf-8 -*-
"""Central configuration for the TC-MMKG prototype.

All secrets are read from environment variables.  Defaults are intentionally
limited to local, non-secret development values so the code can be moved
between machines without editing source files.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "output"
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

for _p in (DATA_DIR, OUTPUT_DIR, TEMP_UPLOAD_DIR, STATIC_DIR, TEMPLATE_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---- LLM / VLM -----------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
# Respect the model explicitly configured by the user.  Do not silently remap
# model names here: the manuscript records the experiment-time model setting,
# and runtime model selection should remain transparent and reproducible.
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

QWEN_API_KEY = os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")).strip()
QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).strip()
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen3.5-plus").strip()
QWEN_VL_FALLBACK_MODEL = os.getenv("QWEN_VL_FALLBACK_MODEL", "qwen3-vl-plus").strip()

# ---- Neo4j ---------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "").strip()

# ---- Sentence-BERT -------------------------------------------------------
SBERT_LOCAL_PATH = os.getenv(
    "SBERT_LOCAL_PATH",
    r"D:/models/paraphrase-multilingual-MiniLM-L12-v2",
).strip()
SBERT_MODEL_NAME = os.getenv(
    "SBERT_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"
).strip()

# ---- Method parameters matching the manuscript --------------------------
# Relation/entity semantic clustering threshold in Sec. 3.1.4.
RELATION_ALIGNMENT_THRESHOLD = float(os.getenv("RELATION_ALIGNMENT_THRESHOLD", "0.80"))
ENTITY_ALIGNMENT_THRESHOLD = float(os.getenv("ENTITY_ALIGNMENT_THRESHOLD", "0.80"))
# Tail-entity semantic-similarity threshold for conflict arbitration.
TAIL_SIMILARITY_THRESHOLD = float(os.getenv("TAIL_SIMILARITY_THRESHOLD", "0.65"))
# Producer agreement threshold delta. The manuscript leaves delta configurable.
DEBATE_TRIGGER_THRESHOLD = float(os.getenv("DEBATE_TRIGGER_THRESHOLD", "0.85"))
MAX_VALIDATION_ITERATIONS = int(os.getenv("MAX_VALIDATION_ITERATIONS", "2"))
TOP_K_CASES = int(os.getenv("TOP_K_CASES", "5"))

# Optional throttling for paid APIs. Keep 0 for local testing.
API_CALL_INTERVAL = float(os.getenv("API_CALL_INTERVAL", "0"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "2"))
# Offline fallback is disabled by default because the manuscript requires actual LLM/VLM extraction.
ALLOW_LLM_FALLBACK = os.getenv("ALLOW_LLM_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}

# ---- Q690D fatigue parameters used consistently across the project -------
# These values are configurable. Nref=2e6, gamma_m=1.1 and m=3 follow the
# manuscript equations for welded components. The FAT=50 MPa, Rm=850 MPa and
# 590 MPa fallback scaling preserve the parameter set used by the original
# prototype damage module so its reported damage level is not silently changed.
# Override them in .env only when a different verified material/detail class is used.
Q690D_RM_MPA = float(os.getenv("Q690D_RM_MPA", "850"))
Q690D_RE_MPA = float(os.getenv("Q690D_RE_MPA", "720"))
Q690D_FAT_MPA = float(os.getenv("Q690D_FAT_MPA", "50"))
Q690D_SN_SLOPE = float(os.getenv("Q690D_SN_SLOPE", "3"))
Q690D_NREF = float(os.getenv("Q690D_NREF", "2000000"))
Q690D_GAMMA_M = float(os.getenv("Q690D_GAMMA_M", "1.1"))

# Fallback load-to-stress scaling is used only when TC-MMKG has no SMX/SMN
# values for a requested interval.
FALLBACK_STRESS_AT_100_PERCENT_MPA = float(
    os.getenv("FALLBACK_STRESS_AT_100_PERCENT_MPA", "590")
)
FALLBACK_STRESS_RATIO = float(os.getenv("FALLBACK_STRESS_RATIO", "-0.3"))


def require_secret(value: str, name: str) -> str:
    """Raise a clear configuration error only when a secret is actually used."""
    if not value:
        raise RuntimeError(
            f"{name} is not configured. Set environment variable {name} before using this feature."
        )
    return value
