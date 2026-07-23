# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def normalize_time(value: Any) -> str:
    """Return `YYYY-MM-DD HH:MM:SS` when possible, otherwise a stripped string."""
    if value is None:
        return ""
    try:
        import pandas as pd

        ts = pd.to_datetime(value, errors="coerce")
        if not pd.isna(ts):
            return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    text = str(value).strip()
    m = re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", text)
    return m.group(0).replace("T", " ") if m else text


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        import pandas as pd
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        v = float(value)
        return v if math.isfinite(v) else None
    text = str(value).replace(",", "")
    m = NUMBER_RE.search(text)
    if not m:
        return None
    try:
        v = float(m.group(0))
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def canonical_triple(t: Sequence[Any]) -> Tuple[str, str, str]:
    return tuple(re.sub(r"\s+", " ", str(x).strip()) for x in t[:3])  # type: ignore[return-value]


def jaccard(a: Iterable[Sequence[Any]], b: Iterable[Sequence[Any]]) -> float:
    sa = {canonical_triple(x) for x in a}
    sb = {canonical_triple(x) for x in b}
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def extract_json(text: str, expected: type | None = None) -> Any:
    """Extract the first outermost JSON object/list from an LLM response."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.I).replace("```", "").strip()
    candidates = []
    for op, cl in (("{", "}"), ("[", "]")):
        start = cleaned.find(op)
        end = cleaned.rfind(cl)
        if start >= 0 and end > start:
            candidates.append(cleaned[start : end + 1])
    candidates.append(cleaned)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if expected is None or isinstance(obj, expected):
                return obj
        except Exception:
            continue
    return None


@lru_cache(maxsize=1)
def get_sentence_model():
    """Lazily load SBERT so importing the application never triggers downloads."""
    from sentence_transformers import SentenceTransformer
    from config import SBERT_LOCAL_PATH, SBERT_MODEL_NAME

    if SBERT_LOCAL_PATH:
        try:
            return SentenceTransformer(SBERT_LOCAL_PATH)
        except Exception:
            pass
    return SentenceTransformer(SBERT_MODEL_NAME)


def encode_texts(texts: Sequence[str]) -> np.ndarray:
    model = get_sentence_model()
    return np.asarray(model.encode(list(texts), convert_to_numpy=True))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
