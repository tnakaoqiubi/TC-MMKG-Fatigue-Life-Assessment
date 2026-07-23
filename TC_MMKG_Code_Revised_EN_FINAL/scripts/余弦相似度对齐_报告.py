# -*- coding: utf-8 -*-
"""Cross-modal semantic alignment and confidence-guided fusion.

Implements manuscript Sec. 3.1.4:
- global relation alignment by SBERT + cosine clustering (threshold 0.80);
- entity/tail alignment at each time point;
- semantic-similarity split between complementary and conflicting information
  (threshold 0.65);
- confidence-difference arbitration with adaptive threshold 2*std(delta C);
- LLM secondary arbitration for high-conflict / low-consistency cases.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    ENTITY_ALIGNMENT_THRESHOLD,
    OUTPUT_DIR,
    RELATION_ALIGNMENT_THRESHOLD,
    TAIL_SIMILARITY_THRESHOLD,
)
from scripts.common import clean_text, cosine, encode_texts, extract_json, parse_float


CANONICAL_RELATION_ALIASES = {
    "最大应力": "SMX", "最大应力 (SMX)": "SMX", "最大应力值 (SMX)": "SMX", "SMX值": "SMX",
    "最小应力": "SMN", "最小应力 (SMN)": "SMN", "最小应力值 (SMN)": "SMN", "SMN值": "SMN",
    "最大位移": "DMX", "最大位移 (DMX)": "DMX", "DMX值": "DMX",
    "实际载荷": "实际吊重量", "额定载荷": "额定吊重量", "臂长": "主臂长度", "主臂长": "主臂长度",
    "幅度": "工作幅度", "工作半径": "工作幅度", "载荷百分比": "力矩百分比",
}

PROTECTED_RELATIONS = {
    "SMX", "SMN", "DMX", "实际吊重量", "额定吊重量", "主臂长度", "角度",
    "工作幅度", "力矩百分比", "运行时间", "属于车辆", "最大应力位置",
}

ZERO_SHOT_FUSION_PROMPT = """
You are the zero-shot fusion reviewer described in the TC-MMKG pipeline. For one time point, review the
semantically aligned multimodal candidate triples. Apply only these operations: union, format consistency,
exact/semantic redundancy removal. Do NOT invent facts and do NOT resolve genuine cross-modal numeric conflicts;
those are handled by the confidence arbitration stage. Return JSON only: {{"keep_indices":[0,1,...]}}.
Candidates: {candidates}
"""

ARBITRATION_PROMPT = """
You are the secondary arbitrator for multimodal crane knowledge fusion.
At the same time point, two semantically similar tail entities conflict after alignment.
Use engineering consistency and the supplied confidence/evidence metadata. Select one candidate only.
Do not invent a new value. Return JSON only: {{"selected_index": 0}} or {{"selected_index": 1}}.
Time point: {time_point}
Relation: {relation}
Candidates: {candidates}
"""


def _ensure_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    required = {"head", "relation", "tail"}
    if not required.issubset(df.columns):
        if len(df.columns) >= 3:
            df = df.rename(columns={df.columns[0]: "head", df.columns[1]: "relation", df.columns[2]: "tail"})
        else:
            raise ValueError(f"{source} CSV must contain head, relation, tail columns.")
    out = df.copy()
    out["head"] = out["head"].astype(str).str.strip()
    out["relation"] = out["relation"].astype(str).str.strip()
    out["tail"] = out["tail"].astype(str).str.strip()
    out["source"] = out.get("source", source)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.7).clip(0, 1) if "confidence" in out.columns else 0.7
    out["validation_decision"] = out.get("validation_decision", "legacy_input")
    return out


def align_terms(terms: Sequence[str], threshold: float, protected: Iterable[str] | None = None) -> Dict[str, str]:
    unique = list(dict.fromkeys(clean_text(t) for t in terms if clean_text(t)))
    if len(unique) <= 1:
        return {x: x for x in unique}
    protected_set = set(protected or [])
    freq = Counter(clean_text(x) for x in terms)
    mapping = {x: x for x in unique if x in protected_set}
    candidates = [x for x in unique if x not in protected_set]
    if len(candidates) <= 1:
        mapping.update({x: x for x in candidates})
        return mapping
    vectors = encode_texts(candidates)
    used = set()
    for i, term in enumerate(candidates):
        if i in used:
            continue
        cluster = [i]
        for j in range(i + 1, len(candidates)):
            if j in used:
                continue
            if cosine(vectors[i], vectors[j]) >= threshold:
                cluster.append(j)
        members = [candidates[k] for k in cluster]
        standard = max(members, key=lambda x: (freq[x], -unique.index(x)))
        for k in cluster:
            mapping[candidates[k]] = standard
            used.add(k)
    return mapping


def _unit_family(text: str) -> str:
    t = text.lower()
    for u in ("mpa", "pa", "kn", "t", "kg", "m", "%", "h", "rpm", "℃", "°"):
        if u in t:
            return u
    return ""


def tail_similarity(a: str, b: str) -> float:
    """Semantic similarity with a numeric-aware branch for engineering values."""
    fa, fb = parse_float(a), parse_float(b)
    if fa is not None and fb is not None:
        ua, ub = _unit_family(a), _unit_family(b)
        if ua and ub and ua != ub:
            return 0.0
        scale = max(abs(fa), abs(fb), 1e-9)
        rel_err = abs(fa - fb) / scale
        return max(0.0, 1.0 - rel_err)
    vecs = encode_texts([a, b])
    return cosine(vecs[0], vecs[1])


def _llm_arbitrate(time_point: str, relation: str, candidates: List[Dict]) -> int | None:
    if not DEEPSEEK_API_KEY:
        return None
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{
                "role": "user",
                "content": ARBITRATION_PROMPT.format(
                    time_point=time_point,
                    relation=relation,
                    candidates=json.dumps(candidates, ensure_ascii=False),
                ),
            }],
            temperature=0,
            max_tokens=120,
        )
        obj = extract_json(resp.choices[0].message.content or "", dict)
        if obj and obj.get("selected_index") in (0, 1):
            return int(obj["selected_index"])
    except Exception:
        return None
    return None


def _llm_zero_shot_review(group: pd.DataFrame) -> pd.DataFrame:
    if not DEEPSEEK_API_KEY or len(group) <= 1:
        return group
    records = []
    for i, (_, row) in enumerate(group.reset_index(drop=True).iterrows()):
        records.append({"index": i, "head": str(row["head"]), "relation": str(row["relation"]), "tail": str(row["aligned_tail"]), "source": str(row["source"])})
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role":"user","content":ZERO_SHOT_FUSION_PROMPT.format(candidates=json.dumps(records,ensure_ascii=False))}],
            temperature=0, max_tokens=300,
        )
        obj = extract_json(resp.choices[0].message.content or "", dict) or {}
        keep = obj.get("keep_indices")
        if isinstance(keep, list) and keep:
            valid = [int(i) for i in keep if isinstance(i,(int,float)) and 0 <= int(i) < len(group)]
            if valid:
                return group.reset_index(drop=True).iloc[sorted(set(valid))].copy()
    except Exception:
        pass
    return group


def _standardize_relations(df: pd.DataFrame, relation_map: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["original_relation"] = out["relation"]
    out["relation"] = out["relation"].map(lambda x: relation_map.get(x, x))
    return out


def _entity_alignment_for_time(group: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    # Align tails within standardized relation groups to avoid merging unrelated entities.
    for relation, rg in group.groupby("relation"):
        tails = rg["tail"].astype(str).tolist()
        # Numeric values are preserved; descriptive synonyms are clustered.
        descriptive = [x for x in tails if parse_float(x) is None]
        local = align_terms(descriptive, ENTITY_ALIGNMENT_THRESHOLD) if descriptive else {}
        mapping.update(local)
        for t in tails:
            mapping.setdefault(t, t)
    return mapping


def _candidate_dict(row: pd.Series) -> Dict:
    return {
        "tail": str(row["tail"]),
        "confidence": float(row["confidence"]),
        "source": str(row["source"]),
        "validation_decision": str(row.get("validation_decision", "")),
    }


def fuse_multimodal(text_csv: str, image_csv: str, output_csv: str | None = None, report_file: str | None = None):
    started_at = time.perf_counter()
    text_path, image_path = Path(text_csv), Path(image_csv)
    print(f"[Fusion] start: text={text_path} | image={image_path}", flush=True)
    if not text_path.exists() or not image_path.exists():
        raise FileNotFoundError("Text or image triple CSV does not exist.")
    text_df = _ensure_columns(pd.read_csv(text_path, encoding="utf-8-sig"), "text")
    image_df = _ensure_columns(pd.read_csv(image_path, encoding="utf-8-sig"), "image")
    print(f"[Fusion 1/5] loaded: text_triples={len(text_df)}, image_triples={len(image_df)}", flush=True)

    all_relations = text_df["relation"].tolist() + image_df["relation"].tolist()
    # Apply deterministic domain aliases first, then cluster the remaining semantic variants with SBERT.
    alias_normalized = [CANONICAL_RELATION_ALIASES.get(r, r) for r in all_relations]
    clustered = align_terms(alias_normalized, RELATION_ALIGNMENT_THRESHOLD, protected=PROTECTED_RELATIONS)
    relation_map = {r: clustered.get(CANONICAL_RELATION_ALIASES.get(r, r), CANONICAL_RELATION_ALIASES.get(r, r)) for r in set(all_relations)}
    text_df = _standardize_relations(text_df, relation_map)
    image_df = _standardize_relations(image_df, relation_map)
    combined = pd.concat([text_df, image_df], ignore_index=True)
    print(f"[Fusion 2/5] relation alignment complete: {len(set(all_relations))} -> {len(set(relation_map.values()))} relations", flush=True)

    if output_csv is None:
        base = re.sub(r"_triples_merged$|_merged$", "", text_path.stem)
        output_csv = str(Path(OUTPUT_DIR) / f"{base}_triples_multimodal_fused.csv")
    if report_file is None:
        report_file = str(Path(output_csv).with_suffix(".alignment_report.txt"))

    # Per-time entity alignment.
    entity_maps: Dict[str, Dict[str, str]] = {}
    aligned_parts = []
    grouped_heads = list(combined.groupby("head", sort=True))
    total_heads = len(grouped_heads)
    print(f"[Fusion 3/5] entity alignment: {total_heads} time points", flush=True)
    for idx, (time_point, group) in enumerate(grouped_heads, 1):
        emap = _entity_alignment_for_time(group)
        entity_maps[str(time_point)] = emap
        g = group.copy()
        g["aligned_tail"] = g["tail"].map(lambda x: emap.get(str(x), str(x)))
        aligned_parts.append(g)
        if idx == 1 or idx % 25 == 0 or idx == total_heads:
            print(f"[Fusion 3/5] entity alignment progress: {idx}/{total_heads}", flush=True)
    aligned = pd.concat(aligned_parts, ignore_index=True) if aligned_parts else combined.assign(aligned_tail=combined["tail"])
    # Union / format standardization / exact redundancy removal are deterministic here.
    # IMPORTANT: do NOT call an LLM once for every time point. That creates hundreds of
    # unnecessary sequential API calls and makes the Flask request appear frozen.
    # The manuscript-required LLM is reserved for genuinely ambiguous, low-confidence
    # cross-modal conflicts in the secondary-arbitration stage below.

    # First remove exact duplicates while retaining max confidence.
    aligned = (
        aligned.sort_values("confidence", ascending=False)
        .drop_duplicates(subset=["head", "relation", "aligned_tail", "source"], keep="first")
        .reset_index(drop=True)
    )

    # Collect semantically similar cross-modal conflicts and their confidence gaps.
    print("[Fusion 4/5] scanning cross-modal semantic conflicts...", flush=True)
    conflicts: List[Tuple[str, str, pd.Series, pd.Series, float, float]] = []
    for (time_point, relation), group in aligned.groupby(["head", "relation"], sort=True):
        text_rows = [row for _, row in group[group["source"].astype(str).str.contains("text")].iterrows()]
        image_rows = [row for _, row in group[group["source"].astype(str).str.contains("image|filename", regex=True)].iterrows()]
        for tr in text_rows:
            for ir in image_rows:
                sim = tail_similarity(str(tr["aligned_tail"]), str(ir["aligned_tail"]))
                if sim >= TAIL_SIMILARITY_THRESHOLD and str(tr["aligned_tail"]) != str(ir["aligned_tail"]):
                    gap = abs(float(tr["confidence"]) - float(ir["confidence"]))
                    conflicts.append((str(time_point), str(relation), tr, ir, sim, gap))
    gaps = np.array([x[5] for x in conflicts], dtype=float)
    theta_dyn = float(2.0 * np.std(gaps)) if len(gaps) > 1 else 0.0
    print(f"[Fusion 4/5] candidate conflicts={len(conflicts)}, theta_dyn={theta_dyn:.6f}", flush=True)

    consumed = set()
    fused_rows: List[Dict] = []
    conflict_records: List[Dict] = []

    def row_key(row: pd.Series) -> Tuple[str, str, str, str]:
        return (str(row["head"]), str(row["relation"]), str(row["aligned_tail"]), str(row["source"]))

    llm_arbitrations = 0
    for time_point, relation, tr, ir, sim, gap in conflicts:
        tk, ik = row_key(tr), row_key(ir)
        if tk in consumed or ik in consumed:
            continue
        tc, ic = _candidate_dict(tr), _candidate_dict(ir)
        if gap > theta_dyn:
            selected = tr if float(tr["confidence"]) >= float(ir["confidence"]) else ir
            arbitration = "confidence_guided"
            consistency = "resolved"
        else:
            llm_arbitrations += 1
            if llm_arbitrations == 1 or llm_arbitrations % 10 == 0:
                print(f"[Fusion 5/5] LLM secondary arbitration #{llm_arbitrations}: {time_point} / {relation}", flush=True)
            idx = _llm_arbitrate(time_point, relation, [tc, ic])
            if idx is None:
                # Deterministic fallback: higher confidence; ties preserve both evidence by choosing text
                # only for identical semantic content. This avoids random behavior when API is unavailable.
                idx = 0 if float(tr["confidence"]) >= float(ir["confidence"]) else 1
            selected = tr if idx == 0 else ir
            arbitration = "llm_secondary" if DEEPSEEK_API_KEY else "deterministic_secondary"
            consistency = "high-conflict/low-consistency"
        fused_rows.append({
            "head": str(selected["head"]),
            "relation": str(selected["relation"]),
            "tail": str(selected["aligned_tail"]),
            "confidence": float(selected["confidence"]),
            "source": str(selected["source"]),
            "consistency": consistency,
            "arbitration": arbitration,
        })
        consumed.update({tk, ik})
        conflict_records.append({
            "time": time_point,
            "relation": relation,
            "text_tail": str(tr["aligned_tail"]),
            "image_tail": str(ir["aligned_tail"]),
            "semantic_similarity": sim,
            "confidence_gap": gap,
            "theta_dynamic": theta_dyn,
            "selected": str(selected["aligned_tail"]),
            "arbitration": arbitration,
        })

    # Retain all non-conflicting or semantically dissimilar complementary triples.
    for _, row in aligned.iterrows():
        key = row_key(row)
        if key in consumed:
            continue
        fused_rows.append({
            "head": str(row["head"]),
            "relation": str(row["relation"]),
            "tail": str(row["aligned_tail"]),
            "confidence": float(row["confidence"]),
            "source": str(row["source"]),
            "consistency": "complementary_or_unique",
            "arbitration": "retain",
        })

    fused = pd.DataFrame(fused_rows)
    fused = (
        fused.sort_values(["head", "relation", "confidence"], ascending=[True, True, False])
        .drop_duplicates(subset=["head", "relation", "tail"], keep="first")
        .reset_index(drop=True)
    )
    fused.to_csv(output_csv, index=False, encoding="utf-8-sig")

    conflict_rate = len(conflict_records) / max(len(fused), 1)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("TC-MMKG Cross-modal Alignment and Fusion Report\n")
        f.write("=" * 72 + "\n")
        f.write(f"Relation threshold: {RELATION_ALIGNMENT_THRESHOLD:.2f}\n")
        f.write(f"Entity threshold: {ENTITY_ALIGNMENT_THRESHOLD:.2f}\n")
        f.write(f"Tail conflict threshold: {TAIL_SIMILARITY_THRESHOLD:.2f}\n")
        f.write(f"Adaptive confidence threshold theta_dyn = 2*std(DeltaC): {theta_dyn:.6f}\n")
        f.write(f"Original relations: {len(set(all_relations))}\n")
        f.write(f"Standardized relations: {len(set(relation_map.values()))}\n")
        f.write(f"Fused triples: {len(fused)}\n")
        f.write(f"Detected semantically similar conflicts: {len(conflict_records)}\n")
        f.write(f"Observed conflict ratio (conflicts/fused triples): {conflict_rate:.4%}\n\n")
        f.write("[Relation mapping]\n")
        rev = defaultdict(list)
        for orig, std in relation_map.items():
            rev[std].append(orig)
        for std in sorted(rev):
            f.write(f"{std} <- {', '.join(sorted(rev[std]))}\n")
        f.write("\n[Conflict arbitration]\n")
        for rec in conflict_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.write(f"\nLLM secondary arbitrations: {llm_arbitrations}\n")

    elapsed = time.perf_counter() - started_at
    print(f"[Fusion] complete: fused_triples={len(fused)}, conflicts={len(conflict_records)}, LLM_arbitrations={llm_arbitrations}, elapsed={elapsed:.2f}s", flush=True)
    print(f"[Fusion] output={output_csv}", flush=True)
    return str(output_csv), str(report_file)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python 余弦相似度对齐_报告.py <text.csv> <image.csv>")
    print(fuse_multimodal(sys.argv[1], sys.argv[2]))
