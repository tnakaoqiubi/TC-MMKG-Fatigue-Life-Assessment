# -*- coding: utf-8 -*-
"""Text-modality knowledge extraction for TC-MMKG.

Implements the manuscript's Producer–Verifier–Integrator pipeline:
1) each operational record is normalized into a natural-language description;
2) two independent producer sessions extract time-point-centred triples;
3) Jaccard agreement is evaluated and evidence-grounded adversarial verification
   is triggered for inconsistent triples;
4) iterative validation corrects heads, relations, contradictions and redundancy;
5) the integrator outputs a unique triple set with confidence annotations.

The public API `process_text_excel()` is kept compatible with the original code.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from config import (
    API_CALL_INTERVAL,
    API_MAX_RETRIES,
    ALLOW_LLM_FALLBACK,
    DEBATE_TRIGGER_THRESHOLD,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    MAX_VALIDATION_ITERATIONS,
    OUTPUT_DIR,
    require_secret,
)
from scripts.common import canonical_triple, clean_text, extract_json, jaccard, normalize_time

Triple = Tuple[str, str, str]

FIELD_SPECS = [
    ("车辆代码", 0, ""),
    ("型号", 1, ""),
    ("时间", 2, ""),
    ("位于区域", 3, ""),
    ("ACC状态", 4, ""),
    ("油压", 5, "MPa"),
    ("水温", 6, "℃"),
    ("发动机转速", 7, "rpm"),
    ("运行时间", 8, "h"),
    ("车速", 9, "km/h"),
    ("实际吊重量", 10, "t"),
    ("额定吊重量", 11, "t"),
    ("主臂长度", 12, "m"),
    ("角度", 13, "°"),
    ("工作幅度", 14, "m"),
    ("工况代码", 15, ""),
    ("倍率", 16, ""),
    ("力矩百分比", 17, "%"),
    ("力限器故障码", 18, ""),
    ("控制类故障码", 19, ""),
    ("发动机故障码", 20, ""),
    ("绑定状态", 21, ""),
    ("锁车状态", 22, ""),
]

RELATION_ALIASES = {
    "车辆编号": "车辆代码",
    "车辆ID": "车辆代码",
    "车辆型号": "型号",
    "区域": "位于区域",
    "所在区域": "位于区域",
    "冷却液温度": "水温",
    "实际载荷": "实际吊重量",
    "额定载荷": "额定吊重量",
    "臂长": "主臂长度",
    "主臂长": "主臂长度",
    "幅度": "工作幅度",
    "半径": "工作幅度",
    "载荷百分比": "力矩百分比",
    "力矩%": "力矩百分比",
    "力限器故障": "力限器故障码",
    "控制故障码": "控制类故障码",
}
STANDARD_RELATIONS = {x[0] for x in FIELD_SPECS if x[0] != "时间"}


@dataclass
class Candidate:
    head: str
    relation: str
    tail: str
    confidence: float
    decision: str

    def triple(self) -> Triple:
        return self.head, self.relation, self.tail


def _client() -> OpenAI:
    return OpenAI(
        api_key=require_secret(DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
    )


def _call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 3500) -> str:
    """Call the configured DeepSeek model with retry and require a real final answer."""
    last_error = None
    for attempt in range(max(1, API_MAX_RETRIES + 1)):
        try:
            response = _client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if API_CALL_INTERVAL > 0:
                time.sleep(API_CALL_INTERVAL)
            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("DeepSeek returned empty content")
            return content
        except Exception as exc:
            last_error = exc
            if attempt < API_MAX_RETRIES:
                time.sleep(min(1.5 * (attempt + 1), 4.0))
    raise RuntimeError(f"DeepSeek LLM call failed after retries: {last_error}") from last_error


def _find_column(df: pd.DataFrame, relation: str, fallback_index: int) -> str:
    aliases = {
        "车辆代码": ["车辆代码", "起重机代码", "crane code", "vehicle code"],
        "型号": ["型号", "车辆型号", "model"],
        "时间": ["时间", "时间点", "timestamp", "time"],
        "位于区域": ["所在区域", "区域", "location", "region"],
        "水温": ["水温", "冷却液温度", "coolant temperature"],
        "主臂长度": ["主臂长度", "长度", "boom length"],
        "力矩百分比": ["力矩百分比", "moment percentage", "torque percentage"],
    }
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for name in aliases.get(relation, [relation]):
        hit = normalized.get(name.strip().lower())
        if hit is not None:
            return hit
    if fallback_index < len(df.columns):
        return df.columns[fallback_index]
    raise ValueError(
        f"Input Excel does not contain the required field '{relation}' and has only {len(df.columns)} columns."
    )


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {rel: _find_column(df, rel, idx) for rel, idx, _ in FIELD_SPECS}


def normalize_record(row: pd.Series, columns: Dict[str, str]) -> Dict[str, str]:
    record: Dict[str, str] = {}
    for relation, _, unit in FIELD_SPECS:
        raw = row.get(columns[relation])
        if relation == "时间":
            value = normalize_time(raw)
        else:
            value = clean_text(raw)
        if value and unit and not value.endswith(unit):
            value = f"{value} {unit}".strip()
        record[relation] = value
    return record


def row_to_sentence(row: pd.Series, columns: Dict[str, str] | None = None) -> str:
    if columns is None:
        # Backward-compatible positional mode.
        fake = pd.DataFrame(columns=list(range(max(x[1] for x in FIELD_SPECS) + 1)))
        columns = {rel: idx for rel, idx, _ in FIELD_SPECS}  # type: ignore[assignment]
        record = {}
        for rel, idx, unit in FIELD_SPECS:
            raw = row.iloc[idx] if idx < len(row) else ""
            value = normalize_time(raw) if rel == "时间" else clean_text(raw)
            if value and unit and not value.endswith(unit):
                value = f"{value} {unit}"
            record[rel] = value
    else:
        record = normalize_record(row, columns)

    time_point = record.get("时间", "")
    fields = [f"{k}={v}" for k, v in record.items() if k != "时间" and v != ""]
    return f"Time point {time_point}. " + "; ".join(fields) + "."


def deterministic_triples(record: Dict[str, str]) -> List[Triple]:
    """Evidence-preserving fallback and post-validation baseline."""
    head = record.get("时间", "")
    if not head:
        return []
    return [(head, rel, value) for rel, value in record.items() if rel != "时间" and value != ""]


def normalize_relation(relation: str) -> str:
    rel = re.sub(r"\s+", "", str(relation).strip())
    if rel in RELATION_ALIASES:
        return RELATION_ALIASES[rel]
    for std in STANDARD_RELATIONS:
        if rel == std or rel in std or std in rel:
            return std
    return str(relation).strip()


def parse_triples(text: str, default_time: str | None = None) -> List[Triple]:
    """Parse JSON or `(head, relation, tail)` lines from an LLM response."""
    triples: List[Triple] = []
    obj = extract_json(text)
    if isinstance(obj, dict):
        obj = obj.get("triples", obj.get("validated_triples", obj.get("final_triples", [])))
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                h = item.get("head", default_time or "")
                r = item.get("relation", "")
                t = item.get("tail", "")
                if r and t is not None:
                    triples.append((normalize_time(h) or (default_time or ""), normalize_relation(r), clean_text(t)))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                triples.append((normalize_time(item[0]) or (default_time or ""), normalize_relation(item[1]), clean_text(item[2])))
    if triples:
        return list(dict.fromkeys(canonical_triple(x) for x in triples))

    for line in (text or "").splitlines():
        line = line.strip().strip("-•")
        if not (line.startswith("(") and line.endswith(")")):
            continue
        parts = [p.strip() for p in line[1:-1].split(",", 2)]
        if len(parts) != 3:
            continue
        h = normalize_time(parts[0]) or (default_time or "")
        triples.append((h, normalize_relation(parts[1]), parts[2].strip()))
    return list(dict.fromkeys(canonical_triple(x) for x in triples))


PRODUCER_PROMPT = """
You are a truck-crane knowledge extraction agent. Extract all facts explicitly stated in the evidence below.
Every triple MUST use the provided observation time as the head entity. Do not infer facts not present in the evidence.
Use standardized relation names whenever possible: {relations}.
Return JSON only:
{{"triples":[{{"head":"YYYY-MM-DD HH:MM:SS","relation":"...","tail":"..."}}]}}

Observation time: {time_point}
Evidence:
{description}
"""

DEBATE_PROMPT = """
You are the verifier in an evidence-grounded adversarial debate. Two independent producer sessions extracted
knowledge from the same truck-crane observation. Resolve ONLY the disputed triples using the original evidence as ground truth.
For every disputed fact run a two-position debate:
- affirmative agent: argue to retain the candidate using direct evidence;
- opposing agent: argue to revise/delete it using direct evidence;
- verifier: decide retain_consensus, select_higher_confidence, or discard.
Never introduce facts absent from the evidence. Return JSON only:
{{
  "debates":[{{"fact":"...","affirmative_argument":"...","opposing_argument":"...","decision":"..."}}],
  "validated_triples": [{{"head":"...","relation":"...","tail":"...","decision":"retain_consensus|select_higher_confidence"}}],
  "discarded": [{{"head":"...","relation":"...","tail":"..."}}]
}}
Evidence: {description}
Producer A: {a}
Producer B: {b}
Disputed symmetric difference: {diff}
"""

VALIDATOR_PROMPT = """
You are a verifier for truck-crane knowledge triples. Ground every output in the original evidence.
Tasks: (1) force the head entity to the observation time; (2) normalize synonymous relations;
(3) remove triples conflicting with the evidence; (4) remove redundancy.
Return JSON only as {{"validated_triples":[{{"head":"...","relation":"...","tail":"..."}}]}}.
Observation time: {time_point}
Evidence: {description}
Candidates: {candidates}
"""

INTEGRATOR_PROMPT = """
You are the integrator. Consolidate the validated truck-crane triples into a unique, nonredundant set.
Do not alter numeric values or units unless the evidence explicitly supports the change. All heads must equal {time_point}.
Use standardized relation names. Return JSON only as {{"final_triples":[{{"head":"...","relation":"...","tail":"..."}}]}}.
Evidence: {description}
Validated candidates: {candidates}
"""


def _sanitize_against_record(triples: Iterable[Triple], record: Dict[str, str]) -> List[Triple]:
    """Evidence-ground LLM triples without replacing the LLM extraction stage.

    The source record is used only to verify/correct facts that the LLM actually
    extracted. Missing relations are NOT restored automatically; otherwise a
    failed LLM call could masquerade as successful knowledge extraction.
    """
    head = record.get("时间", "")
    baseline = {rel: value for rel, value in record.items() if rel != "时间" and value != ""}
    output: List[Triple] = []
    for _, rel, _tail in triples:
        rel = normalize_relation(rel)
        if rel not in baseline:
            continue
        output.append((head, rel, baseline[rel]))
    return list(dict.fromkeys(output))


def process_one_text(
    text: str,
    default_time: str,
    num_producers: int = 2,
    max_iterations: int = MAX_VALIDATION_ITERATIONS,
    record: Dict[str, str] | None = None,
) -> Tuple[List[Candidate], Dict]:
    """Run the producer–verifier–integrator pipeline for one observation."""
    if num_producers != 2:
        raise ValueError("The manuscript implementation uses exactly two independent producers.")
    record = record or {"时间": default_time}
    baseline = deterministic_triples(record) if len(record) > 1 else []

    producer_outputs: List[List[Triple]] = []
    producer_raw: List[str] = []
    for _ in range(2):
        try:
            raw = _call_llm(
                PRODUCER_PROMPT.format(
                    relations=", ".join(sorted(STANDARD_RELATIONS)),
                    time_point=default_time,
                    description=text,
                ),
                temperature=0.3,
            )
            producer_raw.append(raw)
            producer_outputs.append(parse_triples(raw, default_time))
        except Exception as exc:
            producer_raw.append(f"ERROR: {exc}")
            if ALLOW_LLM_FALLBACK:
                producer_outputs.append(baseline.copy())
            else:
                raise RuntimeError(
                    "Text extraction requires two real LLM producer outputs, but a producer call failed. "
                    f"Check DEEPSEEK_API_KEY/DEEPSEEK_BASE_URL/model access. Original error: {exc}"
                ) from exc

    if ALLOW_LLM_FALLBACK:
        producer_outputs = [x if x else baseline.copy() for x in producer_outputs]
    elif any(not x for x in producer_outputs):
        raise RuntimeError(
            "A DeepSeek producer returned no parseable triples. The pipeline will not silently substitute "
            "deterministic Excel-to-triple conversion because that would not match the manuscript method."
        )
    agreement = jaccard(producer_outputs[0], producer_outputs[1])
    union = list(dict.fromkeys(producer_outputs[0] + producer_outputs[1]))
    intersection = set(producer_outputs[0]) & set(producer_outputs[1])
    diff = list((set(producer_outputs[0]) | set(producer_outputs[1])) - intersection)

    decision_by_triple: Dict[Triple, str] = {t: "retain_consensus" for t in intersection}
    debate_raw = ""
    if agreement < DEBATE_TRIGGER_THRESHOLD and diff:
        try:
            debate_raw = _call_llm(
                DEBATE_PROMPT.format(
                    description=text,
                    a=json.dumps(producer_outputs[0], ensure_ascii=False),
                    b=json.dumps(producer_outputs[1], ensure_ascii=False),
                    diff=json.dumps(diff, ensure_ascii=False),
                ),
                temperature=0.1,
            )
            debate_obj = extract_json(debate_raw, dict) or {}
            accepted = parse_triples(json.dumps(debate_obj.get("validated_triples", []), ensure_ascii=False), default_time)
            current = list(intersection) + accepted
            for t in accepted:
                decision_by_triple[t] = "select_higher_confidence"
        except Exception as exc:
            if not ALLOW_LLM_FALLBACK:
                raise RuntimeError(f"Verifier adversarial-debate call failed: {exc}") from exc
            current = union
            for t in diff:
                decision_by_triple[t] = "select_higher_confidence"
    else:
        current = union
        for t in diff:
            decision_by_triple[t] = "select_higher_confidence"

    # Iterative validation until convergence or max iterations.
    for _ in range(max_iterations):
        try:
            raw = _call_llm(
                VALIDATOR_PROMPT.format(
                    time_point=default_time,
                    description=text,
                    candidates=json.dumps(current, ensure_ascii=False),
                ),
                temperature=0.1,
            )
            validated = parse_triples(raw, default_time)
            if not validated:
                break
        except Exception as exc:
            if not ALLOW_LLM_FALLBACK:
                raise RuntimeError(f"Iterative validator call failed: {exc}") from exc
            break
        if set(validated) == set(current):
            current = validated
            break
        current = validated

    if record and len(record) > 1:
        current = _sanitize_against_record(current, record)

    try:
        raw = _call_llm(
            INTEGRATOR_PROMPT.format(
                time_point=default_time,
                description=text,
                candidates=json.dumps(current, ensure_ascii=False),
            ),
            temperature=0.1,
        )
        integrated = parse_triples(raw, default_time)
        if integrated:
            current = integrated
    except Exception as exc:
        if not ALLOW_LLM_FALLBACK:
            raise RuntimeError(f"Integrator LLM call failed: {exc}") from exc

    if record and len(record) > 1:
        current = _sanitize_against_record(current, record)

    candidates: List[Candidate] = []
    for t in current:
        decision = decision_by_triple.get(t, "retain_consensus" if t in intersection else "select_higher_confidence")
        confidence = 1.0 if agreement >= DEBATE_TRIGGER_THRESHOLD and t in intersection else (0.8 if t in intersection else 0.5)
        candidates.append(Candidate(*t, confidence=confidence, decision=decision))

    metadata = {
        "jaccard_similarity": agreement,
        "debate_triggered": bool(agreement < DEBATE_TRIGGER_THRESHOLD and diff),
        "producer_a_count": len(producer_outputs[0]),
        "producer_b_count": len(producer_outputs[1]),
        "final_count": len(candidates),
        "producer_raw": producer_raw,
        "debate_raw": debate_raw,
        "extraction_mode": "llm" if not ALLOW_LLM_FALLBACK else "llm_with_explicit_offline_fallback",
    }
    return candidates, metadata


def process_text_excel(excel_path: str, sample_limit: int | None = None) -> str:
    # Do not silently generate triples from columns when the LLM is unavailable.
    # The manuscript explicitly uses two independent LLM producers.
    if not ALLOW_LLM_FALLBACK:
        require_secret(DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY")
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(excel_path)
    df = pd.read_excel(path)
    if sample_limit is not None:
        df = df.head(int(sample_limit))
    if df.empty:
        raise ValueError("Input Excel contains no data rows.")
    if len(df.columns) < 23:
        raise ValueError(f"Expected at least 23 fields, got {len(df.columns)}.")

    columns = resolve_columns(df)
    base = path.stem
    raw_csv = Path(OUTPUT_DIR) / f"{base}_triples_raw.csv"
    merged_csv = Path(OUTPUT_DIR) / f"{base}_triples_merged.csv"
    audit_jsonl = Path(OUTPUT_DIR) / f"{base}_text_debate_audit.jsonl"

    rows: List[Dict] = []
    audit_rows: List[Dict] = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Text extraction"):
        record = normalize_record(row, columns)
        head = record["时间"]
        if not head:
            continue
        description = row_to_sentence(row, columns)
        candidates, meta = process_one_text(description, head, record=record)
        for c in candidates:
            rows.append(
                {
                    "head": c.head,
                    "relation": c.relation,
                    "tail": c.tail,
                    "confidence": c.confidence,
                    "source": "text",
                    "validation_decision": c.decision,
                    "row_index": idx,
                }
            )
        audit_rows.append({"row_index": int(idx), "time": head, "description": description, **meta})

    if not rows:
        raise RuntimeError("No valid text triples were generated.")
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    # Keep the highest confidence duplicate and preserve deterministic ordering.
    merged_df = (
        raw_df.sort_values(["head", "relation", "confidence"], ascending=[True, True, False])
        .drop_duplicates(subset=["head", "relation", "tail"], keep="first")
        .reset_index(drop=True)
    )
    merged_df.to_csv(merged_csv, index=False, encoding="utf-8-sig")
    with open(audit_jsonl, "w", encoding="utf-8") as f:
        for item in audit_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return str(merged_csv)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python 文本_时间序列.py <operational-data.xlsx>")
    print(process_text_excel(sys.argv[1]))
