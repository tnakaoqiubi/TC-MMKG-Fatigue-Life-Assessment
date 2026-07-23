# -*- coding: utf-8 -*-
"""Image-modality knowledge extraction for TC-MMKG.

The implementation follows the manuscript's dual-producer / verifier logic:
1. Parse crane code and timestamp deterministically from the image filename.
2. Two independent Qwen vision calls extract SMX, SMN, DMX and max-stress location.
3. Numerical/descriptive agreement is evaluated independently.
4. Disagreements trigger an evidence-grounded verifier call on the same image.
5. Confidence annotations are exported for downstream cross-modal fusion.

Important runtime detail:
BMP stress-contour maps are normalized to PNG/JPEG bytes before API upload. This
avoids failures caused by oversized uncompressed BMP Base64 payloads while
preserving the original image on disk.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from config import (
    API_CALL_INTERVAL,
    API_MAX_RETRIES,
    DEBATE_TRIGGER_THRESHOLD,
    OUTPUT_DIR,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_VL_FALLBACK_MODEL,
    QWEN_VL_MODEL,
    require_secret,
)
from scripts.common import clean_text, extract_json, normalize_time, parse_float

Triple = Tuple[str, str, str]
ALLOWED_RELATIONS = {"SMX", "SMN", "DMX", "最大应力位置"}
NUMERIC_RELATIONS = {"SMX", "SMN", "DMX"}

# Runtime audit counters. These counters make it impossible for the UI to report a
# successful dual-producer run without actually issuing the corresponding VLM requests.
_API_STATS = {"requests": 0, "producer_requests": 0, "verifier_requests": 0, "successful_requests": 0}
_LAST_BATCH_SUMMARY: Dict = {}

def get_last_extraction_summary() -> Dict:
    return dict(_LAST_BATCH_SUMMARY)

EXTRACTION_PROMPT = r"""
你是 Producer {producer_id}，是一名独立工作的专业汽车起重机主臂应力云图知识抽取智能体。请分析当前应力云图，并严格按照原始系统的规则提取知识三元组。

【必须提取的关系（且仅限以下四种）】
1. 关系 "SMX"：宾语为云图中标注的最大应力值。保留图中原始数字格式与科学计数法，例如 ".193E+09"、"193E+09" 或 "240E+09"，不要擅自改写数量级。
2. 关系 "SMN"：宾语为云图中标注的最小应力值，保留原始数字格式，例如 "41.8055"。
3. 关系 "DMX"：宾语为云图中标注的最大位移值，保留原始数字格式，例如 ".308604"。
4. 关系 "最大应力位置"：宾语必须描述最大应力出现在哪两节臂之间，格式固定为 "第X节臂与第Y节臂之间"，例如 "第2节臂与第3节臂之间"。如果无法从图中可靠判断，必须输出 "无法确定"，不得编造位置。

【严格限制】
- 只允许输出上述四种关系：SMX、SMN、DMX、最大应力位置。
- 不要输出任何其他关系，例如：时间、日期、版本、软件信息、图例刻度、STEP/SUB、物理量类型、颜色、形态、坐标轴、几何特征等。
- 每个时间点尽量输出上述四条三元组；若 SMX、SMN 或 DMX 在图中确实不存在或不可读，可省略该项；“最大应力位置”无法判断时输出“无法确定”。
- 所有三元组的头实体必须使用当前时间点：{time_point}。
- 不要根据常识猜测不可见数字；数值必须来自图像可见证据。
- 只输出 JSON，不要输出 Markdown 代码块、解释、分析或额外文字。

输出格式必须为：
[
  {{"head": "{time_point}", "relation": "SMX", "tail": "..."}},
  {{"head": "{time_point}", "relation": "SMN", "tail": "..."}},
  {{"head": "{time_point}", "relation": "DMX", "tail": "..."}},
  {{"head": "{time_point}", "relation": "最大应力位置", "tail": "第X节臂与第Y节臂之间或无法确定"}}
]
"""

VERIFIER_PROMPT = r"""
你是双模型对抗验证流程中的 Verifier。Producer A 和 Producer B 已针对同一张汽车起重机主臂应力云图独立提取三元组。请以当前图像本身作为唯一事实依据，对两者不一致之处进行核验。

【允许的关系仍然严格限定为四种】
SMX、SMN、DMX、最大应力位置。

【验证规则】
- SMX、SMN、DMX 必须以图像中可见的原始数值为依据，尽量保留原始科学计数法与符号，不得自行换算或改写数量级。
- 最大应力位置必须采用 "第X节臂与第Y节臂之间" 的固定格式；若图像不足以可靠判断，输出 "无法确定"。
- 不得新增软件版本、STEP/SUB、图例刻度、颜色、形态、坐标轴、几何特征等其他关系。
- 对一致项直接保留；对不一致项根据原图选择正确值、修正或丢弃不可验证的数值。
- 只输出 JSON，不要输出 Markdown 或额外解释。

返回格式：
{{
  "debates": [{{"relation":"...","affirmative_argument":"...","opposing_argument":"...","decision":"..."}}],
  "validated_triples": [{{"head":"{time_point}","relation":"SMX|SMN|DMX|最大应力位置","tail":"...","decision":"retain_consensus|select_higher_confidence"}}],
  "discarded_relations": ["..."]
}}

Producer A: {a}
Producer B: {b}
"""

def _client() -> OpenAI:
    return OpenAI(
        api_key=require_secret(QWEN_API_KEY, "QWEN_API_KEY (or DASHSCOPE_API_KEY)"),
        base_url=QWEN_BASE_URL,
    )


def extract_vehicle_and_time_from_filename(filename: str) -> Tuple[str, str | None]:
    stem = Path(filename).stem
    m = re.match(r"^(?P<vehicle>.+?)_(?P<date>\d{8})_(?P<time>\d{6})(?:_|$)", stem)
    if m:
        d, t = m.group("date"), m.group("time")
        return m.group("vehicle"), f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
    m2 = re.search(r"(\d{4}-\d{2}-\d{2})[_ T](\d{2})[-:]?(\d{2})[-:]?(\d{2})", stem)
    if m2:
        vehicle = stem[: m2.start()].rstrip("_-") or stem
        return vehicle, f"{m2.group(1)} {m2.group(2)}:{m2.group(3)}:{m2.group(4)}"
    return stem, None


def _prepare_image_data_url(image_path: str) -> Tuple[str, Dict]:
    """Normalize any supported local image to an API-safe Base64 data URL.

    Uncompressed BMP can exceed the 10 MB Base64 input limit even when the source
    file itself looks reasonable. Converting it in-memory avoids that problem and
    does not alter the user's source file.
    """
    path = Path(image_path)
    try:
        with Image.open(path) as im:
            im.load()
            original_size = im.size
            original_format = (im.format or path.suffix.lstrip(".")).upper()
            if im.mode not in ("RGB", "L"):
                # Flatten alpha onto white; contour maps normally have a light background.
                if "A" in im.getbands():
                    rgba = im.convert("RGBA")
                    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    bg.alpha_composite(rgba)
                    im = bg.convert("RGB")
                else:
                    im = im.convert("RGB")
            elif im.mode == "L":
                im = im.convert("RGB")

            # Keep enough pixels for small legend text but avoid pathological payloads.
            max_side = 4096
            if max(im.size) > max_side:
                scale = max_side / max(im.size)
                im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            payload = buf.getvalue()
            mime = "image/png"

            # Base64 expands bytes by roughly 4/3. Stay comfortably below 10 MB.
            if len(payload) > 7_000_000:
                # JPEG is much smaller while quality 97 keeps legend text readable.
                buf = io.BytesIO()
                if max(im.size) > 3072:
                    scale = 3072 / max(im.size)
                    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.LANCZOS)
                im.save(buf, format="JPEG", quality=97, subsampling=0, optimize=True)
                payload = buf.getvalue()
                mime = "image/jpeg"

            if len(payload) > 7_000_000:
                raise ValueError(
                    f"Image remains too large after normalization ({len(payload)/1024/1024:.1f} MB before Base64)."
                )

        data_url = f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"
        meta = {
            "original_format": original_format,
            "original_size": list(original_size),
            "uploaded_mime": mime,
            "normalized_size": list(im.size),
            "normalized_bytes": len(payload),
        }
        return data_url, meta
    except Exception as exc:
        raise RuntimeError(f"Cannot read/normalize image '{path.name}': {exc}") from exc


def _model_candidates() -> List[str]:
    models = [QWEN_VL_MODEL]
    if QWEN_VL_FALLBACK_MODEL and QWEN_VL_FALLBACK_MODEL not in models:
        models.append(QWEN_VL_FALLBACK_MODEL)
    return [m for m in models if m]


def call_multimodal_llm(
    image_data_url: str,
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    call_kind: str = "producer",
) -> Tuple[str, str]:
    """Call a Qwen visual model and return (content, model_used).

    Qwen3.5-Plus enables thinking by default. For structured extraction we disable
    thinking so the token budget is spent on the final JSON rather than hidden
    reasoning; this also prevents empty final content when max_tokens is exhausted.
    """
    client = _client()
    errors: List[str] = []
    for model in _model_candidates():
        for attempt in range(max(1, API_MAX_RETRIES + 1)):
            try:
                kwargs = dict(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_data_url}},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # Qwen3.5/3.6/3.7 and Qwen3-VL support this non-standard parameter
                # through the OpenAI-compatible endpoint. Disable thinking for a
                # deterministic extraction task.
                kwargs["extra_body"] = {"enable_thinking": False}
                _API_STATS["requests"] += 1
                if call_kind == "verifier":
                    _API_STATS["verifier_requests"] += 1
                else:
                    _API_STATS["producer_requests"] += 1
                response = client.chat.completions.create(**kwargs)
                if API_CALL_INTERVAL > 0:
                    time.sleep(API_CALL_INTERVAL)
                msg = response.choices[0].message
                content = (getattr(msg, "content", None) or "").strip()
                if content:
                    _API_STATS["successful_requests"] += 1
                    return content, model
                reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
                raise RuntimeError(
                    "API returned empty final content"
                    + (f" (reasoning_content length={len(reasoning)})" if reasoning else "")
                )
            except Exception as exc:
                errors.append(f"model={model}, attempt={attempt+1}: {type(exc).__name__}: {exc}")
                if attempt < API_MAX_RETRIES:
                    time.sleep(min(1.5 * (attempt + 1), 4.0))
    raise RuntimeError("All configured vision-model calls failed. " + " | ".join(errors[-6:]))


def parse_triples_from_response(response_text: str, head: str = "") -> List[Triple]:
    obj = extract_json(response_text)
    if isinstance(obj, dict):
        obj = obj.get("triples", obj.get("validated_triples", []))
    if not isinstance(obj, list):
        return []
    result: List[Triple] = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        relation = clean_text(item.get("relation"))
        tail = clean_text(item.get("tail"))
        # Tolerate a few obvious model aliases without broadening the ontology.
        relation_aliases = {
            "最大应力": "SMX",
            "最大应力值": "SMX",
            "最小应力": "SMN",
            "最小应力值": "SMN",
            "最大位移": "DMX",
            "最大位移值": "DMX",
            "高应力位置": "最大应力位置",
        }
        relation = relation_aliases.get(relation, relation)
        if relation in ALLOWED_RELATIONS and tail:
            result.append((head, relation, tail))
    return list(dict.fromkeys(result))


def _numeric_equal(a: str, b: str, rel_tol: float = 0.02) -> bool:
    fa, fb = parse_float(a), parse_float(b)
    if fa is None or fb is None:
        return a.strip() == b.strip()
    scale = max(abs(fa), abs(fb), 1e-12)
    return abs(fa - fb) / scale <= rel_tol


def _agreement(a: List[Triple], b: List[Triple]) -> Tuple[float, float, float]:
    da = {r: t for _, r, t in a}
    db = {r: t for _, r, t in b}
    numeric_scores: List[float] = []
    descriptive_scores: List[float] = []
    for rel in ALLOWED_RELATIONS:
        if rel not in da and rel not in db:
            continue
        if rel in NUMERIC_RELATIONS:
            numeric_scores.append(1.0 if rel in da and rel in db and _numeric_equal(da[rel], db[rel]) else 0.0)
        else:
            descriptive_scores.append(1.0 if rel in da and rel in db and da[rel].strip() == db[rel].strip() else 0.0)
    num = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 1.0
    desc = sum(descriptive_scores) / len(descriptive_scores) if descriptive_scores else 1.0
    return (num + desc) / 2.0, num, desc


def process_image(image_path: str) -> Tuple[List[Dict], Dict]:
    started = time.perf_counter()
    filename = Path(image_path).name
    vehicle, time_point = extract_vehicle_and_time_from_filename(filename)
    if not time_point:
        raise ValueError(f"Cannot parse vehicle/time from image filename: {filename}")
    time_point = normalize_time(time_point)
    image_data_url, image_meta = _prepare_image_data_url(image_path)

    producer_outputs: List[List[Triple]] = []
    raw_outputs: List[str] = []
    models_used: List[str] = []
    call_errors: List[str] = []
    for producer_id in ("A", "B"):
        try:
            raw, model_used = call_multimodal_llm(
                image_data_url,
                EXTRACTION_PROMPT.format(producer_id=producer_id, time_point=time_point),
                temperature=0.1,
                call_kind="producer",
            )
            raw_outputs.append(raw)
            models_used.append(model_used)
            producer_outputs.append(parse_triples_from_response(raw, time_point))
        except Exception as exc:
            call_errors.append(f"Producer {producer_id}: {exc}")
            raw_outputs.append(f"ERROR: {exc}")
            models_used.append("")
            producer_outputs.append([])

    # Manuscript-consistent strict mode: both independent producers must return at
    # least one supported image semantic. Do not silently accept a one-producer
    # shortcut, because that would make extraction suspiciously fast and would no
    # longer implement the stated dual-model verification design.
    if any(not triples for triples in producer_outputs):
        details = " | ".join(call_errors) if call_errors else " | ".join(
            f"Producer {name} raw={raw[:300]!r}" for name, raw in zip(("A", "B"), raw_outputs)
        )
        raise RuntimeError(
            "Dual-producer requirement not satisfied: both vision producers must return supported stress semantics. "
            f"Image preprocessing={image_meta}. Details: {details}"
        )

    overall, num_agree, desc_agree = _agreement(producer_outputs[0], producer_outputs[1])
    by_rel: Dict[str, List[str]] = {}
    for triples in producer_outputs:
        for _, rel, tail in triples:
            by_rel.setdefault(rel, [])
            if tail not in by_rel[rel]:
                by_rel[rel].append(tail)

    if not by_rel:
        details = " | ".join(call_errors) if call_errors else " | ".join(
            f"Producer {name} raw={raw[:300]!r}" for name, raw in zip(("A", "B"), raw_outputs)
        )
        raise RuntimeError(
            "Both vision producers failed or returned no supported stress semantics. "
            f"Image preprocessing={image_meta}. Details: {details}"
        )

    final: Dict[str, Tuple[str, float, str]] = {}
    for rel, values in by_rel.items():
        support_count = sum(1 for triples in producer_outputs if any(r == rel for _, r, _ in triples))
        if len(values) == 1 and support_count == 2:
            final[rel] = (values[0], 1.0, "retain_consensus")

    debate_triggered = overall < DEBATE_TRIGGER_THRESHOLD or any(rel not in final for rel in by_rel)
    verifier_raw = ""
    verifier_model = ""
    if debate_triggered and by_rel:
        try:
            verifier_raw, verifier_model = call_multimodal_llm(
                image_data_url,
                VERIFIER_PROMPT.format(
                    time_point=time_point,
                    a=json.dumps(producer_outputs[0], ensure_ascii=False),
                    b=json.dumps(producer_outputs[1], ensure_ascii=False),
                ),
                temperature=0.0,
                call_kind="verifier",
            )
            verified = parse_triples_from_response(verifier_raw, time_point)
            for _, rel, tail in verified:
                if rel not in final:
                    final[rel] = (tail, 0.8 if tail in by_rel.get(rel, []) else 0.5, "select_higher_confidence")
        except Exception:
            # Preserve a usable unimodal result when the verifier alone fails.
            for rel, values in by_rel.items():
                if rel in final:
                    continue
                first = next((t for _, r, t in producer_outputs[0] if r == rel), values[0])
                final[rel] = (first, 0.5, "select_higher_confidence")
    else:
        for rel, values in by_rel.items():
            if rel not in final:
                final[rel] = (values[0], 0.5, "select_higher_confidence")

    rows = [
        {
            "head": time_point,
            "relation": rel,
            "tail": value,
            "confidence": confidence,
            "source": "image",
            "validation_decision": decision,
            "image_file": filename,
        }
        for rel, (value, confidence, decision) in final.items()
    ]
    rows.append(
        {
            "head": time_point,
            "relation": "属于车辆",
            "tail": vehicle,
            "confidence": 1.0,
            "source": "filename",
            "validation_decision": "deterministic_filename_parse",
            "image_file": filename,
        }
    )
    audit = {
        "image": filename,
        "vehicle": vehicle,
        "time": time_point,
        "image_preprocessing": image_meta,
        "models_used": models_used,
        "verifier_model": verifier_model,
        "agreement_overall": overall,
        "agreement_numerical": num_agree,
        "agreement_descriptive": desc_agree,
        "debate_triggered": debate_triggered,
        "producer_raw": raw_outputs,
        "verifier_raw": verifier_raw,
        "call_errors": call_errors,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "producer_success_count": sum(1 for x in producer_outputs if x),
    }
    return rows, audit


def batch_process(image_folder: str, raw_responses_file: str, output_csv: str) -> str:
    global _LAST_BATCH_SUMMARY
    batch_started = time.perf_counter()
    for key in _API_STATS:
        _API_STATS[key] = 0
    _LAST_BATCH_SUMMARY = {}
    # Fail fast instead of processing every image with a missing key.
    require_secret(QWEN_API_KEY, "QWEN_API_KEY (or DASHSCOPE_API_KEY)")
    folder = Path(image_folder)
    if not folder.exists():
        raise FileNotFoundError(image_folder)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})
    if not images:
        raise ValueError(f"No supported images found in {image_folder}")

    rows: List[Dict] = []
    audits: List[Dict] = []
    errors: List[str] = []
    for image in tqdm(images, desc="Image extraction"):
        try:
            image_rows, audit = process_image(str(image))
            rows.extend(image_rows)
            audits.append(audit)
        except Exception as exc:
            errors.append(f"{image.name}: {exc}")

    if not rows:
        # Preserve the first real API error; this is far more actionable than a generic message.
        raise RuntimeError("No valid image triples were generated. " + "\n".join(errors[:3]))
    df = pd.DataFrame(rows).drop_duplicates(subset=["head", "relation", "tail"], keep="first")
    df.sort_values(["head", "relation"]).to_csv(output_csv, index=False, encoding="utf-8-sig")
    with open(raw_responses_file, "w", encoding="utf-8") as f:
        for audit in audits:
            f.write(json.dumps(audit, ensure_ascii=False) + "\n")
        for err in errors:
            f.write(json.dumps({"error": err}, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - batch_started
    _LAST_BATCH_SUMMARY = {
        "images_received": len(images),
        "images_successful": len(audits),
        "images_failed": len(errors),
        "producer_requests": int(_API_STATS["producer_requests"]),
        "verifier_requests": int(_API_STATS["verifier_requests"]),
        "total_vlm_requests": int(_API_STATS["requests"]),
        "successful_vlm_requests": int(_API_STATS["successful_requests"]),
        "elapsed_seconds": round(elapsed, 3),
        "average_seconds_per_image": round(elapsed / max(len(images), 1), 3),
        "minimum_expected_producer_requests": 2 * len(images),
        "audit_file": raw_responses_file,
    }
    summary_path = str(Path(OUTPUT_DIR) / "image_extraction_summary.json")
    Path(summary_path).write_text(json.dumps(_LAST_BATCH_SUMMARY, ensure_ascii=False, indent=2), encoding="utf-8")
    _LAST_BATCH_SUMMARY["summary_file"] = summary_path
    return output_csv


def process_images(image_folder: str, output_csv: str | None = None, raw_responses_file: str | None = None) -> str:
    output_csv = output_csv or str(Path(OUTPUT_DIR) / "triples_time_from_images.csv")
    raw_responses_file = raw_responses_file or str(Path(OUTPUT_DIR) / "image_debate_audit.jsonl")
    return batch_process(image_folder, raw_responses_file, output_csv)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python 图像_时间序列.py <image-folder>")
    print(process_images(sys.argv[1]))
