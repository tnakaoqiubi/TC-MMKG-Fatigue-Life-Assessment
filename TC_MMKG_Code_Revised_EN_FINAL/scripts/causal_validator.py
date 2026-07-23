# -*- coding: utf-8 -*-
import re
import json
from typing import List, Tuple, Dict
from scripts.causal_rules import CAUSAL_RULES, VAR_SYNONYMS

def normalize_var(name: str) -> str:
    for std, syns in VAR_SYNONYMS.items():
        if any(s in name for s in syns):
            return std
    return name

def parse_assertions(chain: List[str]) -> List[Tuple[str, str, str]]:
    assertions = []
    for step in chain:
        parts = step.split("所以")
        if len(parts) < 2:
            continue
        cause_part, effect_part = parts[0], parts[1]
        cause_match = re.search(r"([\u4e00-\u9fa5a-zA-Z0-9]+)[↑增长升高增大]", cause_part)
        if not cause_match:
            continue
        cause = normalize_var(cause_match.group(1))
        effect_match = re.search(r"([\u4e00-\u9fa5a-zA-Z0-9]+)[↑增长升高增大]", effect_part)
        if effect_match:
            assertions.append((cause, normalize_var(effect_match.group(1)), "positive"))
        else:
            effect_match = re.search(r"([\u4e00-\u9fa5a-zA-Z0-9]+)[↓减少降低减小]", effect_part)
            if effect_match:
                assertions.append((cause, normalize_var(effect_match.group(1)), "negative"))
    return assertions

def check_direction(assertions):
    rule_dict = {(c, e): r for c, e, r in CAUSAL_RULES}
    errors = []
    for c, e, d in assertions:
        if (c, e) in rule_dict and rule_dict[(c, e)] != d:
            errors.append(f"{c}→{e} 方向应为{rule_dict[(c,e)]}, 实际{d}")
    return len(errors) == 0, errors

def check_sufficiency(assertions, retrieved_cases):
    if not retrieved_cases:
        return False, ["无相似案例"]
    errors = []
    for c, e, d in assertions:
        pairs = []
        for case in retrieved_cases:
            text = case.get("text_desc", "") + " " + case.get("stress_desc", "")
            c_val = re.search(rf"{c}[:：]\s*([\d\.]+)", text)
            e_val = re.search(rf"{e}[:：]\s*([\d\.]+)", text)
            if c_val and e_val:
                pairs.append((float(c_val.group(1)), float(e_val.group(1))))
        if len(pairs) < 2:
            errors.append(f"{c}→{e} 证据不足")
            continue
        pairs.sort(key=lambda x: x[0])
        trend = all(pairs[i][1] <= pairs[i+1][1] for i in range(len(pairs)-1))
        if d == "positive" and not trend:
            errors.append(f"{c}→{e} 无正相关趋势")
        elif d == "negative" and trend:
            errors.append(f"{c}→{e} 无负相关趋势")
    return len(errors) == 0, errors

def extract_load_level(text: str) -> float:
    actual_load = None
    boom_length = None
    radius = None
    match = re.search(r"实际吊重量[:\s]*([\d\.]+)", text)
    if match:
        actual_load = float(match.group(1))
    match = re.search(r"主臂长度[:\s]*([\d\.]+)", text)
    if match:
        boom_length = float(match.group(1))
    match = re.search(r"工作幅度[:\s]*([\d\.]+)", text)
    if match:
        radius = float(match.group(1))
    score = 0.0
    if actual_load is not None:
        score += 0.6 * actual_load
    if boom_length is not None:
        score += 0.2 * boom_length
    if radius is not None:
        score += 0.2 * radius
    return score if score > 0 else None

def extract_damage(generated_text: str) -> float:
    if not generated_text:
        return None
    try:
        obj = json.loads(generated_text)
        if "damage_value" in obj:
            return float(obj["damage_value"])
        if "life_value" in obj:
            return float(obj["life_value"])
    except:
        pass
    match = re.search(r"损伤值[:\s]*([\d\.]+(?:e[+-]?\d+)?)", generated_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"([\d\.]+(?:e[+-]?\d+)?)", generated_text)
    if match:
        return float(match.group(1))
    return None

def check_load_damage_proportion(input_text: str, generated_text: str, retrieved_cases: List[Dict]) -> Tuple[bool, List[str]]:
    input_load = extract_load_level(input_text)
    if input_load is None:
        return True, []
    generated_damage = extract_damage(generated_text)
    if generated_damage is None:
        return True, []
    case_loads = []
    for case in retrieved_cases:
        case_text = case.get("text_desc", "") + " " + case.get("stress_desc", "")
        load = extract_load_level(case_text)
        if load is not None and case.get("life_value") is not None:
            case_loads.append((load, case["life_value"]))
    if not case_loads:
        return True, []
    case_loads.sort(key=lambda x: x[0])
    lower_cases = [l for l, d in case_loads if l <= input_load]
    if lower_cases:
        ref_damage = max([d for l, d in case_loads if l <= input_load], default=None)
    else:
        ref_damage = case_loads[0][1]
    if ref_damage is not None:
        if generated_damage < 0.1 * ref_damage:
            return False, [f"载荷水平 ({input_load}) 下的生成损伤值 ({generated_damage:.2e}) 远小于参考损伤 ({ref_damage:.2e})，违反载荷-损伤正比规则"]
    return True, []

def counterfactual_stability(original_input, input_modal, reasoning_func, target_modals):
    numbers = re.findall(r"\d+(?:\.\d+)?", original_input)
    if not numbers:
        return True, []
    base_num = float(numbers[0])
    up_input = original_input.replace(numbers[0], str(base_num*1.1), 1)
    down_input = original_input.replace(numbers[0], str(base_num*0.9), 1)
    def extract_life(text):
        m = re.search(r"寿命[^\d]*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)", text)
        return float(m.group(1)) if m else None
    try:
        base_resp = reasoning_func(input_modal, original_input, "life")
        up_resp = reasoning_func(input_modal, up_input, "life")
        down_resp = reasoning_func(input_modal, down_input, "life")
        base_life = extract_life(base_resp.get("generated_text", ""))
        up_life = extract_life(up_resp.get("generated_text", ""))
        down_life = extract_life(down_resp.get("generated_text", ""))
        if None in (base_life, up_life, down_life):
            return True, []
        if up_life > base_life and base_life > down_life:
            return False, ["反事实测试失败：增大载荷后寿命反而增加"]
        if down_life < base_life and base_life < up_life:
            return False, ["反事实测试失败：减小载荷后寿命反而减少"]
        return True, []
    except Exception as e:
        return True, [f"反事实测试异常: {e}"]

class CausalValidator:
    def validate(self, reasoning_result: Dict, reasoning_func=None) -> Dict:
        chain = reasoning_result.get("reasoning_chain", [])
        retrieved = reasoning_result.get("retrieved_cases", [])
        for case in retrieved:
            case.pop("text_vector", None)
            case.pop("stress_vector", None)
        assertions = parse_assertions(chain)
        dir_pass, dir_err = check_direction(assertions)
        if not dir_pass:
            return {"passed": False, "layer": "direction", "errors": dir_err}
        suff_pass, suff_err = check_sufficiency(assertions, retrieved)
        if not suff_pass:
            return {"passed": False, "layer": "sufficiency", "errors": suff_err}
        load_damage_pass, load_damage_err = check_load_damage_proportion(
            reasoning_result["input_content"],
            reasoning_result.get("generated_text", ""),
            retrieved
        )
        if not load_damage_pass:
            return {"passed": False, "layer": "load_damage_proportion", "errors": load_damage_err}
        if reasoning_func:
            cf_pass, cf_err = counterfactual_stability(
                reasoning_result["input_content"], reasoning_result["input_modal"],
                reasoning_func, reasoning_result["target_modals"])
            if not cf_pass:
                return {"passed": False, "layer": "counterfactual", "errors": cf_err}
        return {
            "passed": True,
            "layer": "all",
            "errors": [],
            "final_output": reasoning_result["generated_text"],
            "reasoning_chain": chain,
            "retrieved_cases": retrieved
        }