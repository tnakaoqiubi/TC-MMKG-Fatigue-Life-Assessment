# -*- coding: utf-8 -*-
import os
import json
import re
from openai import OpenAI
from typing import Dict, List, Optional
from scripts.case_library import get_case_library
from scripts.causal_validator import CausalValidator

# Runtime configuration only; the reasoning algorithm below is the original version.
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


class AnalogicalReasoner:
    def __init__(self):
        self.case_lib = get_case_library()
        self.validator = CausalValidator()

    def _call_llm_and_parse(self, prompt: str) -> Dict:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = resp.choices[0].message.content
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            try:
                data = json.loads(content[start:end])
                if "reasoning_chain" not in data:
                    data["reasoning_chain"] = []
                if "generated_text" in data:
                    gt = data["generated_text"]
                    if isinstance(gt, dict):
                        data["generated_text"] = json.dumps(gt, ensure_ascii=False)
                    elif not isinstance(gt, str):
                        data["generated_text"] = str(gt)
                else:
                    data["generated_text"] = ""
                return data
            except json.JSONDecodeError:
                pass
        return {"reasoning_chain": [], "generated_text": content}

    def _build_prompt(self, input_modal: str, input_content: str,
                      target_modals: List[str], similar: List[Dict],
                      neighbor: List, error_hint: str = "") -> str:
        case_lines = []
        for i, c in enumerate(similar, 1):
            time_str = c.get("time", "未知")
            sim = c.get("similarity", 0)
            case_lines.append(f"Case {i}: time={time_str}, similarity={sim:.3f}")
        cases_str = "\n".join(case_lines)

        rules_hint = """
【核心因果规则（必须严格遵循）】：
- 实际吊重量↑ → 应力↑ → 损伤↑
- 工作幅度↑ → 应力↑ → 损伤↑
- 主臂长度↑ → 应力↑ → 损伤↑

【载荷-损伤正比关系】：
- 当输入的实际吊重量、主臂长度或工作幅度较大时，生成的损伤值必须比载荷较小的案例的损伤值更大。
- 损伤值必须使用科学计数法，且数值大小应与载荷水平正相关。
- 示例：若案例1载荷10t损伤1e-6，案例2载荷20t损伤应大于1e-6（如2e-6或更高），绝不能小于1e-6。
- 严禁出现高载荷对应极小损伤（如1e-9）的逻辑错误。
"""

        text_format = '"text_desc": "Actual load: <value> t; Boom length: <value> m; Working radius: <value> m; Torque percentage: <value>%; Running hours: <value> h"'
        damage_format = '"damage_value": <scientific notation, e.g., high load may be 2.34e-4, low load 1.23e-6>'
        stress_format = '"stress_desc": "Mean stress: <value> MPa; Max stress: <value> MPa; Min stress: <value> MPa"'

        output_fields = []
        if 'text' in target_modals:
            output_fields.append(text_format)
        if 'stress' in target_modals:
            output_fields.append(stress_format)
        if 'life' in target_modals:
            output_fields.append(damage_format)

        output_format = "{\n    " + ",\n    ".join(output_fields) + "\n}"

        prompt = f"""你是起重机疲劳专家。根据历史案例进行类比推理。

**输入模态**: {input_modal}
**输入内容**: {input_content}
**需要生成的模态**: {', '.join(target_modals)}

**参考案例（仅时间点）**:
{cases_str}

**关联知识（部分三元组）**: {neighbor[:20]}

{rules_hint}

**输出要求**：
- The text description must **strictly** follow the format: "Actual load: X t; Boom length: Y m; Working radius: Z m; Torque percentage: W%; Running hours: T h"
- The damage value must be in scientific notation and must be proportional to the load level (larger load → larger damage).
- Do not add any additional analysis, explanations, or comments.
- In the reasoning chain, refer to case time points and briefly explain the causal relationships.
- **All reasoning chain steps must be written in English.**

请输出如下JSON格式，不要有任何额外解释：
{{
    "reasoning_chain": ["步骤1", "步骤2", ...],
    "generated_text": {output_format}
}}

{error_hint}
"""
        return prompt

    def reason(self, input_modal: str, input_content: str,
               target_modal: Optional[str] = None) -> Dict:
        all_modals = {"text", "stress", "life"}
        if target_modal:
            target_modals = [target_modal]
        else:
            target_modals = list(all_modals - {input_modal})

        if input_modal == "text":
            similar = self.case_lib.retrieve_similar(query_text=input_content, modal="text", top_k=5)
        elif input_modal == "stress":
            similar = self.case_lib.retrieve_similar(query_text=input_content, modal="stress", top_k=5)
        else:  # life
            match = re.search(r"[\d\.]+(?:e[+-]?\d+)?", input_content)
            life_val = float(match.group()) if match else 1e-6
            candidates = []
            for c in self.case_lib.cases:
                diff = abs(c["life_value"] - life_val) / max(1e-12, c["life_value"])
                candidates.append((diff, c))
            candidates.sort(key=lambda x: x[0])
            similar = []
            for diff, case in candidates[:5]:
                c_clean = {k: v for k, v in case.items() if k not in ["text_vector", "stress_vector"]}
                c_clean["similarity"] = 1.0 - diff
                similar.append(c_clean)

        neighbor = self.case_lib.get_neighbor_triples(similar[0]["time"]) if similar else []

        prompt = self._build_prompt(input_modal, input_content, target_modals, similar, neighbor)

        max_retries = 2
        result = None
        for attempt in range(max_retries + 1):
            llm_result = self._call_llm_and_parse(prompt)

            temp_result = {
                "reasoning_chain": llm_result.get("reasoning_chain", []),
                "generated_text": llm_result.get("generated_text", ""),
                "retrieved_cases": similar,
                "input_modal": input_modal,
                "input_content": input_content,
                "target_modals": target_modals
            }
            validation = self.validator.validate(temp_result, reasoning_func=None)

            if validation["passed"]:
                result = llm_result
                result["retrieved_cases"] = similar
                result["retrieved_case_times"] = [c.get("time") for c in similar if c.get("time")]
                result["neighbor_triples"] = neighbor
                result["input_modal"] = input_modal
                result["input_content"] = input_content
                result["target_modals"] = target_modals
                result["validation_passed"] = True
                result["validation_layer"] = validation.get("layer", "all")
                break
            else:
                if attempt < max_retries:
                    error_hint = f"\n上一次推理未通过因果校验，错误：{validation.get('errors', [])}。请严格遵守格式和因果规则。"
                    prompt = self._build_prompt(input_modal, input_content, target_modals, similar, neighbor, error_hint)
                else:
                    result = llm_result
                    result["retrieved_cases"] = similar
                    result["retrieved_case_times"] = [c.get("time") for c in similar if c.get("time")]
                    result["neighbor_triples"] = neighbor
                    result["input_modal"] = input_modal
                    result["input_content"] = input_content
                    result["target_modals"] = target_modals
                    result["validation_passed"] = False
                    result["validation_errors"] = validation.get("errors", [])
                    break

        # 只移除向量字段（内部使用），保留文本描述供前端显示
        for case in result.get("retrieved_cases", []):
            case.pop("text_vector", None)
            case.pop("stress_vector", None)
        # 不再移除 text_desc 和 stress_desc

        if "generated_text" in result and not isinstance(result["generated_text"], str):
            result["generated_text"] = json.dumps(result["generated_text"], ensure_ascii=False)

        # 英文格式化（仅针对 generated_text）
        raw_text = result.get("generated_text", "")
        if isinstance(raw_text, str):
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, dict):
                    parts = []
                    if "text_desc" in parsed:
                        parts.append(parsed["text_desc"])
                    if "stress_desc" in parsed:
                        parts.append(parsed["stress_desc"])
                    if "damage_value" in parsed:
                        damage_val = parsed["damage_value"]
                        parts.append(f"Damage: {damage_val:.6f}")
                    if parts:
                        result["generated_text"] = ", ".join(parts)
            except json.JSONDecodeError:
                pass

        return result