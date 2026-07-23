# -*- coding: utf-8 -*-
import os
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Any
import copy

# Runtime configuration only; case construction/retrieval logic below is the original version.
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, SBERT_LOCAL_PATH, SBERT_MODEL_NAME

try:
    model = SentenceTransformer(SBERT_LOCAL_PATH)
except:
    model = SentenceTransformer(SBERT_MODEL_NAME)

# ---------- 新增：中文关系名 → 英文显示名映射 ----------
REL_TO_EN = {
    "实际吊重量": "Actual load",
    "额定吊重量": "Rated load",
    "力矩百分比": "Torque percentage",
    "载荷百分比": "Load percentage",
    "主臂长度": "Boom length",
    "角度": "Angle",
    "工作幅度": "Working radius",
    "长度": "Length",
    "主臂长": "Boom length",
    "幅": "Working radius",
    "最大应力": "Max stress",
    "最小应力": "Min stress",
    "平均应力": "Mean stress",
    "应力幅": "Stress amplitude",
    "图像最大应力值": "Image max stress",
    "图像应力极值差": "Image stress range",
    "图像应力均值": "Image mean stress",
    "最大应力 (SMX)": "SMX",
    "最小应力 (SMN)": "SMN",
    "最大值 (SMX)": "SMX",
    "最小值 (SMN)": "SMN",
    "最大应力值 (SMX)": "SMX",
    "最小应力值 (SMN)": "SMN",
    "SMN数值": "SMN",
    "SMX值": "SMX",
    "SMN值": "SMN",
    "最大值": "Max",
    "最小值": "Min",
    "上限": "Upper limit",
    "下限": "Lower limit",
    "数值范围上限": "Upper limit",
    "数值范围下限": "Lower limit",
    "应力范围上限": "Upper limit",
    "应力范围下限": "Lower limit",
    "主要应力状态": "Stress state",
    "数值 (SMN)": "SMN value",
    "数值 (SMX)": "SMX value",
    "数值 (DMX)": "DMX value",
    "次大值刻度": "Second max scale",
    "次小值刻度": "Second min scale",
    "最大值刻度": "Max scale",
    "最小值刻度": "Min scale",
    "高应力集中区域": "High stress concentration",
    "高应力区域颜色": "High stress color",
    "几何分布": "Geometry distribution",
    "位于": "Located at",
    "DMX值": "DMX",
    "图例最大值": "Legend max",
    "图例最小值": "Legend min",
    "最大应力位置": "Max stress location",
    "整体应力范围": "Overall stress range",
    "最大应力位置标记": "Max mark",
    "高应力区域": "High stress zone",
    "低应力区域位置": "Low stress zone",
    "主要颜色": "Main color",
    "颜色图例最小值": "Color legend min",
    "分布区域": "Distribution zone",
    "几何特征": "Geometry feature",
    "结构形态": "Structure shape",
    "应力集中状态": "Stress concentration",
    "标记": "Label",
    "类别": "Category",
    "编号": "Number",
    "位置标记": "Marker",
    "视觉特征": "Visual feature",
    "低应力区域": "Low stress zone",
    "最大值对应颜色": "Max color",
    "高应力区域位置": "High stress zone",
    "最小值颜色": "Min color",
    "标记符号": "Symbol",
    "整体应力分布": "Overall stress distribution",
    "主要应力颜色": "Main color",
    "低应力分布区域": "Low stress zone",
    "刻度单位": "Scale unit",
    "显示变量": "Display variable",
    "低应力区域颜色": "Low stress color",
    "区域": "Region",
    "最大值颜色": "Max color",
    "最大值标记": "Max mark",
    "最小值标记": "Min mark",
    "使用软件": "Software",
    "发动机转速": "Engine speed",
    "油压": "Oil pressure",
    "水温": "Water temp",
    "运行时间": "Running hours",
    "车速": "Vehicle speed",
    "绑定状态": "Binding status",
    "锁车状态": "Lock status",
    "工况代码": "Case code",
    "倍率": "Multiplier",
    "区域代码": "Region code",
    "故障码": "Fault code",
    "力限器故障码": "Limiter fault",
    "控制类故障码": "Control fault",
    "发动机故障码": "Engine fault",
    "属于车辆": "Belongs to vehicle",
    "ACC状态": "ACC status",
    "型号": "Model",
    "车辆代码": "Vehicle code",
    "版本": "Version",
    "标识": "Label",
    "名称": "Name",
    "类型": "Type",
    "含义": "Meaning",
    "形态": "Shape",
    "位置": "Location",
    "数值": "Value",
    "时间 (TIME)": "Time",
    "具体时间": "Time",
    "TIME": "Time",
    "包含轴": "Axes",
    "几何形态": "Geometry",
    "处理方式": "Processing",
    "结果类型": "Result type",
    "求解类型": "Solution type",
    "物理量": "Physical quantity",
    "显示物理量": "Display quantity",
    "分析软件": "Analysis software",
    "所属软件": "Software",
    "包含数值": "Contains value",
    "位于区域": "Region",
    "显示内容": "Display content",
    "DMX": "DMX",
    "最大位移 (DMX)": "DMX",
    "日期时间": "Date time",
    "时刻": "Time moment",
    "形态特征": "Shape feature",
    "SMN": "SMN",
    "SMX": "SMX",
    "DMX": "DMX",
    "STEP": "STEP",
    "SUB": "Substep",
    "具体时刻": "Time moment",
    "名称及版本": "Name & version",
    "日期": "Date",
    "时间戳": "Timestamp",
    "子步": "Substep",
    "子步 (SUB)": "Substep",
    "步长 (STEP)": "Step",
    "生成软件": "Software",
    "单位": "Unit",
    "解类型": "Solution type",
    "分析类型": "Analysis type",
    "最低刻度值": "Min scale value",
    "物理量名称": "Physical quantity",
    "下限值": "Lower limit",
    "上限值": "Upper limit",
    "颜色图例最大值": "Color legend max",
    "应力范围": "Stress range",
    "SMX": "SMX",
    "SMN": "SMN",
    "DMX": "DMX",
}
# --------------------------------------------------

class CaseLibrary:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.cases = []
        self.text_vectors = None
        self.stress_vectors = None
        self._build()

    def _build(self):
        query_timepoints = """
        MATCH (t:TimePoint)
        RETURN t.time AS time, 
               t.瞬时损伤 AS damage_value,
               t.实际吊重量 AS actual_load,
               t.主臂长度 AS boom_length,
               t.工作幅度 AS working_radius,
               t.力矩百分比 AS torque_percent,
               t.运行时间 AS running_hours,
               t.额定吊重量 AS rated_load,
               t.角度 AS angle
        ORDER BY t.time
        """

        query_entities = """
        MATCH (t:TimePoint {time: $time})-[r]->(e:Entity)
        RETURN r.relation AS rel, e.value AS value
        """

        with self.driver.session() as session:
            timepoints = session.run(query_timepoints)
            for record in timepoints:
                time = record["time"]
                damage_val = record["damage_value"]
                if damage_val is None:
                    print(f"警告：时间点 {time} 没有 瞬时损伤 属性，跳过")
                    continue

                params = {
                    "actual_load": record.get("actual_load"),
                    "boom_length": record.get("boom_length"),
                    "working_radius": record.get("working_radius"),
                    "torque_percent": record.get("torque_percent"),
                    "running_hours": record.get("running_hours"),
                    "rated_load": record.get("rated_load"),
                    "angle": record.get("angle"),
                }

                entities = session.run(query_entities, time=time)
                text_parts = []
                stress_parts = []
                for ent in entities:
                    rel = ent["rel"]
                    val = ent["value"]
                    # 补充参数（如果从关系获取到值，且未在timepoint中直接存在）
                    if rel == "实际吊重量" and params["actual_load"] is None:
                        params["actual_load"] = val
                    elif rel == "主臂长度" and params["boom_length"] is None:
                        params["boom_length"] = val
                    elif rel == "工作幅度" and params["working_radius"] is None:
                        params["working_radius"] = val
                    elif rel == "力矩百分比" and params["torque_percent"] is None:
                        params["torque_percent"] = val
                    elif rel == "运行时间" and params["running_hours"] is None:
                        params["running_hours"] = val
                    elif rel == "额定吊重量" and params["rated_load"] is None:
                        params["rated_load"] = val
                    elif rel == "角度" and params["angle"] is None:
                        params["angle"] = val

                    # ---------- 核心改动：关系名翻译为英文 ----------
                    en_rel = REL_TO_EN.get(rel, rel)  # 如果未映射则保留原关系名（但多数已覆盖）
                    # 判断是否应力相关（根据关系名或值中包含"MPa"或"应力"）
                    if "应力" in rel or "MPa" in str(val) or "应力" in str(val):
                        stress_parts.append(f"{en_rel}: {val}")
                    else:
                        text_parts.append(f"{en_rel}: {val}")

                # 补充直接参数（使用英文标签）
                if params["actual_load"] is not None:
                    text_parts.append(f"Actual load: {params['actual_load']}")
                if params["boom_length"] is not None:
                    text_parts.append(f"Boom length: {params['boom_length']}")
                if params["working_radius"] is not None:
                    text_parts.append(f"Working radius: {params['working_radius']}")
                if params["torque_percent"] is not None:
                    text_parts.append(f"Torque percentage: {params['torque_percent']}")
                if params["running_hours"] is not None:
                    text_parts.append(f"Running hours: {params['running_hours']}")
                if params["rated_load"] is not None:
                    text_parts.append(f"Rated load: {params['rated_load']}")
                if params["angle"] is not None:
                    text_parts.append(f"Angle: {params['angle']}")

                text_desc = "；".join(text_parts) if text_parts else f"Time point {time} condition data"
                stress_desc = "；".join(stress_parts) if stress_parts else f"Time point {time} stress data"

                self.cases.append({
                    "time": time,
                    "text_desc": text_desc,
                    "stress_desc": stress_desc,
                    "life_value": float(damage_val),
                    "params": params,
                    "text_vector": model.encode(text_desc).reshape(1, -1),
                    "stress_vector": model.encode(stress_desc).reshape(1, -1),
                })

        if self.cases:
            self.text_vectors = np.vstack([c["text_vector"] for c in self.cases])
            self.stress_vectors = np.vstack([c["stress_vector"] for c in self.cases])
        print(f"案例库构建完成，共 {len(self.cases)} 个案例")

    def retrieve_similar(self, query_text: str = None, query_vector=None, modal="text", top_k=5) -> List[Dict]:
        if query_text and query_vector is None:
            query_vector = model.encode(query_text).reshape(1, -1)
        if query_vector is None:
            return []
        vectors = self.text_vectors if modal == "text" else self.stress_vectors
        if vectors is None or len(vectors) == 0:
            return []
        sims = cosine_similarity(query_vector, vectors)[0]
        indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in indices:
            c = copy.deepcopy(self.cases[idx])
            c.pop("text_vector", None)
            c.pop("stress_vector", None)
            c["similarity"] = float(sims[idx])
            results.append(c)
        return results

    def get_neighbor_triples(self, time: str) -> List[Tuple[str, str, str]]:
        query = "MATCH (t:TimePoint {time: $time})-[r]->(e:Entity) RETURN r.relation AS rel, e.value AS val LIMIT 30"
        with self.driver.session() as session:
            return [(time, rec["rel"], rec["val"]) for rec in session.run(query, time=time)]

    def close(self):
        self.driver.close()

_case_lib = None
def get_case_library():
    global _case_lib
    if _case_lib is None:
        _case_lib = CaseLibrary()
    return _case_lib