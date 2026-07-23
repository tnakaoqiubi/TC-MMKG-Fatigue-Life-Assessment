# -*- coding: utf-8 -*-
"""Import fused TC-MMKG triples into Neo4j with a schema shared by all modules."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Type

import pandas as pd
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
try:
    from langchain_core.tools import BaseTool
except Exception:  # pragma: no cover
    from langchain.tools import BaseTool

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from scripts.common import TIME_RE, parse_float

RELATION_EN_MAP = {
    "车辆代码": "VEHICLE_CODE", "型号": "MODEL", "位于区域": "REGION",
    "ACC状态": "ACC_STATUS", "油压": "OIL_PRESSURE", "水温": "WATER_TEMP",
    "发动机转速": "ENGINE_SPEED", "运行时间": "RUNNING_HOURS", "车速": "VEHICLE_SPEED",
    "实际吊重量": "ACTUAL_LOAD", "额定吊重量": "RATED_LOAD", "主臂长度": "BOOM_LENGTH",
    "角度": "ANGLE", "工作幅度": "WORKING_RADIUS", "工况代码": "CASE_CODE",
    "倍率": "REEVING_RATIO", "力矩百分比": "TORQUE_PERCENT", "力限器故障码": "LIMITER_FAULT",
    "控制类故障码": "CONTROL_FAULT", "发动机故障码": "ENGINE_FAULT",
    "绑定状态": "BINDING_STATUS", "锁车状态": "LOCK_STATUS", "属于车辆": "BELONGS_TO",
    "SMX": "SMX", "SMN": "SMN", "DMX": "DMX", "最大应力位置": "MAX_STRESS_LOCATION",
    "最大应力": "MAX_STRESS", "最小应力": "MIN_STRESS", "平均应力": "MEAN_STRESS",
    "应力幅": "STRESS_AMPLITUDE", "图像最大应力值": "IMAGE_MAX_STRESS",
    "图像应力极值差": "IMAGE_STRESS_RANGE", "图像应力均值": "IMAGE_MEAN_STRESS",
}
TIMEPOINT_PROPERTY_MAP = {
    "车辆代码": "车辆代码", "型号": "型号", "实际吊重量": "实际吊重量",
    "额定吊重量": "额定吊重量", "主臂长度": "主臂长度", "角度": "角度",
    "工作幅度": "工作幅度", "力矩百分比": "力矩百分比", "运行时间": "运行时间",
}
NUMERIC_RELATIONS = {"实际吊重量", "额定吊重量", "主臂长度", "角度", "工作幅度", "力矩百分比", "运行时间", "SMX", "SMN", "DMX"}


def get_english_relation(relation: str) -> str:
    if relation in RELATION_EN_MAP:
        return RELATION_EN_MAP[relation]
    safe = re.sub(r"[^A-Za-z0-9_]", "_", relation.upper()).strip("_")
    return safe or "RELATES"


def get_label_for_tail(relation: str) -> str:
    if relation in {"实际吊重量", "额定吊重量", "力矩百分比"}:
        return "LoadParam"
    if relation in {"主臂长度", "角度", "工作幅度"}:
        return "GeomParam"
    if relation in {"SMX", "SMN", "最大应力", "最小应力", "平均应力", "应力幅", "图像最大应力值", "图像应力极值差", "图像应力均值"}:
        return "StressValue"
    if relation in {"DMX", "最大应力位置"}:
        return "ImageFeature"
    return "Entity"


def import_triples(triples_csv: str) -> dict:
    path = Path(triples_csv)
    if not path.exists():
        raise FileNotFoundError(triples_csv)
    if not NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD is not configured.")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"head", "relation", "tail"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {sorted(required - set(df.columns))}")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.7) if "confidence" in df.columns else 0.7
    df["source"] = df.get("source", "unknown")
    df["consistency"] = df.get("consistency", "unspecified")
    valid = df[
        df["head"].astype(str).str.match(TIME_RE) &
        df["relation"].notna() & df["tail"].notna()
    ].copy()
    if valid.empty:
        raise ValueError("No valid time-point-centred triples found in CSV.")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    imported = 0
    try:
        with driver.session() as session:
            session.run("CREATE INDEX timepoint_time IF NOT EXISTS FOR (t:TimePoint) ON (t.time)").consume()
            session.run("CREATE INDEX entity_value IF NOT EXISTS FOR (e:Entity) ON (e.value)").consume()
            for _, row in valid.iterrows():
                head, relation, tail = str(row["head"]).strip(), str(row["relation"]).strip(), str(row["tail"]).strip()
                rel_type = get_english_relation(relation)
                label = get_label_for_tail(relation)
                # Dynamic type and label are generated from controlled mappings/sanitized identifiers.
                cypher = f"""
                MERGE (t:TimePoint {{time:$head}})
                SET t.name=$head
                MERGE (e:Entity {{value:$tail, semantic_type:$label}})
                SET e:{label}, e.name=$tail
                MERGE (t)-[r:`{rel_type}`]->(e)
                SET r.relation=$relation, r.confidence=$confidence,
                    r.source=$source, r.consistency=$consistency
                """
                session.run(
                    cypher,
                    head=head, tail=tail, label=label, relation=relation,
                    confidence=float(row["confidence"]), source=str(row["source"]), consistency=str(row["consistency"]),
                ).consume()
                if relation in TIMEPOINT_PROPERTY_MAP:
                    prop = TIMEPOINT_PROPERTY_MAP[relation]
                    numeric = parse_float(tail) if relation in NUMERIC_RELATIONS else None
                    # Property names come from a fixed whitelist above.
                    session.run(
                        f"MATCH (t:TimePoint {{time:$head}}) SET t.`{prop}`=$value",
                        head=head, value=numeric if numeric is not None else tail,
                    ).consume()
                imported += 1
            # Graph-construction status statistics used by the QA/application layer.
            graph_nodes = session.run(
                "MATCH (n) RETURN count(n) AS c"
            ).single()["c"]
            graph_relationships = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS c"
            ).single()["c"]
            graph_time_points = session.run(
                "MATCH (t:TimePoint) RETURN count(t) AS c"
            ).single()["c"]
    finally:
        driver.close()
    return {
        "success": True,
        "imported_triples": imported,
        "time_points": int(valid["head"].nunique()),
        "graph_nodes": int(graph_nodes),
        "graph_relationships": int(graph_relationships),
        "graph_time_points": int(graph_time_points),
        "message": (
            f"Successfully imported {imported} triples covering {valid['head'].nunique()} time points. "
            f"Current graph statistics: {graph_nodes} nodes, {graph_relationships} relationships, "
            f"{graph_time_points} TimePoint nodes."
        ),
    }


class KGImportInput(BaseModel):
    triples_csv: str = Field(description="Fused triple CSV path containing head, relation and tail columns")


class KGImportTool(BaseTool):
    name: str = "kg_import"
    description: str = "Import time-point-centred fused triples into the TC-MMKG Neo4j database."
    args_schema: Type[BaseModel] = KGImportInput

    def _run(self, triples_csv: str) -> str:
        return import_triples(triples_csv)["message"]

    async def _arun(self, triples_csv: str):
        raise NotImplementedError
