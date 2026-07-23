# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from scripts.common import normalize_time


def update_damage_to_neo4j(damage_excel: str) -> dict:
    path = Path(damage_excel)
    if not path.exists():
        return {"success": False, "message": f"File does not exist: {damage_excel}"}
    df = pd.read_excel(path)
    required = {"时间", "瞬时损伤"}
    if not required.issubset(df.columns):
        return {"success": False, "message": f"Excel missing required columns: {sorted(required - set(df.columns))}"}
    if not NEO4J_PASSWORD:
        return {"success": False, "message": "NEO4J_PASSWORD is not configured"}

    valid = df.dropna(subset=["时间", "瞬时损伤"]).copy()
    if valid.empty:
        return {"success": False, "message": "No valid time-damage data"}
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    updated = 0
    try:
        with driver.session() as session:
            for _, row in valid.iterrows():
                time_str = normalize_time(row["时间"])
                instant = float(row["瞬时损伤"])
                cumulative = float(row["累积损伤"]) if "累积损伤" in row and pd.notna(row["累积损伤"]) else None
                remaining = float(row["剩余寿命_循环"]) if "剩余寿命_循环" in row and pd.notna(row["剩余寿命_循环"]) else None
                rec = session.run(
                    """
                    MATCH (t:TimePoint {time:$time})
                    MERGE (d:DamageResult {time:$time})
                    SET d.instant_damage=$instant,
                        d.cumulative_damage=coalesce($cumulative,d.cumulative_damage),
                        d.remaining_life_cycles=coalesce($remaining,d.remaining_life_cycles),
                        d.computed_by='Haigh_SN_Miner', d.material='Q690D',
                        d.value=toString($instant), d.name=toString($instant)
                    MERGE (t)-[:HAS_DAMAGE {relation:'瞬时损伤'}]->(d)
                    SET t.`瞬时损伤`=$instant,
                        t.`累积损伤`=coalesce($cumulative,t.`累积损伤`),
                        t.`剩余寿命_循环`=coalesce($remaining,t.`剩余寿命_循环`)
                    RETURN t.time AS time
                    """,
                    time=time_str, instant=instant, cumulative=cumulative, remaining=remaining,
                ).single()
                if rec:
                    updated += 1
    finally:
        driver.close()
    return {"success": True, "message": f"Updated {updated} time points", "updated_count": updated, "total_rows": len(valid)}
