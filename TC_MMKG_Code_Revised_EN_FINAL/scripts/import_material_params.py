# -*- coding: utf-8 -*-
"""Import the Q690D parameter set used by the fatigue module into Neo4j."""
from __future__ import annotations

from neo4j import GraphDatabase

from config import (
    NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER,
    Q690D_FAT_MPA, Q690D_GAMMA_M, Q690D_NREF, Q690D_RE_MPA,
    Q690D_RM_MPA, Q690D_SN_SLOPE,
)

MATERIAL_DATA = {
    "name": "Q690D",
    "Rm": Q690D_RM_MPA,
    "Re": Q690D_RE_MPA,
    "FAT": Q690D_FAT_MPA,
    "m": Q690D_SN_SLOPE,
    "Nref": Q690D_NREF,
    "gamma_m": Q690D_GAMMA_M,
    "E": 206000.0,
    "rho": 7850.0,
}


def import_material_params() -> dict:
    if not NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD is not configured.")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            session.run(
                """
                MERGE (m:Material {name:$name})
                SET m.Rm_MPa=$Rm, m.Re_MPa=$Re, m.FAT_MPa=$FAT,
                    m.SN_slope_m=$m, m.Nref=$Nref, m.gamma_m=$gamma_m,
                    m.E_MPa=$E, m.density_kg_m3=$rho
                MERGE (b:Boom {model:'QY25K'})
                MERGE (b)-[:MADE_OF]->(m)
                """,
                **MATERIAL_DATA,
            ).consume()
    finally:
        driver.close()
    return {"success": True, "material": MATERIAL_DATA.copy()}


if __name__ == "__main__":
    print(import_material_params())
