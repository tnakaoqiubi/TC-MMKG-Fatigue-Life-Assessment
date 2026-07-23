# -*- coding: utf-8 -*-
"""LangGraph-based main agent coordinating five specialist agents.

The five specialist nodes match the manuscript: text extraction, image
extraction, knowledge fusion, graph ingestion, and life prediction.  A
coordinator node dynamically routes each request according to the information
already available in state, so existing text/image/fused files can be reused
without forcing unnecessary upstream steps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

from scripts.文本_时间序列 import process_text_excel
from scripts.图像_时间序列 import process_images
from scripts.余弦相似度对齐_报告 import fuse_multimodal
from scripts.rainflow2_use import run_rainflow
from scripts.try6_use import run_damage_analysis
from tools.kg_import_tool import import_triples
from tools.damage_to_neo4j import update_damage_to_neo4j

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:
    END = "__end__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False


class PipelineState(TypedDict, total=False):
    excel_path: str
    image_folder: str
    text_csv: str
    image_csv: str
    fused_csv: str
    alignment_report: str
    import_to_neo4j: bool
    kg_result: Dict[str, Any]
    rainflow_excel: str
    damage_report: str
    chart_path: str
    result_excel: str
    damage_graph_update: Dict[str, Any]
    steps: list[str]


def _add_step(state: PipelineState, step: str) -> list[str]:
    return [*state.get("steps", []), step]


def _coordinator(state: PipelineState) -> PipelineState:
    """Message/state aggregation node used for dynamic scheduling."""
    return state


def _text_agent(state: PipelineState) -> PipelineState:
    if state.get("text_csv") or state.get("fused_csv"):
        return state
    if not state.get("excel_path"):
        raise ValueError("excel_path is required when neither text_csv nor fused_csv is supplied")
    text_csv = process_text_excel(state["excel_path"])
    return {**state, "text_csv": text_csv, "steps": _add_step(state, "Text extraction")}


def _image_agent(state: PipelineState) -> PipelineState:
    if state.get("image_csv") or state.get("fused_csv"):
        return state
    if not state.get("image_folder"):
        raise ValueError("image_folder is required when neither image_csv nor fused_csv is supplied")
    image_csv = process_images(state["image_folder"])
    return {**state, "image_csv": image_csv, "steps": _add_step(state, "Image extraction")}


def _fusion_agent(state: PipelineState) -> PipelineState:
    if state.get("fused_csv"):
        return state
    if not state.get("text_csv") or not state.get("image_csv"):
        raise ValueError("text_csv and image_csv are required for cross-modal fusion")
    fused, report = fuse_multimodal(state["text_csv"], state["image_csv"])
    return {
        **state,
        "fused_csv": fused,
        "alignment_report": report,
        "steps": _add_step(state, "Cross-modal alignment and fusion"),
    }


def _graph_agent(state: PipelineState) -> PipelineState:
    if not state.get("import_to_neo4j", True) or state.get("kg_result"):
        return state
    if not state.get("fused_csv"):
        raise ValueError("fused_csv is required for graph ingestion")
    result = import_triples(state["fused_csv"])
    return {**state, "kg_result": result, "steps": _add_step(state, "TC-MMKG ingestion")}


def _life_agent(state: PipelineState) -> PipelineState:
    if state.get("result_excel") or not state.get("excel_path"):
        return state
    rainflow = run_rainflow(state["excel_path"])
    report, chart, result = run_damage_analysis(rainflow)
    out: PipelineState = {
        **state,
        "rainflow_excel": rainflow,
        "damage_report": report,
        "chart_path": chart,
        "result_excel": result,
        "steps": [*_add_step(state, "Rainflow counting"), "Haigh/S-N/Miner damage calculation"],
    }
    per_row = str(result).replace("_损伤结果.xlsx", "_每行损伤.xlsx")
    if Path(per_row).exists() and state.get("import_to_neo4j", True):
        out["damage_graph_update"] = update_damage_to_neo4j(per_row)
    return out


def _route_next(state: PipelineState) -> str:
    """Choose the next specialist from current request/state.

    This implements the manuscript's dynamic scheduling idea while preserving
    the natural dependency order: extraction -> fusion -> graph -> life.
    Existing intermediate files skip their corresponding stages.
    """
    # A fused file is a valid quick-mode input and skips extraction/fusion.
    if not state.get("fused_csv"):
        if not state.get("text_csv"):
            if state.get("excel_path"):
                return "text_extraction_agent"
            raise ValueError("Raw Excel or an existing text/fused triple file is required")
        if not state.get("image_csv"):
            if state.get("image_folder"):
                return "image_extraction_agent"
            raise ValueError("Stress images or an existing image/fused triple file are required")
        return "knowledge_fusion_agent"

    # Graph ingestion precedes life calculation for a complete new-data run so
    # SMX/SMN can be retrieved from TC-MMKG as the primary stress source.
    if state.get("import_to_neo4j", True) and not state.get("kg_result"):
        return "graph_ingestion_agent"
    if state.get("excel_path") and not state.get("result_excel"):
        return "life_prediction_agent"
    return END


def build_workflow():
    if not LANGGRAPH_AVAILABLE:
        return None
    graph = StateGraph(PipelineState)
    graph.add_node("coordinator", _coordinator)
    graph.add_node("text_extraction_agent", _text_agent)
    graph.add_node("image_extraction_agent", _image_agent)
    graph.add_node("knowledge_fusion_agent", _fusion_agent)
    graph.add_node("graph_ingestion_agent", _graph_agent)
    graph.add_node("life_prediction_agent", _life_agent)
    graph.set_entry_point("coordinator")
    graph.add_conditional_edges(
        "coordinator",
        _route_next,
        {
            "text_extraction_agent": "text_extraction_agent",
            "image_extraction_agent": "image_extraction_agent",
            "knowledge_fusion_agent": "knowledge_fusion_agent",
            "graph_ingestion_agent": "graph_ingestion_agent",
            "life_prediction_agent": "life_prediction_agent",
            END: END,
        },
    )
    for node in (
        "text_extraction_agent",
        "image_extraction_agent",
        "knowledge_fusion_agent",
        "graph_ingestion_agent",
        "life_prediction_agent",
    ):
        graph.add_edge(node, "coordinator")
    return graph.compile()


WORKFLOW = build_workflow()


def _initial_state(
    excel_path: Optional[str] = None,
    image_folder: Optional[str] = None,
    text_csv: Optional[str] = None,
    image_csv: Optional[str] = None,
    fused_csv: Optional[str] = None,
    import_to_neo4j: bool = True,
) -> PipelineState:
    return {
        "import_to_neo4j": import_to_neo4j,
        "steps": [],
        **({"excel_path": excel_path} if excel_path else {}),
        **({"image_folder": image_folder} if image_folder else {}),
        **({"text_csv": text_csv} if text_csv else {}),
        **({"image_csv": image_csv} if image_csv else {}),
        **({"fused_csv": fused_csv} if fused_csv else {}),
    }


def run_pipeline(
    excel_path: Optional[str] = None,
    image_folder: Optional[str] = None,
    text_csv: Optional[str] = None,
    image_csv: Optional[str] = None,
    fused_csv: Optional[str] = None,
    import_to_neo4j: bool = True,
) -> PipelineState:
    state = _initial_state(excel_path, image_folder, text_csv, image_csv, fused_csv, import_to_neo4j)
    if WORKFLOW is not None:
        return WORKFLOW.invoke(state)

    # Compatibility fallback when LangGraph is unavailable. Use the same dynamic
    # routing semantics rather than a fixed chain, including quick fused-file mode.
    while True:
        nxt = _route_next(state)
        if nxt == END:
            return state
        fn = {
            "text_extraction_agent": _text_agent,
            "image_extraction_agent": _image_agent,
            "knowledge_fusion_agent": _fusion_agent,
            "graph_ingestion_agent": _graph_agent,
            "life_prediction_agent": _life_agent,
        }[nxt]
        state = fn(state)


async def run_pipeline_async(
    excel_path: Optional[str] = None,
    image_folder: Optional[str] = None,
    text_csv: Optional[str] = None,
    image_csv: Optional[str] = None,
    fused_csv: Optional[str] = None,
    import_to_neo4j: bool = True,
) -> PipelineState:
    """Asynchronous LangGraph entry point for message-driven orchestration."""
    state = _initial_state(excel_path, image_folder, text_csv, image_csv, fused_csv, import_to_neo4j)
    if WORKFLOW is not None:
        return await WORKFLOW.ainvoke(state)
    return run_pipeline(excel_path, image_folder, text_csv, image_csv, fused_csv, import_to_neo4j)


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--excel")
    p.add_argument("--images")
    p.add_argument("--text-csv")
    p.add_argument("--image-csv")
    p.add_argument("--fused-csv")
    p.add_argument("--no-neo4j", action="store_true")
    args = p.parse_args()
    result = run_pipeline(
        args.excel,
        args.images,
        args.text_csv,
        args.image_csv,
        args.fused_csv,
        not args.no_neo4j,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
