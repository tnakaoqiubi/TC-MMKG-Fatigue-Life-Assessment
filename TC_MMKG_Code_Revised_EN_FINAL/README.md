# TC-MMKG Truck-Crane Telescopic-Boom Remaining Fatigue Life Assessment System (English Version)

This project follows the overall workflow described in **Remaining Fatigue Life Assessment of Truck-Crane Telescopic Booms via Multimodal Knowledge Graphs and Physics-Constrained Reasoning**.

## Main modules

- `scripts/文本_时间序列.py`: dual-Producer text extraction, Jaccard agreement, Verifier debate, iterative validation, and integration.
- `scripts/图像_时间序列.py`: dual-VLM stress-contour extraction, separate numerical/descriptive agreement checks, and Verifier arbitration.
- `scripts/余弦相似度对齐_报告.py`: SBERT relation/entity alignment and confidence-guided conflict resolution.
- `tools/kg_import_tool.py`: time-point-centered TC-MMKG/Neo4j ingestion.
- `scripts/rainflow2_use.py` + `scripts/try6_use.py`: rainflow counting, graph-stress retrieval, Haigh/Goodman correction, S-N relation, Miner cumulative damage, and remaining-life estimation.
- `main_agent.py`: LangGraph coordinator with five specialist agents.
- `scripts/case_library.py` + `scripts/analogical_reasoner.py` + `scripts/causal_validator.py`: Top-5 case retrieval, neighboring knowledge, analogical generation, and the restored original validation logic.
- `scripts/qa_1.py`: restored original knowledge-graph QA logic.
- `app.py` + `templates/index3.html`: English Web interface, QA/graph visualization, analogical reasoning, and report downloads.

## Run

1. Python 3.13 is recommended.
2. Install dependencies: `pip install -r requirements.txt`.
3. Configure DeepSeek, Qwen, Neo4j, and the SBERT path according to `.env.example`.
4. Start Neo4j.
5. Run: `python app.py`.

## Language-version note

This English package and the paired Chinese package use the same algorithms, API routes, data structures, and workflow. Only user-visible interface text, runtime messages, report/chart labels, and export titles are localized. Internal JSON keys, CSV/Excel schema fields, function/API names, and Chinese semantic relation names used by the existing Neo4j graph are intentionally preserved for compatibility.

## Restored modules

As requested, the following two areas retain the core algorithm/call flow of the user's originally uploaded version, with only current-project configuration loading preserved:

- 🧠 Cross-modal analogical reasoning and causal validation
- 💬 Smart assistant (KG QA + natural-language visualization)

All other text/image extraction, fusion, Neo4j ingestion, fatigue calculation, and LangGraph components remain at the current revised version.
