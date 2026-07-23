# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from config import (OUTPUT_DIR, STATIC_DIR, TEMP_UPLOAD_DIR, TEMPLATE_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL)
from scripts.文本_时间序列 import process_text_excel
from scripts.图像_时间序列 import process_images, get_last_extraction_summary
from scripts.余弦相似度对齐_报告 import fuse_multimodal
from scripts.rainflow2_use import run_rainflow
from scripts.try6_use import run_damage_analysis
from scripts.qa_1 import query_knowledge_graph
from scripts.analogical_reasoner import AnalogicalReasoner
from scripts.causal_validator import CausalValidator
from scripts.report_generator import generate_assessment_report
from tools.kg_import_tool import import_triples
from tools.damage_to_neo4j import update_damage_to_neo4j
from main_agent import run_pipeline

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024

# Local single-user runtime guard: a fatigue calculation is allowed only after
# a fused triple set has been imported into Neo4j during the current server run.
# This prevents stale graph data or the load-percentage fallback from silently
# bypassing the manuscript workflow.
ACTIVE_GRAPH_TOKENS: dict[str, dict] = {}


def _job_dir(kind: str) -> Path:
    p = Path(TEMP_UPLOAD_DIR) / kind / uuid.uuid4().hex
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_file(storage, folder: Path) -> str:
    name = secure_filename(storage.filename or '')
    if not name:
        raise ValueError('Filename is empty')
    path = folder / name
    storage.save(path)
    return str(path)


def _json_error(exc, code=500):
    return jsonify({'success': False, 'error': str(exc)}), code


@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST'
    return response


@app.route('/')
def index():
    return render_template('index3.html')


@app.route('/api/text_extract', methods=['POST'])
def text_extract():
    try:
        file = request.files.get('excel')
        if not file: return _json_error('No Excel file uploaded', 400)
        path = _save_file(file, _job_dir('excel'))
        csv_path = process_text_excel(path)
        return jsonify({'success': True, 'csv_path': csv_path, 'excel_path': path})
    except Exception as exc: return _json_error(exc)


@app.route('/api/image_extract', methods=['POST'])
def image_extract():
    try:
        files = [f for f in request.files.getlist('images') if f.filename]
        if not files: return _json_error('No stress contour images uploaded', 400)
        folder = _job_dir('images')
        for f in files: _save_file(f, folder)
        csv_path = process_images(str(folder))
        summary = get_last_extraction_summary()
        return jsonify({'success': True, 'csv_path': csv_path, 'extraction_summary': summary})
    except Exception as exc: return _json_error(exc)


@app.route('/api/fusion', methods=['POST'])
def fusion():
    try:
        data = request.get_json(silent=True) or {}
        text_csv = data.get('text_csv', '')
        image_csv = data.get('image_csv', '')
        print(f"[API /api/fusion] request received: text_csv={text_csv}, image_csv={image_csv}", flush=True)
        fused, report = fuse_multimodal(text_csv, image_csv)
        print(f"[API /api/fusion] success: {fused}", flush=True)
        return jsonify({'success': True, 'fused_csv': fused, 'report': report})
    except Exception as exc:
        print(f"[API /api/fusion] ERROR: {type(exc).__name__}: {exc}", flush=True)
        return _json_error(exc)


@app.route('/api/kg_import', methods=['POST'])
def kg_import():
    try:
        excel_path = None
        if request.is_json:
            data = request.get_json() or {}
            fused_csv = data.get('fused_csv')
            excel_path = data.get('excel_path')
        else:
            file = request.files.get('fused_file')
            if not file: return _json_error('No fused-triple CSV uploaded', 400)
            fused_csv = _save_file(file, _job_dir('fused'))
        result = import_triples(fused_csv)
        graph_token = uuid.uuid4().hex
        ACTIVE_GRAPH_TOKENS[graph_token] = {
            'fused_csv': str(fused_csv),
            'imported_at': time.time(),
        }
        return jsonify({
            'success': True, 'message': result['message'],
            'graph_token': graph_token,
            'damage_updated': False,
            'damage_message': 'Graph import completed. Run load analysis and life prediction to calculate and write fatigue damage.'
        })
    except Exception as exc: return _json_error(exc)


@app.route('/api/rainflow', methods=['POST'])
def rainflow():
    try:
        file = request.files.get('excel')
        if not file: return _json_error('No Excel file uploaded',400)
        path = _save_file(file, _job_dir('excel'))
        result = run_rainflow(path)
        return jsonify({'success': True, 'rainflow_excel': result, 'excel_path': path})
    except Exception as exc: return _json_error(exc)


@app.route('/api/damage', methods=['POST'])
def damage():
    try:
        data = request.get_json(silent=True) or {}
        rainflow_excel = data.get('rainflow_excel')
        graph_token = data.get('graph_token')
        if not rainflow_excel or not Path(rainflow_excel).exists():
            return _json_error('Rainflow result file does not exist', 400)
        if not graph_token or graph_token not in ACTIVE_GRAPH_TOKENS:
            return _json_error(
                'The current fused triples have not been imported into Neo4j, so fatigue calculation cannot run yet. '
                'Complete text/image knowledge extraction → cross-modal fusion → Neo4j import first.',
                409,
            )
        report, chart, result_excel = run_damage_analysis(
            rainflow_excel,
            use_graph_stress=True,
            allow_stress_fallback=False,
        )
        per_row = result_excel.replace('_损伤结果.xlsx', '_每行损伤.xlsx')
        damage_graph_update = update_damage_to_neo4j(per_row) if Path(per_row).exists() else {'success': False, 'message': 'Per-row damage result file was not found'}
        stamp = int(time.time())
        chart_name = f'damage_{stamp}.png'
        shutil.copy2(chart, Path(STATIC_DIR)/chart_name)
        report_docx = Path(OUTPUT_DIR)/f'Truck_Crane_Boom_Remaining_Fatigue_Life_Assessment_Report_{stamp}.docx'
        generate_assessment_report(str(report_docx), damage_report=report, chart_path=chart)
        return jsonify({
            'success':True, 'report':report, 'chart_url':f'/static/{chart_name}',
            'result_excel':result_excel, 'report_url':f'/api/download/{report_docx.name}', 'damage_graph_update': damage_graph_update
        })
    except Exception as exc: return _json_error(exc)


@app.route('/api/download/<path:filename>')
def download_output(filename):
    return send_from_directory(str(OUTPUT_DIR), filename, as_attachment=True)


@app.route('/api/kg_query', methods=['POST'])
def kg_query():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400
    try:
        answer = query_knowledge_graph(question)
        return jsonify({'success': True, 'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def extract_cypher(text: str) -> str:
    text = text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        if len(lines) > 2:
            text = '\n'.join(lines[1:-1])
        else:
            text = text.strip('`').strip()
    match = re.search(r'```(?:\w+)?\s*(.*?)```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text.strip()


# ---------- NL Query & Visualization ----------
@app.route('/api/kg_nl_query', methods=['POST'])
def kg_nl_query():
    from neo4j import GraphDatabase
    from langchain_openai import ChatOpenAI
    from langchain_community.graphs import Neo4jGraph

    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'Please provide a natural language question'}), 400

    uri = NEO4J_URI
    user = NEO4J_USER
    password = NEO4J_PASSWORD
    deepseek_key = DEEPSEEK_API_KEY

    try:
        graph = Neo4jGraph(url=uri, username=user, password=password)
        schema = graph.schema
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=deepseek_key,
            openai_api_base=DEEPSEEK_BASE_URL,
            temperature=0
        )

        cypher_prompt = f"""
You are a Neo4j expert. Convert the user's question into a Cypher query according to the graph schema below.

Important rules:
- Property/relationship semantic names in this TC-MMKG are stored in Chinese. Use backticks for Chinese or special-character property names when needed.
- Map English engineering terms to the correct Chinese semantic names.
- Use `ORDER BY`, never `ORDER TO`.
- When a value contains units (for example "2.5 t" or "340 MPa") and numeric comparison/sorting is needed, use `toFloat(split(property, ' ')[0])`.
- Return only the Cypher query, with no explanation.

Common English → graph-semantic mappings:
- "actual load" → '实际吊重量'
- "working radius" → '工作幅度'
- "boom length" → '主臂长度'
- "maximum stress" → '最大应力'
- "torque percentage" → '力矩百分比'
- "rated load" → '额定吊重量'
- "oil pressure" → '油压'
- "engine speed" → '发动机转速'

Example:
Question: What is the maximum stress at 2024-12-15 10:27:49?
Cypher:
MATCH (t:TimePoint)-[r:RELATES]->(e:Entity)
WHERE t.time = '2024-12-15 10:27:49' AND r.relation = '最大应力'
RETURN e.value AS maximum_stress

Question: At which time point is the actual load the largest?
Cypher:
MATCH (t:TimePoint)-[r:RELATES]->(e:Entity)
WHERE r.relation = '实际吊重量'
RETURN t.time AS time_point, toFloat(e.value) AS actual_load
ORDER BY actual_load DESC
LIMIT 1

Graph schema:
{schema}

Question: {question}
Cypher query:
"""
        response = llm.invoke(cypher_prompt)
        raw_cypher = response.content.strip()
        cypher = extract_cypher(raw_cypher)

        driver = GraphDatabase.driver(uri, auth=(user, password))
        nodes = []
        edges = []
        node_ids = set()
        with driver.session() as session:
            result = session.run(cypher)
            for record in result:
                for value in record.values():
                    if hasattr(value, 'element_id'):
                        if hasattr(value, 'labels'):
                            n_id = value.element_id
                            if n_id not in node_ids:
                                props = dict(value)
                                label = list(value.labels)[0] if value.labels else "Node"

                                if label == 'TimePoint':
                                    display_name = value.get('time') or props.get('time')
                                    if not display_name:
                                        display_name = label
                                elif label == 'DamageResult':
                                    display_name = value.get('time') or props.get('time')
                                    if not display_name:
                                        display_name = value.get('value') or props.get('value')
                                    if not display_name:
                                        display_name = f"{props.get('instant_damage', 0):.6e}"
                                else:
                                    display_name = value.get('value') or props.get('value')
                                    if not display_name:
                                        display_name = value.get('name') or props.get('name')
                                    if not display_name:
                                        display_name = value.get('model') or props.get('model')
                                    if not display_name:
                                        display_name = label

                                if not display_name:
                                    display_name = label

                                nodes.append({
                                    "id": n_id,
                                    "label": f"{label}: {display_name}",
                                    "title": f"{label}\n" + "\n".join([f"{k}: {v}" for k, v in props.items()])
                                })
                                node_ids.add(n_id)
                        elif hasattr(value, 'type'):
                            edges.append({
                                "from": value.start_node.element_id,
                                "to": value.end_node.element_id,
                                "label": value.type,
                                "title": value.type
                            })
        driver.close()
        return jsonify({
            "success": True,
            "cypher": cypher,
            "nodes": nodes,
            "edges": edges
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- Cross-modal Reasoning ----------
@app.route('/api/crossmodal_reason', methods=['POST'])
def crossmodal_reason():
    from scripts.analogical_reasoner import AnalogicalReasoner
    from scripts.causal_validator import CausalValidator
    data = request.json
    modal = data.get('modal')
    content = data.get('content')
    target = data.get('target')
    if not modal or not content:
        return jsonify({
            'success': False,
            'validation_errors': ['Missing modal or content parameter'],
            'validation_layer': 'input_error'
        }), 400
    try:
        reasoner = AnalogicalReasoner()
        result = reasoner.reason(modal, content, target)
        def reason_wrapper(m, c, t):
            return reasoner.reason(m, c, t)
        validator = CausalValidator()
        validation = validator.validate(result, reasoning_func=reason_wrapper)
        response_data = {
            'success': validation['passed'],
            'generated_text': validation.get('final_output', ''),
            'reasoning_chain': validation.get('reasoning_chain', []),
            'retrieved_cases': validation.get('retrieved_cases', []),
            'validation_errors': validation.get('errors', []),
            'validation_layer': validation.get('layer', 'none')
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({
            'success': False,
            'validation_errors': [str(e)],
            'validation_layer': 'exception',
            'generated_text': '',
            'reasoning_chain': [],
            'retrieved_cases': []
        }), 500


@app.route('/api/full_pipeline', methods=['POST'])
def full_pipeline():
    """Run the complete manuscript workflow through the LangGraph coordinator."""
    try:
        excel_file = request.files.get('excel')
        fused_file = request.files.get('fused_file')
        image_files = [f for f in request.files.getlist('images') if f.filename]
        if not excel_file and not fused_file:
            return _json_error('Provide raw Excel or an existing fused-triple CSV', 400)

        excel_path = _save_file(excel_file, _job_dir('excel')) if excel_file else None
        fused_csv = _save_file(fused_file, _job_dir('fused')) if fused_file else None
        image_folder = None

        if fused_csv is None:
            if not excel_path:
                return _json_error('Raw Excel is required when no fused CSV is provided', 400)
            if not image_files:
                return _json_error('Stress contour images are required in full extraction mode', 400)
            # The engineering validation set uses one contour map per time point.
            row_count = len(pd.read_excel(excel_path))
            if row_count != len(image_files):
                return _json_error(
                    f'Number of stress contour images ({len(image_files)}) does not match the Excel row count ({row_count}).', 400
                )
            folder = _job_dir('images')
            for f in image_files:
                _save_file(f, folder)
            image_folder = str(folder)

        state = run_pipeline(
            excel_path=excel_path,
            image_folder=image_folder,
            fused_csv=fused_csv,
            import_to_neo4j=True,
        )
        result = dict(state)

        # Expose the generated chart in the same way as the individual damage API.
        chart = result.get('chart_path')
        if chart and Path(chart).exists():
            stamp = int(time.time())
            chart_name = f'full_pipeline_damage_{stamp}.png'
            shutil.copy2(chart, Path(STATIC_DIR) / chart_name)
            result['chart_url'] = f'/static/{chart_name}'

        return jsonify({'success': True, **result})
    except ValueError as exc:
        return _json_error(exc, 400)
    except Exception as exc:
        return _json_error(exc, 500)


if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG','0')=='1',host='0.0.0.0',port=int(os.getenv('PORT','5000')))
