# -*- coding: utf-8 -*-
"""
基于 LangChain 的 Neo4j 知识图谱问答系统
支持自然语言查询，自动生成 Cypher 查询并返回结果。
针对中文属性名和带单位的数值比较进行了优化。
"""

import os
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

# ==================== 配置区域 ====================
# Runtime configuration only; QA prompt/chain logic below is the original version.
from config import NEO4J_URI as NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
MODEL_NAME = "deepseek-chat"

VERBOSE = False
# ==================================================

def create_chain():
    """创建并返回问答链（单例模式）"""
    if not hasattr(create_chain, "chain"):
        print("正在连接 Neo4j...")
        graph = Neo4jGraph(
            url=NEO4J_URL,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD
        )
        graph.refresh_schema()
        print("✅ 连接成功，图模式如下：")
        print(graph.schema)
        print("-" * 60)

        llm = ChatOpenAI(
            model=MODEL_NAME,
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_BASE_URL,
            temperature=0,
        )

        # ========== 增强提示词（中文，包含映射示例） ==========
        CYPHER_GENERATION_TEMPLATE = """
你是一个 Neo4j 专家。请根据以下图模式将用户问题转换为 Cypher 查询。

重要规则：
- 所有属性名都是中文，如果属性名包含特殊字符或中文，请用反引号 ` 括起来。
- 用户可能用英文提问，但你必须将英文术语映射到正确的**中文**属性名（参考下方映射）。
- 排序必须使用 `ORDER BY`，而不是 `ORDER TO`。
- 如果属性值包含单位（如 "2.5 t" 或 "340 MPa"），且需要进行数值比较或排序，请使用 `toFloat(split(属性名, ' ')[0])` 提取数字部分进行数值操作。
- **只返回 Cypher 查询语句，不要包含任何额外解释。**

常见英文→中文映射（必须使用）：
- "actual load" → '实际吊重量'
- "working radius" → '工作幅度'
- "boom length" → '主臂长度'
- "maximum stress" → '最大应力'
- "torque percentage" → '力矩百分比'
- "rated load" → '额定吊重量'
- "oil pressure" → '油压'
- "engine speed" → '发动机转速'

示例：
问题：2024-12-15 10:27:49 的最大应力是多少？
Cypher：
MATCH (t:TimePoint)-[r:RELATES]->(e:Entity)
WHERE t.time = '2024-12-15 10:27:49' AND r.relation = '最大应力'
RETURN e.value AS 最大应力值

问题：哪个时间点的实际吊重量最大？
Cypher：
MATCH (t:TimePoint)-[r:RELATES]->(e:Entity)
WHERE r.relation = '实际吊重量'
RETURN t.time AS 时间点, toFloat(e.value) AS 吊重量
ORDER BY 吊重量 DESC
LIMIT 1

图模式:
{schema}

问题: {question}
Cypher 查询:
"""
        # ================================================================

        cypher_prompt = PromptTemplate(
            template=CYPHER_GENERATION_TEMPLATE,
            input_variables=["schema", "question"]
        )

        chain = GraphCypherQAChain.from_llm(
            graph=graph,
            llm=llm,
            cypher_prompt=cypher_prompt,
            verbose=VERBOSE,
            allow_dangerous_requests=True,
            return_intermediate_steps=False,
        )
        create_chain.chain = chain
    return create_chain.chain


def query_knowledge_graph(question: str) -> str:
    """
    对外提供的函数接口：输入自然语言问题，返回答案文本。
    """
    chain = create_chain()
    result = chain.invoke(question)
    return result['result']


if __name__ == "__main__":
    test_q = "2024-03-19 09:27:44 的油压是多少？"
    ans = query_knowledge_graph(test_q)
    print(f"问题: {test_q}\n答案: {ans}")