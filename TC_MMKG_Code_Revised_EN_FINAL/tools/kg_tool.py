from typing import Type
from pydantic import BaseModel, Field
try:
    from langchain_core.tools import BaseTool
except Exception:
    from langchain.tools import BaseTool
from scripts.qa_1 import query_knowledge_graph

class KGQueryInput(BaseModel): question:str=Field(description="Natural-language TC-MMKG question")
class KGQueryTool(BaseTool):
    name:str="kg_query"; description:str="Read-only natural-language query over TC-MMKG."; args_schema:Type[BaseModel]=KGQueryInput
    def _run(self,question:str)->str:return query_knowledge_graph(question)
    async def _arun(self,question:str):raise NotImplementedError
