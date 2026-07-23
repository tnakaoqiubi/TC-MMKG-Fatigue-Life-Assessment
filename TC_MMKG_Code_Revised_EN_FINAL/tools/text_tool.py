from typing import Type
from pydantic import BaseModel, Field
try:
    from langchain_core.tools import BaseTool
except Exception:
    from langchain.tools import BaseTool
from scripts.文本_时间序列 import process_text_excel

class TextToTriplesInput(BaseModel):
    excel_path: str = Field(description="Operational-data Excel path")

class TextToTriplesTool(BaseTool):
    name: str = "text_to_triples"
    description: str = "Run dual-producer text extraction, evidence-grounded verification and integration."
    args_schema: Type[BaseModel] = TextToTriplesInput
    def _run(self, excel_path: str) -> str: return process_text_excel(excel_path)
    async def _arun(self, excel_path: str): raise NotImplementedError
