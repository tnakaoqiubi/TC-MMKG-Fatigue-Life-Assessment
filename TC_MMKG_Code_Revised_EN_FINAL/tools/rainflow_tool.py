from typing import Type
from pydantic import BaseModel, Field
try:
    from langchain_core.tools import BaseTool
except Exception:
    from langchain.tools import BaseTool
from scripts.rainflow2_use import run_rainflow

class RainflowInput(BaseModel):
    input_excel: str = Field(description="Operational-data Excel path")
class RainflowTool(BaseTool):
    name: str = "rainflow_analysis"
    description: str = "Extract turning points and perform ASTM-style rainflow cycle counting."
    args_schema: Type[BaseModel] = RainflowInput
    def _run(self,input_excel:str)->str:return run_rainflow(input_excel)
    async def _arun(self,input_excel:str):raise NotImplementedError
