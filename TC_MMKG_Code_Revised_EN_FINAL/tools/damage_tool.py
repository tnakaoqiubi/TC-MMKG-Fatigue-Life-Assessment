from typing import Type
from pydantic import BaseModel, Field
try:
    from langchain_core.tools import BaseTool
except Exception:
    from langchain.tools import BaseTool
from scripts.try6_use import run_damage_analysis

class DamageInput(BaseModel):
    rainflow_excel: str = Field(description="Rainflow result Excel path")
class DamageTool(BaseTool):
    name: str = "damage_calculation"
    description: str = "Retrieve graph stresses, apply Haigh/S-N/Miner calculation, and estimate remaining cycles."
    args_schema: Type[BaseModel] = DamageInput
    def _run(self,rainflow_excel:str)->str:
        report,chart,result=run_damage_analysis(rainflow_excel);return f"{report}\nchart={chart}\nresult={result}"
    async def _arun(self,rainflow_excel:str):raise NotImplementedError
