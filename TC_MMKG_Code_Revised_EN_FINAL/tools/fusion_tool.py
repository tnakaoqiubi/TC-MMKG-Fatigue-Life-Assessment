from typing import Type
from pydantic import BaseModel, Field
try:
    from langchain_core.tools import BaseTool
except Exception:
    from langchain.tools import BaseTool
from scripts.余弦相似度对齐_报告 import fuse_multimodal

class FusionInput(BaseModel):
    text_csv: str = Field(description="Text-modality triple CSV")
    image_csv: str = Field(description="Image-modality triple CSV")

class FusionTool(BaseTool):
    name: str = "multimodal_fusion"
    description: str = "SBERT alignment and confidence-guided cross-modal conflict arbitration."
    args_schema: Type[BaseModel] = FusionInput
    def _run(self, text_csv: str, image_csv: str) -> str:
        fused, report=fuse_multimodal(text_csv,image_csv); return f"{fused}\n{report}"
    async def _arun(self, text_csv: str, image_csv: str): raise NotImplementedError
