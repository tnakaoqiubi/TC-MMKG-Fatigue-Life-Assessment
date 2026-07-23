from typing import Type
from pydantic import BaseModel, Field
try:
    from langchain_core.tools import BaseTool
except Exception:
    from langchain.tools import BaseTool
from scripts.图像_时间序列 import process_images

class ImageToTriplesInput(BaseModel):
    image_folder: str = Field(description="Folder containing stress contour images")

class ImageToTriplesTool(BaseTool):
    name: str = "image_to_triples"
    description: str = "Run dual-VLM image extraction and adversarial verification with time-point heads."
    args_schema: Type[BaseModel] = ImageToTriplesInput
    def _run(self, image_folder: str) -> str: return process_images(image_folder)
    async def _arun(self, image_folder: str): raise NotImplementedError
