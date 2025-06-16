from langchain.tools import BaseTool
from typing import List, Optional, Type
from pydantic import BaseModel, Field, PrivateAttr
import logging

from agents.qc_agent import QCAgent
from core.llm import OpenRouterClient, LLMClient


class QcControllerInput(BaseModel):
    analysis_description: str = Field(..., description="The analysis of the transformation.")
    generated_program: dict = Field(..., description="The program/code implementing the transformation.")
    input_grid: Optional[List[List[int]]] = Field(None, description="Optional input grid for context.")
    output_grid: Optional[List[List[int]]] = Field(None, description="Optional output grid for context.")
    prompt_hint: Optional[str] = Field(None, description="Optional extra instructions or feedback.")


class QcControllerTool(BaseTool):
    name: str = "qc_controller"
    description: str = "Verifies if the generated program matches the described analysis by comparing against expected behavior."
    args_schema: Type[BaseModel] = QcControllerInput
    _agent: QCAgent = PrivateAttr()

    class Config:
        """This allows extra parameters not in the pydantic model, like the llm if injected"""
        extra = 'allow'

    def __init__(self, llm: LLMClient = None, **kwargs):
        super().__init__(**kwargs)
        llm = llm or OpenRouterClient()
        self._agent = QCAgent(llm=llm)

    def _run(
        self,
        analysis_description: str,
        generated_program: dict,
        input_grid: Optional[List[List[int]]] = None,
        output_grid: Optional[List[List[int]]] = None,
        prompt_hint: Optional[str] = None
    ):
        try:
            return self._agent.verify(
                analysis_description=analysis_description,
                generated_program=generated_program,
                input_grid=input_grid,
                output_grid=output_grid,
                hint=prompt_hint
            )
        except Exception as e:
            return f"Quality check failed: {e}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("QcControllerTool does not support async mode yet.")