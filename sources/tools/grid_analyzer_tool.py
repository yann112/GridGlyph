# arc_solver/tools/grid_analyzer_tool.py

from langchain.tools import BaseTool
from typing import List, Optional, Type
from pydantic import BaseModel, Field, PrivateAttr
import numpy as np
import logging

from agents.analyze_agent import AnalyzeAgent
from core.llm import OpenRouterClient, LLMClient
from core.features_analysis import ProblemAnalyzer

class GridAnalyzerInput(BaseModel):
    input_grid: List[List[int]] = Field(..., description="The input ARC grid.")
    output_grid: List[List[int]] = Field(..., description="The target output ARC grid.")
    prompt_hint: Optional[str] = Field(None, description="Optional extra instructions or feedback.")


class GridAnalyzerTool(BaseTool):
    name: str = "grid_analyzer"
    description: str = "Analyzes the input/output grids and describes potential transformations."
    args_schema: Type[BaseModel] = GridAnalyzerInput
    _agent: AnalyzeAgent = PrivateAttr()

    def __init__(self, llm: LLMClient = None, **kwargs):
        super().__init__(**kwargs)
        llm = llm or OpenRouterClient()
        self._agent = AnalyzeAgent(mode='single', llm=llm, analyzer=ProblemAnalyzer())

    def _run(self, input_grid: List[List[int]], output_grid: List[List[int]], prompt_hint: Optional[str] = None):
        try:
            # Convert to numpy arrays if needed
            input_np = np.copy(input_grid)
            output_np = np.copy(output_grid)

            # Run analysis
            return self._agent.analyze(input_np, output_np, hint=prompt_hint)

        except Exception as e:
            return f"Analysis failed: {e}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("GridAnalyzerTool does not support async mode yet.")
