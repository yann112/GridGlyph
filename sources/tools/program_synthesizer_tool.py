# arc_solver/tools/program_synthesizer_tool.py

from typing import List, Optional, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
import numpy as np

from core.llm import OpenRouterClient, LLMClient
from core.synthesis_engine import SynthesisEngine
from agents.synthesize_agent import SynthesizeAgent

class ProgramSynthesizerInput(BaseModel):
    input_grid: List[List[int]] = Field(..., description="Input grid as 2D list")
    output_grid: List[List[int]] = Field(..., description="Target output grid as 2D list")
    analysis_summary: str = Field(..., description="Analysis from grid analyzer")

class ProgramSynthesizerTool(BaseTool):
    name: str = "program_synthesizer"
    description: str = "Generates and validates DSL programs to transform input→output grids"
    args_schema: Type[BaseModel] = ProgramSynthesizerInput
    _agent: SynthesizeAgent = PrivateAttr()

    def __init__(self, llm: LLMClient = None, synthesizer: SynthesisEngine = None, **kwargs):
        super().__init__(**kwargs)
        llm = llm or OpenRouterClient()
        synthesizer = synthesizer or SynthesisEngine()
        self._agent = SynthesizeAgent(llm=llm, synthesizer=synthesizer)

    def _run(self, input_grid: List[List[int]], output_grid: List[List[int]], analysis_summary: str) -> str:
        try:
            input_np = np.array(input_grid)
            output_np = np.array(output_grid)

            valid_programs = self._agent.synthesize(input_np, output_np, analysis_summary)
            if not valid_programs:
                return "Error: No valid programs found"

            result_grid = valid_programs[0].execute(input_np)
            return str(result_grid.tolist())

        except Exception as e:
            return f"Program synthesis failed: {e}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("ProgramSynthesizerTool does not support async mode yet.")
