# arc_solver/tools/program_synthesizer_tool.py
from typing import List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import numpy as np

class ProgramSynthesizerInput(BaseModel):
    input_grid: List[List[int]] = Field(..., description="Input grid as 2D list")
    output_grid: List[List[int]] = Field(..., description="Target output grid as 2D list") 
    analysis_summary: str = Field(..., description="Analysis from grid analyzer")

class ProgramSynthesizerTool(BaseTool):
    name: str = "program_synthesizer"  # Add type annotation
    description: str = "Generates and validates DSL programs to transform input→output grids"
    args_schema: Type[BaseModel] = ProgramSynthesizerInput  # Add type annotation
    
    def __init__(self, llm=None, synthesizer=None, **kwargs):
        super().__init__(**kwargs)
        from core.llm import OpenRouterClient
        from core.synthesis_engine import SynthesisEngine
        from agents.synthesize_agent import SynthesizeAgent
        
        self.agent = SynthesizeAgent(
            llm=llm or OpenRouterClient(),
            synthesizer=synthesizer or SynthesisEngine()
        )

    def _run(self, input_grid, output_grid, analysis_summary):
        input_np = np.array(input_grid)
        output_np = np.array(output_grid)
        
        valid_programs = self.agent.synthesize(input_np, output_np, analysis_summary)
        if not valid_programs:
            return "Error: No valid programs found"
        
        result_grid = valid_programs[0].execute(input_np)
        return str(result_grid.tolist())

    def _arun(self, *args, **kwargs):
        raise NotImplementedError()