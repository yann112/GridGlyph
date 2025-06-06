from typing import List, Dict, Any, Type
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from core.synthesis_engine import SynthesisEngine
from core.llm import OpenRouterClient, LLMClient
from agents.synthesize_agent import SynthesizeAgent


class ProgramSynthesizerInput(BaseModel):
    """Input schema for the program synthesizer tool."""
    input_grid: List[List[int]] = Field(..., description="2D input grid as list of lists")
    output_grid: List[List[int]] = Field(..., description="2D desired output grid as list of lists")
    analysis_summary: str = Field(..., description="Text analysis of the transformation pattern")

class ProgramSynthesizerTool(BaseTool):
    name: str = "program_synthesizer"
    description: str = (
        "Generates and validates DSL programs to transform input grids to output grids. "
        "Returns both the transformed grid and the program that produced it."
    )
    args_schema: Type[BaseModel] = ProgramSynthesizerInput
    return_direct: bool = True  # Return results directly without agent post-processing
    _agent: SynthesizeAgent = PrivateAttr()

    def __init__(self, llm: LLMClient = None, synthesizer: SynthesisEngine = None, **kwargs):
        super().__init__(**kwargs)
        llm = llm or OpenRouterClient()
        synthesizer = synthesizer or SynthesisEngine()
        self._agent = SynthesizeAgent(llm=llm, synthesizer=synthesizer)

    def _run(self, input_grid: List[List[int]], output_grid: List[List[int]], analysis_summary: str) -> Dict[str, Any]:
        """
        Executes the program synthesis and returns structured results.
        
        Returns:
            Dictionary containing:
            - success: bool indicating if synthesis succeeded
            - result_grid: transformed grid (if successful)
            - program: string representation of the program
            - score: similarity score (0-1)
            - alternatives: list of alternative programs
            - error: error message (if failed)
        """
        try:
            # Convert to numpy arrays
            input_np = np.array(input_grid)
            output_np = np.array(output_grid)

            # Synthesize programs
            valid_programs = self._agent.synthesize(input_np, output_np, analysis_summary)
            if not valid_programs:
                return {
                    "success": False,
                    "error": "No valid transformation programs found"
                }

            # Get top program and execute it
            top_program, top_score = valid_programs[0]
            result_grid = top_program.execute(input_np)

            # Prepare alternatives (skip the top one we're already returning)
            alternatives = [
                {
                    "program": str(program),
                    "score": float(score)
                }
                for program, score in valid_programs[1:]  # Skip first element
            ]

            return {
                "success": True,
                "result_grid": result_grid.tolist(),
                "program": str(top_program),
                "score": float(top_score),
                "alternatives": alternatives
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Program synthesis failed: {str(e)}"
            }

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("ProgramSynthesizerTool does not support async operations")