# arc_solver/tools/grid_glyph_analyzer_tool.py

from langchain.tools import BaseTool
from typing import List, Optional, Union, Dict, Type, Any
from pydantic import BaseModel, Field, PrivateAttr



from agents.gridglyphs_analyze_agent import GridGlyphAnalyzerAgent
from core.llm import OpenRouterClient, LLMClient
from assets.symbols import SYMBOL_SETS_JSON


class GridGlyphInput(BaseModel):
    """
    Input schema for the GridGlyphTool.
    
    Accepts either a full ARC-style puzzle dict or a single input/output pair.
    """

    puzzle_data: Union[Dict[str, List[List[List[int]]]], None] = Field(
        None,
        description="Full ARC-style puzzle data with train and/or test inputs."
    )

    input_grid: Optional[List[List[int]]] = Field(
        None,
        description="Single input grid (use if no puzzle_data)"
    )

    output_grid: Optional[List[List[int]]] = Field(
        None,
        description="Single output grid (use if no puzzle_data)"
    )

    test_grid: Optional[List[List[int]]] = Field(
        None,
        description="Test input grid to apply transformation"
    )

    glyphset_id: Optional[List] = Field(
        None,
        description="Symbol set ID to use for mapping (default: katakana)"
    )

    llm_response_mode: str = Field(
        "function_only",
        description="Mode of LLM response parsing: 'function_only', 'rule_and_function'"
    )
    

class GridGlyphsAnalyzerTool(BaseTool):
    name: str = "grid_glyph"
    description: str = (
        "Converts numeric grids into symbolic glyphs using ancient/runes-inspired symbol sets "
        "and prompts an LLM to return a transformation function or rule description."
    )
    args_schema: Type[BaseModel] = GridGlyphInput

    # Private attributes
    _llm_client: LLMClient = PrivateAttr()
    _agent: GridGlyphAnalyzerAgent = PrivateAttr()

    class Config:
        """This allow extra parameters that not in pydantic model like the llm if injected"""
        extra = 'allow'
        
    def __init__(self, glyphsets=None, llm: LLMClient = None, **kwargs):
        super().__init__(**kwargs)
        self._llm_client = llm or OpenRouterClient()
        self._agent = GridGlyphAnalyzerAgent(llm=llm)

    def _run(
        self,
        puzzle_data: Optional[Dict[str, Any]] = None,
        input_grid: Optional[List[List[int]]] = None,
        output_grid: Optional[List[List[int]]] = None,
        test_grid: Optional[List[List[int]]] = None,
        glyphset_ids: List[str] = None,
        n_variants_per_set: int = 20,
        mode: str = "python_function"
        ):
        self._agent.analyze(
        puzzle_data,
        input_grid,
        output_grid,
        test_grid,
        glyphset_ids,
        n_variants_per_set,
        mode
            )

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("GridGlyphTool does not support async mode yet.")
    
    



        