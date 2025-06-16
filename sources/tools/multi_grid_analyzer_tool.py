from typing import List, Dict, Any, Type, Optional
from core.llm import LLMClient
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
import logging

from agents.analyze_agent import AnalyzeAgent
from core.llm import OpenRouterClient, LLMClient
from core.features_analysis import ProblemAnalyzer




class MultiGridAnalyzerInput(BaseModel):
    data: Dict[str, Any] = Field(..., description="Full ARC task dictionary including train and test grids")
    train_results: Optional[Dict[str, Any]] = Field(None, description="Results from solving each puzzle individually")
    prompt_hint: Optional[str] = Field(None, description="Optional extra instructions or feedback.")

class MultiGridAnalyzerTool(BaseTool):
    name: str = "multi_grid_analyzer"
    description: str = "Analyzes multiple grid transformations to find a generalizable pattern."
    args_schema: Type[BaseModel] = MultiGridAnalyzerInput
    _agent: AnalyzeAgent = PrivateAttr()
    class Config:
        extra = 'allow' # Allows passing llm even though not in schema

    def __init__(self, llm: LLMClient = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or OpenRouterClient() 
        self._agent = AnalyzeAgent(mode='multi', llm=llm, analyzer=ProblemAnalyzer())

    def _run(
        self,
        data: Dict[str, Any],  # Contains 'train' and optionally 'test'
        train_results: Optional[Dict[str, Any]],
        prompt_hint: Optional[str] = None,
        analysis_mode: str = "both"  # could be "both", "features_only", "results_only"
    ) -> dict:
        """
        Runs multi-grid analysis on the full task dictionary.
        
        Args:
            data: Full ARC task {'train': [...], 'test': [...]}
            train_results: Results from solving puzzles individually
            prompt_hint: Optional instruction or feedback
            
        Returns:
            dict: Generalized transformation rule + metadata
        """
        return self._agent.analyze(
            data=data,
            train_results=train_results,
            prompt_hint=prompt_hint,
            analysis_mode=analysis_mode
        )

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async mode not supported")
    