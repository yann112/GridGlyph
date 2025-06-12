from typing import List, Dict, Any, Type, Optional
from core.llm import LLMClient
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
import logging

from agents.analyze_agent import AnalyzeAgent
from core.llm import OpenRouterClient, LLMClient
from core.features_analysis import ProblemAnalyzer




class MultiGridAnalyzerInput(BaseModel):
    examples: List[Dict[str, List[List[int]]]] = Field(..., description="The input ARC grid.")
    train_results: Dict[str, Any] = Field(..., description="Result from the single grid strategy")
    prompt_hint: Optional[str] = Field(None, description="Optional extra instructions or feedback.")


class MultiGridAnalyzerTool(BaseTool):
    name: str = "multi_grid_analyzer"
    description: str = "Analyzes multiple grid transformations to find a generalizable pattern."
    args_schema: Type[BaseModel] = MultiGridAnalyzerInput
    _agent: AnalyzeAgent = PrivateAttr()
    
    def __init__(self, llm: LLMClient = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or OpenRouterClient() 
        self._agent = AnalyzeAgent(mode='multi', llm=llm, analyzer=ProblemAnalyzer())

    def _run(
        self,
        examples: List[Dict[str, List[List[int]]]],
        train_results: Dict[str, Any],
        prompt_hint: Optional[str] = None
    ) -> dict:
        """
        Analyzes multiple examples to find a unified transformation rule.
        
        Args:
            examples: List of input/output grid pairs
            train_results: Results from solving each puzzle individually
            
        Returns:
            dict: Contains generalized program, explanation, confidence, etc.
        """
        return self._agent.analyze_multi(examples, train_results, prompt_hint)

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async mode not supported")
    