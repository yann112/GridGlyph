import logging
from abc import ABC, abstractmethod
import numpy as np
import pytest
from typing import List, Dict, Any

from agents.agents_utils import MultiGridFeatureCollector
from core.features_analysis import ProblemAnalyzer
class MultiGridStrategy(ABC):
    """
    Abstract base class for strategies that solve using multiple examples.
    
    When implementing:
        1. Subclass this class
        2. Implement `.generalize(...)`
        3. Provide metadata via `.describe()` and `.get_metadata()`
    """
    
    def __init__(self, analyzer_tool=None, synthesizer_tool=None, logger: logging.Logger = None):
        self.analyzer_tool = analyzer_tool
        self.synthesizer_tool = synthesizer_tool
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def analyze(self, train_examples: List[Dict[str, Any]], test_input: np.ndarray) -> Dict[str, Any]:
        """
        Takes multiple train examples and tries to generalize to test input.
        
        Args:
            train_examples: List of {"input", "output"} dicts
            test_input: Input grid to transform
            
        Returns:
            Dictionary with:
                - program: executable transformation object
                - score: float (0.0–1.0 match with output)
                - explanation: natural language description
                - program_str: string representation of the program
        """
        pass

    @classmethod
    @abstractmethod
    def describe(cls) -> str:
        """Return a description of what this strategy does."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> dict:
        """Return metadata including name, type, parameters, and description."""
        raise NotImplementedError()
    

class GeneralizingStrategy(MultiGridStrategy):
    def describe(self) -> str:
        return "Generalizes a solution across multiple input-output grid examples."

    def get_metadata(self) -> dict:
        return {
            "name": "generalize",
            "description": "Finds a unified program that works across all training examples.",
            "supports_multi_input": True,
            "type": "generalization"
        }

    def analyze(self, train_examples: List[dict], train_results: dict) -> dict:
        """
        Generalizes a transformation rule from multiple train examples and their solutions.
        
        Args:
            train_examples: List of {'input': [[...]], 'output': [[...]]}
            train_results: Dict mapping puzzle keys to individual results with programs
            
        Returns:
            dict: Contains generalized solution, confidence, success flag, etc.
        """

        # Step 1: Analyze if we can extract an universal solution base on the unperfect or partial solutions we found before
        analysis = self.analyzer_tool._run(
            examples = train_examples,
            train_results = train_results,
            prompt_hint = None
        )

        # Step 2: ask the synthesizer to find a generic program and score it on all grid

        # Step 3: Validate and iterate until found or timeout

        # Step 4: Generate candidate solutions for test input

        # Step 6: Return result with metadata
