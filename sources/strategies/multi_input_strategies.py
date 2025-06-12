import logging
from abc import ABC, abstractmethod
import numpy as np
import pytest
from typing import List, Dict, Any


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
    def generalize(self, train_examples: List[Dict[str, Any]], test_input: np.ndarray) -> Dict[str, Any]:
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

    def generalize(self, train_examples: List[dict], train_results: dict) -> dict:
        """
        Generalizes a transformation rule from multiple train examples and their solutions.
        
        Args:
            train_examples: List of {'input': [[...]], 'output': [[...]]}
            train_results: Dict mapping puzzle keys to individual results with programs
            
        Returns:
            dict: Contains generalized solution, confidence, success flag, etc.
        """
        # Step 1: Collect all individual solutions from train_results
        all_candidates_by_puzzle, all_unique_programs = self._collect_all_programs(train_results)

        # Step 2: Extract transformation patterns or programs from each solution
        extracted_patterns = self._extract_patterns(all_unique_programs)

        # Step 2.5: Check if any single solution already works universally
        universal_solution = self._check_universal_candidate(
            all_candidates_by_puzzle,
            train_examples
        )
        if universal_solution:
            return self._create_success_result(universal_solution)

        # Step 3: Find commonalities between the extracted patterns
        common_pattern = self._find_common_pattern(train_examples, extracted_patterns)

        # Step 4: Build a generalized transformation rule or program
        generalized_program = self._synthesize_generalized_program(common_pattern)

        # Step 4.5: Validate and iterate until found or timeout

        # Step 5: Generate candidate solutions for test input
        final_solution = self._score_and_select_final_solution(
            generalized_program,
            train_examples
        )

        # Step 6: Return result with metadata
        return final_solution