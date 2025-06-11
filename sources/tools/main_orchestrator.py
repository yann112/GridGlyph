import logging
from typing import List, Dict, Any
import numpy as np

from strategies.strategy_factory import SingleInputStrategyFactory


class ARCSolverOrchestrator:
    """
    Orchestrates ARC grid transformation synthesis using a named strategy.
    
    Strategy is selected by name and loaded via StrategyFactory.
    """

    def __init__(self, strategy_name: str, analyzer=None, synthesizer=None, logger: logging.Logger = None):
        self.strategy_name = strategy_name
        self.analyzer = analyzer
        self.synthesizer = synthesizer
        self.logger = logger or logging.getLogger(__name__)
        self.strategy = self._load_strategy()

    def _load_strategy(self):
        """Load strategy class from name using factory"""
        try:
            return SingleInputStrategyFactory.create_strategy(
                self.strategy_name,
                analyzer=self.analyzer,
                synthesizer=self.synthesizer,
                logger=self.logger
            )
        except ValueError as e:
            self.logger.error(f"Failed to load strategy '{self.strategy_name}': {str(e)}")
            raise

    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main solving interface.
        
        Args:
            input_grid: Input grid as 2D list
            output_grid: Target output grid as 2D list
            
        Returns:
            Dictionary containing:
                - 'solution': Best program found
                - 'confidence': Score between 0.0 and 1.0
                - 'success': Whether perfect match was found
                - 'alternatives': Other valid programs
        """
        results = {}

        # Step 1: Solve all train examples
        train_examples = data.get("train", [])
        for idx, example in enumerate(train_examples):
            input_grid = np.array(example["input"])
            output_grid = np.array(example["output"])

            self.logger.info(f"Solving train example {idx}")
            single_puzzle_results = self.strategy.synthesize(input_grid, output_grid)

            if not single_puzzle_results:
                self.logger.warning(f"No valid solution found for train example {idx}")
                continue

            results[f"puzzle_{idx}"] = single_puzzle_results
        
        return results
       