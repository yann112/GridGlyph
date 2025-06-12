import logging
from typing import List, Dict, Any
import numpy as np

from strategies.strategy_factory import SingleInputStrategyFactory, MultiInputStrategyFactory


class ARCSolverOrchestrator:
    """
    Orchestrates ARC grid transformation synthesis using a named strategy.
    
    Strategy is selected by name and loaded via StrategyFactory.
    """

    def __init__(
        self,
        single_strategy_name: str,
        multi_strategy_name:str,
        analyzer_tool=None,
        synthesizer_tool=None,
        logger: logging.Logger = None
            ):
        self.single_strategy_name = single_strategy_name
        self.multi_strategy_name = multi_strategy_name
        self.analyzer_tool = analyzer_tool
        self.synthesizer_tool = synthesizer_tool
        self.logger = logger or logging.getLogger(__name__)
        self.single_strategy = self._load_single_grid_strategy()
        self.single_strategy = self._load_single_grid_strategy()

    def _load_single_grid_strategy(self):
        return SingleInputStrategyFactory.create_strategy(
            self.single_strategy_name,
            analyzer_tool=self.analyzer_tool,
            synthesizer_tool=self.synthesizer_tool,
            logger=self.logger
        )

    def _load_multi_input_strategy(self):
        return MultiInputStrategyFactory.create_strategy(
            self.multi_strategy_name,
            analyzer_tool=self.analyzer_tool,
            synthesizer_tool=self.synthesizer_tool,
            logger=self.logger
        )

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
        train_results = {}
        test_result = {}

        # Step 1: Solve all train examples individually
        train_examples = data.get("train", [])
        for idx, example in enumerate(train_examples):
            input_grid = np.array(example["input"])
            output_grid = np.array(example["output"])

            self.logger.info(f"Solving train example {idx}")
            single_puzzle_results = self.single_strategy.synthesize(input_grid, output_grid, previous_solutions=results)

            if not single_puzzle_results:
                self.logger.warning(f"No valid solution found for train example {idx}")
                continue

            train_results[f"puzzle_{idx}"] = single_puzzle_results

        results['train_results'] = train_results
        
        # step 2 find a generics solution
        generic_solution = self._find_generics_solution(data, train_results)
        # step 3 provide solution for tests
        return results

    def _find_generics_solution(self, data, train_results):
        multi_strategy = self._load_multi_input_strategy()
        
        # Pass full train set + their individual results to generalize
        generic_solution = multi_strategy.generalize(
            train_examples=data["train"],
            train_results=train_results
        )
        
        return generic_solution
        
        
         
        
       