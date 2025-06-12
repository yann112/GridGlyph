import logging
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any


class SingleGridStrategy(ABC):
    """
    Abstract base class for strategies that solve single input → output grid transformations.
    
    When implementing a concrete strategy:
        1. Subclass this class
        2. Implement `.synthesize()`, `.describe()`, and `.get_metadata()`
        3. update the factory registry
    
    """
    def __init__(self, analyzer_tool=None, synthesizer_tool=None, logger: logging.Logger = None):
        self.analyzer_tool = analyzer_tool
        self.synthesizer_tool = synthesizer_tool
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def synthesize(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Dict[str, Any]]:
        pass

    @classmethod
    def describe(cls) -> str:
        """Return a description of what this strategy does."""
        raise NotImplementedError()

    @classmethod
    def get_metadata(cls) -> dict:
        """Return metadata including name, type, parameters, and description."""
        raise NotImplementedError()


class GreedySynthesisStrategy(SingleGridStrategy):
    def __init__(
        self,
        analyzer_tool=None,
        synthesizer_tool=None,
        logger: logging.Logger = None,
        max_initial_attempts: int = 10,
        max_refinement_attempts: int = 10,
        max_refinement_iterations: int = 3,
        score_threshold_for_refinement: float = 0.95,
        min_score_improvement: float = 0.05,
        use_feedback_on_retry: bool = True
    ):
        super().__init__(analyzer_tool=analyzer_tool, synthesizer_tool=synthesizer_tool, logger=logger)
        
        # Strategy-specific configuration
        self.max_initial_attempts = max_initial_attempts
        self.max_refinement_attempts = max_refinement_attempts
        self.max_refinement_iterations = max_refinement_iterations
        self.score_threshold_for_refinement = score_threshold_for_refinement
        self.min_score_improvement = min_score_improvement
        self.use_feedback_on_retry = use_feedback_on_retry

    def _test_previous_solutions(self, input_grid, output_grid, previous_solutions):
        self.logger.info(f"Trying {len(previous_solutions)} previous solutions before synthesis")

        for puzlle_number in previous_solutions:
            prev_solution = previous_solutions[puzlle_number].copy()
            if prev_solution[-1].get("score", 0) < 0.95:
                continue
            previous_grid = input_grid.copy()
            new_solution = []
            for step in prev_solution:
                try:
                    new_result = step.copy()
                    new_result['alternatives'] = []
                    program, new_result['score'], _  = self.synthesizer_tool._agent.parse_and_validate(step['program_str'], previous_grid, output_grid)
                    new_result['result_grid'] = program.execute(previous_grid)
                    previous_grid = new_result['result_grid']
                    new_solution.append(new_result)
                except Exception as e:
                    self.logger.warning(f"Failed to reuse previous solution '{step}...': {str(e)}")
                    continue
            return new_solution



    def synthesize(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        previous_solutions: List[Dict[str, Any]]
            ) -> List[Dict[str, Any]]:
        """
        Implements current workflow:
        1. Initial analysis → generate first attempt
        2. If not perfect, find differences
        3. Refine until match or max iterations
        
        Returns list of solution dicts with scores and explanations
        """
        combination_solutions = []

        # step 0: test previous solutions if exists
        if previous_solutions:
            return self._test_previous_solutions(input_grid, output_grid, previous_solutions)
        # Step 1: Initial Analysis
        analysis = self._perform_initial_analysis(input_grid, output_grid)
        self.logger.debug(f"Initial analysis: {analysis}")

        # Step 2: Initial Synthesis
        for _ in range(self.max_initial_attempts):  # Try up to 10 times to get valid initial solution
            initial_solution = self._generate_solution(input_grid, output_grid, analysis)
            if initial_solution["success"]:
                break
        else:
            raise RuntimeError("Failed after max attempts")

        self.logger.debug(f"Initial solution: {initial_solution}")
        combination_solutions.append(initial_solution)

        # Step 3: Iterative Refinement
        for iteration in range(self.max_refinement_iterations):
            if combination_solutions[-1].get('score', 0) >= 1.0:  # Perfect match
                return combination_solutions

            # Find differences between current best output and target
            differences = self._find_differences(output_grid, combination_solutions[-1])
            self.logger.debug(f"differences between current best output and target: {differences}")

            # Generate refined solution
            for _ in range(self.max_refinement_attempts):
                refined_solution = self._generate_solution(combination_solutions[-1]['result_grid'], output_grid, differences)
                if refined_solution["success"]:
                    break
            else:
                raise RuntimeError("Failed after 10 attempts")

            self.logger.debug(f"Iteration solution: {refined_solution}")
            if refined_solution.get('score', 0) > combination_solutions[-1].get('score', 0):
                combination_solutions.append(refined_solution)

        return combination_solutions

    def _perform_initial_analysis(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """Internal version of analysis step."""
        if self.analyzer_tool:
            return self.analyzer_tool._run(input_grid.tolist(), output_grid.tolist())
        return f"Analyzing grid transformation from {input_grid.shape} to {output_grid.shape}"

    def _generate_solution(self, input_grid: np.ndarray, output_grid: np.ndarray, analysis: str) -> Dict[str, Any]:
        """Internal version of solution generation."""
        if self.synthesizer_tool:
            return self.synthesizer_tool._run(input_grid, output_grid, analysis)
        return {"success": False, "error": "Fallback synthesizer not implemented"}

    def _find_differences(self, output_grid: np.ndarray, current_solution: Dict[str, Any]) -> Dict[str, Any]:
        """Internal version of difference detection."""
        if self.analyzer_tool:
            hint = f"""
                These grids are not identical — their similarity score is {current_solution['score']}.
                Differences could be anywhere:
                
                Compare cell by cell.
                Check for broken patterns (sequences, symmetry, repetition).
                Even small mismatches matter.
                """
            return self.analyzer_tool._run(
                input_grid=output_grid,
                output_grid=current_solution["result_grid"],
                prompt_hint=hint
            )
        return {"differences": "unknown", "error_type": "generic"}

    @classmethod
    def describe(cls) -> str:
        return """
            Implements greedy iterative refinement strategy.
            Tries initial solution then improves based on differences.
        """

    @classmethod
    def get_metadata(cls) -> dict:
        return {
            "name": "greedy",
            "type": "synthesis",
            "description": cls.describe(),
            "parameters": {
                "max_initial_attempts": 10,
                "max_refinement_attempts": 10,
                "max_refinement_iterations": 3,
                "score_threshold_for_refinement": 0.95,
                "use_feedback_on_retry": True
            }
        }