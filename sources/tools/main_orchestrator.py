import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np


class ARCProblemInput(BaseModel):
    """Input data model for ARC problem."""
    input_grid: List[List[int]] = Field(..., description="The input ARC grid as 2D list of integers")
    output_grid: List[List[int]] = Field(..., description="The target output ARC grid as 2D list of integers")
    max_iterations: Optional[int] = Field(3, description="Maximum refinement iterations to perform")


class ARCProblemOrchestrator:
    """Simplified orchestrator implementing greedy strategy workflow for ARC problems.
    
    Follows the workflow:
    1. First Analysis: Ask analyzer to study input → output grids
    2. Initial Synthesis: Ask synthesizer to generate programs + score them
    3. Iteration Loop: Compare differences and refine until perfect match or max iterations
    """
    
    def __init__(self, analyzer=None, synthesizer=None, logger: logging.Logger = None):
        """Initialize the orchestrator with required components.
        
        Args:
            analyzer: Tool for analyzing grid patterns and differences
            synthesizer: Tool for generating and scoring programs
            logger: Logger instance for tracking progress
        """
        self.logger = logger or logging.getLogger(__name__)
        self.analyzer = analyzer
        self.synthesizer = synthesizer
        
    def solve(self, input_grid: List[List[int]], 
              output_grid: List[List[int]], 
              max_iterations: int = 3) -> Dict[str, Any]:
        """Main solving interface implementing greedy strategy workflow.
        
        Args:
            input_grid: 2D list representing input grid
            output_grid: 2D list representing target output grid
            max_iterations: Maximum number of refinement iterations
            
        Returns:
            Dictionary containing:
                - solution: Best program found
                - confidence: Confidence score (0.0-1.0)
                - iterations: Number of iterations performed
                - success: Whether perfect match was found
        """
        # Convert to numpy arrays for easier manipulation
        input_np = np.array(input_grid)
        output_np = np.array(output_grid)
        combination_solutions = []

        # Step 1: Initial Analysis TODO add a try except block in case http error
        analysis = self._perform_initial_analysis(input_np, output_np)
        self.logger.debug(f"Initial analysis: {analysis}")
        
        # Step 2: Initial Synthesis
        for _ in range(10):
            initial_solution = self._generate_solution(input_np, output_np, analysis)
            if initial_solution["success"]:
                break
            else:  # Only runs if loop completes without breaking
                raise RuntimeError("Failed after 10 attempts")

        self.logger.debug(f"Initial solution: {initial_solution}")
        combination_solutions.append(initial_solution)
        # Step 3: Iterative Refinement
        for iteration in range(max_iterations):
            if combination_solutions[-1].get('score',0) >= 1.0:  # Perfect match
                return combination_solutions
                
            # Find differences between current best output and target
            differences = self._find_differences(output_np, combination_solutions[-1])
            self.logger.debug(f"differences between current best output and target: {differences}")
                
            # Generate refined solution
            for _ in range(10):
                refined_solution = self._generate_solution(combination_solutions[-1]['result_grid'], output_np, differences)
                if refined_solution["success"]:
                    break
            else:  # Only runs if loop completes without breaking
                raise RuntimeError("Failed after 10 attempts")

            self.logger.debug(f"Iteration solution: {refined_solution}")
            if refined_solution.get('score', 0) > combination_solutions[-1].get('score',0):
                combination_solutions.append(refined_solution)

        return combination_solutions
    
    def _perform_initial_analysis(self, input_grid: np.ndarray, 
                                output_grid: np.ndarray) -> str:
        """Perform initial pattern analysis of input/output grids.
        
        Args:
            input_grid: Numpy array of input grid
            output_grid: Numpy array of target output grid
            
        Returns:
            Analysis text describing observed patterns
        """
        if self.analyzer:
            return self.analyzer._run(input_grid.tolist(), output_grid.tolist())
        
        # Fallback basic analysis
        return f"Analyzing grid transformation from {input_grid.shape} to {output_grid.shape}"

    def _generate_solution(self, input_grid: np.ndarray,
                                 output_grid: np.ndarray,
                                 analysis: str) -> tuple[str, float]:
        """Generate solution program.
        
        Args:
            input_grid: Numpy array of input grid
            output_grid: Numpy array of target output grid
            analysis: Analysis text from initial analysis
            
        Returns:
            Tuple of (program, confidence_score)
        """
        if self.synthesizer:
            return self.synthesizer._run(input_grid, output_grid, analysis)
        
        # Fallback dummy solution
        return []
    
    def _find_differences(self,
                         output_grid: np.ndarray,
                         current_solution: str) -> Dict[str, Any]:
        """Identify differences between current solution output and target.
        
        Args:
            input_grid: Original input grid
            output_grid: Target output grid
            current_solution: Current best solution program
            
        Returns:
            Dictionary describing differences found
        """
        if self.analyzer:
            hint = f"""
                These grids are not identical (this is a certitude)—their similarity score is {current_solution['score']} (1.0 = perfect match). Differences could be anywhere:

                Compare cell by cell (very small mismatches matter).

                Check for broken patterns (sequences, symmetry, repetitions).

                Even one altered value can change the score.
                """
            return self.analyzer._run(
                input_grid = output_grid,
                output_grid = current_solution["result_grid"],
                prompt_hint = hint
            )
            
        # Fallback dummy differences
        return {"differences": "unknown", "error_type": "generic"}
    
    def _refine_solution(self, input_grid: np.ndarray,
                        output_grid: np.ndarray,
                        current_solution: str,
                        differences: Dict[str, Any],
                        iteration: int) -> tuple[str, float]:
        """Refine solution based on identified differences.
        
        Args:
            input_grid: Original input grid
            output_grid: Target output grid
            current_solution: Current best solution program
            differences: Identified differences from comparison
            iteration: Current iteration number
            
        Returns:
            Tuple of (refined_program, new_confidence_score)
        """
        if self.synthesizer:
            return self.synthesizer.refine(
                input_grid.tolist(),
                output_grid.tolist(),
                current_solution,
                differences,
                iteration
            )
            
        # Fallback - slightly improve confidence
        return (current_solution, 0.5 + min(0.5, iteration * 0.1))
