import itertools
import logging
from typing import List, Optional, Tuple, Type

import numpy as np

from core.dsl_nodes import AbstractTransformationCommand, Identity
from core.dsl_interpreter import DslInterpreter
from core.difference_utils import compute_difference_mask
from core.transformation_factory import TransformationFactory

class SynthesisEngine:
    """Enumerates DSL programs and selects those that match the output."""

    def __init__(self, logger: logging.Logger = None):
        """
        Args:
            logger (logging.Logger, optional): Logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.interpreter = DslInterpreter(self.logger)

    def synthesize_matching_programs(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        operation_names: List[str] = None,
        max_repeat: int = 4,
        top_k: int = 3
    ) -> List[Tuple[AbstractTransformationCommand, float]]:
        """
        Enumerates programs and returns best matching ones, scored by similarity.

        Args:
            input_grid (np.ndarray): The input grid.
            output_grid (np.ndarray): The desired output grid.
            operation_names (List[str], optional): List of operation names to use.
            max_repeat (int, optional): Maximum number of repeats for RepeatGrid operations.
            top_k (int, optional): Number of top candidates to return.

        Returns:
            List of tuples (program, score), sorted by score descending.
        """
        if operation_names is None:
            operation_names = ['repeat_grid']

        candidates_with_scores: List[Tuple[AbstractTransformationCommand, float]] = []

        try:
            difference_mask = compute_difference_mask(input_grid, output_grid, self.logger)
        except ValueError:
            self.logger.warning("Skipping mask-based pruning due to shape mismatch.")
            difference_mask = None

        for operation_name in operation_names:
            if operation_name == 'repeat_grid':
                for vertical_repeats, horizontal_repeats in itertools.product(range(1, max_repeat + 1), repeat=2):
                    program = TransformationFactory.create_operation(
                        'repeat_grid',
                        inner_command=TransformationFactory.create_operation('identity', logger=self.logger),
                        vertical_repeats=vertical_repeats,
                        horizontal_repeats=horizontal_repeats,
                        logger=self.logger
                    )

                    try:
                        candidate_output = self.interpreter.execute_program(program, input_grid)
                    except Exception as e:
                        self.logger.debug("Program execution failed: %s", str(e))
                        continue

                    if candidate_output.shape != output_grid.shape:
                        continue

                    # Compute a simple similarity score:
                    total_cells = output_grid.size
                    matching_cells = np.sum(candidate_output == output_grid)
                    score = matching_cells / total_cells

                    if difference_mask is not None:
                        changed = candidate_output != output_grid
                        if np.any(difference_mask != changed):
                            self.logger.debug("Pruned candidate with incorrect cell modifications.")
                            continue

                    self.logger.debug(f"Candidate with V={vertical_repeats} H={horizontal_repeats} scored {score:.3f}")
                    candidates_with_scores.append((program, score))

        # Sort descending by score
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k candidates
        return candidates_with_scores[:top_k]

    def run_synthesized_programs(
        self,
        programs_with_scores: List[Tuple[AbstractTransformationCommand, float]],
        input_grid: np.ndarray
    ) -> List[Tuple[AbstractTransformationCommand, float, np.ndarray]]:
        """
        Executes the synthesized programs on the input grid and returns the results.

        Args:
            programs_with_scores (List[Tuple[AbstractTransformationCommand, float]]): List of tuples (program, score).
            input_grid (np.ndarray): The input grid to run the programs on.

        Returns:
            List[Tuple[AbstractTransformationCommand, float, np.ndarray]]: List of tuples (program, score, output_grid).
        """
        results = []
        for program, score in programs_with_scores:
            try:
                output_grid = self.interpreter.execute_program(program, input_grid)
                results.append((program, score, output_grid))
            except Exception as e:
                self.logger.error(f"Failed to execute program: {str(e)}")
                results.append((program, score, None))
        return results
