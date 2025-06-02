import itertools
import logging
from typing import List, Optional, Tuple

import numpy as np

from core.dsl_nodes import AbstractTransformationCommand, Identity, RepeatGrid
from core.dsl_interpreter import DslInterpreter
from core.difference_utils import compute_difference_mask


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
        max_repeat: int = 4,
        top_k: int = 3
    ) -> List[Tuple[AbstractTransformationCommand, float]]:
        """
        Enumerates programs and returns best matching ones, scored by similarity.

        Returns:
            List of tuples (program, score), sorted by score descending.
        """
        candidates_with_scores: List[Tuple[AbstractTransformationCommand, float]] = []

        try:
            difference_mask = compute_difference_mask(input_grid, output_grid, self.logger)
        except ValueError:
            self.logger.warning("Skipping mask-based pruning due to shape mismatch.")
            difference_mask = None

        for vertical_repeats, horizontal_repeats in itertools.product(range(1, max_repeat + 1), repeat=2):
            program = RepeatGrid(
                inner_command=Identity(self.logger),
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
            # Here, higher is better (1.0 means perfect match)
            # You can customize this metric as needed
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
