import itertools
import logging
from typing import List, Optional, Tuple

import numpy as np

from dsl_nodes import AbstractTransformationCommand, Identity, RepeatGrid
from dsl_interpreter import DslInterpreter
from difference_utils import compute_difference_mask


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
        max_repeat: int = 4
    ) -> List[AbstractTransformationCommand]:
        """Enumerates programs and returns those that match the output exactly.

        Args:
            input_grid (np.ndarray): Input 2D grid.
            output_grid (np.ndarray): Output 2D grid.
            max_repeat (int): Maximum repeat factor for grid repetition.

        Returns:
            List[AbstractTransformationCommand]: Programs that match output exactly.
        """
        successful_programs: List[AbstractTransformationCommand] = []

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

            if np.array_equal(candidate_output, output_grid):
                self.logger.info("Found exact match with V=%d, H=%d", vertical_repeats, horizontal_repeats)
                successful_programs.append(program)
            elif difference_mask is not None:
                changed = candidate_output != output_grid
                if np.any(difference_mask != changed):
                    self.logger.debug("Pruned candidate with incorrect cell modifications.")
                    continue

        return successful_programs
