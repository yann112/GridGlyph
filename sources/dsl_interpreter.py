import logging
import numpy as np
from dsl_nodes import AbstractTransformationCommand


class DslInterpreter:
    """Executes DSL programs represented as AST nodes."""

    def __init__(self, logger: logging.Logger = None):
        """
        Args:
            logger (logging.Logger, optional): Logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)

    def execute_program(
        self,
        transformation_command: AbstractTransformationCommand,
        input_grid: np.ndarray
    ) -> np.ndarray:
        """Executes a DSL command on the input grid.

        Args:
            transformation_command (AbstractTransformationCommand): DSL command to execute.
            input_grid (np.ndarray): The input 2D grid.

        Returns:
            np.ndarray: Transformed grid.
        """
        self.logger.debug("Executing program using DSL interpreter.")
        return transformation_command.execute(input_grid)
