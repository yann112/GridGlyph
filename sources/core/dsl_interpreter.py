import logging
import numpy as np
import json
from core.transformation_factory import TransformationFactory
from core.dsl_nodes import AbstractTransformationCommand


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

    def parse_program(self, program_str: str) -> AbstractTransformationCommand:
        """Parse a JSON DSL program string into a command object."""
        try:
            # Parse JSON
            program_dict = json.loads(program_str)
            
            # Recursively build command
            return self._build_command_from_dict(program_dict)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in program: {program_str} - {str(e)}")
            raise ValueError(f"Failed to parse program JSON: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to build command from: {program_str} - {str(e)}")
            raise ValueError(f"Failed to build command: {str(e)}")

    def _build_command_from_dict(self, cmd_dict: dict) -> AbstractTransformationCommand:
        """Recursively build command from dictionary."""
        operation = cmd_dict["operation"]
        parameters = cmd_dict.get("parameters", {})
        
        # Handle inner commands recursively
        processed_params = {}
        for key, value in parameters.items():
            if isinstance(value, dict) and "operation" in value:
                # This is a nested command
                processed_params[key] = self._build_command_from_dict(value)
            else:
                processed_params[key] = value
        
        # Create command using factory
        return TransformationFactory.create_operation(operation, **processed_params)