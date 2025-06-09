import logging
import ast
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
        """Recursively build command from dictionary with improved type handling."""
        operation = cmd_dict["operation"]
        parameters = cmd_dict.get("parameters", {})

        processed_params = {}

        for key, value in parameters.items():
            if isinstance(value, dict) and "operation" in value:
                # Recursively build nested commands
                processed_params[key] = self._build_command_from_dict(value)

            elif key == "mask_func" and isinstance(value, str):
                # Special case: evaluate mask lambda
                try:
                    # Restrict builtins for security
                    namespace = {"np": np}
                    func = eval(value, {"__builtins__": {}}, namespace)
                    if callable(func):
                        processed_params[key] = func
                    else:
                        raise ValueError(f"'{value}' is not a callable function")
                except Exception as e:
                    self.logger.error(f"Failed to parse lambda '{value}': {str(e)}")
                    raise

            else:
                # Try safe evaluation of stringified values
                if isinstance(value, str):
                    try:
                        evaluated = ast.literal_eval(value)
                        if isinstance(evaluated, list):
                            processed_params[key] = tuple(evaluated)  # normalize to tuple
                        else:
                            processed_params[key] = evaluated
                    except (SyntaxError, ValueError):
                        # Not a Python literal, keep original string
                        processed_params[key] = value
                elif isinstance(value, list):
                    # Normalize all lists to tuples
                    processed_params[key] = tuple(value)
                else:
                    processed_params[key] = value

        return TransformationFactory.create_operation(operation, **processed_params)