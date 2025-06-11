import logging
import ast
import numpy as np
import json
ast.literal_eval
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
        """
        Recursively builds a transformation command from a dictionary.
        Validates that the operation name is valid before proceeding.
        """
        # Ensure input is a dict with "operation" key
        if not isinstance(cmd_dict, dict):
            raise TypeError(f"Expected dict for command, got {type(cmd_dict)}: {cmd_dict}")
        if "operation" not in cmd_dict:
            raise ValueError(f"Missing 'operation' key in command dict: {cmd_dict}")

        op_name = cmd_dict["operation"]
        valid_ops = TransformationFactory.OPERATION_MAP.keys()

        if not isinstance(op_name, str):
            raise TypeError(f"Operation name must be a string, got {type(op_name)}: {op_name}")
        if op_name not in valid_ops:
            raise ValueError(f"Unknown operation '{op_name}'. Valid operations are: {list(valid_ops)}")

        # Process parameters
        parameters = cmd_dict.get("parameters", {})

        processed_params = {}
        for key, value in parameters.items():
            processed_params[key] = self._process_value(value)

        # Build and return the actual command
        command = TransformationFactory.create_operation(op_name, **processed_params)

        if not isinstance(command, AbstractTransformationCommand):
            raise ValueError(
                f"Expected AbstractTransformationCommand, got {type(command)} from {op_name}"
            )

        return command

    def _process_value(self, value):
        """
        Recursively process a value, turning any nested command dicts into real command objects.
        Also handles lists and safely evaluates strings.
        """
        # Base case: None or primitive types
        if value is None:
            return value

        # Handle lists recursively
        if isinstance(value, list):
            return [self._process_value(item) for item in value]

        # Handle dicts (could be nested commands)
        if isinstance(value, dict):
            if "operation" in value and isinstance(value["operation"], str):
                return self._build_command_from_dict(value)
            else:
                # Regular dict - process its values
                return {k: self._process_value(v) for k, v in value.items()}

        # Handle strings (numbers, booleans, nulls, or potentially stringified commands)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict) and "operation" in parsed:
                        return self._build_command_from_dict(parsed)
                except json.JSONDecodeError:
                    pass  # Not a JSON object, fall through
            
            if stripped.startswith("lambda "):
                lambda_func = self._parse_lambda_string(stripped)
                if lambda_func is not None:
                    return lambda_func

            return self._safe_eval(stripped)

        # Anything else (int, float, bool, etc.)
        return value

    def _safe_eval(self, value: str) -> object:
        """
        Safely convert common string representations into Python primitives.
        Handles:
        - "true"/"True"/"TRUE" → True
        - "false"/"False"/"FALSE" → False
        - "none"/"None"/"NULL" → None
        - Numeric strings → int or float
        - Stringified command dicts → parsed via _build_command_from_dict
        - All others → original string
        """
        value = value.strip()
        lower_val = value.lower()

        if lower_val in ("none", "null"):
            return None
        elif lower_val == "true":
            return True
        elif lower_val == "false":
            return False

        # Try numeric conversion
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass  # Not a number, continue checking

        # Try parsing as a command dict
        if value.startswith("{") and value.endswith("}"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict) and "operation" in parsed:
                    return self._build_command_from_dict(parsed)
            except json.JSONDecodeError:
                pass  # Not valid JSON, treat as string

        # Default: return original string
            # Try parsing as a list/tuple
        if value.startswith("[") and value.endswith("]") or value.startswith("(") and value.endswith(")"):
            try:
                # Use literal_eval to safely parse list/tuple strings
                return ast.literal_eval(value)
            except (SyntaxError, ValueError) as e:
                self.logger.warning(f"Failed to parse list-like string '{value}': {e}")
                # Fall through and return raw string if needed
                return value

        return value

    def _parse_lambda_string(self, s: str):
        """
        Helper to evaluate a string as a lambda function, making 'np' available.
        Returns the callable function or None if not a valid lambda string or fails evaluation.
        """
        # This function is now only called if `s.startswith("lambda ")` is true
        try:
            # Provide 'np' in the global scope for the evaluated lambda
            allowed_builtins = {
                "bool": bool,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "tuple": tuple,
                "dict": dict,
                "set": set
            }

            restricted_globals = {
                "__builtins__": {},
                "np": np,
                **allowed_builtins
            }
            evaluated_func = eval(s, restricted_globals, {})
            if callable(evaluated_func):
                self.logger.debug(f"Successfully evaluated lambda string: {s}")
                return evaluated_func
            else:
                self.logger.error(f"Lambda string '{s}' did not evaluate to a callable.")
                # Raise an error here, as this helper is specifically for parsing a lambda
                raise ValueError("Lambda string did not evaluate to a callable.")
        except Exception as e:
            self.logger.error(f"Failed to evaluate lambda string '{s}': {e}", exc_info=True)
            # Re-raise the error to be caught by the higher-level _parse_string_value or _process_value
            raise ValueError(f"Invalid lambda function string: {e}")
