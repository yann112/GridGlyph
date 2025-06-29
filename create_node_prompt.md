You are an AI agent specialized in modifying Python codebases for a Domain-Specific Language (DSL).
Your task is to integrate a new DSL command into the existing project structure.

Here are the details of the new command and the required code changes. Please ensure imports are correctly managed (added if missing, not duplicated if existing).

---
# New DSL Command Details
- Command Name: [FILL_IN_COMMAND_NAME_E.G._"Extract Bounding Box"]
- Symbol: [FILL_IN_SYMBOL_E.G._"⧈"]
- Purpose: [FILL_IN_QUICK_EXPLANATION_OF_COMMAND_PURPOSE_E.G._"Extracts the smallest rectangular subgrid that contains all non-background pixels."]

---
# File Modifications Instructions

## File 1: sources/core/dsl_nodes.py
### Action: Add a new Python class definition for the `[FILL_IN_CLASS_NAME_E.G._"ExtractBoundingBox"]` command.
### Parent Class: `AbstractTransformationCommand`
### Code to add (ensure proper indentation and placement at the module level):
```python
# [FILL_IN_COMPLETE_PYTHON_CLASS_DEFINITION_HERE]
# Example for an atomic command:
# class MyAtomicCommand(AbstractTransformationCommand):
#     synthesis_rules = {
#         "type": "atomic",
#         "requires_inner": False,
#         "parameter_ranges": {}
#     }
#     def __init__(self, logger: Optional[logging.Logger] = None):
#         super().__init__(logger)
#     def execute(self, input_grid: np.ndarray) -> np.ndarray:
#         return input_grid.copy()
#     @classmethod
#     def describe(cls) -> str:
#         return "A placeholder command."

# Example for a combinator command (requires 'interpreter: Any' in __init__):
# class MyCombinatorCommand(AbstractTransformationCommand):
#     synthesis_rules = {
#         "type": "combinator",
#         "requires_inner": True,
#         "parameter_ranges": {}
#     }
#     def __init__(self, inner_command_str: str, interpreter: Any, logger: Optional[logging.Logger] = None):
#         super().__init__(logger)
#         self.inner_command_str = inner_command_str
#         self.interpreter = interpreter
#     def execute(self, input_grid: np.ndarray) -> np.ndarray:
#         inner_result = self.interpreter.parse_and_execute(self.inner_command_str, input_grid)
#         return inner_result
#     @classmethod
#     def describe(cls) -> str:
#         return "A placeholder combinator command."

Required Imports for this file (add if not already present):

    import numpy as np

    import logging

    from typing import Optional, Any (only if Any or Optional are used in your class)

    from abc import ABC, abstractmethod

File 2: sources/core/transformation_factory.py

Action: Add a new entry to the OPERATION_MAP dictionary within the TransformationFactory class.

Key: [FILL_IN_OPERATION_MAP_KEY_E.G._'extract_bounding_box'] (usually lowercase command name)

Value: [FILL_IN_OPERATION_MAP_VALUE_CODE] (This will either be the class name directly, or a lambda for commands with special constructor needs like combinators, e.g., passing interpreter=self).

Code to add (insert this key-value pair within the OPERATION_MAP dictionary, ensuring proper comma separation):

Python

        '[FILL_IN_OPERATION_MAP_KEY]': [FILL_IN_OPERATION_MAP_VALUE_CODE],

Important Notes for this file:

    Ensure from core.dsl_nodes import [FILL_IN_CLASS_NAME] is present at the top of the file.

    Critical Adjustment for Combinator Commands: If the new command is a "combinator" (i.e., its class's synthesis_rules["requires_inner"] is True), the create_command method within TransformationFactory must be updated to pass the TransformationFactory instance itself (self) as an interpreter argument to the command's constructor. For example:
    Python

    # Example modification for create_command (adapt to your existing method)
    def create_command(self, op_name: str, **params) -> AbstractTransformationCommand:
        command_factory_or_class = self.OPERATION_MAP.get(op_name)
        if command_factory_or_class:
            if callable(command_factory_or_class): # This handles lambdas (e.g., for combinators or special initializations)
                return command_factory_or_class(interpreter=self, **params) # Pass self as interpreter
            else: # This handles direct class references (typically for atomic commands)
                return command_factory_or_class(**params)
        raise ValueError(f"Unknown operation: {op_name}")

File 3: sources/core/dsl_symbolic_interpreter.py

Action: Add a new rule to the SYMBOL_RULES dictionary.

Key: [FILL_IN_SYMBOL_RULES_KEY_E.G._'extract_bounding_box'] (usually lowercase command name)

Code to add (insert this key-value pair within the SYMBOL_RULES dictionary, ensuring proper comma separation):

Python

    "[FILL_IN_SYMBOL_RULES_KEY]": {
        "pattern": r"[FILL_IN_REGEX_PATTERN_FOR_SYMBOL_AND_PARAMS_E.G._'^⧈$']",
        "transform_params": [FILL_IN_TRANSFORM_PARAMS_LAMBDA_CODE_OR_NONE_IF_NO_PARAMS_E.G._'lambda m: {}' or 'lambda m: {"factor": roman_to_int(m["factor"])}']
    },

Important Notes for this file:

    Ensure the roman_to_int function (if used for parameter parsing) is accessible and correctly imported/defined in this file.

File 4: tests/test_dsl_symbolic_interpreter.py

Action: Add new one-liner test cases to the TEST_CASES list.

Code to add (append these lines to the TEST_CASES list, ensuring proper comma separation. Provide at least one test case.):

Python

    # [OPTIONAL_COMMENT_FOR_TEST_CASES_E.G._"Test for MyNewCommand - basic case"]
    ("[FILL_IN_DSL_STRING_FOR_TEST_1]", [FILL_IN_NUMPY_ARRAY_INPUT_1], [FILL_IN_NUMPY_ARRAY_EXPECTED_OUTPUT_1]),
    # [ADD_MORE_TEST_CASES_AS_NEEDED_FOLLOWING_THE_SAME_ONE-LINER_FORMAT]
    # Example:
    # ("⧈", np.array([[0, 1, 0], [0, 0, 0]], dtype=int), np.array([[1]], dtype=int)),

Important Notes for this file:

    Ensure import numpy as np is present at the top of the file.

    All test cases must be strictly one-liners, enclosed in parentheses () as a tuple, and ending with a comma ,.

Please generate the required pull request based on this information.