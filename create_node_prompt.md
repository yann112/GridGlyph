You are an AI agent specialized in modifying Python codebases for a Domain-Specific Language (DSL).
Your task is to integrate *multiple* new DSL commands into the existing project structure. For each command, you will be given only its symbol and a quick explanation.

For each command in the provided list, you will perform the following steps:

1.  **Infer Full Command Details:** Based *solely* on the provided symbol and quick explanation, infer all the necessary details for implementing the command:
    * **Full Command Name:** A human-readable name for the command (e.g., "Rotate Grid").
    * **Class Name:** A valid Python class name for the `AbstractTransformationCommand` subclass (e.g., "RotateGrid90Clockwise").
    * **Complete Class Code:** The full Python code for the `AbstractTransformationCommand` subclass implementation. This must include `synthesis_rules` (correctly setting `"type"` to "atomic" or "combinator" and `"requires_inner"` to `True`/`False`), the `__init__` method (including `interpreter: Any` if it's a combinator), the `execute` method with robust NumPy logic, and the `describe` class method.
    * **Operation Map Key:** The lowercase, snake_case string key for the `OPERATION_MAP` dictionary (e.g., "rotate_90_clockwise").
    * **Operation Map Value Code:** The Python code for the value in `OPERATION_MAP`. This will typically be the `ClassName` for atomic commands, or a `lambda` expression for combinators that need to inject the `interpreter` or handle specific parameter mappings at instantiation.
    * **Symbol Rules Key:** The lowercase, snake_case string key for the `SYMBOL_RULES` dictionary (usually matches the Operation Map Key).
    * **Regex Pattern:** The regular expression string to match the DSL symbol and extract any parameters. Use Roman numerals (`[IVX]+`) for integer parameters and `∅` for zero if applicable, consistently following existing DSL patterns.
    * **Transform Params Lambda Code:** The Python `lambda` expression that processes regex matches into a dictionary of command parameters, using `roman_to_int` if Roman numerals are involved. If no parameters, use `lambda m: {}`.
    * **Test Cases:** A list of at least **3** representative one-liner test cases. Each test case must be a Python tuple string: `("DSL_string", "np.array_input_as_string", "np.array_expected_output_as_string")`. Ensure `np.array(...)` is represented as a string that can be evaluated to a numpy array. Include diverse scenarios (e.g., different sizes, edge cases, zero/non-zero values).

2.  **Apply Changes to Files:** Integrate all inferred details into the following files, ensuring correct placement, indentation, and import management (add if missing, do not duplicate any imports):
    * `sources/core/dsl_nodes.py`: Add the inferred class definition.
    * `sources/core/transformation_factory.py`: Add the inferred `OPERATION_MAP` entry. (Remember the `create_command` method in `TransformationFactory` *must* be correctly updated to pass `interpreter=self` for combinator commands).
    * `sources/core/dsl_symbolic_interpreter.py`: Add the inferred `SYMBOL_RULES` entry.
    * `tests/test_dsl_symbolic_interpreter.py`: Append the inferred one-liner test cases to the `TEST_CASES` list.

---
# List of DSL Commands to Implement (Symbol and Explanation Only)

```json
[
    {
        "symbol": "↻",
        "quick_explanation": "Rotates the entire grid 90 degrees clockwise."
    }

]