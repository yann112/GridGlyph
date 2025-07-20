## DSL Command Integration: A Straightforward Guide

This guide explains how to add new commands to your DSL. It focuses on the essential parts and how they work together to turn your DSL text into executable code.

-----

### 1\. Define Your Command Class (`core/dsl_nodes.py`)

Every command is a Python class inheriting from `AbstractTransformationCommand`.

#### `AbstractTransformationCommand` (Base Class)

  * **Purpose:** The blueprint for all executable DSL operations.
  * **Key Methods You Implement:**
      * `__init__(self, logger, ...)`: Sets up your command. Parameters can be:
          * **Direct values:** `int`, `str`, `np.ndarray` (e.g., `fill_value: int`).
          * **Other commands:** `AbstractTransformationCommand` instances (e.g., `target_grid_command: AbstractTransformationCommand`).
      * `execute(self, input_grid: np.ndarray) -> Union[np.ndarray, int]`: The command's core logic. It processes `input_grid` and returns a `np.ndarray` (for grid ops) or an `int` (for scalar ops).
      * `get_children_commands(self) -> Iterator['AbstractTransformationCommand']`: **Crucial.** Yields *only* `AbstractTransformationCommand` instances that are direct arguments (children) of this command. **Do NOT yield simple `int`s or `str`s.**
      * `describe(cls) -> str`: A short, human-readable description.
      * `synthesis_rules (dict)`: Metadata for program synthesis.

#### Types of Commands (How They Take Arguments)

  * **Simple Value Commands:**

      * **Description:** Take direct Python literals (like integers from Roman numerals, strings). The parser converts these strings to Python types *before* passing them to `__init__`.
      * **Examples:** `FillRegion` (for `fill_value`, coordinates), `CreateSolidColorGrid`.
      * `__init__`: Parameters are direct types (e.g., `fill_value: int`).
      * `execute()`: Uses these values directly.
      * `get_children_commands()`: Yields only other command objects, or nothing if all arguments are direct values.

  * **Nested Command Commands (Combinators):**

      * **Description:** Take other commands as arguments, building a command tree. The parser provides `AbstractTransformationCommand` objects to `__init__`.
      * **Examples:** `Sequence` (⟹), `ConditionalTransform` (¿), `MaskCombinator` (⧎).
      * `__init__`: Parameters are `AbstractTransformationCommand` instances (e.g., `true_command: AbstractTransformationCommand`).
      * `execute()`: Calls `.execute()` on its child command arguments.
      * `get_children_commands()`: **Must yield all nested `AbstractTransformationCommand` arguments.**

-----

### 2\. Understand DSL Syntax

Your DSL uses specific symbols and structure:

  * **Single Symbols:** `Ⳁ`, `↔`, `↕`, `↢`, `⤨`, `⧈`, `⧀`, `⌂`, `∅`, `¬`.
  * **Parameterized Symbols:** `SYMBOL(arg1, arg2, ... argN)`. Examples: `⟹`, `¿`, `◫`, `⊕`, `■`, `⧎`, `→`, `⇄`, `⮝`, `⮞`, `✂`, `⌖`, `⬒`, `◨`, `∧`, `∨`, `ⓑ`, `◎`, `⏚`, `⊡`, `≡`, `≗`, `⍰`, `▦`, `⇒`.
  * **Constants:**
      * **Roman Numerals (I, II, ..., X):** Converted to `int`s (1-10).
      * **∅:** Converted to `int` `0`.
      * **? (Wildcard):** For patterns.
  * **Structure:** Arguments separated by `,`, grouped by `()`. Lists/Grids use `[]`. Arguments can be nested commands.

-----

### 3\. Configure the Parser (`SymbolicRuleParser` in `core/dsl_parser.py`)

This tells the parser how to turn DSL strings into command objects. Add a new entry to the `SYMBOL_RULES` dictionary for each new command.

Each entry needs:

  * **`"pattern"`:** A regex to match your command's DSL string. Use `(?P<name>...)` to capture arguments.

  * **`"op_name"`:** The exact string name of your command class (e.g., `"FillRegion"`).

  * **Argument Processing:** This is how captured strings become Python values or command objects.

    **a. For Simple Values (Direct Conversion in `transform_params`):**

      * **Use when:** Your command class `__init__` expects a direct `int`, `str`, etc.
      * **How:** In `transform_params`, convert the regex-captured string directly.
      * **Example (for `FillRegion`'s `fill_value: int`):**
        ```python
        # In SYMBOL_RULES for "fill_region"
        "pattern": fr"^\s*■\((?P<target_grid_str>.+?),\s*(?P<fill_value>{ROMAN_VALUE_PATTERN}),...)\s*$",
        "transform_params": lambda m: {
            "target_grid_str": m["target_grid_str"], # String for nested command
            "fill_value": roman_to_int(m["fill_value"]), # <--- Directly convert to int
            # ... other int conversions
        },
        "nested_commands": {
            "target_grid_command": "target_grid_str", # Only this argument is a nested command
        },
        # ...
        ```
          * **Note:** `_split_balanced_args` is generally NOT used here, as the regex directly captures each literal argument.

    **b. For Nested Commands (via `nested_commands`):**

      * **Use when:** Your command class `__init__` expects another `AbstractTransformationCommand` object.
      * **How:**
        1.  In `transform_params`, capture the full string of the nested command (using `.+` or `_split_balanced_args` if there are multiple comma-separated nested commands).
        2.  List this captured string in `"nested_commands"`. The parser will then recursively parse this string into a command object.
      * **Example (for `ConditionalTransform`'s `condition_command`):**
        ```python
        # In SYMBOL_RULES for "conditional_transform"
        "pattern": r"^¿\((?P<all_args>.+)\)$",
        "transform_params": lambda m: (
            args := _split_balanced_args(m["all_args"], num_args=None),
            {"condition_cmd_str": args[0], ...}
        )[1],
        "nested_commands": {
            "condition_command": "condition_cmd_str", # <--- Tells parser to parse this string
            # ... other nested commands
        },
        # ...
        ```

  * **`"param_processors"` (Advanced):** Rarely used for new commands; prefer `"nested_commands"` for recursive parsing.

-----

### 4\. Understand the Executor (`DSLExecutor`)

The executor runs your parsed command tree with an input grid.

  * `__init__(self, root_command, initial_puzzle_input, logger)`:
      * Gets the top-level command and the input grid.
      * It recursively sets up commands using `get_children_commands()` to inject `initial_puzzle_input` (for `InputGridReference`) and context.
  * `execute_program() -> np.ndarray`:
      * Simply calls `self.root_command.execute(self.initial_puzzle_input)`. Execution flows from this single call, as commands recursively call `execute()` on their children.

-----

### 5\. New Command Checklist (Step-by-Step)

1.  **Design It:**

      * What does it do?
      * What arguments? Are they `int`s/`str`s or other commands?
      * What type does `execute()` return (`np.ndarray` or `int`)?
      * Pick a unique symbol.

2.  **Create Class (`core/dsl_nodes.py`):**

      * Inherit `AbstractTransformationCommand`.
      * Implement `__init__`: Use correct type hints (`int`, `str`, `AbstractTransformationCommand`).
      * Implement `get_children_commands()`: **Yield *only* `AbstractTransformationCommand` arguments.**
      * Implement `execute()`: Define logic. Call `.execute()` on command arguments.
      * Implement `describe()`.
      * (Optional) `synthesis_rules`.

3.  **Define Symbol Rule (`SYMBOL_RULES` in `core/dsl_parser.py`):**

      * Add entry to `SYMBOL_RULES`.
      * Write `"pattern"` regex (use `(?P<name>...)`).
      * Set `"op_name"` to your class name.
      * Define `"transform_params"`: Convert regex captures to Python types (e.g., `roman_to_int`).
      * Define `"nested_commands"` (if arguments are other commands).

4.  **Update `TransformationFactory.OPERATION_MAP`:**

      * Map your `op_name` string to the new command class (e.g., `'MyNewCommand': MyNewCommand`).

5.  **Test It (`tests/test_dsl_symbolic_executor.py`):**

      * Add tests with DSL string, `initial_input_grid`, `expected_output_grid`/value.
      * **Common Error Solutions:**
          * **`ValueError: Could not parse symbolic token: 'XYZ'`**: `SYMBOL_RULES` pattern for `XYZ` is wrong/missing, or test string has unhandled leading/trailing whitespace.
          * **`AttributeError: 'str' object has no attribute 'execute'`**: You passed a string where a command object was expected. Fix `nested_commands` in `SYMBOL_RULES`.
          * **`TypeError: __init__ got an unexpected keyword argument`**: Mismatch between `transform_params` keys and `__init__` parameters.
          * **Errors related to `InputGridReference`**: Missing commands in `get_children_commands()` of parent commands.