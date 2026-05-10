# GridGlyph Core Package

The `core` package is the engine of the DSL. It handles the lifecycle of a rule from a raw string to a structured tree, and finally to a computed output grid.

---

## 2. File Responsibilities

### `dsl_nodes.py` (The DNA)
**Primary Responsibility:** Implementation of Transformation Logic.
- **Node Definitions:** Contains the `execute()` methods for every transformation (e.g., `FlipGridVertically`, `RepeatGrid`).
- **Composite Structure:** Implements `get_children_commands()` to allow for recursive tree traversal.

### `dsl_symbolic_executor.py` (The Runner & Context Injector)
**Primary Responsibility:** Life-cycle Management and Execution.
- **Context Injection:** Injects the `initial_puzzle_input` and the `executor_context` (for variable sharing) into the command tree before execution.
- **Tree Traversal:** Recursively initializes all nodes (especially `InputGridReference` nodes) to ensure they have the necessary data to run.
- **Error Handling:** Provides a centralized logging and exception-handling wrapper for the execution flow.

### `transformation_factory.py` (The Dispatcher)
**Primary Responsibility:** Class Mapping & Instantiation.
- **The Central Registry:** Maps string identifiers/sigils (e.g., `flip_h`, `scale_grid`) to their corresponding Python classes in `dsl_nodes.py`.
- **Dynamic Creation:** Provides the `create_operation` entry point to instantiate nodes anonymously.

### `dsl_symbolic_interpreter.py` (The Translator)
**Primary Responsibility:** Lexing, Parsing, and Tree Construction.
- **Regex-Driven Parsing:** Uses a dictionary of patterns to identify DSL commands and extract their parameters via capture groups.
- **Recursive Descent:** Handles nested rules (e.g., `↻(↔(⌂))`) by recursively calling the parsing logic on inner parentheses.
- **Symbolic Utility:**
    - **Roman Numerals:** Converts Roman values (I-XXX) and the null symbol (`∅`) into Python integers.
    - **Balanced Splitting:** Uses a custom `_split_balanced_args` function to correctly handle commas inside nested brackets and parentheses.
- **Literal Support:** Parses string-based grid definitions (e.g., `[[I,II],[?,IX]]`) into NumPy arrays for the `BlockGridBuilder`.

### `ground_truth.py` (The Anchor)
**Primary Responsibility:** Verification.
- **Reference Data:** Stores the "Golden Set" of rules and their expected results to ensure the core logic remains stable.

---

## 3. The Transversal Execution Flow

1. **Static Build (Interpreter):**
   - The Interpreter parses the string `⇅(I,II)`.
   - It asks the **Factory** for the class.
   - It builds the command tree (uninitialized).

2. **Contextualization (Executor):**
   - The **Executor** receives the tree and the specific `input_grid`.
   - It traverses the tree (`_initialize_command_tree`) to pass the grid to any node that needs it.
   - It injects itself (`_inject_executor_context`) into nodes to allow for variable storage/retrieval.

3. **Active Run:**
   - The Executor calls `root_command.execute()`.
   - The logic flows through the nodes, applying NumPy transformations until the final grid is returned.

---

## 4. Design Principles

- **Late Binding:** The logic (Nodes) and the data (Grid) are kept separate until the `DSLExecutor` joins them. This allows the same rule to be tested against multiple grids easily.
- **Traceability:** The Executor provides deep logging, allowing you to debug complex nested rules by watching the tree initialization and context injection in real-time.