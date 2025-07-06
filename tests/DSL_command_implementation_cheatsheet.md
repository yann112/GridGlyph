DSL Command Implementation Cheatsheet

This guide covers the core architecture for adding new commands to your DSL, focusing on the interplay between the DSL syntax, the SymbolicRuleParser, AbstractTransformationCommand nodes, and the DSLExecutor.

1. Command Definition & Architecture (core/dsl_nodes.py)

All commands in your DSL should adhere to a common structure based on the AbstractTransformationCommand base class.

    AbstractTransformationCommand (Base Class):

        Purpose: Defines the interface for all executable commands in your DSL tree.

        Key Methods:

            __init__(self, logger): Standard constructor.

            execute(self, input_grid: np.ndarray) -> np.ndarray (or Any for constants/scalars): The core logic of your command. This is where the transformation happens. Crucially, grid-transforming commands must return a np.ndarray.

            get_children_commands(self) -> Iterator['AbstractTransformationCommand'] (or List): Absolutely vital for the DSLExecutor's traversal. This method must yield/return every AbstractTransformationCommand instance that is a direct child (argument) of the current command. If a command holds other commands as arguments (e.g., ConditionalTransform holding true_command, condition_command), they must be returned here.

            describe(cls) -> str: Provides a human-readable description (useful for debugging and documentation).

            synthesis_rules: A dictionary for meta-information, useful if you're building a program synthesis component (e.g., {"type": "combinator", "requires_inner": True}).

    Types of Commands:

        Atomic/Leaf Commands: Commands that don't take other commands as arguments (e.g., Identity (Ⳁ), FlipGridLeftRight (↔), InputGridReference (⌂), IntLiteral (I, II, etc.)).

            execute() will directly perform an operation or return a value.

            get_children_commands() will return an empty list/iterator.

        Combinator Commands: Commands that take one or more other commands as arguments (e.g., SequentialComposition (⟹), ConditionalTransform (¿C), MatchPattern (◫), HorizontalConcatenation (◨)).

            __init__ will receive AbstractTransformationCommand instances as parameters.

            execute() will typically call execute() on its child commands to get their results, then combine or use those results.

            get_children_commands() must yield/return all these nested AbstractTransformationCommand arguments. This is where issues like the InputGridReference problem often arise if missed.

2. DSL Syntax & Constants (core/dsl_nodes.py & SymbolicRuleParser context)

Your DSL uses specific symbols and a structured format.

    Single-character symbols: Ⳁ, ↔, ↕, ↢, ⤨, ⧈, ⧀, ⌂, ∅.

    Combinator symbols: ⟹, ¿C, ◫, ⊕, ⧎, →, ⇄, ⮝, ⮞. These typically follow the pattern SYMBOL(arg1, arg2, ..., argN).

    Constants/Literals:

        I, II, III, IV, V, VI, VII, VIII, IX, X: Map to IntLiteral (1 to 10).

        ∅: Can map to IntLiteral(0) for colors/sizes, or EmptyGridLiteral.

        V, X: Often used for specific colors (e.g., 5 for orange, 10 for blue).

    Argument Handling:

        Arguments are separated by commas ,.

        Arguments can be other commands, including nested combinators.

        Parentheses () are used for grouping arguments of a combinator.

        Square brackets [] are used for lists, e.g., for MatchPattern's cases parameter ([(pattern_cmd, action_cmd), (..., ...)]).

3. The Parser (SymbolicRuleParser)

The parser is responsible for converting a DSL string into a tree of AbstractTransformationCommand objects.

    SYMBOL_RULES (core/dsl_parser.py):

        This dictionary defines how the parser recognizes and constructs commands.

        Adding a new command means adding an entry here.

        Key components for each entry:

            "pattern": A regular expression that matches the DSL string for your command. Use named capture groups (?P<arg_name>...) to extract arguments.

            "op_name": The string name of the AbstractTransformationCommand class (e.g., "Identity", "FlipGridLeftRight", "ConditionalTransform"). This is used by the TransformationFactory.

            "param_processors": This is crucial for handling arguments. It's a dictionary mapping your regex capture group names (e.g., condition_command_str, action_command_str) to functions that process those string arguments into the correct Python types (typically AbstractTransformationCommand instances or literal values).

                For nested commands: Use lambda s, factory: factory.create_command_from_rule_string(s) to recursively parse sub-rules.

                For splitting arguments: _split_balanced_args(arg_string) is essential for parsing arguments within parentheses, especially for combinators with multiple arguments.

                For lists of tuples (e.g., MatchPattern.cases): This is complex. You need custom logic to split the list string, then for each tuple string, split that into two parts and use create_command_from_rule_string for each part. Example:
                Python

                "cases_str": lambda s, factory: [
                    (factory.create_command_from_rule_string(arg1_str),
                     factory.create_command_from_rule_string(arg2_str))
                    for arg1_str, arg2_str in your_split_function(s) # your_split_function splits "[(a,b),(c,d)]" into pairs
                ]

                For literals/constants (I, II, V, X, ∅): You'll need _parse_literal_argument(s, factory) or similar helper functions that map these symbols to IntLiteral or ColorLiteral objects.

4. The Executor (DSLExecutor)

The executor is the runtime environment that takes your parsed command tree and executes it with an input grid.

    __init__(self, root_command, initial_puzzle_input, logger):

        Receives the top-level command of the parsed DSL program.

        Receives the actual input grid for the current task.

        Crucially calls:

            _initialize_command_tree(self.root_command): This method recursively traverses the command tree using command.get_children_commands() and calls set_initial_puzzle_input() on any InputGridReference (⌂) instances it finds. If InputGridReference errors, check this traversal!

            _inject_executor_context(self.root_command): Similarly, traverses the tree to inject a reference to the DSLExecutor itself into commands that might need it (e.g., for variable access, though you haven't implemented that yet).

    execute_program() -> np.ndarray:

        Simply calls self.root_command.execute(self.initial_puzzle_input). The entire program execution flows from this single call, as commands recursively call execute() on their children.

5. Implementing a New Command: Checklist

When adding a new command, follow these steps:

    Design the Command:

        What is its purpose?

        What arguments does it take? Are they grids, integers, colors, or other commands?

        What should its execute() method return (usually np.ndarray)?

    Create the Command Class (core/dsl_nodes.py):

        Inherit from AbstractTransformationCommand.

        Implement __init__: Define parameters for its arguments (e.g., command1: AbstractTransformationCommand, value: int).

        Implement get_children_commands(): Yield/return all AbstractTransformationCommand instances that are arguments to this command. If you miss one, the executor won't initialize it (e.g., InputGridReference won't get its puzzle input).

        Implement execute(): Define the core logic. Ensure it correctly calls execute() on its command arguments to get their results. Ensure it returns the correct type (e.g., np.ndarray).

        Implement describe(): A simple string.

        (Optional) Define synthesis_rules.

    Define the Symbol Rule (SYMBOL_RULES in core/dsl_parser.py):

        Choose a unique symbol (e.g., ↦).

        Write a regular expression ("pattern") to match the DSL syntax. Use named capture groups for arguments.

        Set "op_name" to the exact class name of your new command (e.g., "MapGridColor").

        Define "param_processors": Map regex capture group names to lambda functions or helper functions that convert the captured strings into the correct Python objects (commands, integers, etc.). Use factory.create_command_from_rule_string for nested commands.

    Testing (tests/test_dsl_symbolic_executor.py):

        Add new test cases in the test_dsl_executor_execution function.

        Each test should have:

            The DSL string for your new command.

            An initial_input_grid (even if the command doesn't use it directly, the executor expects it).

            The expected_output_grid for that specific input.

        Run your tests! Pay close attention to any AttributeError, TypeError, or InputGridReference errors, as these usually point to issues in execute() or get_children_commands().