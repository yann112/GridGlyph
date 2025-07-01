# symbolic_interpreter.py

import re
import logging
from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
from core.transformation_factory import TransformationFactory
from core.dsl_nodes import AbstractTransformationCommand
from assets.symbols import ROM_VAL_MAP 


def _split_balanced_args(s: str, num_args: int = -1) -> List[str]:
    """
    Splits a string of comma-separated arguments, respecting balanced parentheses.
    If num_args is specified, it will attempt to return exactly that many arguments.
    """
    args = []
    balance = 0
    current_arg = []
    
    for char in s:
        if char == '(' :
            balance += 1
        elif char == ')':
            balance -= 1
        
        if char == ',' and balance == 0:
            args.append("".join(current_arg).strip())
            current_arg = []
        else:
            current_arg.append(char)
            
    args.append("".join(current_arg).strip()) # Add the last argument

    if num_args != -1 and len(args) != num_args:
        # This part handles cases where the top-level regex might be too greedy
        # leading to an incorrect number of arguments initially.
        # It attempts a fallback split on the last argument if needed.
        if len(args) < num_args and len(args) > 0:
            last_arg = args.pop()
            split_last = [x.strip() for x in last_arg.split(',')]
            args.extend(split_last)
            
    if num_args != -1 and len(args) != num_args:
        raise ValueError(f"Expected {num_args} arguments but found {len(args)} after balanced split for string: '{s}'")

    return args

# Roman numerals to int helper
def  roman_to_int(s: str) -> int:
    """Converts a Roman numeral string (up to XXX) to an integer."""
    # A complete map is simpler for this limited range than a full parsing algorithm

    # Convert to upper case for case-insensitivity
    val = ROM_VAL_MAP.get(s.upper())
    if val is None:
        raise ValueError(f"Invalid or out-of-range Roman numeral: {s}. Max supported is XXX.")
    return val


# Define a safe eval environment for lambdas (optional but recommended for security)
# This prevents arbitrary code execution if symbolic strings are from untrusted sources
_SAFE_EVAL_GLOBALS = {"np": np}
_SAFE_EVAL_LOCALS = {}


SYMBOL_RULES = {
    "identity": {"sigil": "Ⳁ"},
    "get_constant": {
        "pattern": r"^↱(?P<value>\d+)$",
        "transform_params": lambda m: {"value": int(m["value"])} # m is match object here
    },
    "map_numbers": {
        "pattern": r"^⇒\((?P<old>[IVX∅]+),(?P<new>[IVX∅]+)\)$", # Changed \d+ to [IVX∅]+
        "transform_params": lambda m: {
            "mapping": {
                roman_to_int(m["old"]): roman_to_int(m["new"]) # Use roman_to_int for both
            }
        }
    },
    "flip_h": {"sigil": "↔"},
    "flip_v": {"sigil": "↕"},
    "reverse_row": {"sigil": "↢"},
    "shift_row_or_column": {
        "pattern": r"^(?P<direction>⮝|⮞)\((?P<idx>[IVX]+),\s*(?P<shift_val>[IVX]+)\)$", # Named group for direction
        "transform_params": lambda m: { # m is the re.Match object here
            "row_index": roman_to_int(m["idx"]) - 1 if m["direction"] == '⮝' else None,
            "col_index": roman_to_int(m["idx"]) - 1 if m["direction"] == '⮞' else None,
            "shift_amount": roman_to_int(m["shift_val"]),
            "wrap": True
        }
    },
    "apply_to_row": {
        "pattern": r"^→\((?P<row_idx>[IVX]+),\s*(?P<inner_command_str>.+)\)$",
        "transform_params": lambda m: {
            "row_index": roman_to_int(m["row_idx"]) - 1,
            "inner_command_str": m["inner_command_str"]
        },
        "nested_commands": {
            "inner_command": "inner_command_str"
        }
    },
    "swap_rows_or_columns": {
        "pattern": r"^⇄\((?P<idx1>[IVX]+),(?P<idx2>[IVX]+)\)$",
        "transform_params": lambda m: {
            "row_swap": (roman_to_int(m["idx1"]) - 1, roman_to_int(m["idx2"]) - 1),
            "col_swap": None,
            "swap_type": "rows"
        }
    },
    "repeat_grid_horizontal": {
        "pattern": r"^◨\((?P<count>[IVX]+)\)$",
        "transform_params": lambda m: {
            "inner_command_str": "Ⳁ", # Injected Identity command string
            "vertical_repeats": 1,
            "horizontal_repeats": roman_to_int(m["count"])
        },
        "nested_commands": {
            "inner_command": "inner_command_str"
        },
        "target_op_name": "repeat_grid" # This rule maps to the 'repeat_grid' factory operation
    },
    "repeat_grid_vertical": {
        "pattern": r"^⬒\((?P<count>[IVX]+)\)$",
        "transform_params": lambda m: {
            "inner_command_str": "Ⳁ", # Injected Identity command string
            "vertical_repeats": roman_to_int(m["count"]),
            "horizontal_repeats": 1
        },
        "nested_commands": {
            "inner_command": "inner_command_str"
        },
        "target_op_name": "repeat_grid" # This rule maps to the 'repeat_grid' factory operation
    },
    "sequence": {
        "pattern": r"^⟹\((?P<commands_list_str>.+)\)$",
        "transform_params": lambda m: {
            "commands_list_str": m["commands_list_str"]
        },
        "nested_commands": {
            "commands": ("commands_list_str", "list")
        }
    },
    "create_solid_color_grid": {
        "pattern": r"^⊕\((?P<rows>[IVX]+),(?P<cols>[IVX]+),(?P<fill_color>[IVX]+|∅)\)$",
        "transform_params": lambda m: {
            "rows": roman_to_int(m["rows"]),
            "cols": roman_to_int(m["cols"]),
            "fill_color": roman_to_int(m["fill_color"])
        }
    },
    "scale_grid": {
        "pattern": r"^⤨\((?P<factor>[IVX]+)\)$",
        "transform_params": lambda m: {
            "scale_factor": roman_to_int(m["factor"])
        }
    },
    "extract_bounding_box": {"sigil": "⧈"},
    "flatten_grid": {"sigil": "⧀"},

    "alternate": {
        "pattern": r"^⇌\((?P<first_command_str>.+?)\s*,\s*(?P<second_command_str>.+)\)$",
        "transform_params": lambda m: {
            "first_command_str": m["first_command_str"],
            "second_command_str": m["second_command_str"]
        },
        "nested_commands": {
            "first": "first_command_str",
            "second": "second_command_str"
        }
    },
    "conditional_transform": {
        "pattern": r"^¿C\((?P<inner_command_str>.+?)\s*,\s*(?P<condition_func_literal>.+)\)$",
        "transform_params": lambda m: {
            "inner_command_str": m["inner_command_str"],
            "condition_func": eval(m["condition_func_literal"], _SAFE_EVAL_GLOBALS, _SAFE_EVAL_LOCALS)
        },
        "nested_commands": {
            "inner_command": "inner_command_str"
        }
    },
    "get_element": {
        "pattern": r"^⊡\((?P<row_idx>[IVX]+),\s*(?P<col_idx>[IVX]+)\)$",
        "transform_params": lambda m: {
            "row_index": roman_to_int(m["row_idx"]) - 1,
            "col_index": roman_to_int(m["col_idx"]) - 1
        }
    },
    "get_constant": {
    "pattern": r"^↱(?P<value>[IVX∅]+)$",
    "transform_params": lambda m: {"value": roman_to_int(m["value"])},
    "returns_literal": True
},
    "compare_equality": {
        "pattern": r"^≡\((?P<all_commands_str>.+)\)$",
        "transform_params": lambda m: {
            "command1_str": _split_balanced_args(m["all_commands_str"], 2)[0],
            "command2_str": _split_balanced_args(m["all_commands_str"], 2)[1]
        },
        "nested_commands": {
            "command1": "command1_str",
            "command2": "command2_str"
        }
    },
    "compare_grid_equality": {
        "pattern": r"^≗\((?P<all_commands_str>.+)\)$",
        "transform_params": lambda m: {
            "command1_str": _split_balanced_args(m["all_commands_str"], 2)[0],
            "command2_str": _split_balanced_args(m["all_commands_str"], 2)[1]
        },
        "nested_commands": {
            "command1": "command1_str",
            "command2": "command2_str"
        }
    },
    "if_else_condition": {
        "pattern": r"^⍰\((?P<all_commands_str>.+)\)$",
        "transform_params": lambda m: {
            "condition_str": _split_balanced_args(m["all_commands_str"], 3)[0],
            "true_branch_str": _split_balanced_args(m["all_commands_str"], 3)[1],
            "false_branch_str": _split_balanced_args(m["all_commands_str"], 3)[2]
        },
        "nested_commands": {
            "condition": "condition_str",
            "true_branch": "true_branch_str",
            "false_branch": "false_branch_str"
        }
    },
    'block_pattern_mask': {
        'pattern': r'^▦\((?P<block_rows>[IVX]+),\s*(?P<block_cols>[IVX]+),\s*"(?P<pattern_str>[I∅;]+)"\)$',
        'transform_params': lambda m: {
            "block_rows": roman_to_int(m["block_rows"]),
            "block_cols": roman_to_int(m["block_cols"]),
            "pattern_matrix": np.array([
                [True if char == 'I' else False for char in block_row_str]
                for block_row_str in m["pattern_str"].split(';')
            ], dtype=bool)
        }
    },
"mask_combinator": {
    "pattern": r"^⧎\((?P<all_commands_str>.+)\)$",
    "transform_params": lambda m: (
        parts := _split_balanced_args(m["all_commands_str"], num_args=3),
        {
            "inner_command_str": parts[0],
            "mask_command_str": parts[1],
            "false_value_command_str": parts[2]
        }
    )[1],
    "nested_commands": {
        "inner_command": "inner_command_str",
        "mask_command": "mask_command_str",
        "false_value_command": "false_value_command_str"
    },
    "target_op_name": "mask_combinator"
}
}

class SymbolicRuleParser:
    def __init__(self, factory: TransformationFactory = None):
        self.factory = factory or TransformationFactory()
        self.logger = logging.getLogger(__name__)
        self._compiled_rules = self._compile_rules(SYMBOL_RULES)

    def _compile_rules(self, raw_rules: dict) -> dict:
        compiled = {}
        for op_name, rule in raw_rules.items():
            if "sigil" in rule:
                pattern = re.compile(f"^{re.escape(rule['sigil'])}$")
                compiled[pattern] = {
                    "operation": op_name,
                    "type": "atomic",
                    "original_sigil": rule['sigil']
                }
            elif "pattern" in rule:
                try:
                    pattern = re.compile(rule["pattern"])
                except Exception as e:
                    self.logger.warning(f"Failed to compile pattern '{rule['pattern']}': {str(e)}")
                    continue
                entry = {
                    "operation": op_name,
                    "type": rule.get("type", "atomic"),
                    "parser": "regex"
                }
                if "transform_params" in rule:
                    entry["transform_params"] = rule["transform_params"]
                if "nested_commands" in rule:
                    entry["nested_commands"] = rule["nested_commands"]
                if "target_op_name" in rule: # Store target_op_name
                    entry["target_op_name"] = rule["target_op_name"]
                compiled[pattern] = entry
            else:
                raise ValueError(f"Rule must define 'sigil' or 'pattern': {op_name}")
        return compiled

    def tokenize_expression(self, expr: Union[str, List[str]]) -> List[str]:
        if isinstance(expr, str):
            return [token.strip() for token in expr.split("+")]
        else:
            return expr

    def parse_rule(self, rule: Union[str, List[str]]) -> AbstractTransformationCommand:
        tokens = self.tokenize_expression(rule)
        commands = []

        for token in tokens:
            cmd = self.parse_token(token)
            if cmd:
                commands.append(cmd)

        if len(commands) == 1:
            return commands[0]
        elif len(commands) > 1:
            return self.factory.create_operation("sequence", commands=commands)
        else:
            return self.factory.create_operation("identity")

    def _split_sequence_commands(self, command_string: str) -> List[str]:
        commands = []
        balance = 0
        current_command_chars = []

        for i, char in enumerate(command_string):
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1

            if char == ',' and balance == 0:
                cmd_part = "".join(current_command_chars).strip()
                if cmd_part:
                    commands.append(cmd_part)
                current_command_chars = []
            else:
                current_command_chars.append(char)

        if current_command_chars:
            final_part = "".join(current_command_chars).strip()
            if final_part:
                commands.append(final_part)

        final_commands = [cmd for cmd in commands if cmd]
        return final_commands

    def parse_token(self, token: str) -> Optional[AbstractTransformationCommand]:
        token = token.strip()
        self.logger.debug(f"Parsing token: '{token}'")

        # Try exact sigil match first for simple atomic commands
        for pattern, data in self._compiled_rules.items():
            if data.get("original_sigil") and "pattern" not in SYMBOL_RULES.get(data["operation"], {}):
                if token == data["original_sigil"]:
                    self.logger.debug(f"Matched sigil '{token}' for operation '{data['operation']}'")
                    return self.factory.create_operation(data["operation"])

        # Try regex patterns for all operations
        for pattern, data in self._compiled_rules.items():
            # Skip sigil-only rules already handled above
            if data.get("original_sigil") and "pattern" not in SYMBOL_RULES.get(data["operation"], {}):
                continue

            match = pattern.match(token)
            if match:
                self.logger.debug(f"Matched pattern for operation '{data['operation']}': '{token}'")

                # IMPORTANT: Pass the full match object to transform_params
                if "transform_params" in data:
                    processed_params = data["transform_params"](match) # Pass 'match' object directly
                    self.logger.debug(f"Transformed params: {processed_params}")
                else:
                    processed_params = match.groupdict() # Use groupdict if no transform_params

                # Generic handling of nested commands based on 'nested_commands' rule
                if "nested_commands" in data:
                    for param_name_in_init, config in data["nested_commands"].items():
                        # Determine if config is just a string (regex group name) or a tuple (group name + type)
                        if isinstance(config, tuple):
                            regex_group_name, parse_type = config
                        else: # Default to "single" if just group name provided
                            regex_group_name = config
                            parse_type = "single"

                        if regex_group_name not in processed_params:
                            raise ValueError(f"Rule for '{data['operation']}' specifies nested command '{param_name_in_init}' linked to regex group '{regex_group_name}', but it was not found in parsed parameters from '{token}'. Raw params: {raw_params}")

                        command_string_to_parse = processed_params.pop(regex_group_name) # Remove the raw string

                        if parse_type == "single":
                            parsed_cmd_obj = self.parse_token(command_string_to_parse)
                            if not parsed_cmd_obj:
                                raise ValueError(f"Failed to parse inner command '{command_string_to_parse}' for '{param_name_in_init}' in '{data['operation']}'")
                            processed_params[param_name_in_init] = parsed_cmd_obj # Assign the parsed object
                        elif parse_type == "list":
                            raw_cmd_strings = self._split_sequence_commands(command_string_to_parse)
                            parsed_commands_list = []
                            for cmd_str in raw_cmd_strings:
                                cmd_obj = self.parse_token(cmd_str)
                                if not cmd_obj:
                                    raise ValueError(f"Failed to parse sequence command '{cmd_str}' for '{param_name_in_init}' in '{data['operation']}'")
                                parsed_commands_list.append(cmd_obj)
                            processed_params[param_name_in_init] = parsed_commands_list # Assign the list of objects
                        else:
                            raise ValueError(f"Unknown nested command parse_type: {parse_type} for '{param_name_in_init}' in '{data['operation']}'")

                # Determine the actual operation name to pass to the factory
                final_op_name = data.get("target_op_name", data["operation"])

                self.logger.debug(f"Final params for '{final_op_name}': {processed_params}")
                return self.factory.create_operation(final_op_name, **processed_params)

        raise ValueError(f"Could not parse symbolic token: '{token}'")