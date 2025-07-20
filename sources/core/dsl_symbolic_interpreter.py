
import re
import logging
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import numpy as np
from core.transformation_factory import TransformationFactory
from core.dsl_nodes import AbstractTransformationCommand
from assets.symbols import ROM_VAL_MAP 

_WILDCARD_VALUE = -1
_SAFE_EVAL_GLOBALS = {"np": np}
_SAFE_EVAL_LOCALS = {}
ROMAN_INDEX_PATTERN = r"-?(?:I|II|III|IV|V|VI|VII|VIII|IX|X)"
ROMAN_VALUE_PATTERN = r"-?(?:I|II|III|IV|V|VI|VII|VIII|IX|X|∅)"


def _process_match_pattern_full_params(
    raw_params: Dict[str, Any],
    parser: 'SymbolicRuleParser'
) -> Dict[str, Any]:
    full_params_str = raw_params.pop('full_params_str', "").strip()
    main_params_strs = _split_balanced_args(full_params_str, num_args=3)

    if len(main_params_strs) != 3:
        raise ValueError(f"Malformed ◫ command: Expected 3 main arguments (grid, cases, default_action) but found {len(main_params_strs)} in '{full_params_str}'")

    grid_to_evaluate_str = main_params_strs[0].strip()
    cases_list_str = main_params_strs[1].strip()
    default_action_str = main_params_strs[2].strip()

    grid_to_evaluate_cmd = parser.parse_token(grid_to_evaluate_str)

    if not (cases_list_str.startswith('[') and cases_list_str.endswith(']')):
        raise ValueError(f"Malformed ◫ cases list: Expected cases to be enclosed in square brackets '[]' but found '{cases_list_str}'")
    
    cases_str_for_processor = cases_list_str[1:-1].strip()

    parsed_cases: List[Tuple['AbstractTransformationCommand', 'AbstractTransformationCommand']] = []
    if cases_str_for_processor:
        individual_case_strs = []
        balance_paren = 0
        balance_bracket = 0
        current_case_chars = []

        for char in cases_str_for_processor:
            if char == '(':
                balance_paren += 1
            elif char == ')':
                balance_paren -= 1
            elif char == '[':
                balance_bracket += 1
            elif char == ']':
                balance_bracket -= 1

            if char == ',' and balance_paren == 0 and balance_bracket == 0:
                part = "".join(current_case_chars).strip()
                if part:
                    individual_case_strs.append(part)
                current_case_chars = []
            else:
                current_case_chars.append(char)
        
        final_part = "".join(current_case_chars).strip()
        if final_part:
            individual_case_strs.append(final_part)
        
        for raw_tuple_str in individual_case_strs:
            stripped_tuple_str = raw_tuple_str.strip()
            
            if not (stripped_tuple_str.startswith('(') and stripped_tuple_str.endswith(')')):
                raise ValueError(f"Malformed case tuple format: '{raw_tuple_str}'. Expected (condition, action).")
            
            inner_tuple_content = stripped_tuple_str[1:-1].strip()

            condition_action_strs = _split_balanced_args(inner_tuple_content, num_args=2)

            if len(condition_action_strs) != 2:
                raise ValueError(f"Malformed case tuple: {raw_tuple_str}. Could not parse into condition and action.")

            condition_cmd_str = condition_action_strs[0].strip()
            action_cmd_str = condition_action_strs[1].strip()

            condition_cmd = parser.parse_token(condition_cmd_str)
            action_cmd = parser.parse_token(action_cmd_str)

            parsed_cases.append((condition_cmd, action_cmd))
    
    default_action_cmd = parser.parse_token(default_action_str)

    return {
        'grid_to_evaluate_cmd': grid_to_evaluate_cmd,
        'cases': parsed_cases,
        'default_action_cmd': default_action_cmd
    }


def parse_symbolic_grid_literal(grid_list_str: str) -> np.ndarray:
    if not (grid_list_str.startswith('[[') and grid_list_str.endswith(']]')):
        raise ValueError(f"Invalid grid literal format: '{grid_list_str}'. Expected '[[...]]'.")
    inner_str = grid_list_str[2:-2]

    row_strings = re.split(r'\],\[', inner_str)

    parsed_rows = []
    for row_str in row_strings:
        symbol_list = _split_balanced_args(row_str)

        parsed_row = []
        for symbol in symbol_list:
            symbol = symbol.strip()

            if symbol == '?':
                parsed_row.append(_WILDCARD_VALUE)
            elif symbol in ROM_VAL_MAP:
                parsed_row.append(ROM_VAL_MAP[symbol])
            elif symbol.isdigit() or (symbol.startswith('-') and symbol[1:].isdigit()):
                parsed_row.append(int(symbol))
            else:
                raise ValueError(f"Unknown symbolic color value: '{symbol}' in grid literal.")
        parsed_rows.append(parsed_row)

    if not parsed_rows:
        return np.array([], dtype=int).reshape(0, 0)

    first_row_len = len(parsed_rows[0])
    for i, row in enumerate(parsed_rows):
        if len(row) != first_row_len:
            raise ValueError(f"Rows in grid literal have inconsistent lengths. Row {i} has {len(row)} elements, expected {first_row_len}.")

    return np.array(parsed_rows, dtype=int)


def _split_balanced_args(s: str, num_args: int = None) -> List[str]:
    args = []
    balance_paren = 0
    balance_bracket = 0
    current_arg = []

    for char in s:
        if char == '(':
            balance_paren += 1
        elif char == ')':
            balance_paren -= 1
        elif char == '[':
            balance_bracket += 1
        elif char == ']':
            balance_bracket -= 1

        if char == ',' and balance_paren == 0 and balance_bracket == 0:
            args.append("".join(current_arg).strip())
            current_arg = []
        else:
            current_arg.append(char)

    args.append("".join(current_arg).strip())

    if num_args is not None and len(args) != num_args:
        raise ValueError(f"Expected {num_args} arguments but found {len(args)} after balanced split for string: '{s}'")
    return args

def roman_to_int(s: str) -> int:
    """
    Converts a Roman numeral string (up to XXX) to an integer,
    supporting an optional leading negative sign.
    """
    is_negative = False
    if s.startswith('-'):
        is_negative = True
        s = s[1:] # Remove the negative sign for lookup

    # Convert to uppercase for map lookup
    s_upper = s.upper()

    val = ROM_VAL_MAP.get(s_upper)

    if val is None:
        # Re-add the '-' to the error message if it was present
        display_s = f"-{s}" if is_negative and s else s
        raise ValueError(f"Invalid or out-of-range Roman numeral: {display_s}. Max supported is XXX.")

    return -val if is_negative else val


SYMBOL_RULES = {
    "input_grid_reference": {
        "sigil": "⌂",
    },
    "identity": {"sigil": "Ⳁ"},
    "map_numbers": {
        "pattern": fr"^⇒\((?P<old>{ROMAN_VALUE_PATTERN}),\s*(?P<new>{ROMAN_VALUE_PATTERN})\)$",
        "transform_params": lambda m: {
            "mapping": {
                roman_to_int(m["old"]): roman_to_int(m["new"])
            }
        }
    },
    "flip_h": {
        "pattern": r"^↔(?:\((?P<arg_content>.+)\))?$",
        "transform_params": lambda m: {
            "argument_command_str": m.group("arg_content") if m.group("arg_content") else None
        },
        "nested_commands": {
            "argument_command": "argument_command_str"
        },
    },
    "flip_v": {
        "pattern": r"^↕(?:\((?P<arg_content>.+)\))?$",
        "transform_params": lambda m: {
            "argument_command_str": m.group("arg_content") if m.group("arg_content") else None
        },
        "nested_commands": {
            "argument_command": "argument_command_str"
        },
    },
    "reverse_row": {
        "pattern": r"^↢(?:\((?P<arg_content>.+)\))?$",
        "transform_params": lambda m: {
            "argument_command_str": m.group("arg_content") if m.group("arg_content") else None
        },
        "nested_commands": {
            "argument_command": "argument_command_str"
        },
    },
    "shift_row_or_column": {
        "pattern": fr"^(?P<direction>⮝|⮞)\((?P<idx>{ROMAN_INDEX_PATTERN}),\s*(?P<shift_val>{ROMAN_VALUE_PATTERN})\)$",
        "transform_params": lambda m: {
            "row_index": roman_to_int(m["idx"]) - 1 if m["direction"] == '⮝' else None,
            "col_index": roman_to_int(m["idx"]) - 1 if m["direction"] == '⮞' else None,
            "shift_amount": roman_to_int(m["shift_val"]),
            "wrap": True
        }
    },
    "apply_to_row": {
        "pattern": fr"^→\((?P<row_idx>{ROMAN_INDEX_PATTERN}),\s*(?P<inner_command_str>.+)\)$",
        "transform_params": lambda m: {
            "row_index": roman_to_int(m["row_idx"]) - 1,
            "inner_command_str": m["inner_command_str"]
        },
        "nested_commands": {
            "inner_command": "inner_command_str"
        }
    },
    "swap_rows_or_columns": {
        "pattern": fr"^⇄\((?P<idx1>{ROMAN_INDEX_PATTERN}),\s*(?P<idx2>{ROMAN_INDEX_PATTERN})\)$",
        "transform_params": lambda m: {
            "row_swap": (roman_to_int(m["idx1"]) - 1, roman_to_int(m["idx2"]) - 1),
            "col_swap": None,
            "swap_type": "rows"
        }
    },
    "repeat_grid_horizontal": {
        "pattern": fr"^◨\((?P<count>{ROMAN_VALUE_PATTERN})\)$",
        "transform_params": lambda m: {
            "inner_command_str": "Ⳁ",
            "vertical_repeats": 1,
            "horizontal_repeats": roman_to_int(m["count"])
        },
        "nested_commands": {
            "inner_command": "inner_command_str"
        },
        "target_op_name": "repeat_grid"
    },
    "repeat_grid_vertical": {
        "pattern": fr"^⬒\((?P<count>{ROMAN_VALUE_PATTERN})\)$",
        "transform_params": lambda m: {
            "inner_command_str": "Ⳁ",
            "vertical_repeats": roman_to_int(m["count"]),
            "horizontal_repeats": 1
        },
        "nested_commands": {
            "inner_command": "inner_command_str"
        },
        "target_op_name": "repeat_grid"
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
        "pattern": fr"^⊕\((?P<rows>{ROMAN_INDEX_PATTERN}),\s*(?P<cols>{ROMAN_INDEX_PATTERN}),\s*(?P<fill_color>{ROMAN_VALUE_PATTERN})\)$",
        "transform_params": lambda m: {
            "rows": roman_to_int(m["rows"]),
            "cols": roman_to_int(m["cols"]),
            "fill_color": roman_to_int(m["fill_color"])
        }
    },
    "scale_grid": {
        "pattern": fr"^⤨\((?P<factor>{ROMAN_VALUE_PATTERN})\)$",
        "transform_params": lambda m: {
            "scale_factor": roman_to_int(m["factor"])
        }
    },
    "extract_bounding_box": {
        "pattern": r"^⧈(?:\((?P<arg_content>.+)\))?$",
        "transform_params": lambda m: {
            "argument_command_str": m.group("arg_content") if m.group("arg_content") else None
        },
        "nested_commands": {
            "argument_command": "argument_command_str"
        },
    },
    "flatten_grid": {
        "pattern": r"^⧀(?:\((?P<arg_content>.+)\))?$",
        "transform_params": lambda m: {
            "argument_command_str": m.group("arg_content") if m.group("arg_content") else None
        },
        "nested_commands": {
            "argument_command": "argument_command_str"
        },
    },
    "alternate": {
        "pattern": r"^⇌\((?P<all_commands_str>.+)\)$",
        "transform_params": lambda m: (
            parts := _split_balanced_args(m["all_commands_str"], num_args=2),
            {
                "first_command_str": parts[0],
                "second_command_str": parts[1]
            }
        )[1],
        "nested_commands": {
            "first": "first_command_str",
            "second": "second_command_str"
        }
    },
    "conditional_transform": {
        "pattern": r"^¿\((?P<all_args>.+)\)$",
        "transform_params": lambda m: (
            args := _split_balanced_args(m["all_args"], num_args=None),
            {
                "true_command_str": args[0],
                "condition_command_str": args[1],
                "false_command_str": args[2] if len(args) > 2 else None
            }
        )[1],
        "nested_commands": {
            "true_command": "true_command_str",
            "condition_command": "condition_command_str",
            "false_command": "false_command_str"
        }
    },
    "get_element": {
        "pattern": fr"^⊡\((?P<row_idx>{ROMAN_INDEX_PATTERN}),\s*(?P<col_idx>{ROMAN_INDEX_PATTERN})\)$",
        "transform_params": lambda m: {
            "row_index": roman_to_int(m["row_idx"]) - 1,
            "col_index": roman_to_int(m["col_idx"]) - 1
        }
    },
    "get_constant": {
        "pattern": fr"^↱(?P<value>{ROMAN_VALUE_PATTERN})$",
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
    'block_grid_builder': { # Renamed rule key for internal clarity
        'pattern': fr'^▦\((?P<block_rows>{ROMAN_INDEX_PATTERN}),\s*(?P<block_cols>{ROMAN_INDEX_PATTERN}),\s*(?P<pattern_list_str>\[\[.+\]\])\)$',
        'transform_params': lambda m: {
            "block_rows": roman_to_int(m["block_rows"]),
            "block_cols": roman_to_int(m["block_cols"]),
            "pattern_matrix": parse_symbolic_grid_literal(m["pattern_list_str"]) # Cleanly call the helper
        },
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
    },
    'match_pattern': {
        'pattern': r'^◫\((?P<full_params_str>.*)\)$', # Capture everything inside ◫(...)
        'transform_params': lambda m: {
            "full_params_str": m["full_params_str"]
        },
        'nested_commands': {},
        'param_processors': {
            'full_params': '_process_match_pattern_full_params'
        }
    },
    "filter_grid_by_color": {
        "pattern": fr"^◎\((?P<color_str>{ROMAN_VALUE_PATTERN})\)$",
        "op_name": "FilterGridByColor",
        "transform_params": lambda m: {
            "target_color": roman_to_int(m["color_str"])
        }
    },
    "get_external_background_mask": {
        "pattern": fr"^⏚\((?P<background_color>{ROMAN_VALUE_PATTERN})\)$",
        "transform_params": lambda m: {
            "background_color": roman_to_int(m["background_color"])
        }
    },
    "mask_not": {
        "pattern": r"^¬\((?P<mask_cmd_str>.+)\)$", # Using direct Unicode character
        "transform_params": lambda m: {
            "mask_cmd_str": m["mask_cmd_str"]
        },
        "nested_commands": {
            "mask_cmd": "mask_cmd_str"
        },
        "target_op_name": "mask_not"
    },
    "mask_and": {
        "pattern": r"^∧\((?P<all_commands_str>.+)\)$", # Using direct Unicode character
        "transform_params": lambda m: {
            "mask_cmd1_str": _split_balanced_args(m["all_commands_str"], 2)[0],
            "mask_cmd2_str": _split_balanced_args(m["all_commands_str"], 2)[1]
        },
        "nested_commands": {
            "mask_cmd1": "mask_cmd1_str",
            "mask_cmd2": "mask_cmd2_str"
        },
        "target_op_name": "mask_and"
    },
    "mask_or": {
        "pattern": r"^∨\((?P<all_commands_str>.+)\)$", # Using direct Unicode character
        "transform_params": lambda m: {
            "mask_cmd1_str": _split_balanced_args(m["all_commands_str"], 2)[0],
            "mask_cmd2_str": _split_balanced_args(m["all_commands_str"], 2)[1]
        },
        "nested_commands": {
            "mask_cmd1": "mask_cmd1_str",
            "mask_cmd2": "mask_cmd2_str"
        },
        "target_op_name": "mask_or"
    },
    "binarize": {
        "pattern": r"^ⓑ\((?P<cmd_str>.+)\)$",
        "op_name": "Binarize",
        "transform_params": lambda m: {
            "cmd_str": m["cmd_str"]
        },
        "nested_commands": {
            "cmd": "cmd_str"
        }
    },
    "locate_pattern": {
        "pattern": r"^⌖\((?P<all_commands_str>.+)\)$",
        "transform_params": lambda m: (
            parts := _split_balanced_args(m["all_commands_str"], num_args=2),
            {
                "grid_to_search_cmd_str": parts[0],
                "pattern_to_find_cmd_str": parts[1]
            }
        )[1],
        "nested_commands": {
            "grid_to_search_cmd": "grid_to_search_cmd_str",
            "pattern_to_find_cmd": "pattern_to_find_cmd_str"
        },
        "target_op_name": "locate_pattern"
    },
    "slice_grid": {
        "pattern": fr"^✂\((?P<row_start>{ROMAN_VALUE_PATTERN}),\s*(?P<col_start>{ROMAN_VALUE_PATTERN}),\s*(?P<row_end>{ROMAN_VALUE_PATTERN}),\s*(?P<col_end>{ROMAN_VALUE_PATTERN})\)$",
        "transform_params": lambda m: {
            "row_start": roman_to_int(m["row_start"]),
            "col_start": roman_to_int(m["col_start"]),
            "row_end": roman_to_int(m["row_end"]),
            "col_end": roman_to_int(m["col_end"])
        },
    },
    "fill_region": {
        "pattern": fr"^\s*■\((?P<target_grid_str>.+?),\s*(?P<fill_value>{ROMAN_VALUE_PATTERN}),\s*(?P<row_start>{ROMAN_VALUE_PATTERN}),\s*(?P<col_start>{ROMAN_VALUE_PATTERN}),\s*(?P<row_end>{ROMAN_VALUE_PATTERN}),\s*(?P<col_end>{ROMAN_VALUE_PATTERN})\)\s*$",
        "transform_params": lambda m: {
            "target_grid_str": m["target_grid_str"],
            "fill_value": roman_to_int(m["fill_value"]),
            "row_start": roman_to_int(m["row_start"]),
            "col_start": roman_to_int(m["col_start"]),
            "row_end": roman_to_int(m["row_end"]),
            "col_end": roman_to_int(m["col_end"]) 
        },
        "nested_commands": {
            "target_grid_command": "target_grid_str", 
        }
    },
    "add_grid_to_canvas": {
        "pattern": fr"^\s*⊞\((?P<all_args>.+)\)\s*$",
        "op_name": "AddGridToCanvas",
        "transform_params": lambda m: (
            args := _split_balanced_args(m["all_args"], num_args=4),
            {"target_grid_cmd_str": args[0],
            "source_grid_cmd_str": args[1],
            "row_offset": roman_to_int(args[2]),
            "col_offset": roman_to_int(args[3])}
        )[1],
        "nested_commands": {
            "target_grid_command": "target_grid_cmd_str",
            "source_grid_command": "source_grid_cmd_str",
        },
    },
    
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
                if "target_op_name" in rule:
                    entry["target_op_name"] = rule["target_op_name"]
                if "param_processors" in rule:
                    entry["param_processors"] = rule["param_processors"]
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
            if token is None or (isinstance(token, str) and token.strip().lower() == 'none'):
                self.logger.warning(f"Attempted to parse a '{token}' token. Returning None.")
                return None
            token = token.strip()
            self.logger.debug(f"Parsing token: '{token}'")

            for pattern, data in self._compiled_rules.items():
                if data.get("original_sigil") and "pattern" not in SYMBOL_RULES.get(data["operation"], {}):
                    if token == data["original_sigil"]:
                        self.logger.debug(f"Matched sigil '{token}' for operation '{data['operation']}'")
                        return self.factory.create_operation(data["operation"])

            for pattern, data in self._compiled_rules.items():
                if data.get("original_sigil") and "pattern" not in SYMBOL_RULES.get(data["operation"], {}):
                    continue

                match = pattern.match(token)
                if match:
                    self.logger.debug(f"Matched pattern for operation '{data['operation']}': '{token}'")

                    if "transform_params" in data:
                        processed_params = data["transform_params"](match)
                        self.logger.debug(f"Transformed params: {processed_params}")
                    else:
                        processed_params = match.groupdict()

                    if 'param_processors' in data:
                        for target_param_name, processor_func_name in data['param_processors'].items():
                            if processor_func_name not in globals() or not callable(globals()[processor_func_name]):
                                self.logger.error(f"Param processor function '{processor_func_name}' not found or not callable for '{data['operation']}'.")
                                raise ValueError(f"Invalid parameter processor: '{processor_func_name}' for {data['operation']}.")
                            
                            processed_params = globals()[processor_func_name](processed_params, self)
                            self.logger.debug(f"Params after '{processor_func_name}': {processed_params}")

                    if "nested_commands" in data:
                        for param_name_in_init, config in data["nested_commands"].items():
                            if isinstance(config, tuple):
                                regex_group_name, parse_type = config
                            else:
                                regex_group_name = config
                                parse_type = "single"

                            command_string_to_parse = processed_params.pop(regex_group_name, None)

                            if command_string_to_parse is None:
                                self.logger.debug(f"Optional nested command '{regex_group_name}' not found or was None for '{data['operation']}'. Setting {param_name_in_init}=None.")
                                processed_params[param_name_in_init] = None
                                continue

                            if isinstance(command_string_to_parse, str) and command_string_to_parse.strip().lower() == 'none':
                                self.logger.debug(f"Parsed inner command string was literal 'None' for '{param_name_in_init}' in '{data['operation']}'. Assigning None.")
                                processed_params[param_name_in_init] = None
                                continue

                            if parse_type == "single":
                                parsed_cmd_obj = self.parse_token(command_string_to_parse)
                                
                                if not parsed_cmd_obj:
                                    if param_name_in_init == "argument_command":
                                        self.logger.debug(f"Parsed inner command '{command_string_to_parse}' for optional argument_command in '{data['operation']}' resulted in None. Assigning None.")
                                        processed_params[param_name_in_init] = None
                                    else:
                                        raise ValueError(f"Failed to parse inner command '{command_string_to_parse}' for '{param_name_in_init}' in '{data['operation']}'")
                                else:
                                    processed_params[param_name_in_init] = parsed_cmd_obj
                            elif parse_type == "list":
                                raw_cmd_strings = self._split_sequence_commands(command_string_to_parse)
                                parsed_commands_list = []
                                for cmd_str in raw_cmd_strings:
                                    cmd_obj = self.parse_token(cmd_str)
                                    if not cmd_obj:
                                        raise ValueError(f"Failed to parse sequence command '{cmd_str}' for '{param_name_in_init}' in '{data['operation']}'")
                                    parsed_commands_list.append(cmd_obj)
                                processed_params[param_name_in_init] = parsed_commands_list
                            else:
                                raise ValueError(f"Unknown nested command parse_type: {parse_type} for '{param_name_in_init}' in '{data['operation']}'")

                    final_op_name = data.get("target_op_name", data["operation"])

                    self.logger.debug(f"Final params for '{final_op_name}': {processed_params}")
                    return self.factory.create_operation(final_op_name, **processed_params)

            raise ValueError(f"Could not parse symbolic token: '{token}'")