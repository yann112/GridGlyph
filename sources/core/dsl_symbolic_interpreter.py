# symbolic_interpreter.py

import re
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from core.transformation_factory import TransformationFactory
from core.dsl_nodes import AbstractTransformationCommand


# Roman numerals to int helper
def roman_to_int(s: str) -> int:
    """Converts a Roman numeral string (up to XXX) to an integer."""
    # A complete map is simpler for this limited range than a full parsing algorithm
    rom_val_map = {
        '∅':0, 'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9,
        'X': 10, 'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19,
        'XX': 20, 'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24, 'XXV': 25, 'XXVI': 26, 'XXVII': 27, 'XXVIII': 28, 'XXIX': 29,
        'XXX': 30
    }
    # Convert to upper case for case-insensitivity
    val = rom_val_map.get(s.upper())
    if val is None:
        raise ValueError(f"Invalid or out-of-range Roman numeral: {s}. Max supported is XXX.")
    return val


SYMBOL_RULES = {
    "identity": {"sigil": "Ⳁ"},
    "map_numbers": {
        "pattern": r"^⇒\((?P<old>\d+)→(?P<new>\d+)\)$",
        "transform_params": lambda m: {"mapping": {int(m["old"]): int(m["new"])}}
    },
    "flip_h": {"sigil": "↔"},
    "flip_v": {"sigil": "↕"},
    "reverse_row": {"sigil": "↢"},
    "shift_row": {
        "pattern": r"^⮝\((?P<row_idx>[IVX]+),\s*(?P<shift_val>[IVX]+)\)$",
        "transform_params": lambda m: {
            "row_index": roman_to_int(m["row_idx"]) - 1,
            "shift_amount": roman_to_int(m["shift_val"]),
            "col_index": None,
            "wrap": True
        }
    },
    "shift_column": {
        "pattern": r"^⮞\((?P<col_idx>[IVX]+),\s*(?P<shift_val>[IVX]+)\)$",
        "transform_params": lambda m: {
            "col_index": roman_to_int(m["col_idx"]) - 1,
            "shift_amount": roman_to_int(m["shift_val"]),
            "row_index": None,
            "wrap": True
        }
    },
    "apply_to_row": {
        "pattern": r"^→\((?P<row_idx>[IVX]+),\s*(?P<inner_command>.+)\)$",
        "transform_params": lambda m: {
            "row_index": roman_to_int(m["row_idx"]) - 1,
            "inner_command_str": m["inner_command"]
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
            "vertical_repeats": 1,
            "horizontal_repeats": roman_to_int(m["count"])
        }
    },
    "repeat_grid_vertical": {
        "pattern": r"^⬒\((?P<count>[IVX]+)\)$",
        "transform_params": lambda m: {
            "vertical_repeats": roman_to_int(m["count"]),
            "horizontal_repeats": 1
        }
    },
    "sequence": {
        "pattern": r"^⟹\((?P<cmds>.+)\)$",
        "transform_params": lambda m: {
            "commands_str": m["cmds"]
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
    "rotate_grid_90_clockwise": {"sigil": "↻"},
    "transpose_grid": {"sigil": "⤫"},
    "flip_grid_anti_diagonal": {"sigil": "╳"},
    "crop_to_bounding_box": {"sigil": "⊟"},
    "add_padding": {
        "pattern": r"^⌗\((?P<amount>[IVX]+|∅),\s*(?P<color>[IVX]+|∅)\)$",
        "transform_params": lambda m: {"padding_amount": roman_to_int(m["amount"]), "padding_color": roman_to_int(m["color"])}
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
                compiled[pattern] = entry
            else:
                raise ValueError(f"Rule must define 'sigil' or 'pattern': {op_name}")
        return compiled

    # This method is for top-level '+' split only, not for nested commas
    def tokenize_expression(self, expr: Union[str, List[str]]) -> List[str]:
        if isinstance(expr, str):
            # This split assumes '+' is the only top-level sequence delimiter.
            return [token.strip() for token in expr.split("+")]
        else:
            return expr

    # This parse_rule handles the top-level '+' sequences
    def parse_rule(self, rule: Union[str, List[str]]) -> AbstractTransformationCommand:
        tokens = self.tokenize_expression(rule)
        commands = []

        for token in tokens:
            cmd = self.parse_token(token) # parse_token is now recursive!
            if cmd:
                commands.append(cmd)

        if len(commands) == 1:
            return commands[0]
        elif len(commands) > 1:
            # Here, the 'commands' list already contains parsed command objects
            return self.factory.create_operation("sequence", commands=commands)
        else:
            return self.factory.create_operation("identity")

    # This is the helper for splitting comma-separated commands *within* a sequence.
    # It must handle nested parentheses correctly.
    def _split_sequence_commands(self, command_string: str) -> List[str]:
        commands = []
        balance = 0
        current_command_chars = []
        
        # self.logger.debug(f"Splitting sequence inner string: '{command_string}'")
        
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
        # self.logger.debug(f"Split result for sequence inner: {final_commands}")
        return final_commands

    def parse_token(self, token: str) -> Optional[AbstractTransformationCommand]:
        token = token.strip()
        # self.logger.debug(f"Parsing token: '{token}'")

        # Try exact sigil match first (atomic commands)
        for pattern, data in self._compiled_rules.items():
            if data.get("original_sigil"):
                if token == data["original_sigil"]:
                    return self.factory.create_operation(data["operation"])

        # Try regex patterns for all operations (including compound ones)
        for pattern, data in self._compiled_rules.items():
            # Skip sigil-based rules as they were handled
            if data.get("original_sigil"):
                continue

            match = pattern.match(token)
            if match:
                raw_params = match.groupdict()
                
                # Apply transform_params to get structured parameters
                if "transform_params" in data:
                    processed_params = data["transform_params"](raw_params)
                else:
                    processed_params = raw_params # No transformation needed

                # --- RECURSIVE PARSING LOGIC HERE ---
                if data["operation"] == "apply_to_row":
                    inner_command_str = processed_params.pop("inner_command_str") # Get raw string
                    # Recursively parse the inner command
                    inner_command_obj = self.parse_token(inner_command_str)
                    if not inner_command_obj:
                        raise ValueError(f"Failed to parse inner command '{inner_command_str}' for apply_to_row")
                    processed_params["inner_command"] = inner_command_obj
                
                elif data["operation"] == "sequence":
                    commands_str = processed_params.pop("commands_str") # Get raw string
                    # Use the special splitter for sequence commands
                    raw_cmd_strings = self._split_sequence_commands(commands_str)

                    parsed_commands = []
                    for cmd_str in raw_cmd_strings:
                        cmd_obj = self.parse_token(cmd_str) # Recursively parse each command
                        if not cmd_obj:
                             raise ValueError(f"Failed to parse sequence command '{cmd_str}'")
                        parsed_commands.append(cmd_obj)
                    processed_params["commands"] = parsed_commands

                # For other commands (atomic or simple combinators), processed_params are already final
                return self.factory.create_operation(data["operation"], **processed_params)

        raise ValueError(f"Could not parse symbolic token: '{token}'")