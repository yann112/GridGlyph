import logging
import re
from typing import Optional, List # Ensure List is imported

from sources.core.dsl_nodes import (
    AbstractTransformationCommand,
    DSL_REGISTRY,
    Identity,
    RepeatGrid,
    FlipGridHorizontally,
    FlipGridVertically,
)


class SynthesizeAgent:
    def __init__(self, llm, synthesizer):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self.synthesizer = synthesizer

    def _parse_program_string(self, program_string: str) -> Optional[AbstractTransformationCommand]:
        """Parses a single program string and returns an AST command object."""
        program_string = program_string.strip()
        match = re.match(r"(\w+)\((.*)\)", program_string)

        if not match:
            self.logger.warning(f"Could not parse command string: {program_string}")
            return None

        command_name = match.group(1)
        args_str = match.group(2)

        if command_name not in DSL_REGISTRY:
            self.logger.warning(f"Unknown command: {command_name} in {program_string}")
            return None

        command_class = DSL_REGISTRY[command_name]

        try:
            if command_name == "repeat_grid":
                # repeat_grid(identity(), 2, 3)
                parts = [p.strip() for p in args_str.split(",")]
                if len(parts) != 3:
                    self.logger.warning(f"Incorrect number of arguments for repeat_grid: {args_str}")
                    return None
                
                inner_command_str = parts[0]
                inner_command = self._parse_program_string(inner_command_str)
                if not inner_command:
                    self.logger.warning(f"Could not parse inner command for repeat_grid: {inner_command_str}")
                    return None

                vertical_repeats = int(parts[1])
                horizontal_repeats = int(parts[2])
                return RepeatGrid(
                    inner_command=inner_command,
                    vertical_repeats=vertical_repeats,
                    horizontal_repeats=horizontal_repeats,
                )
            elif command_name in ["flip_h", "flip_v"]:
                # flip_h() or flip_h(identity())
                # The constructors for FlipGridHorizontally and FlipGridVertically do not take arguments.
                # We can ignore the presence of identity() if the LLM includes it.
                return command_class()
            elif command_name == "identity":
                 return Identity()
            else:
                # For other commands, if they expect arguments, this part needs to be generalized.
                # Assuming simple commands without arguments for now if not repeat_grid or flips.
                # This might need adjustment if new commands with different arg patterns are added.
                if args_str == "" or args_str == "identity()":
                    return command_class()
                else:
                    self.logger.warning(f"Command {command_name} called with unexpected arguments: {args_str}")
                    return None

        except ValueError as e:
            self.logger.warning(f"Error parsing arguments for {command_name} in '{program_string}': {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error parsing {program_string}: {e}", exc_info=True)
            return None

    def generate_program_candidates(self, input_grid, output_grid, analysis_summary):
        """Generates multiple DSL program candidates via LLM."""

        # Constructing DSL command descriptions dynamically
        available_commands_desc = []
        if "identity" in DSL_REGISTRY:
            available_commands_desc.append("- identity(): No operation, returns the grid as is.")
        if "flip_h" in DSL_REGISTRY:
            available_commands_desc.append("- flip_h(): Flips the grid horizontally.")
        if "flip_v" in DSL_REGISTRY:
            available_commands_desc.append("- flip_v(): Flips the grid vertically.")
        # Add other simple commands here if they are parameterless
        # Example:
        # if "rotate_90_cw" in DSL_REGISTRY:
        #     available_commands_desc.append("- rotate_90_cw(): Rotates grid 90 degrees clockwise.")
        
        # Special handling for commands with parameters like repeat_grid
        if "repeat_grid" in DSL_REGISTRY:
            desc = (
                "- repeat_grid(inner_command, vertical_repeats, horizontal_repeats): "
                "Repeats the output of the inner_command. "
                "'inner_command' must be another valid DSL command from this list (e.g., identity()). "
                "'vertical_repeats' and 'horizontal_repeats' must be integers (e.g., 1, 2, 3)."
            )
            available_commands_desc.append(desc)
        # Add other complex commands with parameters here, detailing their arguments.

        dsl_command_info = "\n        ".join(available_commands_desc) # Indent for prompt readability

        prompt = f"""You are an expert DSL program writer.
        Your goal is to suggest 3-5 DSL programs that transform an input grid into an output grid, based on an analysis summary.

        Input Grid:
        {input_grid}

        Output Grid:
        {output_grid}

        Analysis:
        {analysis_summary}

        Available DSL commands and their syntax:
        {dsl_command_info}

        Important Rules:
        1. Return ONLY valid DSL commands from the list above, one command per line.
        2. Ensure all arguments for commands match the specified syntax precisely.
        3. For 'repeat_grid', the 'inner_command' MUST be one of the other available commands (e.g., 'identity()').
        4. Do NOT use commands not listed (e.g., `rotate_cw(identity())` is invalid if `rotate_cw` is not listed).
        5. For commands like `flip_h()` or `flip_v()`, do not pass any arguments inside the parentheses.

        Examples of VALID lines in your output:
        repeat_grid(identity(), 2, 3)
        flip_v()
        identity()

        Example of an INVALID line (if `flip_h` takes no arguments):
        flip_h(identity())

        Provide your response as a list of commands, each on a new line.
        """
        self.logger.debug(f"Generated LLM prompt:\n{prompt}")
        raw_response = self.llm(prompt)
        self.logger.debug(f"LLM raw response:\n{raw_response}")
        
        # Filter out empty lines that might result from LLM response format
        candidates = [line.strip() for line in raw_response.strip().split('\n') if line.strip()]
        return candidates

    def synthesize(self, input_grid, output_grid, analysis_summary, top_k_programs: int = 5):
        """Full pipeline: Generate candidates → Parse → Evaluate & Sort → Return best programs."""
        candidates_str = self.generate_program_candidates(input_grid, output_grid, analysis_summary)
        
        parsed_programs: List[AbstractTransformationCommand] = []
        for candidate_str in candidates_str:
            program = self._parse_program_string(candidate_str)
            if program:
                parsed_programs.append(program)
        
        if not parsed_programs:
            self.logger.info("No programs were successfully parsed from LLM candidates.")
            return []

        # Assuming input_grid and output_grid are already numpy arrays
        # as per typical usage and type hints in SynthesisEngine.
        # If conversion is needed, it would be:
        # import numpy as np
        # input_grid_np = np.array(input_grid) if not isinstance(input_grid, np.ndarray) else input_grid
        # output_grid_np = np.array(output_grid) if not isinstance(output_grid, np.ndarray) else output_grid

        self.logger.info(f"Evaluating {len(parsed_programs)} parsed programs.")
        scored_candidates = self.synthesizer.evaluate_and_sort_candidates(
            program_candidates=parsed_programs,
            input_grid=input_grid,
            output_grid=output_grid,
            top_k=top_k_programs
        )
        
        self.logger.info(f"Returning {len(scored_candidates)} programs after evaluation and sorting.")
        # Extract program objects from (program, score) tuples
        best_programs = [program for program, score in scored_candidates]
        
        return best_programs