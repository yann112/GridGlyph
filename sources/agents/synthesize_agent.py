from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import logging
import json
import inspect
from core.dsl_nodes import AbstractTransformationCommand
from core.transformation_factory import TransformationFactory


class SynthesizeAgent:
    def __init__(self, llm, synthesizer, logger: logging.Logger = None, mandatory_operations: List = None):
        """
        Args:
            llm: Language model interface for generating program candidates
            synthesizer: SynthesisEngine instance for validating programs
            logger: Optional logger instance
            mandatory_operations: Optional, set a list of operations that will always be given to the synthesizer 
        """
        self.llm = llm
        self.synthesizer = synthesizer
        self.number_program_candidates = 10
        self.logger = logger or logging.getLogger(__name__)
        self.mandatory_operations = mandatory_operations or ['identity', 'sequence', 'mask_combinator']

    def _build_operation_documentation(self, allowed_operations: List[str] = None) -> str:
        """
        Build JSON-formatted operation documentation.
        
        Args:
            allowed_operations (List[str], optional): If provided, only include these operations.
                                                    Defaults to None (include all operations).
        
        Returns:
            str: JSON string of operation documentation.
        """
        docs = []

        for op_name, op_class in TransformationFactory.OPERATION_MAP.items():
            # Skip if filtering is active and this op isn't in the allow list
            if allowed_operations is not None and op_name not in allowed_operations:
                self.logger.debug(f"Skipping operation '{op_name}' - not in allowed list.")
                continue

            try:
                rules = op_class.synthesis_rules

                # Get parameter info
                sig = inspect.signature(op_class.__init__)
                param_names = [name for name in sig.parameters.keys()
                            if name not in ['self', 'logger']]

                dict_doc = {
                    "operation name": op_name,
                    "list parameters:": param_names,
                    "synthesis rules": rules,
                    "decription": op_class.describe()
                }
                docs.append(dict_doc)
            except Exception as e:
                self.logger.error(f"Failed to build documentation for operation '{op_name}': {str(e)}", exc_info=True)
                continue

        return json.dumps(docs, indent=2)

    def _filter_relevant_operations(self, input_grid: np.ndarray, output_grid: np.ndarray,
                                    analysis_summary: str) -> List[str]:
        """
        Ask the LLM to narrow down which operations are relevant based on analysis.
        """
        operation_docs = self._build_operation_documentation()
        self.logger.debug("Starting operation filtering...")
        self.logger.debug(f"Input Grid:\n{input_grid.tolist()}")
        self.logger.debug(f"Output Grid:\n{output_grid.tolist()}")
        self.logger.debug(f"Analysis Summary:\n{analysis_summary}")

        prompt = f"""
            You are an operation filterer. Your task is to identify all operations from the available list that could potentially be used to transform the Input Grid into the Output Grid.

            Available Operations:
            {operation_docs}

            Input Grid: {input_grid.tolist()}
            Output Grid: {output_grid.tolist()}

            Analysis Summary:
            {analysis_summary}

            Instructions:
            - Focus on operations that can **implement the core transformation described in the Analysis Summary**.
            - Actively explore and include any **alternative operational approaches** that could also achieve the transformation.
            - Include operations that might serve as an inner command within a combinator (e.g., `flip_h` inside `apply_to_row`).
            - Keep combinator operations like `sequence` and `mask_combinator` if multiple smaller or targeted changes appear necessary.
            - Only exclude operations that are clearly not applicable (e.g., those that would have no effect on this specific grid, or operate on entirely irrelevant axes).
            - Return only a JSON list of operation names.

            Example response:
            ```json
            ["swap_rows_or_columns", "apply_to_row", "flip_h", "sequence", "mask_combinator"]
            ```
            """

        try:
            response = self.llm(prompt).strip()
            self.logger.info("Operation filtering response received.")
            self.logger.debug(f"Raw filtering response:\n{response}")

            start = response.find("```json") + 7
            end = response.rfind("```")
            if start == -1 or end == -1:
                raise ValueError("Could not find JSON block in filtering response.")
            content = response[start:end].strip()
            filtered_ops = json.loads(content)
            # Add mandatory operations by name
            filtered_ops.extend(self.mandatory_operations)
            # Remove duplicates while preserving order
            seen = set()
            filtered_ops = [x for x in filtered_ops if x not in seen and not seen.add(x)]
            self.logger.info(f"Filtered operations: {filtered_ops}")
            return filtered_ops
        except Exception as e:
            self.logger.error(f"Error during operation filtering: {e}", exc_info=True)
            return []

    def _filter_valid_json_dicts(self, candidate_strings: List[str]) -> List[str]:
        """
        Filters a list of strings, returning only those that are valid JSON dictionaries.
        
        Uses self.logger for logging warnings.
        
        Args:
            candidate_strings: List of raw string candidates (possibly JSON-formatted)
            
        Returns:
            List of valid JSON dictionary objects
        """
        valid_dicts = []
        
        for idx, line in enumerate(candidate_strings):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    valid_dicts.append(line)
                else:
                    self.logger.debug(f"Line {idx}: JSON is not a dictionary: {data}")
            except json.JSONDecodeError as e:
                self.logger.debug(f"Line {idx}: Invalid JSON - {e}: '{line}'")
        
        return valid_dicts

    def _generate_filtered_programs(
            self,
            input_grid: np.ndarray,
            output_grid: np.ndarray,
            allowed_operations: List[str],
            analysis_summary) -> List[str]:
            """
            Generate candidate programs using only the filtered set of operations.
            """
            operation_docs = self._build_operation_documentation()

            # Filter operation docs to include only relevant ones
            allowed_ops_set = set(allowed_operations)
            try:
                full_ops_list = json.loads(operation_docs)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse operation docs JSON: {e}")
                return []

            filtered_docs = [
                op for op in full_ops_list
                if op["operation name"] in allowed_ops_set
            ]
            filtered_docs_str = json.dumps(filtered_docs, indent=2)

            self.logger.debug("Generating programs with allowed operations:")
            self.logger.debug(f"Allowed Operations: {allowed_ops_set}")
            self.logger.debug(f"Filtered Operation Docs:\n{filtered_docs_str}")

            prompt = f"""
                You are a DSL program synthesizer. Generate around {self.number_program_candidates} transformation programs in JSON format.
                Notes: We always zero-based indexing. All row and column indices in this document are zero-based, When we refer to "row 4", we mean the row at index 4
                Available Operations:
                {filtered_docs_str}

                Input Grid: {input_grid.tolist()}
                Output Grid: {output_grid.tolist()}
                Analysis: {analysis_summary}
                                
                - Follow the provided analysis closely. At least half of the generated programs must reflect the main hypothesis from the analysis.
                - Prefer simple atomic operations unless multiple steps are clearly required.
                - Do not wrap a single operation inside a `sequence` unnecessarily.
                - Avoid using identity transformations or mask combinators unless they serve a purpose.
                - Avoid generating programs that cancel each other out (e.g., swapping rows back and forth).
                - If unsure, prioritize minimalism and correctness over complexity.

                Return each program as a JSON object with this structure:
                {{"operation": "operation_name", "parameters": {{...}}}}

                For operations requiring inner commands, nest them:
                {{"operation": "repeat_grid", "parameters": {{"inner_command": {{"operation": "identity", "parameters": {{}}}}, "vertical_repeats": 2, "horizontal_repeats": 3}}}}
                
                Parameter Types (to avoid breaking parsing):
                - All numeric values must be numbers, not strings.
                - Boolean values must be `true` or `false`, not strings.
                - Null values must be `null`, not `"None"` or other strings.

                IMPORTANT: Return only raw JSON objects, one per line. 
                DO NOT use markdown code blocks, backticks, or any formatting.
                DO NOT add explanations or comments.
                Each line must be a valid JSON object that starts with {{ and ends with }}.
                Ensure all keys are double-quoted.
                Avoid trailing commas.

                Example output format:
                {{"operation": "flip_h", "parameters": {{}}}}
                {{"operation": "repeat_grid", "parameters": {{"inner_command": {{"operation": "identity", "parameters": {{}}}}, "vertical_repeats": 2, "horizontal_repeats": 2}}}}
            """
            try:
                response = self.llm(prompt).strip()
                self.logger.info("Program generation completed.")
                self.logger.debug(f"Raw generation response:\n{response}")

                candidates = [line.strip() for line in response.split('\n') if line.strip()]
                self.logger.info(f"Generated {len(candidates)} program candidates.")
                json_candidates = self._filter_valid_json_dicts(candidates)

                return json_candidates
            except Exception as e:
                self.logger.error(f"Error during program generation: {e}", exc_info=True)
                return []
        
    def _generate_program_candidates(self, input_grid: np.ndarray, output_grid: np.ndarray,
                                    analysis_summary: str) -> List[str]:
        """
        Main entry point: two-stage synthesis pipeline.
        """
        self.logger.info("Starting two-stage program synthesis...")
        self.logger.debug("Stage 1: Filtering relevant operations")

        # Stage 1: Filter relevant operations
        relevant_ops = self._filter_relevant_operations(input_grid, output_grid, analysis_summary)

        if not relevant_ops:
            self.logger.warning("No relevant operations found. Falling back to all operations.")
            try:
                full_ops = json.loads(self._build_operation_documentation())
                relevant_ops = [op["operation name"] for op in full_ops]
                self.logger.info(f"Fallback to full operation list: {relevant_ops}")
            except Exception as e:
                self.logger.error(f"Failed to fallback to full operation list: {e}")
                return []

        self.logger.debug(f"Stage 2: Generating programs with filtered operations: {relevant_ops}")
        # Stage 2: Generate programs using only relevant operations
        candidates = self._generate_filtered_programs(input_grid, output_grid, relevant_ops, analysis_summary)

        if not candidates:
            self.logger.warning("No valid program candidates were generated.")

        self.logger.info("Program synthesis complete.")
        return candidates

    def parse_and_validate(self, program_str: str, input_grid: np.ndarray, 
                         output_grid: np.ndarray) -> Optional[Tuple[AbstractTransformationCommand, float]]:
        """Attempts to parse and validate a single program candidate.
        
        Args:
            program_str: DSL program string to evaluate
            input_grid: Input grid to test against
            output_grid: Expected output grid
            
        Returns:
            Tuple of (program, score) if valid, None otherwise
        """
        try:
            # Parse the program string into an actual command object
            program = self.synthesizer.interpreter.parse_program(program_str)
            
            # Execute the program
            candidate_output = self.synthesizer.interpreter.execute_program(program, input_grid)
            
            # Verify shape matches
            if candidate_output.shape != output_grid.shape:
                return None
                
            # Calculate score
            matching_cells = np.sum(candidate_output == output_grid)
            score = matching_cells / output_grid.size
            
            return (program, score, program_str)
            
        except Exception as e:
            self.logger.debug(f"Program validation failed for '{program_str}': {str(e)}")
            return None

    def explain_program(self, program_str: str) -> str:
        try:
            operation_docs = self._build_operation_documentation()
            prompt = f"""
                You are given the following DSL operations and a program written in that DSL.
                Your task is to briefly explain what the program does in simple terms.
                Rules:
                - Be concise.
                - Do not describe each operation step-by-step.
                - Do not include examples.
                - Output only the explanation.
                Available Operations:
                {operation_docs}

                Program:
                {program_str}

                Explanation:
                """
            return self.llm(prompt).strip()
        except Exception as e:
            return f"[Error explaining program: {str(e)}]"
    
    def _validate_programs(self, candidate_strings, input_grid, output_grid):
        """Validates a list of candidate strings and returns only those that match output."""
        valid_batch = []
        for program_str in candidate_strings:
            result = self.parse_and_validate(program_str, input_grid, output_grid)
            if result:
                valid_batch.append(result)
        return valid_batch

    def _build_enhanced_analysis(self, original_summary, stored_candidates):
        """Builds enhanced prompt from previous failures to guide next generation."""
        failed_examples = '\n'.join(f'- {c}' for c in stored_candidates)
        return (
            f"{original_summary}\n\n"
            f"Previous unsuccessful attempts (learn from these):\n"
            f"{failed_examples}"
        )
    
    def _execute_top_program(self, program, input_grid):
        """Safely executes the top transformation on the input grid."""
        try:
            return program.execute(input_grid)
        except Exception as e:
            raise RuntimeError(f"Failed to execute program: {str(e)}")
    
    def _format_output_on_failure(self, stored_candidates):
        """Returns structured failure response."""
        return {
            "success": False,
            "error": f"Failed to generate a valid program after max attempts",
            "stored_candidates": stored_candidates
        }

    def _format_output(self, unique_programs, input_grid, top_k=3):
        """Formats final structured output including top program and alternatives."""
        sorted_programs = sorted(
            unique_programs.values(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        top_program, top_score, top_str, top_explanation = sorted_programs[0]
        result_grid = self._execute_top_program(top_program, input_grid)

        alternatives = [
            {
                "program": p,
                "score": float(s),
                "program_str": str(ps),
                "explanation": e
            }
            for p, s, ps, e in sorted_programs[1:]
        ]

        return {
            "success": True,
            "result_grid": result_grid.tolist(),
            "program": top_program,
            "score": float(top_score),
            "program_str": top_str,
            "explanation": top_explanation,
            "alternatives": alternatives
        }
   
    def _deduplicate_and_enrich(self, validated_programs):
        """Deduplicates programs and enriches with explanation."""
        unique_programs = {}
        for program, score, prg_str in validated_programs:
            key = str(program)
            explanation = self.explain_program(prg_str)
            if key not in unique_programs or score > unique_programs[key][1]:
                unique_programs[key] = (program, score, prg_str, explanation)
        return unique_programs

    def synthesize(self, input_grid: np.ndarray, output_grid: np.ndarray, 
                analysis_summary: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Full synthesis pipeline combining LLM generation, program validation,
        deduplication, execution, and structured output formatting.
        
        Returns:
            Dict[str, Any]: Structured result with success flag, result grid,
                            top program, alternatives, etc.
        """
        stored_candidates = []
        validated_programs = []
        max_attempts = 10
        found_valid = False

        for attempt in range(max_attempts):
            # Step 1: Generate candidate programs
            if attempt == 0:
                candidate_strings = self._generate_program_candidates(input_grid, output_grid, analysis_summary)
            else:
                enhanced_analysis = self._build_enhanced_analysis(analysis_summary, stored_candidates)
                candidate_strings = self._generate_program_candidates(input_grid, output_grid, enhanced_analysis)

            stored_candidates.extend(candidate_strings)

            # Step 2: Validate generated programs
            batch_validated = self._validate_programs(candidate_strings, input_grid, output_grid)
            validated_programs.extend(batch_validated)

            if batch_validated:
                found_valid = True

            if found_valid and len(validated_programs) >= top_k:
                break  # Stop early if we already have enough top-quality candidates

        if not found_valid:
            return self._format_output_on_failure(stored_candidates)

        # Step 3: Deduplicate and enrich with explanations
        unique_programs = self._deduplicate_and_enrich(validated_programs)

        # Step 4: Format final output
        formated_output = self._format_output(unique_programs, input_grid, top_k=top_k)
        
        return formated_output
