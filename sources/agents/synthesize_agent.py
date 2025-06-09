from typing import List, Tuple, Optional
import numpy as np
import logging
import json
import inspect
from core.dsl_nodes import AbstractTransformationCommand
from core.transformation_factory import TransformationFactory


class SynthesizeAgent:
    def __init__(self, llm, synthesizer, logger: logging.Logger = None):
        """
        Args:
            llm: Language model interface for generating program candidates
            synthesizer: SynthesisEngine instance for validating programs
            logger: Optional logger instance
        """
        self.llm = llm
        self.synthesizer = synthesizer
        self.logger = logger or logging.getLogger(__name__)


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
            You are an operation filterer. Based on the analysis, identify which operations are most relevant for transforming the input grid to match the output grid.

            Available Operations:
            {operation_docs}

            Input Grid: {input_grid.tolist()}
            Output Grid: {output_grid.tolist()}

            Analysis Summary:
            {analysis_summary}

            Instructions:
            - Use the analysis as one possible interpretation, but also consider alternative transformations that could produce the same result.
            - Include any operation that could reasonably be part of a transformation — even if it's only useful as an inner command (e.g., identity).
            - Only exclude operations that are clearly not applicable (e.g., those that have no effect on this grid or operate on irrelevant axes).
            - Return only a list of operation names that could explain the transformation.
            - Exclude any operations that are clearly not applicable.
            - Wrap your answer in triple backticks like this:
              ```json
              ["swap_rows_or_columns", "apply_to_row", "mask_combinator"]
              ```

            Example response:
            ```json
            ["swap_rows_or_columns", "apply_to_row", "flip_h"]
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
            self.logger.info(f"Filtered operations: {filtered_ops}")
            return filtered_ops
        except Exception as e:
            self.logger.error(f"Error during operation filtering: {e}", exc_info=True)
            return []
        

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
                You are a DSL program synthesizer. Generate 5–10 transformation programs in JSON format.

                Available Operations:
                {filtered_docs_str}

                Input Grid: {input_grid.tolist()}
                Output Grid: {output_grid.tolist()}
                Analysis: {analysis_summary}

                Generate transformation programs that match the Input → Output mapping.
                Some should closely follow the provided analysis.
                Others may propose alternative interpretations — as long as they result in the correct output.

                Do not propose duplicated programs.

                Return each program as a JSON object with this structure:
                {{"operation": "operation_name", "parameters": {{...}}}}

                For operations requiring inner commands, nest them:
                {{"operation": "repeat_grid", "parameters": {{"inner_command": {{"operation": "identity", "parameters": {{}}}}, "vertical_repeats": 2, "horizontal_repeats": 3}}}}

                IMPORTANT: Return only raw JSON objects, one per line. 
                DO NOT use markdown code blocks, backticks, or any formatting.
                DO NOT add explanations or comments.
                Each line must be a valid JSON object that starts with {{ and ends with }}.

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
                return candidates
            except Exception as e:
                self.logger.error(f"Error during program generation: {e}", exc_info=True)
                return []
        

    def generate_program_candidates(self, input_grid: np.ndarray, output_grid: np.ndarray,
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
            
            return (program, score)
            
        except Exception as e:
            self.logger.debug(f"Program validation failed for '{program_str}': {str(e)}")
            return None

    def synthesize(self, input_grid: np.ndarray, output_grid: np.ndarray, 
                 analysis_summary: str, top_k: int = 3) -> List[Tuple[AbstractTransformationCommand, float]]:
        """Full synthesis pipeline combining LLM generation and program validation.
        
        Args:
            input_grid: Input grid as numpy array
            output_grid: Desired output grid as numpy array
            analysis_summary: Text analysis of the transformation
            top_k: Number of top programs to return
            
        Returns:
            List of (program, score) tuples sorted by score descending
        """
        # Step 1: Generate initial candidates via LLM
        candidate_strings = self.generate_program_candidates(input_grid, output_grid, analysis_summary)
        stored_candidates = []  # To keep track of all candidate strings tried

        max_attempts = 10
        validated_programs = []
        found_valid = False

        for attempt in range(max_attempts):
            # On first attempt, use the initial candidates
            # On subsequent attempts, generate new candidates
            if attempt > 0:
                # Prepare enhanced analysis with previous attempts
                enhanced_analysis = (
                    f"{analysis_summary}\n\n"
                    f"Previous unsuccessful attempts with outputed shape mismatch (learn from these):\n"
                    f"{'\n'.join(f'- {candidate}' for candidate in stored_candidates)}"
                )
                candidate_strings = self.generate_program_candidates(input_grid, output_grid, enhanced_analysis)
            
            # Store all candidate strings in memory
            stored_candidates.extend(candidate_strings)
            
            # Validate all candidates in this batch
            for program_str in candidate_strings:
                result = self.parse_and_validate(program_str, input_grid, output_grid)
                if result:
                    validated_programs.append(result)
                    found_valid = True
            
            # If we found at least one valid program, break out of the retry loop
            if found_valid:
                break

        if not found_valid:
            raise ValueError(f"Failed to generate a valid program after {max_attempts} attempts. Stored candidates: {stored_candidates}")

        # Step 3: Add synthesized programs from the engine
        synthesized_programs = self.synthesizer.synthesize_matching_programs(
            input_grid, 
            output_grid,
            top_k=top_k
        )
        validated_programs.extend(synthesized_programs)
        
        # Step 4: Deduplicate and sort by score
        unique_programs = {}
        for program, score in validated_programs:
            program_key = str(program)  # Simple deduplication
            if program_key not in unique_programs or score > unique_programs[program_key][1]:
                unique_programs[program_key] = (program, score)
        
        # Get top programs sorted by score
        top_programs = sorted(unique_programs.values(), key=lambda x: x[1], reverse=True)[:top_k]
        
        self.logger.info(f"Found {len(top_programs)} valid programs (top score: {top_programs[0][1] if top_programs else 0})")
        return top_programs