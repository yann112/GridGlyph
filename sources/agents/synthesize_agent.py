from typing import List, Tuple, Optional
import numpy as np
import logging
from core.dsl_nodes import AbstractTransformationCommand

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

    def generate_program_candidates(self, input_grid: np.ndarray, output_grid: np.ndarray, 
                                 analysis_summary: str) -> List[str]:
        """Generates multiple DSL program candidates via LLM.
        
        Args:
            input_grid: Input grid as numpy array
            output_grid: Desired output grid as numpy array
            analysis_summary: Text analysis of the transformation
            
        Returns:
            List of candidate program strings in DSL format
        """
        prompt = f"""
        Given these grids and analysis, suggest 3-5 DSL programs that could transform:
        
        Input Grid: 
        {input_grid}
        
        Output Grid: 
        {output_grid}
        
        Analysis: 
        {analysis_summary}
        
        Return ONLY valid DSL commands, one per line. Example:
        repeat_grid(identity(), 3, 3)
        flip_h(identity())
        rotate90(identity())
        """
        response = self.llm(prompt).strip()
        candidates = [line.strip() for line in response.split('\n') if line.strip()]
        self.logger.debug(f"Generated {len(candidates)} program candidates: {candidates}")
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
        
        # Step 2: Validate all candidates
        validated_programs = []
        for program_str in candidate_strings:
            result = self.parse_and_validate(program_str, input_grid, output_grid)
            if result:
                validated_programs.append(result)
        
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