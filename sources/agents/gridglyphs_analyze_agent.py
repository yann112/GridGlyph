# arc_solver/agents/grid_glyph_analyzer_agent.py

import logging
from typing import Optional, List, Dict, Any
from agents.agents_utils import SymbolicGridMapper


class GridGlyphAnalyzerAgent:
    def __init__(self, llm: Any, logger: logging.Logger = None):
        """
        Analyzes visual puzzles by mapping numeric grids into symbolic representations
        and prompting an LLM to infer transformation logic.
        
        Args:
            llm: LLMClient instance
            analyzer: Optional feature analyzer (not used yet)
            logger: Optional custom logger
        """
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze(
        self,
        puzzle_data: Optional[Dict[str, Any]] = None,
        input_grid: Optional[List[List[int]]] = None,
        output_grid: Optional[List[List[int]]] = None,
        test_grid: Optional[List[List[int]]] = None,
        glyphset_ids: List[str] = None,
        n_variants_per_set: int = None,
        mode: str = "python_function"
    ) -> Dict[str, Any]:
        """
        Analyze the given puzzle using symbolic representation and LLM prompting.
        
        Args:
            puzzle_data: Full ARC-style puzzle dict (with train/test)
            input_grid: Fallback input grid
            output_grid: Fallback output grid
            test_grid: Test input to apply transformation
            glyphset_ids: Symbol sets to try
            n_variants_per_set: How many symbolic variants per set
            mode: Response type from LLM ('python_function', 'rule_only', 'rule_and_function')
            
        Returns:
            Dict containing analysis results including best attempt, raw responses, mapped variants
        """
            
        mapper = SymbolicGridMapper()

        variants = mapper.generate_n_variants(
            puzzle_data,
            n=n_variants_per_set,
        )
    
        prompt = """Below are several puzzle examples.\n"""
        # prompt += """All use abstract symbols.\n"""
        # prompt += """The transformation is the *same* in all examples.\n"""
        # prompt += """Study the input/output pairs carefully and extract the underlying rule.\n\n"""
        # Add variants
        prompt += mapper.format_variants_list(variants, include_variant_headers=True)
        prompt += """The transformation are not exactly the same for each puzzle explain the transformation from input to output"""
        # Final instruction
        # prompt += """
        # Now apply the same transformation to the final test input and return only the Python function that performs it.
        # Do not add any extra text — return only the code.
        # Use clean, generalizable logic — avoid hardcoding values.
        # Make sure the function works across symbol sets.
        # """
        # prompt += """Now respond with the following instructions:
        # 1. Ignore what the symbols might represent in human language
        # 2. Treat each symbol as an abstract visual element
        # 3. Study how the input becomes the output across all variants
        # 4. Return only the Python function that performs this transformation
        # 5. Make sure it works across different symbol sets (not tied to one script)

        # Do not explain — return only the code.
        # Make sure the function transforms grids correctly regardless of what the symbols mean.
        # Use general logic — avoid hardcoding glyph behavior.""" 
                      
        raw_response = self.llm(prompt)

        return {
            "raw_response": raw_response,
        }

