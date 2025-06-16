import logging
from typing import Optional, List, Dict, Any
from core.llm import LLMClient


class QCAgent:
    def __init__(self, llm: LLMClient, logger: logging.Logger = None):
        self.llm: LLMClient = llm
        self.logger = logger or logging.getLogger(__name__)
    
    def verify(
        self,
        analysis_description: str,
        generated_program: Dict[str, Any],
        input_grid: Optional[List[List[int]]] = None,
        output_grid: Optional[List[List[int]]] = None,
        hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Uses the LLM to validate whether the generated program aligns with the analysis description.
        
        Args:
            analysis_description: Natural language description of expected transformation
            generated_program: Dictionary representation of the generated program
            input_grid: Optional input grid for context
            output_grid: Optional expected output grid for validation
            hint: Optional extra instruction or feedback
            
        Returns:
            Dict containing 'is_valid', 'reasoning', and 'score'
        """
        try:
            # Convert program dict to string representation
            program_str = generated_program.get('program_str', str(generated_program))
            explanation = generated_program.get('explanation', '')

            # Build prompt for LLM
            prompt = self._build_verification_prompt(
                analysis=analysis_description,
                program=program_str,
                explanation=explanation,
                input_grid=input_grid,
                output_grid=output_grid,
                hint=hint
            )

            # Query LLM
            raw_response = self.llm(prompt)

            # Parse and return result
            result = self._parse_llm_response(raw_response)
            return result

        except Exception as e:
            self.logger.error(f"QC verification failed: {e}", exc_info=True)
            return {
                "is_valid": False,
                "reasoning": f"Verification error: {e}",
                "score": 0.0
            }

    def _build_verification_prompt(
        self,
        analysis: str,
        program: str,
        explanation: str,
        input_grid: Optional[List[List[int]]],
        output_grid: Optional[List[List[int]]],
        hint: Optional[str]
    ) -> str:
        prompt = (
            "You are a quality control assistant. Your job is to determine whether the given program correctly implements "
            "the transformation described in the analysis.\n\n"
            "Here's what you need to evaluate:\n\n"
            f"ANALYSIS DESCRIPTION:\n{analysis}\n\n"
            f"PROGRAM (in JSON format):\n{program}\n\n"
            f"EXPLANATION OF PROGRAM:\n{explanation}\n\n"
        )

        if input_grid is not None:
            prompt += f"INPUT GRID:\n{input_grid}\n\n"

        if output_grid is not None:
            prompt += f"EXPECTED OUTPUT GRID:\n{output_grid}\n\n"

        if hint:
            prompt += f"HINT/ADDITIONAL INSTRUCTIONS:\n{hint}\n\n"

        prompt += (
            "Now answer the following question strictly in JSON format:\n"
            "{\n"
            '  "is_valid": true or false,\n'
            '  "reasoning": "Why do you think it does or does not match?",\n'
            '  "score": "Confidence score between 0.0 and 1.0"\n'
            "}\n"
            "Only return the JSON, no extra text."
        )

        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parses the LLM's response into a structured QC result.
        In production, this could use JSON parsing + schema validation.
        """
        try:
            # Simulate simple parsing here; real version would parse actual JSON
            # For now, assume LLM returns valid JSON structure
            import json
            result = json.loads(response.strip())
            return {
                "is_valid": bool(result.get("is_valid", False)),
                "reasoning": str(result.get("reasoning", "No reasoning provided")),
                "score": float(result.get("score", 0.0))
            }
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return {
                "is_valid": False,
                "reasoning": "Failed to parse response from LLM",
                "score": 0.0
            }