import numpy as np
import json
from abc import ABC
from core.features_analysis import ProblemAnalyzer
from  core.llm import LLMClient

class BaseAnalyzeAgent(ABC):
    def __init__(self, llm: LLMClient = None, analyzer=None):
        self.llm = llm
        self.analyzer = analyzer or ProblemAnalyzer()

    def _format_grid(self, grid: np.ndarray) -> str:
        return "\n".join(" ".join(str(int(cell)) for cell in row) for row in grid)

    def _clean_feature_dict(self, d):
        if isinstance(d, dict):
            return {k: self._clean_feature_dict(v) for k, v in d.items()}
        elif isinstance(d, (np.floating, np.integer)):
            return d.item()
        elif isinstance(d, list):
            return [self._clean_feature_dict(i) for i in d]
        else:
            return d

class SingleGridAnalyzeAgent(BaseAnalyzeAgent):
    def analyze(self, input_grid, output_grid, hint=None):
        analysis_result = self.analyzer.analyze(input_grid, output_grid)
        summary = self._compose_summary(analysis_result, hint, input_grid, output_grid)
        return self.llm.generate(summary)

    def _compose_summary(self, analysis_result, hint, input_grid=None, output_grid=None):
        instruction = (
            """
            You are an ARC puzzle detective and strategist. Your job is to examine the input and output grids like clues in a mystery.

            The transformation could be simple (like pattern repetition), complex (like value swapping), or deeply abstract (like simulated behavior or rule application).

            Think like a scientist observing data:
            
            1. What seems to be the base structure or repeating pattern in the input?
            - Is there symmetry?
            - Are values arranged in predictable sequences?
            - Does it look like a blank canvas before some rule was applied?

            2. Where does the output deviate from that structure?
            - Show what changed, in detail.
            - Be precise: which rows/columns? what values? how did they change?

            3. Could this change be explained in multiple ways?
            - Give 1–3 different interpretations:
                - One most likely explanation
                - One alternative view
                - One speculative or exploratory idea

            4. What kinds of transformations definitely don't fit?
            - Rule out flawed assumptions early

            5. If you were to test these ideas, how would you do it?
            - Describe expected results if each theory were true

            Don't propose DSL commands or programs — just describe what happened and what might explain it.

            Stay curious and open-minded. If multiple explanations fit, describe them all.
            Your goal is to help the next step find the right path — not to guess the final answer.
            """
        )

        summary = ""
        if input_grid is not None and output_grid is not None:
            summary += f"Input Grid:\n{self._format_grid(input_grid)}\n\n"
            summary += f"Output Grid:\n{self._format_grid(output_grid)}\n\n"

        summary += "Analysis Summary:\n"
        clean_features = self._clean_feature_dict(analysis_result.features)
        for feature_type, features in clean_features.items():
            summary += f"\n{feature_type}:\n{features}\n"

        if hint:
            summary += f"\nHints: {hint}\n"

        return instruction + summary
    

class MultiGridGeneralizingAgent(BaseAnalyzeAgent):
    def analyze_multi(self, examples, train_results, prompt_hint=None):
        """
        Analyzes multiple input-output-program triplets to find a general pattern.
        
        Args:
            examples: List of {'input': [[...]], 'output': [[...]]}
            train_results: From solving each puzzle individually
            prompt_hint: Optional extra instructions
            
        Returns:
            dict: Unified transformation rule + explanation
        """
        # Step 1: Extract programs and features
        structured_input = self._prepare_structured_input(examples, train_results)

        # Step 2: Build rich prompt with all examples
        prompt = self._build_generalization_prompt(structured_input, prompt_hint)

        # Step 3: Ask LLM for generalized pattern
        response = self.llm.generate(prompt)

        # Step 4: Parse and return result
        return self._parse_llm_response(response)

    def _prepare_structured_input(self, examples, train_results):
        """Build list of {input, output, program, features}"""
        main_programs = [
            train_results[puzzle_key][0]['program']
            for puzzle_key in sorted(train_results.keys())
        ]
        features_list = [
            self.analyzer.analyze(np.array(example['input']), np.array(example['output']))
            for example in examples
        ]

        return [
            {
                "input": np.array(example["input"]),
                "output": np.array(example["output"]),
                "program": program,
                "features": features
            }
            for example, program, features in zip(examples, main_programs, features_list)
        ]

    def _build_generalization_prompt(self, structured_input, prompt_hint=None):
        """
        Builds a prompt that includes:
        - All input/output grids
        - Their individual programs
        - Extracted features
        - Request for generalized transformation
        """
        prompt = """
            You are given multiple ARC grid transformations. Your task is to:
            
            1. Identify what's consistent across all examples
            2. Describe the general transformation pattern
            3. Suggest a single DSL program that would work for all inputs
            
            Each example contains:
            - Input grid
            - Output grid
            - Program used
            - Extracted features
            
            Provide your final answer in JSON format:
            {
              "pattern_description": "...",
              "generalized_program": { ... },
              "confidence": 0.95,
              "common_features": { ... }
            }
        """

        # Add each example
        for idx, item in enumerate(structured_input):
            prompt += f"\n\n--- Example {idx + 1} ---\n"
            prompt += f"Input:\n{self._format_grid(item['input'])}\n"
            prompt += f"Output:\n{self._format_grid(item['output'])}\n"
            prompt += f"Program: {item['program']}\n"
            prompt += "Features:\n"
            for ft, val in self._clean_feature_dict(item['features'].features).items():
                prompt += f"- {ft}: {val}\n"

        if prompt_hint:
            prompt += f"\nHint: {prompt_hint}"

        return prompt

    def _parse_llm_response(self, response: str) -> dict:
        """Parse and validate LLM's JSON-formatted response"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse LLM response", "raw": response}
        

class AnalyzeAgent:
    def __init__(self, mode: str = "single", llm=None, analyzer=None):
        if mode == "single":
            self._impl = SingleGridAnalyzeAgent(llm=llm, analyzer=analyzer)
        elif mode == "multi":
            self._impl = MultiGridGeneralizingAgent(llm=llm, analyzer=analyzer)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def analyze(self, *args, **kwargs):
        return self._impl.analyze(*args, **kwargs)

    def analyze_multi(self, *args, **kwargs):
        return self._impl.analyze_multi(*args, **kwargs)

# class AnalyzeAgent:
#     def __init__(self, llm, analyzer=None):
#         self.llm = llm
#         self.analyzer = analyzer or ProblemAnalyzer()

#     def analyze(self, input_grid, output_grid, hint=None):
#         analysis_result = self.analyzer.analyze(input_grid, output_grid)
#         summary = self._compose_summary(analysis_result, hint, input_grid, output_grid)
#         return self.llm(summary)

#     def _format_grid(self, grid: np.ndarray) -> str:
#         return "\n".join(" ".join(str(int(cell)) for cell in row) for row in grid)

#     def _clean_feature_dict(self, d):
#         if isinstance(d, dict):
#             return {k: self._clean_feature_dict(v) for k, v in d.items()}
#         elif isinstance(d, (np.floating, np.integer)):
#             return d.item()
#         elif isinstance(d, list):
#             return [self._clean_feature_dict(i) for i in d]
#         else:
#             return d

#     def _compose_summary(self, analysis_result, hint, input_grid=None, output_grid=None):
#         instruction = (
#             """
#             You are an ARC puzzle detective and strategist. Your job is to examine the input and output grids like clues in a mystery.

#             The transformation could be simple (like pattern repetition), complex (like value swapping), or deeply abstract (like simulated behavior or rule application).

#             Think like a scientist observing data:
            
#             1. What seems to be the base structure or repeating pattern in the input?
#             - Is there symmetry?
#             - Are values arranged in predictable sequences?
#             - Does it look like a blank canvas before some rule was applied?

#             2. Where does the output deviate from that structure?
#             - Show what changed, in detail.
#             - Be precise: which rows/columns? what values? how did they change?

#             3. Could this change be explained in multiple ways?
#             - Give 1–3 different interpretations:
#                 - One most likely explanation
#                 - One alternative view
#                 - One speculative or exploratory idea

#             4. What kinds of transformations definitely don't fit?
#             - Rule out flawed assumptions early

#             5. If you were to test these ideas, how would you do it?
#             - Describe expected results if each theory were true

#             Don't propose DSL commands or programs — just describe what happened and what might explain it.

#             Stay curious and open-minded. If multiple explanations fit, describe them all.
#             Your goal is to help the next step find the right path — not to guess the final answer.
#             """
#         )

#         summary = ""
#         if input_grid is not None and output_grid is not None:
#             summary += f"Input Grid:\n{self._format_grid(input_grid)}\n\n"
#             summary += f"Output Grid:\n{self._format_grid(output_grid)}\n\n"

#         summary += "Analysis Summary:\n"
#         clean_features = self._clean_feature_dict(analysis_result.features)
#         for feature_type, features in clean_features.items():
#             summary += f"\n{feature_type}:\n{features}\n"

#         if hint:
#             summary += f"\nHints: {hint}\n"

#         return instruction + summary
