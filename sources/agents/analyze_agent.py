import numpy as np
from core.features_analysis import ProblemAnalyzer

class AnalyzeAgent:
    def __init__(self, llm, analyzer=None):
        self.llm = llm
        self.analyzer = analyzer or ProblemAnalyzer()

    def analyze(self, input_grid, output_grid, hint=None):
        analysis_result = self.analyzer.analyze(input_grid, output_grid)
        summary = self._compose_summary(analysis_result, hint, input_grid, output_grid)
        return self.llm(summary)

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
