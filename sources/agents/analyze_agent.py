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
        instruction = ("""
            As an ARC puzzle strategist:

            1. FIRST identify the most obvious whole-grid transformation
            2. THEN examine where the output deviates from this simple pattern
            3. FINALLY propose minimal operations to explain deviations

            For each step:
            - Quantify explanatory power (% of output explained)
            - Describe deviations precisely (location and nature)
            - Suggest specific, testable operations

            Format response as:
            1. Primary Pattern: [description] (confidence: X%, coverage: Y%)
            2. Key Deviation: [location and type] 
            3. Suggested Operation: [concrete transform]
            4. Verification: [how to test this hypothesis]\n\n
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
