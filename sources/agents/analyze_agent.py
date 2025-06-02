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
            "You are an expert ARC puzzle analyzer. "
            "Given the analysis results below comparing the input and output grids, "
            "please explain in clear, concise language the main transformations, patterns, "
            "or relationships that led from input to output. "
            "Focus on relevant features and suggested transforms. "
            "If there are hints, incorporate them to refine your explanation.\n\n"
        )

        summary = ""
        if input_grid is not None and output_grid is not None:
            summary += f"Input Grid:\n{self._format_grid(input_grid)}\n\n"
            summary += f"Output Grid:\n{self._format_grid(output_grid)}\n\n"

        summary += "Analysis Summary:\n"
        clean_features = self._clean_feature_dict(analysis_result.features)
        for feature_type, features in clean_features.items():
            summary += f"\n{feature_type}:\n{features}\n"

        if analysis_result.possible_transforms:
            transforms_conf = list(zip(analysis_result.possible_transforms, analysis_result.confidence_scores))
            summary += f"\nSuggested transforms (confidence): {transforms_conf}\n"

        if hint:
            summary += f"\nHints: {hint}\n"

        return instruction + summary
