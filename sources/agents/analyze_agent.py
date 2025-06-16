import numpy as np
import json
from abc import ABC

from agents.agents_utils import MultiGridFeatureCollector
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
        analysis_result = self.analyzer.analyze(input_grid, output_grid, mode='io')
        summary = self._compose_summary(analysis_result, hint, input_grid, output_grid)
        return self.llm(summary)

    def _compose_summary(self, analysis_result, hint, input_grid=None, output_grid=None):
        instruction = (
            """
            You are an ARC puzzle detective and strategist. Your job is to examine the input and output grids like clues in a mystery.

            The transformation could be simple (like pattern repetition), complex (like value swapping), or deeply abstract (like simulated behavior or rule application).

            Think like a scientist observing data:
            
            0. **Rules for Analysis:**
            - This analysis uses zero-based indexing. All row and column indices in this document are zero-based (e.g., "row 2" means the row at index 2).
            
            1. **Observed Base Structure and Patterns in Input:**
            - Is there symmetry?
            - Are values arranged in predictable sequences?
            - Does it look like a blank canvas before some rule was applied?

            2. **Detailed Deviations from Input to Output:**
            - Show precisely what changed.
            - Be specific: identify which rows/columns, what values, and how they transformed (e.g., "Input Row X [A,B,C] transformed to Output Row X [C,B,A]").
            - Highlight any repeating patterns in the changes.

            3. **Plausible Transformation Hypotheses:**
            - Propose 1–3 distinct and plausible explanations for the observed changes. Each explanation should be a high-level concept (e.g., "horizontal flip," "row swap," "color change rule," "pattern repetition").
            - Clearly state **one most direct and likely explanation** that directly maps to the observed changes.
            - Provide **one alternative view** if another distinct high-level transformation could also yield the same result.
            - (Optional/Speculative) Add **one exploratory idea** for very abstract or complex transformations if simple ones don't fully capture it.

            4. **Ruled-Out Transformations:**
            - Identify any common transformation types that definitely do not fit the observed changes, and why.

            5. **Key Transformation Summary (for downstream use):**
            - State the most concise and accurate description of the core transformation observed. This should be a single, clear sentence or phrase summarizing the main change(s).

            **Important Directives:**
            - Do NOT propose DSL commands or specific programs; focus solely on describing what happened and what might explain it.
            - Be curious and open-minded. Describe all explanations that genuinely fit.
            - Your goal is to provide a comprehensive and accurate analysis that guides subsequent steps towards the correct path.
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
    def analyze(self, data=None, train_results=None, prompt_hint=None, analysis_mode="both"):
        """
        Analyzes multiple grid transformations to find a general pattern.
        
        Args:
            examples: List of {'input': [[...]], 'output': [[...]]} training examples
            train_results: Results from individual puzzle solving
            prompt_hint: Optional extra instructions or feedback
            analysis_mode: One of "both", "features_only", "results_only"
                - "both": Use features and train results
                - "features_only": Use only raw input-output comparisons
                - "results_only": Use only program chains from train results
                
        Returns:
            dict: Unified transformation rule + explanation
        """
        # Validate input based on mode
        if analysis_mode in ("both", "features_only") and data is None:
            raise ValueError("examples must be provided for 'both' or 'features_only' analysis")
        if analysis_mode in ("both", "results_only") and train_results is None:
            raise ValueError("train_results must be provided for 'both' or 'results_only' analysis")

        # Step 1: Extract features if needed
        features_analysis = None
        if analysis_mode in ("both", "features_only"):
            collector = MultiGridFeatureCollector(analyzer=self.analyzer)
            extracted_features = collector.extract_features_from_task(data)
            features_analysis_prompt = self._build_feature_analysis_prompt(data, extracted_features)
            features_analysis = self.llm(features_analysis_prompt)
        
        # Step 2: Build prompt with correct context
        prompt = self._build_generalization_prompt(
            data=data,
            features_analysis=features_analysis,
            train_results=train_results,
            prompt_hint=prompt_hint,
            analysis_mode=analysis_mode
        )

        # Step 3: Send to LLM
        raw_output = self.llm(prompt)
        
        # Step 4: Parse response
        return self._parse_llm_response(raw_output)

    def _clean_train_results(self, train_results):
        """Removes unserializable objects and keeps only meaningful keys"""
        cleaned = {}
        for puzzle_key, results in train_results.items():
            cleaned[puzzle_key] = []
            for result in results:
                main_result = {
                    'score': result.get('score'),
                    'program_str': result.get('program_str'),
                    'explanation': result.get('explanation')
                }
                cleaned[puzzle_key].append(main_result)
        return cleaned
    
    def _build_feature_analysis_prompt(self,data, extracted_features):
        """
        Builds the prompt for analyzing feature data only.
        """
        prompt = "You are an ARC puzzle detective and strategist.\n\n"
        prompt += "=== DATA ===\n"
        prompt += f"{data}\n"
        # Include feature analysis to provide context
        prompt += "=== FEATURE DATA ===\n"
        prompt += """
        You are given a feature analysis of multiple grid transformations.

        This data includes comparisons between:
        - Train input grids (original)
        - Train output grids (transformed)
        - Test inputs vs train inputs

        Your task is to provide a concise, high-level interpretation based *strictly* on the provided data. You are the first step in a complex system; therefore, it is crucial to **only state what can be directly inferred or is explicitly mentioned**. Do not make assumptions or infer rules beyond what the data strongly supports. Highlight any areas where the provided data might suggest ambiguity or contradiction.

        **Analyze the provided data by:**
        1.  **Input Commonalities:** Identify **common characteristics and consistent patterns** observed across *all* input grids.
        2.  **Output Commonalities:** Identify **common characteristics and consistent patterns** observed across *all* output grids.
        3.  **Input-Output Relationship:** Identify **direct and consistent relationships or transformations** that map input grids to their corresponding output grids. Focus on observable changes in properties like dimensions, color usage, component characteristics, and symmetry.

        **Structure your analysis with the following headings:**
        -   `### High-Level Interpretation`
        -   `#### Input Commonalities`
        -   `#### Output Commonalities`
        -   `#### Input-Output Transformations`
        -   `### Summary of Certain Rules`
        -   `### Identified Ambiguities/Contradictions`
        -   `### Next Steps`

        Prioritize **accuracy, direct inference, and conciseness**. Explicitly note any points where the data provided appears contradictory or implies missing information for a complete rule.
        """
        prompt += json.dumps(extracted_features, indent=2) + "\n\n"
        return prompt

    def _build_generalization_prompt(
        self, 
        data,
        features_analysis,
        train_results=None,
        prompt_hint=None,
        analysis_mode="both"
            ):
        prompt = "You are an ARC puzzle detective and strategist.\n\n"
        prompt += "=== DATA ===\n"
        prompt += f"{data}\n"
        # Section A: Feature Data (Optional)
        if analysis_mode in ("both", "features_only"):
            prompt += "=== FEATURE DATA ===\n"
            prompt += """
                You are given a feature analysis of multiple grid transformations.
                
                This data includes comparisons between:
                - Train input grids (original)
                - Train output grids (transformed)
                - Test inputs vs train inputs
            """
            prompt += json.dumps(features_analysis, indent=2) + "\n\n"

        # Section B: Train Results (Optional)
        if analysis_mode in ("both", "results_only"):
            if train_results:
                cleaned_train_results = self._clean_train_results(train_results)
                prompt += "=== CLEANED TRAIN RESULTS ===\n"
                prompt += """
                    Each puzzle was solved using a single transformation chain built over multiple steps.
                    - The first step creates a base pattern (e.g., tiling or repetition)
                    - Later steps refine this base with additional transformations
                    - Each step is scored (0.0–1.0), indicating its effectiveness
                    - A score of 1.0 means the full chain worked perfectly
                    
                    These are not alternatives — they are iterations toward one working solution.

                    Your job:
                    1. Identify transformations that appear consistently across puzzles
                    2. Build a unified rule by combining base steps and common refinements
                    3. Include confidence based on consistency across examples
                """
                prompt += json.dumps(cleaned_train_results, indent=2) + "\n\n"
            else:
                raise ValueError("train_results required for 'results_only' or 'both' analysis_mode")

        # Section C: Final Instructions
        prompt += """
            Now provide your final answer in this format:
            {
              "pattern_description": "...",
              "generalized_program": { ... },
            }

            Only return the JSON object — no extra text.
        """

        return prompt
    def _analyze_inputs(self, data: dict) -> list:
        """Analyzes each input grid independently"""
        result = []
        data['train_input_comparisons']
        return result

    def _analyze_outputs(self, data: dict) -> list:
        """Analyzes each output grid independently"""
        result = []
        return result

    def _analyze_pairs(self, data: dict) -> list:
        """Compare input → output for each example"""
        result = []
        return result

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM response into structured pattern dictionary"""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                response = response.split("{", 1)[1]
                response = "{" + response

            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            return {"error": "Failed to parse LLM response", "raw": response, "exception": str(e)}

class AnalyzeAgent:
    def __init__(self, mode: str = None, llm=None, analyzer=None):
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