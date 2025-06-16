# agents/agent_utils.py

import numpy as np
from typing import Dict, List, Any
from core.features_analysis import ProblemAnalyzer


class MultiGridFeatureCollector:
    def __init__(self, analyzer: ProblemAnalyzer = None):
        self.analyzer = analyzer or ProblemAnalyzer()

    def extract_features_from_task(self, data: dict) -> dict:
        """
        Extracts pairwise transformation features from all grids in the task.
        
        Args:
            data: Full ARC task dictionary with 'train' and optionally 'test'
            
        Returns:
            dict: Structured feature data including:
                - train_input_comparisons: input ↔ input
                - train_output_comparisons: output ↔ output
                - test_to_train_comparisons: test-input ↔ train-input
        """
        result = {}

        # Step 1: Get all train inputs and outputs as numpy arrays
        train_inputs = [np.array(ex["input"]) for ex in data.get("train", [])]
        test_inputs = [np.array(ex["input"]) for ex in data.get("test", [])]
        inputs = train_inputs + test_inputs
        train_outputs = [np.array(ex["output"]) for ex in data.get("train", [])]

        # Step 2: Compare all train inputs to each other
        result["input_comparisons"] = self._extract_pairwise_features(inputs, mode="ii")

        # Step 3: Compare all train outputs to each other
        result["output_comparisons"] = self._extract_pairwise_features(train_outputs, mode="oo")

        # Step 4: Compare each train input to its corresponding output
        result["input_output_comparisons"] = self._extract_input_output_features(train_inputs, train_outputs)
        
        return result

    def _extract_input_output_features(self, inputs: list, outputs: list) -> dict:
        """Compare each input grid with its corresponding output grid in training examples."""
        comparisons = {}
        for idx, (inp, out) in enumerate(zip(inputs, outputs)):
            features = self._analyze_pair(inp, out, mode="io")
            pair_label = f"input{idx + 1}-output{idx + 1}"
            comparisons[pair_label] = features
        return comparisons
    
    def _extract_pairwise_features(self, grids: list, mode: str) -> dict:
        """Compare every unique pair of grids and extract transformation features.
        
        Returns:
            Dict[int, Dict]: Flat dictionary where each key is a comparison ID,
                            and each value is a dictionary that includes:
                            - 'pair': a string like 'grid1-grid2'
                            - 'features': the actual extracted features from _analyze_pair
        """
        comparisons = {}
        n = len(grids)
        comparison_id = 1  # Counter for assigning IDs: 1, 2, 3...
        if mode == "ii":
            grid = 'input'
        elif mode == "oo":
            grid = 'output'
        for i in range(n):
            grid_i = grids[i]

            for j in range(i + 1, n):
                grid_j = grids[j]
                features = self._analyze_pair(grid_i, grid_j, mode)

                pair_label = f"{grid}{i + 1}-{grid}{j + 1}"

                comparisons[f"{pair_label}"] = features

                comparison_id += 1

        return comparisons

    def _analyze_pair(self, grid_a: np.ndarray, grid_b: np.ndarray, mode) -> dict:
        """Use ProblemAnalyzer to find transformation features between two grids"""
        return self.analyzer.analyze(grid_a, grid_b, mode).to_dict()