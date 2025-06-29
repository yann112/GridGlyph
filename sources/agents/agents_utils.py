# agents/agent_utils.py

import numpy as np
import random
import logging
from typing import List, Dict, Any, Union, Optional, Tuple, Mapping
from core.features_analysis import ProblemAnalyzer
from assets.symbols import SYMBOL_SETS_JSON

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
    

class SymbolicGridMapper:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self._symbol_set_dict = {item["id"]: item for item in SYMBOL_SETS_JSON["grid_glyphs"]}
        # It's good practice to make this available if you plan to use it outside
        self.symbol_sets = self._symbol_set_dict # Added for consistency with generate_variants

    def map_grid(self, grid: List[List[int]], symbols: List, symbol_set_id: Optional[str] = None) -> Tuple[List[List[str]], Dict[int, str]]:
        """
        Map a numeric grid into symbolic form using selected set.
        Returns the mapped grid and the value-to-symbol mapping used.
        """
        unique_values = sorted(list(set(cell for row in grid for cell in row))) # Convert set to list for consistent order
        
        value_to_symbol = {}
        for idx, val in enumerate(unique_values):
            if idx < len(symbols):
                value_to_symbol[val] = symbols[idx]
            else:
                # Add symbol_set_id to the warning message for better context
                self.logger.warning(f"Not enough symbols in '{symbol_set_id or 'unknown_set'}' for all unique values. Value {val} could not be mapped.")
                # You might want a default mapping for unmapped values, e.g., value_to_symbol[val] = '?'
                # For now, we'll just break, which means subsequent unique values won't be mapped if symbols run out.
                break 
        
        mapped_grid = [[value_to_symbol.get(cell, '?') for cell in row] for row in grid] # Use .get for safety
        
        return mapped_grid, value_to_symbol

    def map_input_output_pair(
        self, input_grid: List[List[int]], output_grid: List[List[int]], symbol_set_id: str
    ) -> Dict[str, Union[List[List[str]], Dict[int, str]]]:
        """Map a single input/output pair using the specified symbol set."""
        symbols = self._symbol_set_dict[symbol_set_id]["symbols"]

        # Pass symbol_set_id to map_grid for better warning messages
        mapped_input, _ = self.map_grid(input_grid, symbols, symbol_set_id)
        mapped_output, _ = self.map_grid(output_grid, symbols, symbol_set_id)

        return {
            "symbol_set_id": symbol_set_id,
            "input": mapped_input,
            "output": mapped_output,
        }

    def map_grid_with_mapping(self, grid: List[List[int]], mapping: Dict[int, str]) -> List[List[str]]:
        """Map grid using a specific value-to-symbol mapping."""
        # Ensure that if a cell value isn't in the provided mapping, it's handled gracefully
        return [[mapping.get(cell, '?') for cell in row] for row in grid]

    def map_test_input(
        self, test_input: List[List[int]], symbol_set_id: str
    ) -> Dict[str, Union[List[List[str]], Dict[int, str]]]:
        """Map a test input grid using the specified symbol set."""
        symbols = self._symbol_set_dict[symbol_set_id]["symbols"]
        # Corrected: map_grid now returns two values, so unpack them
        mapped, mapping = self.map_grid(test_input, symbols, symbol_set_id)

        return {
            "symbol_set_id": symbol_set_id,
            "input": mapped,
            "mapping": mapping
        }

    def generate_variants(
        self, data: Dict[str, Any], symbol_set_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate symbolic variants of the entire dataset.

        Args:
            data: Dictionary with 'train' and/or 'test' keys
            symbol_set_ids: Override symbol sets used for mapping

        Returns:
            List of variants, one per symbol set
        """
        # Ensure symbol_sets is initialized
        if not hasattr(self, 'symbol_sets'):
            self.symbol_sets = {item["id"]: item for item in SYMBOL_SETS_JSON["grid_glyphs"]}

        if symbol_set_ids is None:
            symbol_set_ids = list(self.symbol_sets.keys())

        variants = []

        for sid in symbol_set_ids:
            self.logger.info(f"Generating symbolic variant: {sid}")
            variant = {
                "symbol_set_id": sid,
                "examples": [],
                "test_inputs": []
            }

            # Map training examples
            if "train" in data:
                for example in data["train"]:
                    mapped = self.map_input_output_pair(
                        example["input"],
                        example["output"],
                        sid
                    )
                    variant["examples"].append({
                        "input": mapped["input"],
                        "output": mapped["output"],
                    })

            # Map test inputs
            if "test" in data:
                for test_example in data["test"]:
                    mapped = self.map_test_input(test_example["input"], sid)
                    variant["test_inputs"].append({
                        "input": mapped["input"],
                    })

            variants.append(variant)

        return variants
    
    def generate_n_variants(
            self,
            puzzle_data: Dict[str, Any],
            n: int = 10,
            shuffle_symbols: bool = True,
            allow_repeats: bool = False
        ) -> List[Dict[str, Any]]:
            """
            Generate N symbolic variants of the full puzzle.
            
            Args:
                puzzle_data: Full puzzle dictionary with train/test grids
                n: Number of symbolic variants to generate
                shuffle_symbols: If True, shuffles glyphs to create different mappings
                allow_repeats: If False, avoids repeating the same symbol mapping
                
            Returns:
                List of symbolic variant dictionaries
            """
            # Step 1: Extract all training examples
            train_examples = puzzle_data.get("train", [])
            if not train_examples:
                raise ValueError("No training examples found in puzzle_data")

            # Step 2: Choose glyphset & generate variants
            available_sets = list(self._symbol_set_dict.keys())
            variants = []
            used_mappings = set()

            while len(variants) < n:
                if not available_sets and not allow_repeats:
                    self.logger.warning("Ran out of unique symbol sets/mappings before generating N variants. Consider setting allow_repeats=True or reducing N.")
                    break

                sid = random.choice(available_sets)
                
                # Step 3: Create symbol mapping
                symbols = self._symbol_set_dict[sid]["symbols"][:]
                if shuffle_symbols:
                    random.shuffle(symbols)
                
                # This key needs to capture the actual mapping, not just the shuffled symbols
                # We need to first determine the unique values in the grid to create the mapping key reliably.
                
                # For `generate_n_variants`, the `map_grid` will return the specific mapping for the unique values in the input grid.
                # So, we need to apply map_grid to get the actual mapping for the first training example (or all of them)
                # to form a consistent key for `used_mappings`.
                
                # To get a consistent mapping for the key, we need to determine all unique values across ALL grids first.
                # This is more robust than relying on just one example.
                all_grid_values = set()
                for example in train_examples:
                    all_grid_values.update(cell for row in example["input"] for cell in row)
                    if "output" in example and example["output"]:
                        all_grid_values.update(cell for row in example["output"] for cell in row)
                for test_example in puzzle_data.get("test", []):
                    all_grid_values.update(cell for row in test_example["input"] for cell in row)

                sorted_unique_values = sorted(list(all_grid_values))
                
                # Create the value_to_symbol mapping based on the sorted unique values and shuffled symbols
                temp_value_to_symbol = {}
                for i, val in enumerate(sorted_unique_values):
                    if i < len(symbols):
                        temp_value_to_symbol[val] = symbols[i]
                    else:
                        self.logger.warning(f"Not enough symbols in '{sid}' for all unique values ({len(sorted_unique_values)} unique values, {len(symbols)} symbols). Some values might not be mapped.")
                        break # Or assign a default symbol
                
                # Create a stable mapping key based on the generated value_to_symbol mapping
                # This ensures that different shuffles of symbols that result in the same value-to-symbol mapping are considered duplicates
                mapping_key_parts = []
                for val in sorted_unique_values:
                    mapping_key_parts.append(f"{val}:{temp_value_to_symbol.get(val, '?')}") # Use .get for safety
                mapping_key = f"{sid}-" + "-".join(mapping_key_parts)


                if not allow_repeats and mapping_key in used_mappings:
                    continue  # Skip duplicate mappings

                # Step 4: Build symbolic version of all train examples
                mapped_examples = []
                for example in train_examples:
                    # Pass the pre-determined mapping to map_grid_with_mapping for consistency
                    mapped_input = self.map_grid_with_mapping(example["input"], temp_value_to_symbol)
                    mapped_output = self.map_grid_with_mapping(example["output"], temp_value_to_symbol) if "output" in example and example["output"] else None

                    mapped_examples.append({
                        "input": mapped_input,
                        "output": mapped_output
                    })

                # Step 5: Map test inputs
                test_inputs = []
                for test in puzzle_data.get("test", []):
                    # Pass the pre-determined mapping to map_grid_with_mapping for consistency
                    test_inputs.append({
                        "input": self.map_grid_with_mapping(test["input"], temp_value_to_symbol)
                    })

                # Step 6: Store variant
                variants.append({
                    "symbol_set_id": sid,
                    "examples": mapped_examples,
                    "test_inputs": test_inputs,
                    "mapping": temp_value_to_symbol # Store the actual mapping used for this variant
                })

                used_mappings.add(mapping_key)

            return variants

    def format_grid(self, grid: List[List[str]]) -> str:
        """Format a single grid into a string with aligned rows."""
        return "\n".join([" ".join(row) for row in grid])

    def format_example(self, example: Dict[str, Any], index: int = 1) -> str:
        """Format one input/output example as string."""
        input_str = self.format_grid(example["input"])
        output_str = self.format_grid(example["output"]) if "output" in example and example["output"] else "" # Ensure output exists and is not empty
        
        lines = [f"Input:\n{input_str}"]
        if output_str:
            lines.append(f"Output:\n{output_str}")
            
        return "\n".join(lines)

    def format_test_input(self, test: Dict[str, Any]) -> str:
        """Format test input for prompting."""
        input_str = self.format_grid(test["input"])
        return f"Test Input:\n{input_str}"

    def format_variant(
        self,
        variant: Dict[str, Any],
        include_variant_header: bool = True
    ) -> str:
        """
        Format an entire symbolic variant into clean prompt-ready string.
        
        Args:
            variant: Dictionary containing symbol_set_id + list of examples
            include_variant_header: Show which symbol set was used
        
        Returns:
            Formatted string for LLM
        """
        sid = variant.get("symbol_set_id", "unknown")
        lines = []

        if include_variant_header:
            lines.append(f"--- VARIANT: {sid} ---\n")

        # Add each example in the variant
        for idx, ex in enumerate(variant["examples"]):
            lines.append(f"Example {idx + 1}:")
            lines.append(self.format_example(ex))
            lines.append("")  # Spacing

        # Add test inputs
        for test in variant.get("test_inputs", []):
            lines.append(self.format_test_input(test))
            lines.append("")  # Spacing

        return "\n".join(lines).strip()
    
    def format_variants_list(
        self,
        variants: List[Dict[str, Any]],
        include_variant_headers: bool = True,
        separator: str = "-" * 50
    ) -> str:
        """
        Format a list of symbolic variants into one clean string.
        
        Args:
            variants: List of variant dictionaries
            include_variant_headers: Whether to show '--- VARIANT: runic ---'
            separator: Separator between variants (only if multiple variants used)

        Returns:
            Full prompt-ready string
        """
        formatted = []
        
        for i, variant in enumerate(variants):
            variant_str = self.format_variant(variant, include_variant_header=include_variant_headers)
            formatted.append(variant_str)

            if i < len(variants) - 1 and separator:
                formatted.append(separator)

        return "\n".join(formatted)