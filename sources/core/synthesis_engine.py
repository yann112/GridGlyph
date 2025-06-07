import itertools
import logging
from typing import List, Optional, Tuple, Type, Dict, Any
import numpy as np

from core.dsl_nodes import AbstractTransformationCommand, Identity, RepeatGrid, FlipGridHorizontally, FlipGridVertically
from core.transformation_factory import TransformationFactory
from core.dsl_interpreter import DslInterpreter


class SynthesisEngine:
    """
    A flexible synthesis engine that enumerates programs using metadata from DSL nodes.
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.interpreter = DslInterpreter(self.logger)

    def synthesize_matching_programs(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        operation_names: List[str] = None,
        max_repeat: int = 4,
        top_k: int = 3
    ) -> List[Tuple[AbstractTransformationCommand, float]]:
        """
        Enumerates programs using metadata-driven synthesis rules.

        Args:
            input_grid (np.ndarray): Input grid.
            output_grid (np.ndarray): Expected output grid.
            operation_names (List[str]): Allowed operations to consider.
            max_repeat (int): Max repeat count for RepeatGrid operations.
            top_k (int): Number of best candidates to return.

        Returns:
            List of tuples (program, score), sorted by score descending.
        """
        if operation_names is None:
            self.logger.info("no operations given")
            return []

        candidates_with_scores: List[Tuple[AbstractTransformationCommand, float]] = []

        # Try each base operation
        for op_name in operation_names:
            cls = TransformationFactory.get_class(op_name)
            rules = getattr(cls, "synthesis_rules", {"type": "atomic"})

            try:
                if rules["type"] == "atomic":
                    # Just create and test it directly
                    program = TransformationFactory.create_operation(op_name, logger=self.logger)
                    candidate_output = self.interpreter.execute_program(program, input_grid)

                    if candidate_output.shape != output_grid.shape:
                        continue

                    score = self._score_candidate(candidate_output, output_grid)
                    candidates_with_scores.append((program, score))

                elif rules["type"] == "combinator":
                    arity = rules.get("arity", 1)

                    inner_ops = [n for n in operation_names if n != op_name]
                    if not inner_ops:
                        inner_ops = ['identity']  # Fallback

                    if arity == 1:
                        # Unary combinators like RepeatGrid(inner_command)
                        for inner_name in inner_ops:
                            inner_command = TransformationFactory.create_operation(inner_name, logger=self.logger)
                            for params in self._generate_params(rules.get("parameter_ranges", {})):
                                program = TransformationFactory.create_operation(
                                    op_name,
                                    inner_command=inner_command,
                                    logger=self.logger,
                                    **params
                                )
                                candidate_output = self.interpreter.execute_program(program, input_grid)
                                if candidate_output.shape == output_grid.shape:
                                    score = self._score_candidate(candidate_output, output_grid)
                                    candidates_with_scores.append((program, score))

                    elif arity >= 2:
                        # Multi-input combinators like Alternate(first, second)
                        from itertools import product
                        for inner_choices in product(inner_ops, repeat=arity):
                            try:
                                inner_commands = [
                                    TransformationFactory.create_operation(name, logger=self.logger)
                                    for name in inner_choices
                                ]
                                program = TransformationFactory.create_operation(
                                    op_name,
                                    *inner_commands,
                                    logger=self.logger
                                )
                                candidate_output = self.interpreter.execute_program(program, input_grid)
                                if candidate_output.shape == output_grid.shape:
                                    score = self._score_candidate(candidate_output, output_grid)
                                    candidates_with_scores.append((program, score))
                            except Exception as e:
                                self.logger.warning(f"Failed to instantiate {op_name} with {inner_choices}: {str(e)}")
                                continue

            except Exception as e:
                self.logger.error(f"Error synthesizing {op_name}: {str(e)}")
                continue

        # Sort by score descending
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)

        return candidates_with_scores[:top_k]

    def _score_candidate(self, candidate: np.ndarray, target: np.ndarray) -> float:
        """Compute match ratio between candidate and target"""
        return float(np.sum(candidate == target)) / float(target.size)

    def _generate_params(self, param_ranges: Dict[str, tuple]) -> Dict[str, int]:
        """Generate combinations of parameters from ranges"""
        keys = list(param_ranges.keys())
        ranges = [range(start, end + 1) for start, end in param_ranges.values()]

        for values in itertools.product(*ranges):
            yield dict(zip(keys, values))

    def run_synthesized_programs(
        self,
        programs_with_scores: List[Tuple[AbstractTransformationCommand, float]],
        input_grid: np.ndarray
    ) -> List[Tuple[AbstractTransformationCommand, float, np.ndarray]]:
        """
        Executes the synthesized programs on the input grid.

        Args:
            programs_with_scores: List of (program, score).
            input_grid: Input grid to apply programs to.

        Returns:
            List of (program, score, output_grid).
        """
        results = []
        for program, score in programs_with_scores:
            try:
                output_grid = self.interpreter.execute_program(program, input_grid)
                results.append((program, score, output_grid))
            except Exception as e:
                self.logger.error(f"Failed to execute program: {str(e)}")
                results.append((program, score, None))
        return results