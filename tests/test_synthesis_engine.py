import numpy as np
import logging
import pytest

from core.synthesis_engine import SynthesisEngine
from core.dsl_nodes import AbstractTransformationCommand, Identity, RepeatGrid, FlipGridHorizontally, Alternate


@pytest.fixture
def logger() -> logging.Logger:
    """Creates a logger instance for the test."""
    logging.basicConfig(level=logging.DEBUG)
    return logging.getLogger(__name__)


def _run_and_assert_match(synthesizer, input_grid, output_grid, operation_names=None):
    """Helper to run synthesis and assert at least one match"""
    matching_programs = synthesizer.synthesize_matching_programs(
        input_grid,
        output_grid,
        operation_names=operation_names or ['repeat_grid', 'identity', 'flip_h', 'flip_v']
    )
    assert matching_programs, "Expected at least one matching program."
    results = synthesizer.run_synthesized_programs(matching_programs, input_grid)
    found_match = False
    for _, _, output in results:
        if output is not None and np.array_equal(output, output_grid):
            found_match = True
            break
    assert found_match, "Expected at least one program to produce the correct output."
    return matching_programs


def test_repeat_grid_finds_correct_program(logger: logging.Logger) -> None:
    """Tests that SynthesisEngine finds a correct RepeatGrid transformation."""
    input_grid = np.array([
        [7, 9],
        [4, 3]
    ])

    output_grid = np.array([
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3],
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3],
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3]
    ])

    synthesizer = SynthesisEngine(logger)

    # Run synthesis
    matching_programs = _run_and_assert_match(synthesizer, input_grid, output_grid, ['repeat_grid'])

    # Check if the correct program is among the results
    found_correct_program = False
    for program, score in matching_programs:
        if (isinstance(program, RepeatGrid) and
            isinstance(program.inner_command, Identity) and
            program.vertical_repeats == 3 and
            program.horizontal_repeats == 3):
            found_correct_program = True
            assert score == 1.0, "Perfect match should have score 1.0"
            break

    assert found_correct_program, "Expected to find RepeatGrid(Identity, 3, 3) among the results."


def test_flip_h_detection(logger: logging.Logger) -> None:
    """Tests that SynthesisEngine detects and applies a horizontal flip."""

    input_grid = np.array([
        [1, 2],
        [3, 4]
    ])

    output_grid = np.fliplr(input_grid)

    synthesizer = SynthesisEngine(logger)

    # Run synthesis
    matching_programs = _run_and_assert_match(synthesizer, input_grid, output_grid, ['flip_h'])

    # Verify it's a FlipGridHorizontally
    found_correct_program = False
    for program, score in matching_programs:
        if isinstance(program, FlipGridHorizontally):
            found_correct_program = True
            assert score == 1.0, "Perfect match should have score 1.0"
            break

    assert found_correct_program, "Expected to find FlipGridHorizontally among the results."


def test_combined_repeat_with_flip(logger: logging.Logger) -> None:
    """Tests that SynthesisEngine can combine repeat and flip operations."""

    input_grid = np.array([
        [1, 2],
        [3, 4]
    ])

    flipped = np.fliplr(input_grid)
    output_grid = np.tile(flipped, (2, 3))

    synthesizer = SynthesisEngine(logger)

    # Run synthesis with all ops enabled
    matching_programs = _run_and_assert_match(synthesizer, input_grid, output_grid, ['repeat_grid', 'flip_h'])

    # Check if we found the right composite transformation
    found_correct_program = False
    for program, score in matching_programs:
        if (isinstance(program, RepeatGrid) and
            isinstance(program.inner_command, FlipGridHorizontally) and
            program.vertical_repeats == 2 and
            program.horizontal_repeats == 3):
            found_correct_program = True
            assert score == 1.0, "Perfect match should have score 1.0"
            break

    assert found_correct_program, "Expected to find RepeatGrid(FlipGridHorizontally, 2, 3) among the results."

def test_alternating_flip_pattern(logger: logging.Logger) -> None:
    """Tests that SynthesisEngine detects and applies alternating flip pattern."""
    
    input_grid = np.array([
        [7, 9],
        [4, 3]
    ])

    # Only alternate between identity and flip — no repetition
    output_grid = np.array([
        [7, 9],
        [3, 4]
    ])

    synthesizer = SynthesisEngine(logger)

    # Run synthesis with support for alternation
    matching_programs = _run_and_assert_match(
        synthesizer,
        input_grid,
        output_grid,
        operation_names=['identity', 'flip_h', 'alternate']
    )

    # Check if we found the right alternating transformation
    found_correct_program = False
    for program, score in matching_programs:
        if (isinstance(program, Alternate) and
            isinstance(program.first, Identity) and
            isinstance(program.second, FlipGridHorizontally)):
            assert score == 1.0, "Perfect match should have score 1.0"
            found_correct_program = True
            break

    assert found_correct_program, (
        "Expected to find Alternate(Identity, FlipGridHorizontally)"
    )