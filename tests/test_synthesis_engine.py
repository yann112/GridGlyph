import numpy as np
import logging
import pytest

from core.synthesis_engine import SynthesisEngine
from core.dsl_nodes import RepeatGrid, Identity

@pytest.fixture
def logger() -> logging.Logger:
    """Creates a logger instance for the test."""
    logging.basicConfig(level=logging.DEBUG)
    return logging.getLogger(__name__)

def test_repeat_grid_finds_correct_program(logger: logging.Logger) -> None:
    """Tests that SynthesisEngine finds a correct RepeatGrid transformation
    that reproduces the desired output from the input.

    This test uses a small grid and expects a repeat transformation to be found.
    """
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

    # Synthesize matching programs
    matching_programs = synthesizer.synthesize_matching_programs(
        input_grid, 
        output_grid,
        operation_names=['repeat_grid'],
        max_repeat=3  # We know the correct answer is 3 repeats
    )

    assert matching_programs, "Expected at least one matching program to be found."

    # Check if the correct program is among the results
    found_correct_program = False
    for program, score in matching_programs:
        if (isinstance(program, RepeatGrid) and
            isinstance(program.inner_command, Identity) and
            program.vertical_repeats == 3 and
            program.horizontal_repeats == 3):
            found_correct_program = True
            # The perfect match should have score 1.0
            assert score == 1.0, "Perfect match should have score 1.0"
            break

    assert found_correct_program, "Expected to find RepeatGrid(Identity, 3, 3) among the results."

    # Run synthesized programs
    results = synthesizer.run_synthesized_programs(matching_programs, input_grid)

    # Check if the results are correct
    for program, score, output in results:
        assert output is not None, "Expected a valid output grid."
        assert np.array_equal(output, output_grid), "The output grid does not match the expected result."