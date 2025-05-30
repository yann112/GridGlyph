import numpy as np
import logging
import pytest

from synthesis_engine import SynthesisEngine
from dsl_nodes import RepeatGrid, Identity


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
    matching_programs = synthesizer.synthesize_matching_programs(input_grid, output_grid)

    assert matching_programs, "Expected at least one matching program to be found."

    # Optional: check if a specific program is in the results
    assert any(
        isinstance(p, RepeatGrid)
        and isinstance(p.inner_command, Identity)
        and p.vertical_repeats == 3
        and p.horizontal_repeats == 3
        for p in matching_programs
    ), "Expected to find RepeatGrid(Identity, 3, 3) among the results."
