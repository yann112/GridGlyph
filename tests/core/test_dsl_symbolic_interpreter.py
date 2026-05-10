# test_symbolic_interpreter.py

import pytest
import numpy as np
from gridglyph.core.dsl_symbolic_interpreter import SymbolicRuleParser, roman_to_int
from gridglyph.core.ground_truth import TEST_CASES 


@pytest.fixture
def parser():
    return SymbolicRuleParser()


@pytest.mark.parametrize("rule, input_grid, expected_output", TEST_CASES)
def test_symbolic_rule(parser, rule, input_grid, expected_output):
    try:
        command = parser.parse_rule(rule)

        # Convert input to ndarray
        input_ndarray = np.array(input_grid)

        # Execute the command
        result = command.execute(input_ndarray)

        # Compare with expected output
        assert result.shape == expected_output.shape, f"Shape mismatch: {result.shape} vs {expected_output.shape}"

        if np.issubdtype(expected_output.dtype, np.number):
            assert np.array_equal(result, expected_output), f"Output mismatch for '{rule}'"
        else:
            # For object arrays (e.g., emoji or strings), compare element-wise
            assert all(
                np.array_equal(r, e) if isinstance(r, np.ndarray) else r == e
                for r, e in zip(result.flatten(), expected_output.flatten())
            ), f"Output mismatch for '{rule}'"

    except Exception as e:
        pytest.fail(f"Failed to parse or execute rule '{rule}': {str(e)}")