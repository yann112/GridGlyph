import pytest
import numpy as np
import logging
from typing import Optional, Union

from gridglyph.core.dsl_symbolic_interpreter import SymbolicRuleParser
from gridglyph.core.dsl_symbolic_executor import DSLExecutor
from gridglyph.core.ground_truth import TEST_CASES  # Use internal ground truth

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')




@pytest.fixture
def parser() -> SymbolicRuleParser:
    return SymbolicRuleParser()

@pytest.fixture
def test_logger() -> logging.Logger:
    return logging.getLogger(__name__)



@pytest.mark.parametrize("rule, initial_input_grid, expected_output", TEST_CASES)
def test_dsl_executor_execution(
    parser: SymbolicRuleParser,
    test_logger: logging.Logger,
    rule: str,
    initial_input_grid: Optional[np.ndarray],
    expected_output: Union[np.ndarray, int] # Changed type hint to accept both
):
    try:
        parsed_command_tree = parser.parse_rule(rule)
        test_logger.info(f"Successfully parsed rule: '{rule}'")

        if initial_input_grid is None:
            executor_input_grid = np.array([[0]], dtype=int)
        else:
            executor_input_grid = np.array(initial_input_grid, dtype=int)

        executor = DSLExecutor(
            root_command=parsed_command_tree,
            initial_puzzle_input=executor_input_grid,
            logger=test_logger
        )
        test_logger.info("Executor instantiated and initialized commands.")

        result = executor.execute_program()
        test_logger.info(f"Execution complete for rule: '{rule}'")

        if isinstance(expected_output, np.ndarray):
            assert isinstance(result, np.ndarray), \
                f"Type mismatch for rule '{rule}': Expected np.ndarray, got {type(result)}"
            assert result.shape == expected_output.shape, \
                f"Shape mismatch for rule '{rule}': {result.shape} vs {expected_output.shape}"
            assert np.array_equal(result, expected_output), \
                f"Output mismatch for rule '{rule}'\nExpected:\n{expected_output}\nGot:\n{result}"
        elif isinstance(expected_output, int):
            assert isinstance(result, int), \
                f"Type mismatch for rule '{rule}': Expected int, got {type(result)}"
            assert result == expected_output, \
                f"Output mismatch for rule '{rule}'\nExpected: {expected_output}\nGot: {result}"
        else:
            pytest.fail(f"Invalid expected_output type in test case for rule '{rule}': {type(expected_output)}")

    except Exception as e:
        pytest.fail(f"Test failed for rule '{rule}': {str(e)}")