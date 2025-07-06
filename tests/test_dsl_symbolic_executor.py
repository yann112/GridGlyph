import pytest
import numpy as np
import logging
from typing import Optional

from core.dsl_symbolic_interpreter import SymbolicRuleParser, roman_to_int
from core.dsl_symbolic_executor import DSLExecutor
from core.dsl_nodes import InputGridReference


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


common_input_grid = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=int)

asymmetric_grid = np.array([
    [1, 2, 0],
    [3, 4, 0],
    [0, 0, 0]
], dtype=int)

tiny_grid_1x1_val1 = np.array([[1]], dtype=int)
tiny_grid_1x1_val2 = np.array([[2]], dtype=int)

TEST_CASES = [
    ("Ⳁ", np.array([[1, 0], [0, 1]], dtype=int), np.array([[1, 0], [0, 1]], dtype=int)),
    ("⇒(I,∅)", np.array([[1, 8], [0, 1]], dtype=int), np.array([[0, 8], [0, 0]], dtype=int)),
    ("↔", np.array([[1, 2], [3, 4]], dtype=int), np.array([[2, 1], [4, 3]], dtype=int)),
    ("→(I, ↢)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[2, 1], [3, 4]], dtype=int)),
    ("⟹(→(I,↢),↔)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[1, 2], [4, 3]], dtype=int)),
    ("⬒(II)",  np.array([[1, 0], [0, 1]], dtype=int), np.tile( np.array([[1, 0], [0, 1]], dtype=int), (2, 1))),
    ("⇄(I,II)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[3, 4], [1, 2]], dtype=int)),
    ("⮝(I,II)", np.array([[1, 2, 3], [4, 5, 6]], dtype=int), np.array([[2, 3, 1], [4, 5, 6]], dtype=int)),
    ("↕", np.array([[1, 2], [3, 4]], dtype=int), np.array([[3, 4], [1, 2]], dtype=int)),
    ("↕", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int), np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=int)),
    ("⮞(I,II)", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int), np.array([[4, 2, 3], [7, 5, 6], [1, 8, 9]], dtype=int)),
    ("⮞(II,I)", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int), np.array([[1, 8, 3], [4, 2, 6], [7, 5, 9]], dtype=int)),
    ("◨(II)", np.array([[1, 0], [0, 1]], dtype=int), np.tile(np.array([[1, 0], [0, 1]], dtype=int), (1, 2))),
    ("◨(III)", np.array([[1, 2]], dtype=int), np.array([[1, 2, 1, 2, 1, 2]], dtype=int)),
    ("→(II, ↢)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[1, 2], [4, 3]], dtype=int)),
    ("→(I, Ⳁ)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[1, 2], [3, 4]], dtype=int)),
    ("⇄(I,III)", np.array([[1, 2], [3, 4], [5, 6]], dtype=int), np.array([[5, 6], [3, 4], [1, 2]], dtype=int)),
    ("⇄(I,I)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[1, 2], [3, 4]], dtype=int)),
    ("⮝(I,I)", np.array([[1, 2, 3], [4, 5, 6]], dtype=int), np.array([[3, 1, 2], [4, 5, 6]], dtype=int)),
    ("⮝(II,I)", np.array([[1, 2, 3], [4, 5, 6]], dtype=int), np.array([[1, 2, 3], [6, 4, 5]], dtype=int)),
    ("⟹(↔)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[2, 1], [4, 3]], dtype=int)),
    ("⟹(⮝(I,I), ↔, ⬒(II))", np.array([[1, 2], [3, 4]], dtype=int), np.tile(np.array([[1, 2], [4, 3]], dtype=int), (2, 1))),
    ("⊕(I,I,V)", None, np.array([[5]], dtype=int)),
    ("⊕(III,II,VII)", None, np.array([[7, 7], [7, 7], [7, 7]], dtype=int)),
    ("⊕(II,II,∅)", None, np.array([[0, 0], [0, 0]], dtype=int)),
    ("⤨(II)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=int)),
    ("⧈", np.array([[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=int), np.array([[1, 1], [1, 0]], dtype=int)),
    ("⧈", np.array([[0, 0], [0, 0]], dtype=int), np.array([[0]], dtype=int)),
    ("⧈", np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=int), np.array([[1, 1], [1, 0]], dtype=int)),
    ("⧀", np.array([[1, 2], [3, 4]]), np.array([1, 2, 3, 4])),
    ("⧀", np.array([[5], [6], [7]]), np.array([5, 6, 7])),
    ("⧀", np.array([[8, 9, 10]]), np.array([8, 9, 10])),
    ("⧀", np.array([[0, 0], [0, 0]]), np.array([0, 0, 0, 0])),
    ("⊡(I,I)", common_input_grid, np.array([[1]], dtype=int)),
    ("⊡(II,III)", common_input_grid, np.array([[6]], dtype=int)),
    ("⊡(III,II)", common_input_grid, np.array([[8]], dtype=int)),
    ("↱V", np.array([[0]]), np.array([[5]], dtype=int)),
    ("≡(↱V,↱V)", common_input_grid, np.array([[1]], dtype=int)),
    ("≡(↱V,↱X)", common_input_grid, np.array([[0]], dtype=int)),
    ("≗(Ⳁ,Ⳁ)", common_input_grid, np.array([[1]], dtype=int)),
    ("≗(Ⳁ,↔)", asymmetric_grid, np.array([[0]], dtype=int)),
    ("≡(⊡(I,I),↱∅)", np.array([[0,0],[0,0]], dtype=int), np.array([[1]], dtype=int)),
    ("≡(⊡(I,I),↱∅)", np.array([[1,1],[1,1]], dtype=int), np.array([[0]], dtype=int)),
    ("⍰(≡(↱I,↱I),Ⳁ,↔)", common_input_grid, common_input_grid),
    ("⍰(≡(↱I,↱II),Ⳁ,↔)", common_input_grid, np.fliplr(common_input_grid)),
    ("⍰(≡(⊡(I,I),↱I),Ⳁ,↔)", common_input_grid, common_input_grid),
    ("⍰(≡(⊡(I,I),↱∅),Ⳁ,↔)", common_input_grid, np.fliplr(common_input_grid)),
    ("⟹(⮝(I,I), ⟹(↔, ⬒(II)))", np.array([[1, 2], [3, 4]], dtype=int), np.tile(np.array([[1, 2], [4, 3]], dtype=int), (2, 1))),
    ("▦(III,III,\"∅∅I;∅II;I∅∅\")", common_input_grid, np.block([[np.full((3, 3), False, dtype=bool), np.full((3, 3), False, dtype=bool), np.full((3, 3), True, dtype=bool)], [np.full((3, 3), False, dtype=bool), np.full((3, 3), True, dtype=bool), np.full((3, 3), True, dtype=bool)], [np.full((3, 3), True, dtype=bool), np.full((3, 3), False, dtype=bool), np.full((3, 3), False, dtype=bool)]])),
    ("⧎(⊕(IX,IX,VII), ▦(III,III,\"I∅I;∅I∅;I∅I\"), ⊕(IX,IX,∅))", np.array([[0]], dtype=int), np.array([[7,7,7,0,0,0,7,7,7],[7,7,7,0,0,0,7,7,7],[7,7,7,0,0,0,7,7,7],[0,0,0,7,7,7,0,0,0],[0,0,0,7,7,7,0,0,0],[0,0,0,7,7,7,0,0,0],[7,7,7,0,0,0,7,7,7],[7,7,7,0,0,0,7,7,7],[7,7,7,0,0,0,7,7,7]], dtype=int)),
    ("⊕(IV,IV,∅)", np.zeros((4,4), dtype=int), np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=int)),
    ("⟹(◨(II), ⬒(II))", np.array([[0, 1], [1, 0]], dtype=int), np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]], dtype=int)),
    ("⧎(⟹(◨(II), ⬒(II)), ⤨(II), ⊕(IV,IV,∅))", np.array([[0, 1], [1, 0]], dtype=int), np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype=int)),
    ("⌂", np.array([[5, 6], [7, 8]], dtype=int), np.array([[5, 6], [7, 8]], dtype=int)),
    ("⌂", np.array([[1, 1, 1], [0, 0, 0]], dtype=int), np.array([[1, 1, 1], [0, 0, 0]], dtype=int)),
    ("↔(⌂)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[2, 1], [4, 3]], dtype=int)),
    ("↔(⌂)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[2, 1], [4, 3]], dtype=int)),
    ("↔(⌂)", np.array([[1, 2, 3], [4, 5, 6]], dtype=int), np.array([[3, 2, 1], [6, 5, 4]], dtype=int)),
    ("↕(⌂)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[3, 4], [1, 2]], dtype=int)),
    ("↕(⌂)", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int), np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=int)),
    ("↢(⌂)", np.array([[1, 2, 3], [4, 5, 6]], dtype=int), np.array([[3, 2, 1], [6, 5, 4]], dtype=int)),
    ("↢(⌂)", np.array([[1, 2]], dtype=int), np.array([[2, 1]], dtype=int)),
    ("⧈(⌂)", np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=int), np.array([[1, 1], [1, 0]], dtype=int)),
    ("⧈(⌂)", np.array([[0, 0], [0, 0]], dtype=int), np.array([[0]], dtype=int)), # Edge case: all background
    ("⧀(⌂)", np.array([[1, 2], [3, 4]], dtype=int), np.array([1, 2, 3, 4], dtype=int)),
    ("⧀(⌂)", np.array([[5], [6], [7]], dtype=int), np.array([5, 6, 7], dtype=int)),
    ("⟹(⌂, ⤨(II))", np.array([[1, 2], [3, 4]], dtype=int), np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=int)),
    ("⟹(⌂, ⤨(III))", np.array([[1]], dtype=int), np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int)),
    ("⟹(⌂, ⇒(I,V))", np.array([[1, 2], [3, 1]], dtype=int), np.array([[5, 2], [3, 5]], dtype=int)),
    ("⟹(⌂, ⇒(V,∅))", np.array([[5, 5], [0, 0]], dtype=int), np.array([[0, 0], [0, 0]], dtype=int)),
    ("⇌(↔, ↕)", common_input_grid, np.array([[3, 2, 1], [4, 5, 6], [9, 8, 7]], dtype=int)),
    ("⇌(↢, Ⳁ)", common_input_grid, np.array([[3, 2, 1], [4, 5, 6], [9, 8, 7]], dtype=int)),
    ("⇌(⇒(I,∅), ↢)", asymmetric_grid, np.array([[0, 2, 0], [0, 4, 3], [0, 0, 0]], dtype=int)),
    ("¿(↔, ≗(↱I, ↱I))", common_input_grid, np.fliplr(common_input_grid)),    # ("¿C(↔, ≡(↱I, ↱V))", common_input_grid, common_input_grid),
    ("¿(↔, ≗(⊡(I,I), ↱I))", common_input_grid, np.fliplr(common_input_grid)), 
    ("¿(↔, ≗(⊡(I,I), ↱V))", common_input_grid, common_input_grid),
    ("¿(↢, ≗(⌂, Ⳁ))", common_input_grid, np.flip(common_input_grid, axis=1)),
    ("¿(↢, ≗(⌂, ↔(⌂)))", asymmetric_grid, asymmetric_grid),
    ("◫(⌂, [(⊕(I,I,I), ↢)], Ⳁ)", tiny_grid_1x1_val1, tiny_grid_1x1_val1),
    ("◫(⌂, [(⊕(I,I,II), ↢)], Ⳁ)", tiny_grid_1x1_val1, tiny_grid_1x1_val1),
    ("◫(⌂, [(⊕(II,II,I), ↔)], ↕)", np.array([[1,1],[1,1]], dtype=int), np.array([[1,1],[1,1]], dtype=int)), 
    ("◫(⌂, [(⊕(II,II,II), ↔)], ↕)", asymmetric_grid, np.flipud(asymmetric_grid)),
]


@pytest.fixture
def parser() -> SymbolicRuleParser:
    return SymbolicRuleParser()

@pytest.fixture
def test_logger() -> logging.Logger:
    return logging.getLogger(__name__)


@pytest.mark.parametrize("rule, initial_input_grid, expected_output_grid", TEST_CASES)
def test_dsl_executor_execution(
    parser: SymbolicRuleParser,
    test_logger: logging.Logger,
    rule: str,
    initial_input_grid: Optional[np.ndarray],
    expected_output_grid: np.ndarray
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

        result_grid = executor.execute_program()
        test_logger.info(f"Execution complete for rule: '{rule}'")

        assert result_grid.shape == expected_output_grid.shape, \
            f"Shape mismatch for rule '{rule}': {result_grid.shape} vs {expected_output_grid.shape}"

        if np.issubdtype(expected_output_grid.dtype, np.number):
            assert np.array_equal(result_grid, expected_output_grid), \
                f"Output mismatch for rule '{rule}'\nExpected:\n{expected_output_grid}\nGot:\n{result_grid}"
        else:
            assert all(
                np.array_equal(r, e) if isinstance(r, np.ndarray) else r == e
                for r, e in zip(result_grid.flatten(), expected_output_grid.flatten())
            ), f"Output mismatch for rule '{rule}'\nExpected:\n{expected_output_grid}\nGot:\n{result_grid}"

    except Exception as e:
        pytest.fail(f"Test failed for rule '{rule}': {str(e)}")