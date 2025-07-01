# test_symbolic_interpreter.py

import pytest
import numpy as np
from core.dsl_symbolic_interpreter import SymbolicRuleParser

TEST_CASES = [
    ("Ⳁ", np.array([[1, 0], [0, 1]], dtype=int), np.array([[1, 0], [0, 1]], dtype=int)),
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
    ("⤨(III)", np.array([[5]], dtype=int), np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]], dtype=int)),
    ("⧈", np.array([[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=int), np.array([[1, 1], [1, 0]], dtype=int)),
    ("⧈", np.array([[0, 0], [0, 0]], dtype=int), np.array([[0]], dtype=int)),
    ("⧈", np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=int), np.array([[1, 1], [1, 0]], dtype=int)),
    ("↻", np.array([[1, 2], [3, 4]], dtype=int), np.array([[3, 1], [4, 2]], dtype=int)),
    ("↻", np.array([[1, 2, 3], [4, 5, 6]], dtype=int), np.array([[4, 1], [5, 2], [6, 3]], dtype=int)),
    ("↻", np.array([[1]], dtype=int), np.array([[1]], dtype=int)),
    ("⤫", np.array([[1, 2], [3, 4]], dtype=int), np.array([[1, 3], [2, 4]], dtype=int)),
    ("⤫", np.array([[1, 2, 3], [4, 5, 6]], dtype=int), np.array([[1, 4], [2, 5], [3, 6]], dtype=int)),
    ("⤫", np.array([[1, 2]], dtype=int), np.array([[1], [2]], dtype=int)),
    ("⤫", np.array([[1], [2]], dtype=int), np.array([[1, 2]], dtype=int)),
    ("⤫", np.array([[7]], dtype=int), np.array([[7]], dtype=int)),
    ("╳", np.array([[1, 2], [3, 4]], dtype=int), np.array([[4, 2], [3, 1]], dtype=int)),
    ("╳", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int), np.array([[9, 6, 3], [8, 5, 2], [7, 4, 1]], dtype=int)),
    ("╳", np.array([[1, 2, 3], [4, 5, 6]], dtype=int), np.array([[6, 3], [5, 2], [4, 1]], dtype=int)),
    ("╳", np.array([[1,2],[3,4],[5,6]], dtype=int), np.array([[6,4,2],[5,3,1]], dtype=int)),
    ("╳", np.array([[1]], dtype=int), np.array([[1]], dtype=int)),
    ("⊟", np.array([[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=int), np.array([[1, 1], [1, 0]], dtype=int)),
    ("⊟", np.array([[0, 0], [0, 0]], dtype=int), np.array([[0]], dtype=int)),
    ("⊟", np.array([[5, 0, 0], [0, 0, 0], [0, 0, 4]], dtype=int), np.array([[5, 0, 0], [0, 0, 0], [0, 0, 4]], dtype=int)),
    ("⊟", np.array([[1, 1, 1], [1, 1, 1]], dtype=int), np.array([[1, 1, 1], [1, 1, 1]], dtype=int)),
    ("⊟", np.array([[7]], dtype=int), np.array([[7]], dtype=int)),
    ("⌗(I,V)", np.array([[1]], dtype=int), np.array([[5,5,5],[5,1,5],[5,5,5]], dtype=int)),
    ("⌗(II,∅)", np.array([[1,2],[3,4]], dtype=int), np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,1,2,0,0],[0,0,3,4,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]], dtype=int)),
    ("⌗(∅,V)", np.array([[1,2],[3,4]], dtype=int), np.array([[1,2],[3,4]], dtype=int)),
    ("⌗(I,II)", np.array([[7,8],[9,1]], dtype=int), np.array([[2,2,2,2],[2,7,8,2],[2,9,1,2],[2,2,2,2]], dtype=int)),
]




@pytest.fixture
def parser():
    return SymbolicRuleParser()


@pytest.mark.parametrize("rule, input_grid, expected_output", TEST_CASES)
def test_symbolic_rule(parser, rule, input_grid, expected_output):
    try:
        command = parser.parse_rule(rule)

        input_ndarray = input_grid

        result = command.execute(input_ndarray)

        expected_ndarray = expected_output

        assert result.shape == expected_ndarray.shape, f"Shape mismatch for rule '{rule}': Result {result.shape} vs Expected {expected_ndarray.shape}"

        if np.issubdtype(expected_ndarray.dtype, np.number):
            assert np.array_equal(result, expected_ndarray), f"Output mismatch for rule '{rule}'"
        else:
            assert all(
                np.array_equal(r, e) if isinstance(r, np.ndarray) else r == e
                for r, e in zip(result.flatten(), expected_ndarray.flatten())
            ), f"Output mismatch for rule '{rule}'"

    except Exception as e:
        pytest.fail(f"Failed to parse or execute rule '{rule}': {str(e)}")