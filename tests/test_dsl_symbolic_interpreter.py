# test_symbolic_interpreter.py

import pytest
import numpy as np
from core.dsl_symbolic_interpreter import SymbolicRuleParser, roman_to_int
from pathlib import Path


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
    
    ]




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