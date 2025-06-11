# test_dsl_interpreter.py
import numpy as np
import logging

from core.dsl_interpreter import DslInterpreter


def test_dsl_interpreter_mask_combinator_with_map_numbers():
    """
    Tests the DslInterpreter's ability to parse and execute the specific
    MaskCombinator program string with an inner MapNumbers command.
    """
    interpreter = DslInterpreter()

    # The exact program string provided by the user
    program_str = (
        '{"operation": "mask_combinator", "parameters": {"inner_command": {"operation": "map_numbers", '
        '"parameters": {"mapping": {"9": 7, "7": 9}}}, "mask_func": "lambda grid: np.array([[False, False, False], '
        '[True, True, True], [False, False, False]])"}}'
    )

    # Input grid designed to show changes due to MapNumbers within the mask
    # For this test, we'll use a smaller 3x3 grid that clearly demonstrates the mapping.
    input_grid = np.array([
        [9, 7, 9],
        [9, 7, 9],
        [9, 7, 9]
    ])
    expected_output_grid = np.array([
        [9, 7, 9],
        [7, 9, 7],
        [9, 7, 9]
    ])
    # Parse the program string using your DslInterpreter
    parsed_command = interpreter.parse_program(program_str)

    # Execute the parsed command on the input grid
    actual_output_grid = parsed_command.execute(input_grid)


    # Assert that the actual output matches the expected output
    np.testing.assert_array_equal(actual_output_grid, expected_output_grid)
    
    
def test_dsl_interpreter_mask_combinator_with_map_numbers():
    """
    Tests the DslInterpreter's ability to parse and execute the specific
    MaskCombinator program string with an inner MapNumbers command.
    """
    interpreter = DslInterpreter()

    # The exact program string provided by the user
    program_str = (
    '{"operation": "swap_rows_or_columns", "parameters": {"row_swap": "[3, 3]", "col_swap": "[0, 5]", "swap_type": "both"}}'
    )

    # Input grid designed to show changes due to MapNumbers within the mask
    # For this test, we'll use a smaller 3x3 grid that clearly demonstrates the mapping.
    input_grid = np.array([
        [0, 1, 2, 3, 4, 5],
        [10, 11, 12, 13, 14, 15],
        [20, 21, 22, 23, 24, 25],
        [30, 31, 32, 33, 34, 35]
    ])
    expected_output_grid = np.array([
        [5, 1, 2, 3, 4, 0],
        [15, 11, 12, 13, 14, 10],
        [25, 21, 22, 23, 24, 20],
        [35, 31, 32, 33, 34, 30]
    ])
    # Parse the program string using your DslInterpreter
    parsed_command = interpreter.parse_program(program_str)

    # Execute the parsed command on the input grid
    actual_output_grid = parsed_command.execute(input_grid)


    # Assert that the actual output matches the expected output
    np.testing.assert_array_equal(actual_output_grid, expected_output_grid)
