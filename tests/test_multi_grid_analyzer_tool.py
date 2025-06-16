import pytest
import json
import logging

from tools.multi_grid_analyzer_tool import MultiGridAnalyzerTool
from core.llm import OpenRouterClient
from core.synthesis_engine import SynthesisEngine
from core.dsl_interpreter import DslInterpreter


@pytest.fixture
def stable_llm_client():
    """
    Fixture providing a stable OpenRouter client optimized for consistent program synthesis.
    
    Uses deterministic settings:
    - temperature=0.0: Eliminates randomness for identical outputs
    - top_p=0.1: Restricts token sampling for consistency  
    - top_k=1: Always picks most likely token
    - repetition_penalty=1.0: Neutral repetition handling
    
    Returns:
        OpenRouterClient: Configured client for stable program generation
    """
    return OpenRouterClient(
        # model="mistralai/devstral-small:free",
        # model=r"deepseek/deepseek-r1-0528-qwen3-8b:free",
        model=r"mistralai/ministral-8b",
        temperature=0.0,      # Completely deterministic
        top_p=0.1,           # Very restrictive sampling
        top_k=1,             # Most likely token only
        repetition_penalty=1.0,  # Neutral repetition
        max_tokens=1000, 
    )


@pytest.fixture
def creative_llm_client():
    """
    Fixture providing a creative OpenRouter client optimized for diverse and creative program synthesis.

    Uses creative settings:
    - temperature=0.7: Introduces randomness for varied outputs
    - top_p=0.9: Allows for a wide range of token sampling
    - top_k=50: Considers a broader set of likely tokens
    - repetition_penalty=1.2: Slightly discourages repetition

    Returns:
        OpenRouterClient: Configured client for creative program generation
    """
    return OpenRouterClient(
        # model="mistralai/devstral-small:free",
        # model=r"deepseek/deepseek-r1-0528-qwen3-8b:free",
        model=r"mistralai/ministral-8b",
        temperature=0.7,      # Introduces randomness
        top_p=0.9,           # Wide token sampling
        top_k=50,            # Considers more tokens
        repetition_penalty=1.2, # Slightly discourages repetition
        max_tokens=1200,
    )


def build_programs_from_train_results(train_results):
    """
    Reconstructs DSL program objects from 'program_str' fields in train_results.
    
    Args:
        train_results (dict): Original train_results with stringified program dicts
        
    Returns:
        dict: Modified train_results with real DSL command objects
    """
    interpreter = DslInterpreter()
    for puzzle_key in train_results:
        puzzle_results = train_results[puzzle_key]

        for result in puzzle_results:
            # Reconstruct main program
            if 'program_str' in result:
                result['program'] = interpreter.parse_program(result['program_str'])

            # Reconstruct alternatives
            if 'alternatives' in result:
                for alt in result['alternatives']:
                    if 'program_str' in alt:
                        alt['program'] = interpreter.parse_program(alt['program_str'])

    return train_results


@pytest.fixture
def train_results():
    # Sample train_results dictionary with mock program objects replaced by strings
    train_results = {
        'puzzle_0': [
            {
                'success': True,
                'result_grid': [
                    [7, 9, 7, 9, 7, 9],
                    [4, 3, 4, 3, 4, 3],
                    [7, 9, 7, 9, 7, 9],
                    [4, 3, 4, 3, 4, 3],
                    [7, 9, 7, 9, 7, 9],
                    [4, 3, 4, 3, 4, 3]
                ],
                'program': 'RepeatGrid(identity, vertical_repeats=3, horizontal_repeats=3)',
                'score': 0.6666666666666666,
                'program_str': '{"operation": "repeat_grid", "parameters": {"inner_command": {"operation": "identity", "parameters": {}}, "vertical_repeats": 3, "horizontal_repeats": 3}}',
                'explanation': 'The program repeats the input grid 3 times vertically and 3 times horizontally, effectively creating a 3x3 tiled version of the original grid.',
                'alternatives': [
                    {
                        'program': 'Sequence(RepeatGrid(vertical), RepeatGrid(horizontal))',
                        'score': 0.6666666666666666,
                        'program_str': '{"operation": "sequence", "parameters": {"commands": [{"operation": "repeat_grid", "parameters": {"inner_command": {"operation": "identity", "parameters": {}}, "vertical_repeats": 3, "horizontal_repeats": 1}}, {"operation": "repeat_grid", "parameters": {"inner_command": {"operation": "identity", "parameters": {}}, "vertical_repeats": 1, "horizontal_repeats": 3}}]}}',
                        'explanation': 'This program applies a sequence of two transformations to the input grid. First, it repeats the grid vertically three times, and then it repeats the resulting grid horizontally three times.'
                    },
                    {
                        'program': 'NestedRepeatGrid',
                        'score': 0.6666666666666666,
                        'program_str': '{"operation": "repeat_grid", "parameters": {"inner_command": {"operation": "repeat_grid", "parameters": {"inner_command": {"operation": "identity", "parameters": {}}, "vertical_repeats": 3, "horizontal_repeats": 1}}, "vertical_repeats": 1, "horizontal_repeats": 3}}',
                        'explanation': 'This program repeats a grid three times vertically and then repeats the result three times horizontally, effectively creating a 3x3 tiling of the original grid.'
                    }
                ]
            },
            {
                'success': True,
                'result_grid': [
                    [7, 9, 7, 9, 7, 9],
                    [4, 3, 4, 3, 4, 3],
                    [9, 7, 7, 9, 7, 9],
                    [3, 4, 4, 3, 4, 3],
                    [7, 9, 7, 9, 7, 9],
                    [4, 3, 4, 3, 4, 3]
                ],
                'program': 'MaskCombinator(swap_rows_or_columns)',
                'score': 0.7777777777777778,
                'program_str': '{"operation": "mask_combinator", "parameters": {"inner_command": {"operation": "swap_rows_or_columns", "parameters": {"row_swap": [0, 1], "col_swap": [0, 1], "swap_type": "both"}}, "mask_func": "lambda grid: np.array([[False, False, False, False, False, False], [False, False, False, False, False, False], [True, True, True, True, True, True], [True, True, True, True, True, True], [False, False, False, False, False, False], [False, False, False, False, False, False]])"}}',
                'explanation': 'This program applies a transformation to specific elements of the grid based on a boolean mask. The transformation swaps rows and columns at indices 0 and 1, and the mask selects the middle two rows of the grid.',
                'alternatives': [
                    {
                        'program': 'ApplyToRow(swap_rows_or_columns, row_index=2)',
                        'score': 0.7222222222222222,
                        'program_str': '{"operation": "apply_to_row", "parameters": {"inner_command": {"operation": "swap_rows_or_columns", "parameters": {"row_swap": [0, 1], "col_swap": [0, 1], "swap_type": "both"}}, "row_index": 2}}',
                        'explanation': 'This program applies a transformation to the third row of the grid, swapping the first and second rows and columns within that row.'
                    },
                    {
                        'program': 'ApplyToRow(swap_rows_or_columns, row_index=3)',
                        'score': 0.7222222222222222,
                        'program_str': '{"operation": "apply_to_row", "parameters": {"inner_command": {"operation": "swap_rows_or_columns", "parameters": {"row_swap": [0, 1], "col_swap": [0, 1], "swap_type": "both"}}, "row_index": 3}}',
                        'explanation': 'Applies a transformation that swaps rows and columns to the 4th row of the grid.'
                    }
                ]
            }
        ],
        'puzzle_1': [
            {
                'success': True,
                'result_grid': [
                    [8, 6, 8, 6, 8, 6],
                    [6, 4, 6, 4, 6, 4],
                    [8, 6, 8, 6, 8, 6],
                    [6, 4, 6, 4, 6, 4],
                    [8, 6, 8, 6, 8, 6],
                    [6, 4, 6, 4, 6, 4]
                ],
                'program': 'RepeatGrid(identity, vertical_repeats=3, horizontal_repeats=3)',
                'score': 0.6666666666666666,
                'program_str': '{"operation": "repeat_grid", "parameters": {"inner_command": {"operation": "identity", "parameters": {}}, "vertical_repeats": 3, "horizontal_repeats": 3}}',
                'explanation': 'The program repeats the input grid 3 times vertically and 3 times horizontally, effectively creating a 3x3 tiled version of the original grid.',
                'alternatives': [
                    {
                        'program': 'Sequence(RepeatGrid(vertical), RepeatGrid(horizontal))',
                        'score': 0.6666666666666666,
                        'program_str': '{"operation": "sequence", "parameters": {"commands": [{"operation": "repeat_grid", "parameters": {"inner_command": {"operation": "identity", "parameters": {}}, "vertical_repeats": 3, "horizontal_repeats": 1}}, {"operation": "repeat_grid", "parameters": {"inner_command": {"operation": "identity", "parameters": {}}, "vertical_repeats": 1, "horizontal_repeats": 3}}]}}',
                        'explanation': 'This program applies a sequence of two transformations to the input grid. First, it repeats the grid vertically three times, and then it repeats the resulting grid horizontally three times.'
                    },
                    {
                        'program': 'Sequence(RepeatGrid(vertical), SwapRows, RepeatGrid(horizontal))',
                        'score': 0.6666666666666666,
                        'program_str': '{"operation": "sequence", "parameters": {"commands": [{"operation": "repeat_grid", "parameters": {"inner_command": {"operation": "identity", "parameters": {}}, "vertical_repeats": 3, "horizontal_repeats": 1}}, {"operation": "apply_to_row", "parameters": {"inner_command": {"operation": "swap_rows_or_columns", "parameters": {"row_swap": [0, 1], "col_swap": [0, 1], "swap_type": "rows"}}, "row_index": 1}}, {"operation": "repeat_grid", "parameters": {"inner_command": {"operation": "identity", "parameters": {}}, "vertical_repeats": 1, "horizontal_repeats": 3}}]}}',
                        'explanation': 'This program applies a sequence of transformations to an input grid. First, it repeats the grid vertically three times. Then, it swaps the first and second rows. Finally, it repeats the resulting grid horizontally three times.'
                    }
                ]
            },
            {
                'success': True,
                'result_grid': [
                    [8, 6, 8, 6, 8, 6],
                    [6, 4, 6, 4, 6, 4],
                    [6, 8, 6, 8, 6, 8],
                    [4, 6, 4, 6, 4, 6],
                    [8, 6, 8, 6, 8, 6],
                    [6, 4, 6, 4, 6, 4]
                ],
                'program': 'Sequence(ApplyToRow(reverse), ApplyToRow(reverse))',
                'score': 1.0,
                'program_str': '{"operation": "sequence", "parameters": {"commands": [{"operation": "apply_to_row", "parameters": {"inner_command": {"operation": "reverse_row", "parameters": {}}, "row_index": 2}}, {"operation": "apply_to_row", "parameters": {"inner_command": {"operation": "reverse_row", "parameters": {}}, "row_index": 3}}]}}',
                'explanation': 'This program applies a sequence of two transformations to a grid. First, it reverses the elements in the third row, and then it reverses the elements in the fourth row.',
                'alternatives': [
                    {
                        'program': 'SwapRows(row2, row3)',
                        'score': 0.8333333333333334,
                        'program_str': '{"operation": "swap_rows_or_columns", "parameters": {"row_swap": [2, 3], "col_swap": null, "swap_type": "rows"}}',
                        'explanation': 'Swaps the third and fourth rows of the grid.'
                    },
                    {
                        'program': 'ApplyToRow(reverse, row_index=2)',
                        'score': 0.8333333333333334,
                        'program_str': '{"operation": "apply_to_row", "parameters": {"inner_command": {"operation": "reverse_row", "parameters": {}}, "row_index": 2}}',
                        'explanation': 'Reverses the elements in the third row of the grid.'
                    }
                ]
            }
        ]
    }
    return build_programs_from_train_results(train_results)

        
def test_multi_grid_analyzer_tool_integration(
    creative_llm_client,
    train_results
):
    """
    Tests MultiGridAnalyzerTool end-to-end with real data.
    
    Goes through:
    - Tool initialization
    - Schema validation
    - Agent delegation
    - LLM interaction
    """
    # Arrange: Get sample data from fixtures
    data = {
        'train': [
            {'input': [[7, 9], [4, 3]],
            'output': [
                [7, 9, 7, 9, 7, 9],
                [4, 3, 4, 3, 4, 3],
                [9, 7, 9, 7, 9, 7],
                [3, 4, 3, 4, 3, 4],
                [7, 9, 7, 9, 7, 9],
                [4, 3, 4, 3, 4, 3]
                ]
            },
            {'input': [[8, 6], [6, 4]],
            'output': [
                [8, 6, 8, 6, 8, 6],
                [6, 4, 6, 4, 6, 4],
                [6, 8, 6, 8, 6, 8],
                [4, 6, 4, 6, 4, 6],
                [8, 6, 8, 6, 8, 6],
                [6, 4, 6, 4, 6, 4]
                ]
            }
            ],
        'test': [{'input': [[3, 2], [7, 8]]}]}
    # Arrange: Create tool with injected LLM
    tool = MultiGridAnalyzerTool(llm=creative_llm_client)


    raw_output = tool._run(
        data = data,
        train_results = train_results,
        prompt_hint = None,
        # analysis_mode="results_only"
        analysis_mode="features_only"
        # analysis_mode="both"
        )
    
    assert raw_output