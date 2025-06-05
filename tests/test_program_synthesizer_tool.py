import pytest
import numpy as np
from tools.program_synthesizer_tool import ProgramSynthesizerTool

@pytest.mark.integration
def test_program_synthesizer_tool():
    # Initialize the tool
    tool = ProgramSynthesizerTool()

    # Define input and output grids
    input_grid = [[0, 0], [1, 1]]
    output_grid = [[1, 1], [0, 0]]
    analysis_summary = "Colors are inverted. Pattern is symmetric. Recoloring may be required."

    # Call the tool's _run method
    result = tool._run(input_grid, output_grid, analysis_summary)

    # Check if the result is not an error message
    assert result != "Error: No valid programs found", "No valid programs found"

    # Convert the result to a list of lists for comparison
    result_grid = eval(result)

    # Check if the result grid matches the expected output grid
    assert result_grid == output_grid, f"Unexpected result grid: {result_grid}"
