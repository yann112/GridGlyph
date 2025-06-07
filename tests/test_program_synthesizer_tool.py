import pytest
import numpy as np
from tools.program_synthesizer_tool import ProgramSynthesizerTool
from core.llm import OpenRouterClient
from core.synthesis_engine import SynthesisEngine

@pytest.mark.integration
def test_program_synthesizer_tool_end_to_end():
    """Test the full tool workflow with real components."""
    # Initialize the tool with real dependencies
    tool = ProgramSynthesizerTool(
        llm=OpenRouterClient(),
        synthesizer=SynthesisEngine()
    )

    # Simple test case - horizontal repeat
    input_grid = [
        [1, 2],
        [3, 4]
    ]
    output_grid = [
        [1, 2, 1, 2],
        [3, 4, 3, 4]
    ]
    analysis = "The pattern shows horizontal repetition"

    # Execute the tool
    result = tool._run(input_grid, output_grid, analysis)

    # Validate the result structure
    assert isinstance(result, dict), "Tool should return a dictionary"
    assert "program" in result, "Missing program in output"
    assert "score" in result, "Missing score in output"
    # Verify the score indicates perfect match
    assert result["score"] == 1.0, (
        f"Expected perfect score (1.0), got {result['score']}"
    )

@pytest.mark.integration
def test_horizontal_flip():
    tool = ProgramSynthesizerTool(llm=OpenRouterClient(), synthesizer=SynthesisEngine())

    input_grid = [[1, 2], [3, 4]]
    output_grid = [[2, 1], [4, 3]]

    analysis = "The grid appears to be horizontally flipped."

    result = tool._run(input_grid, output_grid, analysis)

    assert result["score"] == 1.0
    assert "flip" in result["program"].lower()
    assert "horizontal" in result["program"].lower()

def test_vertical_flip():
    tool = ProgramSynthesizerTool(llm=OpenRouterClient(), synthesizer=SynthesisEngine())

    input_grid = [[1, 2], [3, 4]]
    output_grid = [[3, 4], [1, 2]]

    analysis = "The grid appears to be vertically flipped."

    result = tool._run(input_grid, output_grid, analysis)

    assert result["score"] == 1.0
    assert "flip" in result["program"].lower()
    assert "vertical" in result["program"].lower()

def test_alternating_rows():
    tool = ProgramSynthesizerTool(llm=OpenRouterClient(), synthesizer=SynthesisEngine())

    input_grid = [[7, 9], [4, 3]]
    output_grid = [[7, 9], [3, 4]]  # Identity row, then flipped row

    analysis = "Rows alternate between original and horizontally flipped versions."

    result = tool._run(input_grid, output_grid, analysis)

    assert result["score"] == 1.0

def test_nested_transformation():
    tool = ProgramSynthesizerTool(llm=OpenRouterClient(), synthesizer=SynthesisEngine())

    input_grid = [[1, 2], [3, 4]]
    output_grid = [[2, 1, 2, 1], [4, 3, 4, 3], [2, 1, 2, 1], [4, 3, 4, 3]]

    analysis = "The grid is flipped horizontally and then repeated both vertically and horizontally."

    result = tool._run(input_grid, output_grid, analysis)

    assert result["score"] == 1.0

def test_color_mapping():
    tool = ProgramSynthesizerTool(llm=OpenRouterClient(), synthesizer=SynthesisEngine())

    input_grid = [[1, 2], [2, 1]]
    output_grid = [[9, 8], [8, 9]]

    analysis = "All instances of color 1 are replaced with 9, and 2 with 8."

    result = tool._run(input_grid, output_grid, analysis)

    assert result["score"] == 1.0