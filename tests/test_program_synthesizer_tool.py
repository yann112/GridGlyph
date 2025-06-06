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
    assert "success" in result, "Result should contain success status"
    
    if not result["success"]:
        pytest.fail(f"Tool failed with error: {result.get('error', 'Unknown error')}")

    # Check successful result structure
    assert result["success"] is True, "Tool should report success"
    assert "result_grid" in result, "Missing result_grid in output"
    assert "program" in result, "Missing program in output"
    assert "score" in result, "Missing score in output"
    assert "alternatives" in result, "Missing alternatives in output"

    # Verify the transformed grid matches expected
    assert result["result_grid"] == output_grid, (
        f"Transformed grid doesn't match expected output\n"
        f"Expected: {output_grid}\n"
        f"Got: {result['result_grid']}"
    )

    # Verify the score indicates perfect match
    assert result["score"] == 1.0, (
        f"Expected perfect score (1.0), got {result['score']}"
    )
