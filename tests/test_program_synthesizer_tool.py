import pytest
import numpy as np
from tools.program_synthesizer_tool import ProgramSynthesizerTool
from core.llm import OpenRouterClient
from core.synthesis_engine import SynthesisEngine
from fixtures import stable_llm_client



@pytest.fixture
def synthesis_engine():
    """Fixture providing a synthesis engine instance."""
    return SynthesisEngine()


@pytest.fixture
def program_synthesizer_tool(stable_llm_client, synthesis_engine):
    """
    Fixture providing a ProgramSynthesizerTool with stable LLM configuration.
    
    Args:
        stable_llm_client: Stable OpenRouter client fixture
        synthesis_engine: Synthesis engine fixture
        
    Returns:
        ProgramSynthesizerTool: Configured tool ready for testing
    """
    return ProgramSynthesizerTool(
        llm=stable_llm_client,
        synthesizer=synthesis_engine
    )


@pytest.mark.integration
def test_program_synthesizer_tool_end_to_end(program_synthesizer_tool):
    """Test the full tool workflow with real components."""
    # Simple test case - horizontal repeat
    input_grid = [
                [7, 9, 7, 9, 7, 9],
                [4, 3, 4, 3, 4, 3],
                [7, 9, 7, 9, 7, 9],
                [4, 3, 4, 3, 4, 3],
                [7, 9, 7, 9, 7, 9],
                [4, 3, 4, 3, 4, 3]
        ]
    output_grid = [
                [7, 9, 7, 9, 7, 9],
                [4, 3, 4, 3, 4, 3],
                [9, 7, 9, 7, 9, 7],
                [3, 4, 3, 4, 3, 4],
                [7, 9, 7, 9, 7, 9],
                [4, 3, 4, 3, 4, 3]
        ]
    analysis = """
        #### 0.Rules
        - This analysis uses zero-based indexing. All row and column indices in this document are zero-based, When we refer to "row 2", we mean the row at index 2      
        
        #### 1. Base Structure or Repeating Pattern in the Input
        The input grid is a 6x6 matrix with alternating values of 7 and 9 in each row, and alternating values of 4 and 3 in each row. This creates a repeating pattern of 7, 9, 7, 9, 7, 9 and 4, 3, 4, 3, 4, 3.

        #### 2. Deviations from the Base Structure
        The output grid deviates from the input grid in the following ways:
        - **Row 2:** The values 9, 7, 9, 7, 9, 7 are changed to 7, 9, 7, 9, 7, 9.
        - **Row 3:** The values 3, 4, 3, 4, 3, 4 are changed to 4, 3, 4, 3, 4, 3.

        #### 3. Possible Explanations
        1. **Most Likely Explanation:**
        - The values in rows 2 and 3 were swapped. Specifically, the 9s and 7s in row 2 were swapped with the 3s and 4s in row 3, respectively.

        2. **Alternative View:**
        - The values in rows 2 and 3 were rotated 180 degrees. This would mean that the values in row 2 were reflected across the horizontal axis, and the values in row 3 were reflected across the horizontal axis.

        3. **Speculative or Exploratory Ide a:**
        - The values in rows 2 and 3 were transposed. This would mean that the values in row 2 were moved to the positions of the values in row 3, and vice versa.

        #### 4. Transformations That Don't Fit
        - **Rotation:** The grid does not show any signs of rotation (90, 180, or 270 degrees).
        - **Tiling:** The grid does not show any signs of tiling.
        - **Color Filtering:** The grid does not show any signs of color filtering, as the colors remain the same.

        #### 5. Testing These Ideas
        - **Swapping Rows:**
        - Expected Result: If rows 2 and 3 were swapped, the output grid should match the input grid with the values in row 2 moved to row 3 and vice versa.

        - **Rotating Rows:**
        - Expected Result: If rows 2 and 3 were rotated 180 degrees, the output grid should show the values in row 2 reflected across the horizontal axis and the values in row 3 reflected across the horizontal axis.

        - **Transposing Rows:**
        - Expected Result: If rows 2 and 3 were transposed, the output grid should show the values in row 2 moved to the positions of the values in row 3, and vice versa.

        ### Conclusion
        The most likely explanation is that the values in rows 2 and 3 were swapped. This hypothesis aligns with the observed changes in the output grid and does not contradict any other features or patterns in the grid. Further testing by swapping rows 2 and 3 in the input grid should confirm this hypothesis.
    """

    # Execute the tool
    result = program_synthesizer_tool._run(input_grid, output_grid, analysis)

    # Validate the result structure
    assert isinstance(result, dict), "Tool should return a dictionary"
    assert "program" in result, "Missing program in output"
    assert "score" in result, "Missing score in output"
    # Verify the score indicates perfect match
    assert result["score"] == 1.0, (
        f"Expected perfect score (1.0), got {result['score']}"
    )
  

@pytest.mark.integration
def test_horizontal_flip(program_synthesizer_tool):
    """Test horizontal flip transformation."""
    input_grid = [[1, 2], [3, 4]]
    output_grid = [[2, 1], [4, 3]]
    analysis = "The grid appears to be horizontally flipped."

    result = program_synthesizer_tool._run(input_grid, output_grid, analysis)

    assert result["score"] == 1.0
    assert "flip" in result["program"].lower()
    assert "horizontal" in result["program"].lower()


@pytest.mark.integration
def test_vertical_flip(program_synthesizer_tool):
    """Test vertical flip transformation."""
    input_grid = [[1, 2], [3, 4]]
    output_grid = [[3, 4], [1, 2]]
    analysis = "The grid appears to be vertically flipped."

    result = program_synthesizer_tool._run(input_grid, output_grid, analysis)

    assert result["score"] == 1.0
    assert "flip" in result["program"].lower()
    assert "vertical" in result["program"].lower()


@pytest.mark.integration
def test_alternating_rows(program_synthesizer_tool):
    """Test alternating row transformation."""
    input_grid = [[7, 9], [4, 3]]
    output_grid = [[7, 9], [3, 4]]  # Identity row, then flipped row
    analysis = "Rows alternate between original and horizontally flipped versions."

    result = program_synthesizer_tool._run(input_grid, output_grid, analysis)

    assert result["score"] == 1.0


@pytest.mark.integration
def test_nested_transformation(program_synthesizer_tool):
    """Test complex nested transformation (flip + repeat)."""
    input_grid = [[1, 2], [3, 4]]
    output_grid = [[2, 1, 2, 1], [4, 3, 4, 3], [2, 1, 2, 1], [4, 3, 4, 3]]
    analysis = "The grid is flipped horizontally and then repeated both vertically and horizontally."

    result = program_synthesizer_tool._run(input_grid, output_grid, analysis)

    assert result["score"] == 1.0


@pytest.mark.integration
def test_color_mapping(program_synthesizer_tool):
    """Test color mapping transformation."""
    input_grid = [[1, 2], [2, 1]]
    output_grid = [[9, 8], [8, 9]]
    analysis = "All instances of color 1 are replaced with 9, and 2 with 8."

    result = program_synthesizer_tool._run(input_grid, output_grid, analysis)

    assert result["score"] == 1.0


# Additional test to verify fixture configuration
def test_stable_llm_client_configuration(stable_llm_client):
    """Test that the stable LLM client has correct configuration."""
    assert stable_llm_client.temperature == 0.0
    assert stable_llm_client.top_p == 0.1
    assert stable_llm_client.top_k == 1
    assert stable_llm_client.repetition_penalty == 1.0
    assert stable_llm_client.max_tokens == 800

