import pytest
import numpy as np
from tools.program_synthesizer_tool import ProgramSynthesizerTool
from core.llm import OpenRouterClient
from core.synthesis_engine import SynthesisEngine


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
        model="mistralai/mistral-small-3.1-24b-instruct",
        temperature=0.0,      # Completely deterministic
        top_p=0.1,           # Very restrictive sampling
        top_k=1,             # Most likely token only
        repetition_penalty=1.0,  # Neutral repetition
        max_tokens=800,      # Sufficient for code generation
    )


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
        [1, 2],
        [3, 4]
    ]
    output_grid = [
        [1, 2, 1, 2],
        [3, 4, 3, 4]
    ]
    analysis = "The pattern shows horizontal repetition"

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

