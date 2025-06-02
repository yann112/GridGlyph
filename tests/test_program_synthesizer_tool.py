import pytest
import numpy as np
from core.llm import OpenRouterClient
from agents.synthesize_agent import SynthesizeAgent
from core.synthesis_engine import SynthesisEngine


@pytest.mark.integration
def test_synthesize_agent_proposes_repeatgrid_dsl():
    llm = OpenRouterClient()
    synthesizer = SynthesisEngine()
    agent = SynthesizeAgent(llm, synthesizer)

    # Given - Grids that need 3x3 repetition
    input_grid = np.array([
        [7, 9],
        [4, 3]
    ])
    
    output_grid = np.array([
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3],
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3],
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3]
    ])
    
    analysis_summary = "Grid needs 3x3 repetition"

    # When
    result = agent.synthesize(str(input_grid.tolist()), 
                            str(output_grid.tolist()), 
                            analysis_summary)

    # Then - Check for DSL command format
    assert "repeat_grid" in result.lower(), "Missing repeat_grid command"
    assert "3" in result, "Missing repetition count"
    
    # Optional: Check for proper DSL structure
    assert "(" in result and ")" in result, "Malformed DSL command"
    assert "identity()" in result.lower() or "inner_command" in result.lower(), \
           "Missing base transformation"