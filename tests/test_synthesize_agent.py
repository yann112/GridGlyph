import pytest
from core.llm import OpenRouterClient
from agents.synthesize_agent import SynthesizeAgent
from core.synthesis_engine import SynthesisEngine


@pytest.mark.integration
def test_synthesize_agent_returns_program():
    llm = OpenRouterClient()
    synthesizer = SynthesisEngine()
    agent = SynthesizeAgent(llm, synthesizer)

    input_grid = "[[0, 0], [1, 1]]"
    output_grid = "[[1, 1], [0, 0]]"
    analysis_summary = "Colors are inverted. Pattern is symmetric. Recoloring may be required."

    result = agent.synthesize(input_grid, output_grid, analysis_summary)

    assert result is not None and result.strip() != "", "Synthesize agent returned empty program"
    assert "dsl" in result.lower() or "program" in result.lower() or "transform" in result.lower(), \
        f"Unexpected program output: {result}"
