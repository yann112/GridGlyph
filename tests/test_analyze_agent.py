import pytest
from core.llm import OpenRouterClient
from agents.analyze_agent import AnalyzeAgent

@pytest.mark.integration
def test_analyze_agent_returns_summary():
    llm = OpenRouterClient()
    agent = AnalyzeAgent(llm)

    input_grid = "[[0, 0], [1, 1]]"
    output_grid = "[[1, 1], [0, 0]]"

    result = agent.analyze(input_grid, output_grid)

    assert result is not None and result.strip() != "", "Analyze agent returned empty result"
    assert "recolor" in result.lower() or "transform" in result.lower() or "pattern" in result.lower(), \
        f"Unexpected analysis result: {result}"
