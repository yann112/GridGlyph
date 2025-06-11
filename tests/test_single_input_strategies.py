import pytest
import numpy as np

from strategies.strategy_factory import SingleInputStrategyFactory

# Mocking dependencies
class MockAnalyzer:
    def _run(self, input_grid, output_grid, prompt_hint=None):
        return "Grid appears to require flipping horizontally."

class MockSynthesizer:
    def _run(self, input_grid, output_grid, analysis_summary):
        # Simulate returning a valid program
        return {
            "success": True,
            "program": {"operation": "flip_h", "parameters": {}},
            "result_grid": np.array([[2, 1], [4, 3]]),  # Flip horizontal
            "score": 1.0,
            "program_str": '{"operation": "flip_h", "parameters": {}}',
            "explanation": "Flips the grid horizontally."
        }

# Ensure the strategy is auto-registered
def test_greedy_strategy_registered():
    assert "greedy" in SingleInputStrategyFactory.list_strategies(), "Greedy strategy not registered"

# Now test that it runs successfully
def test_greedy_strategy_synthesize():

    input_grid = np.array([[1, 2], [3, 4]])
    output_grid = np.array([[2, 1], [4, 3]])

    strategy = SingleInputStrategyFactory.create_strategy(
        "greedy",
        analyzer=MockAnalyzer(),
        synthesizer=MockSynthesizer()
    )

    result = strategy.synthesize(input_grid, output_grid)

    assert isinstance(result, list), "synthesize() should return a list"
    assert len(result) > 0, "At least one solution should be found"
    best_solution = result[0]
    assert best_solution["success"] is True, "Should return successful solution"
    assert best_solution["score"] == 1.0, "Solution should perfectly match"
    assert best_solution["program_str"] == '{"operation": "flip_h", "parameters": {}}'