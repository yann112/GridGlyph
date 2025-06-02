import pytest
from unittest.mock import MagicMock
import numpy as np

from sources.agents.synthesize_agent import SynthesizeAgent
from sources.core.synthesis_engine import SynthesisEngine
from sources.core.dsl_nodes import (
    AbstractTransformationCommand,
    FlipGridVertically,
    Identity,
    RepeatGrid
)

# @pytest.mark.integration
# def test_synthesize_agent_returns_program():
#     llm = OpenRouterClient() # This is an integration test, relies on external LLM
#     synthesizer = SynthesisEngine()
#     agent = SynthesizeAgent(llm, synthesizer)
#
#     input_grid = "[[0, 0], [1, 1]]" # Old test expected string grids
#     output_grid = "[[1, 1], [0, 0]]"
#     analysis_summary = "Colors are inverted. Pattern is symmetric. Recoloring may be required."
#
#     result = agent.synthesize(input_grid, output_grid, analysis_summary) # Old agent returned string
#
#     assert result is not None and result.strip() != "", "Synthesize agent returned empty program"
#     assert "dsl" in result.lower() or "program" in result.lower() or "transform" in result.lower(), \
#         f"Unexpected program output: {result}"


def test_synthesize_agent_logic():
    mock_llm = MagicMock()
    # Use a real SynthesisEngine as its logic is part of what we're testing indirectly
    # (i.e., that the agent uses it correctly for evaluation and sorting)
    synthesis_engine = SynthesisEngine() 
    agent = SynthesizeAgent(llm=mock_llm, synthesizer=synthesis_engine)

    # --- Test Scenario 1: Simple Valid Program ---
    input_grid_np_1 = np.array([[0, 1], [2, 3]])
    output_grid_np_1 = np.array([[2, 3], [0, 1]]) # Expected result of flip_v
    analysis_summary_1 = "A simple vertical flip"
    
    # Configure mock LLM to return a specific program string
    # generate_program_candidates splits by '\n', so a trailing newline is good.
    mock_llm.return_value = "flip_v()\n" 
    
    returned_programs_1 = agent.synthesize(input_grid_np_1, output_grid_np_1, analysis_summary_1)
    
    assert isinstance(returned_programs_1, list), "Scenario 1: Should return a list"
    assert len(returned_programs_1) == 1, "Scenario 1: Should return one program"
    assert isinstance(returned_programs_1[0], FlipGridVertically), "Scenario 1: Program should be FlipGridVertically"
    
    executed_output_1 = returned_programs_1[0].execute(input_grid_np_1)
    assert np.array_equal(executed_output_1, output_grid_np_1), "Scenario 1: Executed program output mismatch"

    # --- Test Scenario 2: Multiple Programs, Sorting, and Mismatches ---
    input_grid_np_2 = np.array([[1, 1], [1, 1]])
    # This output is perfectly matched by repeat_grid(identity(), 1, 1) or identity()
    output_grid_np_2_perfect = np.array([[1, 1], [1, 1]]) 
    analysis_summary_2 = "Testing multiple programs, expecting identity-like match first"

    # LLM returns:
    # 1. repeat_grid(identity(), 1, 1) -> should be best match
    # 2. flip_v() -> valid, but not a match
    # 3. malformed_program_() -> should be ignored by parser
    mock_llm.return_value = "repeat_grid(identity(), 1, 1)\nflip_v()\nmalformed_program_()"
    
    returned_programs_2 = agent.synthesize(input_grid_np_2, output_grid_np_2_perfect, analysis_summary_2)
    
    assert isinstance(returned_programs_2, list), "Scenario 2: Should return a list"
    # malformed_program_() is dropped by the parser, flip_v and repeat_grid are parsed
    assert len(returned_programs_2) == 2, \
        f"Scenario 2: Should return 2 programs, got {len(returned_programs_2)}"
    
    # The first program should be RepeatGrid(Identity(),1,1) because it scores perfectly (1.0)
    # The parser creates RepeatGrid(inner_command=Identity(), vertical_repeats=1, horizontal_repeats=1)
    assert isinstance(returned_programs_2[0], RepeatGrid), \
        "Scenario 2: First program should be RepeatGrid after sorting"
    
    executed_output_2_best = returned_programs_2[0].execute(input_grid_np_2)
    assert np.array_equal(executed_output_2_best, output_grid_np_2_perfect), \
        "Scenario 2: Best program output mismatch"

    # The second program should be FlipGridVertically
    assert isinstance(returned_programs_2[1], FlipGridVertically), \
        "Scenario 2: Second program should be FlipGridVertically"
    executed_output_2_second = returned_programs_2[1].execute(input_grid_np_2)
    assert not np.array_equal(executed_output_2_second, output_grid_np_2_perfect), \
        "Scenario 2: Second program output should not match the 'perfect' output"

    # --- Test Scenario 3: LLM returns no valid programs ---
    input_grid_np_3 = np.array([[0, 1], [2, 3]])
    output_grid_np_3 = np.array([[2, 3], [0, 1]]) # content doesn't matter much here
    analysis_summary_3 = "Testing only invalid programs from LLM"
    
    mock_llm.return_value = "unknown_command()\nanother_bad_one(arg1, arg2)\n"
    
    returned_programs_3 = agent.synthesize(input_grid_np_3, output_grid_np_3, analysis_summary_3)
    
    assert isinstance(returned_programs_3, list), "Scenario 3: Should return a list"
    assert len(returned_programs_3) == 0, "Scenario 3: Should return an empty list of programs"

    # --- Test Scenario 4: LLM returns empty string ---
    input_grid_np_4 = np.array([[0, 1], [2, 3]])
    output_grid_np_4 = np.array([[2, 3], [0, 1]])
    analysis_summary_4 = "Testing empty string from LLM"

    mock_llm.return_value = "" # Empty string
    
    returned_programs_4 = agent.synthesize(input_grid_np_4, output_grid_np_4, analysis_summary_4)
    
    assert isinstance(returned_programs_4, list), "Scenario 4: Should return a list"
    assert len(returned_programs_4) == 0, "Scenario 4: Should return an empty list for empty LLM output"

    # --- Test Scenario 5: Testing top_k_programs parameter ---
    # Uses same grids as scenario 2
    # LLM returns 3 valid programs that will all be parsed.
    # identity() will match perfectly.
    # repeat_grid(identity(), 1, 2) will produce a shape mismatch (2x4 from 2x2) -> score 0 / skipped by engine
    # flip_v() will not match.
    mock_llm.return_value = "identity()\nrepeat_grid(identity(), 1, 2)\nflip_v()"
    
    # Request top 1 program
    returned_programs_5_top1 = agent.synthesize(input_grid_np_2, output_grid_np_2_perfect, analysis_summary_2, top_k_programs=1)
    assert len(returned_programs_5_top1) == 1, "Scenario 5: top_k=1 should return 1 program"
    assert isinstance(returned_programs_5_top1[0], Identity), "Scenario 5: top_k=1 should be Identity"

    # Request top 2 programs
    # repeat_grid(identity(), 1, 2) will be parsed, but SynthesisEngine will not give it a good score due to shape mismatch.
    # So, identity() and flip_v() should be the ones remaining and sorted.
    returned_programs_5_top2 = agent.synthesize(input_grid_np_2, output_grid_np_2_perfect, analysis_summary_2, top_k_programs=2)
    assert len(returned_programs_5_top2) == 2, \
        f"Scenario 5: top_k=2 should return 2 programs, got {len(returned_programs_5_top2)}"
    assert isinstance(returned_programs_5_top2[0], Identity), "Scenario 5: top_k=2 first should be Identity"
    assert isinstance(returned_programs_5_top2[1], FlipGridVertically), "Scenario 5: top_k=2 second should be FlipGridVertically"

    # Request top 5 (more than available valid & scorable programs)
    returned_programs_5_top5 = agent.synthesize(input_grid_np_2, output_grid_np_2_perfect, analysis_summary_2, top_k_programs=5)
    assert len(returned_programs_5_top5) == 2, \
        "Scenario 5: top_k=5 should return 2 programs (all valid and scorable)"
    assert isinstance(returned_programs_5_top5[0], Identity)
    assert isinstance(returned_programs_5_top5[1], FlipGridVertically)

    # --- Test Scenario 6: Program that causes execution error ---
    input_grid_np_6 = np.array([[0, 1]])
    output_grid_np_6 = np.array([[0,1]])
    # flip_v would normally work, but let's mock its execute method to raise an error
    # However, we can't easily mock the execute method of a class that's instantiated *inside*
    # the agent's parsing logic and then passed to the engine.
    # Instead, let's rely on a program that might cause an error in the engine if not handled well,
    # e.g. if a DSL node was incorrectly implemented. For now, SynthesisEngine's evaluate_and_sort_candidates
    # already has a try-except around program execution.
    # We can test that this path is handled:
    # If a program is parsed but fails in engine's execution, it should be skipped.

    mock_llm.return_value = "identity()\nflip_v()" # flip_v will be the second program
    
    # Temporarily mock the execute method of FlipGridVertically instances created by SynthesisEngine
    # This is a bit tricky as the instances are created dynamically.
    # A more robust way would be to create a custom DSL node that always fails for testing.
    # For now, let's assume the SynthesisEngine's error handling for execute_program is tested elsewhere
    # or trust its current implementation. The agent itself doesn't execute, it passes to engine.

    # Let's slightly adjust: the SynthesisEngine logs errors and skips.
    # So, if flip_v() was "bad" and failed in engine, it wouldn't be in the output.
    # To test this behavior through the agent:
    
    original_flip_v_execute = FlipGridVertically.execute
    def mock_failing_execute(self, input_grid):
        raise ValueError("Mocked execution error")

    try:
        FlipGridVertically.execute = mock_failing_execute
        
        # identity() will score 1.0. flip_v() will be parsed, but its execution will fail in the engine.
        mock_llm.return_value = "identity()\nflip_v()"
        returned_programs_6 = agent.synthesize(input_grid_np_6, output_grid_np_6, "testing exec failure", top_k_programs=2)
        
        assert len(returned_programs_6) == 1, "Scenario 6: Should only return one program if the other fails execution"
        assert isinstance(returned_programs_6[0], Identity), "Scenario 6: The working program should be Identity"

    finally:
        FlipGridVertically.execute = original_flip_v_execute # Restore original method

```
