import pytest
import numpy as np
from core.llm import OpenRouterClient
from agents.synthesize_agent import SynthesizeAgent
from core.synthesis_engine import SynthesisEngine
from core.dsl_nodes import RepeatGrid, Identity

@pytest.mark.integration
def test_synthesize_agent_finds_perfect_program():
    """Verify the agent can find perfect-match programs for basic transformations."""
    # Setup - use real components
    llm = OpenRouterClient()
    synthesizer = SynthesisEngine()
    agent = SynthesizeAgent(llm, synthesizer)

    # Test case - horizontal repeat (should be easily solvable)
    input_grid = np.array([
        [1, 2],
        [3, 4]
    ])
    output_grid = np.array([
        [1, 2, 1, 2],
        [3, 4, 3, 4]
    ])
    analysis = "Horizontal repetition of the input pattern"

    # Execute
    results = agent.synthesize(input_grid, output_grid, analysis)
    
    # Debug output
    print("\n=== Synthesis Results ===")
    for idx, (program, score) in enumerate(results[:3], 1):  # Show top 3
        output = program.execute(input_grid)
        print(f"\nProgram #{idx} (Score: {score:.2f}):")
        print(f"DSL: {program}")
        print("Output:")
        print(output)

    # Core assertions
    assert results, "No programs were generated"
    
    # Verify we have at least one perfect match
    perfect_programs = [(p, s) for p, s in results if s == 1.0]
    assert perfect_programs, (
        f"No perfect matches found (best score: {results[0][1]:.2f})\n"
        f"Top program output:\n{results[0][0].execute(input_grid)}"
    )

    # Verify at least one correct RepeatGrid exists
    repeat_grid_programs = [
        p for p, _ in results 
        if isinstance(p, RepeatGrid) 
        and p.horizontal_repeats == 2
        and p.vertical_repeats == 1
    ]
    assert repeat_grid_programs, (
        "Expected to find RepeatGrid(Identity(), vertical=1, horizontal=2)\n"
        f"Found program types: {[type(p).__name__ for p, _ in results]}"
    )

    # Verify the perfect program actually works
    best_program, best_score = results[0]
    assert best_score == 1.0, f"Top program score should be 1.0 (got {best_score:.2f})"
    assert np.array_equal(
        best_program.execute(input_grid),
        output_grid
    ), "Best program doesn't produce exact output"