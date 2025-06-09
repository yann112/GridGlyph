import numpy as np
import pytest
from core.features_analysis import ProblemAnalyzer
from tools.program_synthesizer_tool import ProgramSynthesizerTool
from tools.grid_analyzer_tool import GridAnalyzerTool
from core.llm import OpenRouterClient
from core.synthesis_engine import SynthesisEngine
from tools.main_orchestrator import ARCProblemOrchestrator


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
        model="mistralai/devstral-small:free",
        temperature=0.0,      # Completely deterministic
        top_p=0.1,           # Very restrictive sampling
        top_k=1,             # Most likely token only
        repetition_penalty=1.0,  # Neutral repetition
        max_tokens=2000, 
    )

@pytest.fixture
def creative_llm_client():
    """
    Fixture providing a creative OpenRouter client optimized for diverse and creative program synthesis.

    Uses creative settings:
    - temperature=0.7: Introduces randomness for varied outputs
    - top_p=0.9: Allows for a wide range of token sampling
    - top_k=50: Considers a broader set of likely tokens
    - repetition_penalty=1.2: Slightly discourages repetition

    Returns:
        OpenRouterClient: Configured client for creative program generation
    """
    return OpenRouterClient(
        model="mistralai/devstral-small:free",
        temperature=0.7,      # Introduces randomness
        top_p=0.9,           # Wide token sampling
        top_k=50,            # Considers more tokens
        repetition_penalty=1.2, # Slightly discourages repetition
        max_tokens=2000,
    )

@pytest.fixture
def setup_orchestrator(stable_llm_client, creative_llm_client):

    # Use tools instead of agents directly
    analyze_tool = GridAnalyzerTool(llm=creative_llm_client)
    synth_engine = SynthesisEngine()
    synth_tool = ProgramSynthesizerTool(llm=stable_llm_client, synthesizer=synth_engine)
    
    # Create orchestrator with real dependencies
    orchestrator = ARCProblemOrchestrator(
        analyzer=analyze_tool,
        synthesizer=synth_tool
    )
    return orchestrator

def test_orchestrator_with_flipped_repetition(setup_orchestrator):
    # Test data
    input_grid = [[7, 9], [4, 3]]  # Use lists instead of numpy arrays
    output_grid = [
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3],
        [9, 7, 9, 7, 9, 7],
        [3, 4, 3, 4, 3, 4],
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3]
    ]
    
    # Run the orchestrator
    result = setup_orchestrator.solve(input_grid, output_grid, max_iterations=3)
    
    # Print results for debugging
    print("\n" + "="*50)
    print("ORCHESTRATOR TEST RESULTS")
    print("="*50)
    
    print(f"\nFinal Solution:")
    print(result["solution"])
    
    print(f"\nConfidence Score: {result['confidence']:.2f}")
    print(f"Iterations Completed: {result['iterations_completed']}")
    
    print(f"\nKey Insights:")
    for i, insight in enumerate(result.get('insights', []), 1):
        print(f"  {i}. {insight}")
    
    print(f"\nIteration Details:")
    for iteration in result.get('iteration_details', []):
        print(f"  Iteration {iteration['iteration']}: "
              f"Strategy={iteration['strategy'][:30]}..., "
              f"Confidence={iteration['confidence']:.2f}, "
              f"Success={iteration['success']}")
    
    if result.get('failed_approaches'):
        print(f"\nFailed Approaches:")
        for failure in result['failed_approaches']:
            print(f"  - {failure['approach']}: {failure['reason']}")
    
    print(f"\nSummary:")
    print(result.get('summary', 'No summary available'))
    
    # Verify key requirements
    solution = result["solution"]
    
    # 1. Check that a solution was found
    assert solution != "No satisfactory solution found", "Orchestrator should find some solution"
    
    # 2. Check that the solution mentions repetition/tiling patterns
    repetition_keywords = ["repeat", "tiling", "tile", "pattern", "replicate", "duplicate"]
    assert any(word in solution.lower() for word in repetition_keywords), \
        f"Solution should mention repetition patterns. Solution: {solution[:200]}..."
    
    # 3. Check that alternation/flipping is detected
    alternation_keywords = ["alternate", "flip", "every other", "modification", "invert", "reverse"]
    assert any(word in solution.lower() for word in alternation_keywords), \
        f"Solution should detect alternation/flipping. Solution: {solution[:200]}..."
    
    # 4. Verify the orchestrator completed multiple iterations
    assert result['iterations_completed'] >= 1, "Should complete at least one iteration"
    
    # 5. Check that we have some confidence in the solution
    assert result['confidence'] > 0, "Should have some confidence in the solution"
    
    # 6. Verify iteration details are captured
    assert 'iteration_details' in result, "Should capture iteration details"
    assert len(result['iteration_details']) > 0, "Should have at least one iteration detail"
    
    # 7. Check that strategies were used
    strategies_used = [iter_detail['strategy'] for iter_detail in result.get('iteration_details', [])]
    assert len(strategies_used) > 0, "Should have used at least one strategy"
    
    # 8. Verify insights were generated
    assert len(result.get('insights', [])) > 0, "Should generate at least one insight"

    print("\n" + "="*50)
    print("ALL ORCHESTRATOR ASSERTIONS PASSED!")
    print("="*50)

def test_orchestrator_edge_cases(setup_orchestrator):
    """Test orchestrator with edge cases"""
    
    # Test with simple identity transformation
    simple_input = [[1, 2], [3, 4]]
    simple_output = [[1, 2], [3, 4]]
    
    result = setup_orchestrator.solve(simple_input, simple_output, max_iterations=2)
    
    print(f"\nEdge Case Test - Identity Transformation:")
    print(f"Solution: {result['solution'][:100]}...")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Should still find a solution (identity transformation)
    assert result['solution'] != "No satisfactory solution found"
    assert result['confidence'] > 0
    
    print("Edge case test passed!")

def test_orchestrator_error_handling(setup_orchestrator):
    """Test orchestrator error handling"""
    
    # Test with invalid input (empty grids)
    try:
        result = setup_orchestrator.solve([], [], max_iterations=1)
        # Should handle gracefully, not crash
        assert 'error' in result or result['solution'] != "No satisfactory solution found"
        print("Error handling test passed!")
    except Exception as e:
        # If it does throw an exception, it should be informative
        assert len(str(e)) > 0
        print(f"Error handling test passed with exception: {str(e)[:100]}...")
