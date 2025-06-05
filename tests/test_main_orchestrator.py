#!/usr/bin/env python3
"""
Test module for ARC Solver Pipeline
Tests the integration of analyzer and synthesizer agents with real puzzle data.
"""

import pytest
import numpy as np
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch

# Import the pipeline module (adjust import path as needed)
from core.main_orchestrator import ARCSolverPipeline, format_solution_results_for_display


class TestARCSolverPipeline:
    """Test suite for the ARC Solver Pipeline with real puzzle data."""
    
    @pytest.fixture
    def sample_puzzle_data(self) -> Dict:
        """Provide sample ARC puzzle data for testing.
        
        Returns:
            Dictionary containing training and test examples.
        """
        return {
            'train': [
                {
                    'input': [[7, 9], [4, 3]], 
                    'output': [[7, 9, 7, 9, 7, 9], [4, 3, 4, 3, 4, 3], 
                              [9, 7, 9, 7, 9, 7], [3, 4, 3, 4, 3, 4], 
                              [7, 9, 7, 9, 7, 9], [4, 3, 4, 3, 4, 3]]
                }, 
                {
                    'input': [[8, 6], [6, 4]], 
                    'output': [[8, 6, 8, 6, 8, 6], [6, 4, 6, 4, 6, 4], 
                              [6, 8, 6, 8, 6, 8], [4, 6, 4, 6, 4, 6], 
                              [8, 6, 8, 6, 8, 6], [6, 4, 6, 4, 6, 4]]
                }
            ], 
            'test': [{'input': [[3, 2], [7, 8]]}]
        }
    
    @pytest.fixture
    def training_input_grids(self, sample_puzzle_data: Dict) -> List[np.ndarray]:
        """Extract training input grids as numpy arrays.
        
        Args:
            sample_puzzle_data: Sample puzzle data fixture.
            
        Returns:
            List of numpy arrays representing input grids.
        """
        return [np.array(example['input']) for example in sample_puzzle_data['train']]
    
    @pytest.fixture
    def training_output_grids(self, sample_puzzle_data: Dict) -> List[np.ndarray]:
        """Extract training output grids as numpy arrays.
        
        Args:
            sample_puzzle_data: Sample puzzle data fixture.
            
        Returns:
            List of numpy arrays representing output grids.
        """
        return [np.array(example['output']) for example in sample_puzzle_data['train']]
    
    @pytest.fixture
    def test_input_grid(self, sample_puzzle_data: Dict) -> np.ndarray:
        """Extract test input grid as numpy array.
        
        Args:
            sample_puzzle_data: Sample puzzle data fixture.
            
        Returns:
            Numpy array representing the test input grid.
        """
        return np.array(sample_puzzle_data['test'][0]['input'])
    
    @pytest.fixture
    def mock_llm_client(self) -> Mock:
        """Create a mock LLM client for testing.
        
        Returns:
            Mock object representing the LLM client.
        """
        mock_client = Mock()
        mock_client.generate_response.return_value = "Mock LLM response"
        return mock_client
    
    @pytest.fixture
    def mock_synthesis_engine(self) -> Mock:
        """Create a mock synthesis engine for testing.
        
        Returns:
            Mock object representing the synthesis engine.
        """
        mock_engine = Mock()
        mock_engine.synthesize.return_value = "repeat_grid(identity(), 3, 3)"
        return mock_engine
    
    @pytest.fixture
    def test_logger(self) -> logging.Logger:
        """Create a test logger for pipeline testing.
        
        Returns:
            Logger instance configured for testing.
        """
        logger = logging.getLogger('test_pipeline')
        logger.setLevel(logging.DEBUG)
        return logger
    
    def test_pipeline_initialization_with_defaults(self):
        """Test that pipeline initializes correctly with default parameters."""
        pipeline = ARCSolverPipeline()
        
        assert pipeline.llm_client is not None
        assert pipeline.synthesis_engine is not None
        assert pipeline.analyzer_agent is not None
        assert pipeline.synthesizer_agent is not None
        assert pipeline.logger is not None
    
    def test_pipeline_initialization_with_custom_components(self, mock_llm_client: Mock, test_logger: logging.Logger):
        """Test pipeline initialization with custom components.
        
        Args:
            mock_llm_client: Mock LLM client fixture.
            test_logger: Test logger fixture.
        """
        pipeline = ARCSolverPipeline(
            llm_client=mock_llm_client,
            verbose=True,
            logger=test_logger
        )
        
        assert pipeline.llm_client == mock_llm_client
        assert pipeline.logger == test_logger
        assert pipeline.verbose is True
    
    def test_solve_single_puzzle_with_valid_data(self, 
                                               training_input_grids: List[np.ndarray],
                                               training_output_grids: List[np.ndarray],
                                               test_input_grid: np.ndarray,
                                               mock_llm_client: Mock,
                                               test_logger: logging.Logger):
        """Test solving a single puzzle with valid training data.
        
        Args:
            training_input_grids: Training input grids fixture.
            training_output_grids: Training output grids fixture.
            test_input_grid: Test input grid fixture.
            mock_llm_client: Mock LLM client fixture.
            test_logger: Test logger fixture.
        """
        with patch('agents.analyze_agent.AnalyzeAgent') as mock_analyze_agent_class, \
             patch('agents.synthesize_agent.SynthesizeAgent') as mock_synthesize_agent_class:
            
            # Setup mock agent instances
            mock_analyzer = Mock()
            mock_analyzer.analyze.return_value = "Pattern detected: 3x3 grid repetition with alternating rows"
            mock_analyze_agent_class.return_value = mock_analyzer
            
            mock_synthesizer = Mock()
            mock_synthesizer.synthesize.return_value = "repeat_grid(identity(), 3, 3)"
            mock_synthesize_agent_class.return_value = mock_synthesizer
            
            # Initialize pipeline with mocks
            pipeline = ARCSolverPipeline(
                llm_client=mock_llm_client,
                verbose=True,
                logger=test_logger
            )
            
            # Execute solve_single_puzzle
            puzzle_results = pipeline.solve_single_puzzle(
                training_input_grids=training_input_grids,
                training_output_grids=training_output_grids,
                test_input_grid=test_input_grid,
                maximum_synthesis_attempts=2
            )
            
            # Verify results structure
            assert isinstance(puzzle_results, dict)
            assert 'success' in puzzle_results
            assert 'analysis_summary' in puzzle_results
            assert 'synthesis_attempts' in puzzle_results
            assert 'error_messages' in puzzle_results
            
            # Verify successful execution
            assert puzzle_results['success'] is True
            assert puzzle_results['analysis_summary'] is not None
            assert len(puzzle_results['synthesis_attempts']) > 0
            
            # Verify mock calls
            mock_analyzer.analyze.assert_called()
            mock_synthesizer.synthesize.assert_called()
    
    def test_solve_single_puzzle_with_mismatched_training_data(self, 
                                                             training_input_grids: List[np.ndarray],
                                                             test_input_grid: np.ndarray):
        """Test error handling when training input and output grids have different lengths.
        
        Args:
            training_input_grids: Training input grids fixture.
            test_input_grid: Test input grid fixture.
        """
        pipeline = ARCSolverPipeline()
        
        # Create mismatched output grids (different length)
        mismatched_output_grids = [np.array([[1, 2], [3, 4]])]  # Only one output vs two inputs
        
        with pytest.raises(ValueError, match="Training input and output grids must have the same length"):
            pipeline.solve_single_puzzle(
                training_input_grids=training_input_grids,
                training_output_grids=mismatched_output_grids,
                test_input_grid=test_input_grid
            )
    
    def test_load_puzzle_from_json_file_with_valid_data(self, sample_puzzle_data: Dict):
        """Test loading and solving puzzle from a JSON file.
        
        Args:
            sample_puzzle_data: Sample puzzle data fixture.
        """
        with patch('agents.analyze_agent.AnalyzeAgent') as mock_analyze_agent_class, \
             patch('agents.synthesize_agent.SynthesizeAgent') as mock_synthesize_agent_class:
            
            # Setup mock agents
            mock_analyzer = Mock()
            mock_analyzer.analyze.return_value = "Test analysis result"
            mock_analyze_agent_class.return_value = mock_analyzer
            
            mock_synthesizer = Mock()
            mock_synthesizer.synthesize.return_value = "test_program()"
            mock_synthesize_agent_class.return_value = mock_synthesizer
            
            # Create temporary JSON file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(sample_puzzle_data, temp_file)
                temp_file_path = temp_file.name
            
            try:
                pipeline = ARCSolverPipeline()
                puzzle_results = pipeline.load_puzzle_from_json_file(temp_file_path)
                
                assert puzzle_results['success'] is True
                assert 'analysis_summary' in puzzle_results
                assert 'synthesis_attempts' in puzzle_results
                
            finally:
                Path(temp_file_path).unlink()  # Clean up temporary file
    
    def test_load_puzzle_from_nonexistent_file(self):
        """Test error handling when loading from a non-existent file."""
        pipeline = ARCSolverPipeline()
        
        with pytest.raises(FileNotFoundError):
            pipeline.load_puzzle_from_json_file("nonexistent_file.json")
    
    def test_load_puzzle_from_invalid_json_file(self):
        """Test error handling when loading from a file with invalid JSON."""
        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_file.write("{ invalid json content")
            temp_file_path = temp_file.name
        
        try:
            pipeline = ARCSolverPipeline()
            
            with pytest.raises(json.JSONDecodeError):
                pipeline.load_puzzle_from_json_file(temp_file_path)
                
        finally:
            Path(temp_file_path).unlink()  # Clean up temporary file
    
    def test_analyze_training_examples_with_multiple_examples(self, 
                                                            training_input_grids: List[np.ndarray],
                                                            training_output_grids: List[np.ndarray]):
        """Test analysis of multiple training examples.
        
        Args:
            training_input_grids: Training input grids fixture.
            training_output_grids: Training output grids fixture.
        """
        with patch('agents.analyze_agent.AnalyzeAgent') as mock_analyze_agent_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze.side_effect = [
                "First example: horizontal repetition pattern",
                "Second example: similar horizontal repetition pattern"
            ]
            mock_analyze_agent_class.return_value = mock_analyzer
            
            pipeline = ARCSolverPipeline()
            
            analysis_result = pipeline._analyze_training_examples(
                training_input_grids, training_output_grids
            )
            
            assert "Example 1:" in analysis_result
            assert "Example 2:" in analysis_result
            assert "horizontal repetition pattern" in analysis_result
            assert mock_analyzer.analyze.call_count == 2
    
    def test_synthesize_transformation_programs_with_multiple_attempts(self, 
                                                                     training_input_grids: List[np.ndarray],
                                                                     training_output_grids: List[np.ndarray]):
        """Test synthesis of transformation programs with multiple attempts.
        
        Args:
            training_input_grids: Training input grids fixture.
            training_output_grids: Training output grids fixture.
        """
        with patch('agents.synthesize_agent.SynthesizeAgent') as mock_synthesize_agent_class:
            mock_synthesizer = Mock()
            mock_synthesizer.synthesize.side_effect = [
                "repeat_grid(identity(), 3, 3)",
                "repeat_grid(identity(), 3, 3)",  # Duplicate (should be filtered)
                "tile_pattern(input_grid, 3, 3)"   # Different program
            ]
            mock_synthesize_agent_class.return_value = mock_synthesizer
            
            pipeline = ARCSolverPipeline()
            
            candidate_programs = pipeline._synthesize_transformation_programs(
                input_grid=training_input_grids[0],
                output_grid=training_output_grids[0],
                analysis_summary="Test analysis summary",
                maximum_attempts=3
            )
            
            # Should have 2 unique programs (duplicate filtered out)
            assert len(candidate_programs) == 2
            assert "repeat_grid(identity(), 3, 3)" in candidate_programs
            assert "tile_pattern(input_grid, 3, 3)" in candidate_programs
    
    def test_format_solution_results_for_display_success(self):
        """Test formatting of successful solution results for display."""
        successful_results = {
            'success': True,
            'analysis_summary': 'Grid repetition pattern detected',
            'synthesis_attempts': ['program1', 'program2']
        }
        
        formatted_output = format_solution_results_for_display(successful_results)
        
        assert "✅ Puzzle solved successfully!" in formatted_output
        assert "Grid repetition pattern detected" in formatted_output
        assert "Generated 2 programs" in formatted_output
    
    def test_format_solution_results_for_display_failure(self):
        """Test formatting of failed solution results for display."""
        failed_results = {
            'success': False,
            'error_messages': ['Analysis failed', 'Synthesis failed']
        }
        
        formatted_output = format_solution_results_for_display(failed_results)
        
        assert "❌ Failed to solve puzzle" in formatted_output
        assert "Analysis failed" in formatted_output
        assert "Synthesis failed" in formatted_output


@pytest.mark.integration
class TestARCSolverPipelineIntegration:
    """Integration tests for the ARC Solver Pipeline with real components."""
    
    def test_pipeline_integration_with_repeat_grid_pattern(self):
        """Test complete pipeline integration with repeat grid pattern data.
        
        This test requires actual AnalyzeAgent and SynthesizeAgent implementations.
        Mark as integration test to run separately from unit tests.
        """
        # Sample data representing the repeat grid pattern
        training_input_grids = [
            np.array([[7, 9], [4, 3]]),
            np.array([[8, 6], [6, 4]])
        ]
        
        training_output_grids = [
            np.array([
                [7, 9, 7, 9, 7, 9], [4, 3, 4, 3, 4, 3], 
                [9, 7, 9, 7, 9, 7], [3, 4, 3, 4, 3, 4], 
                [7, 9, 7, 9, 7, 9], [4, 3, 4, 3, 4, 3]
            ]),
            np.array([
                [8, 6, 8, 6, 8, 6], [6, 4, 6, 4, 6, 4], 
                [6, 8, 6, 8, 6, 8], [4, 6, 4, 6, 4, 6], 
                [8, 6, 8, 6, 8, 6], [6, 4, 6, 4, 6, 4]
            ])
        ]
        
        test_input_grid = np.array([[3, 2], [7, 8]])
        
        # Skip this test if actual agents are not available
        try:
            pipeline = ARCSolverPipeline(verbose=True)
            
            puzzle_results = pipeline.solve_single_puzzle(
                training_input_grids=training_input_grids,
                training_output_grids=training_output_grids,
                test_input_grid=test_input_grid,
                maximum_synthesis_attempts=3
            )
            
            # Basic checks for integration test
            assert isinstance(puzzle_results, dict)
            assert 'success' in puzzle_results
            assert 'analysis_summary' in puzzle_results
            assert 'synthesis_attempts' in puzzle_results
            
            # Log results for manual inspection
            print(f"Integration test results: {puzzle_results}")
            
        except ImportError as import_error:
            pytest.skip(f"Integration test skipped due to missing dependencies: {import_error}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])