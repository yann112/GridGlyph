#!/usr/bin/env python3
"""
ARC Solver Pipeline - Main Orchestrator
Integrates analyzer and synthesizer agents for end-to-end ARC puzzle solving.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from core.llm import OpenRouterClient
from agents.analyze_agent import AnalyzeAgent
from agents.synthesize_agent import SynthesizeAgent
from core.synthesis_engine import SynthesisEngine


class ARCSolverPipeline:
    """Main orchestrator that coordinates analyzer and synthesizer agents to solve ARC puzzles."""
    
    def __init__(
            self, 
            llm_client: Optional[OpenRouterClient] = None,
            verbose: bool = False,
            logger: logging.Logger = None
            ):
        """Initialize the ARC solver pipeline.
        
        Args:
            llm_client: LLM client for agents (creates default if None).
            verbose: Enable detailed logging.
            logger: Logger instance (creates default if None).
        """
        self.logger = logger or logging.getLogger(__name__)
        self.verbose = verbose
        self._setup_logging()
        
        # Initialize core components
        self.llm_client = llm_client or OpenRouterClient()
        self.synthesis_engine = SynthesisEngine()
        
        # Initialize agents
        self.analyzer_agent = AnalyzeAgent(self.llm_client)
        self.synthesizer_agent = SynthesizeAgent(self.llm_client, self.synthesis_engine)
        
        self.logger.info("ARC Solver Pipeline initialized successfully")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration based on verbosity level."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('arc_solver.log')
            ]
        )
    
    def solve_single_puzzle(self, 
                           training_input_grids: List[np.ndarray],
                           training_output_grids: List[np.ndarray], 
                           test_input_grid: np.ndarray,
                           maximum_synthesis_attempts: int = 3) -> Dict:
        """Solve a single ARC puzzle using the integrated pipeline.
        
        Args:
            training_input_grids: List of input grids for training examples.
            training_output_grids: List of output grids for training examples.
            test_input_grid: Input grid to solve.
            maximum_synthesis_attempts: Maximum number of synthesis attempts.
            
        Returns:
            Dictionary containing solution results and metadata.
            
        Raises:
            ValueError: If training grids lists have different lengths.
        """
        if len(training_input_grids) != len(training_output_grids):
            raise ValueError("Training input and output grids must have the same length")
        
        self.logger.info(f"Starting puzzle solve with {len(training_input_grids)} training pairs")
        
        puzzle_results = {
            'success': False,
            'predicted_output': None,
            'analysis_summary': None,
            'synthesis_attempts': [],
            'error_messages': []
        }
        
        try:
            # Step 1: Analyze training examples
            analysis_summary = self._analyze_training_examples(
                training_input_grids, training_output_grids
            )
            puzzle_results['analysis_summary'] = analysis_summary
            
            if not analysis_summary:
                error_message = "Failed to generate analysis summary from training examples"
                puzzle_results['error_messages'].append(error_message)
                self.logger.error(error_message)
                return puzzle_results
            
            # Step 2: Synthesize transformation programs
            candidate_programs = self._synthesize_transformation_programs(
                training_input_grids[0], training_output_grids[0], 
                analysis_summary, maximum_synthesis_attempts
            )
            puzzle_results['synthesis_attempts'] = candidate_programs
            
            if not candidate_programs:
                error_message = "No candidate transformation programs were generated"
                puzzle_results['error_messages'].append(error_message)
                self.logger.error(error_message)
                return puzzle_results
            
            # For now, just return the first candidate program as "solution"
            # TODO: Implement program execution and validation when DSL interpreter is ready
            puzzle_results['success'] = True
            puzzle_results['predicted_output'] = "Program generated (execution not implemented yet)"
            self.logger.info("✅ Pipeline completed successfully - generated candidate programs")
                
        except Exception as pipeline_error:
            error_message = f"Pipeline execution error: {str(pipeline_error)}"
            self.logger.error(error_message)
            puzzle_results['error_messages'].append(error_message)
        
        return puzzle_results
    
    def _analyze_training_examples(self, 
                                 training_input_grids: List[np.ndarray], 
                                 training_output_grids: List[np.ndarray]) -> str:
        """Analyze training examples to understand the transformation pattern.
        
        Args:
            training_input_grids: List of input grids from training examples.
            training_output_grids: List of output grids from training examples.
            
        Returns:
            Summary of the analysis as a string.
        """
        try:
            individual_analyses = []
            
            for example_index, (input_grid, output_grid) in enumerate(
                zip(training_input_grids, training_output_grids)
            ):
                self.logger.debug(f"Analyzing training example {example_index + 1}")
                
                # Convert numpy arrays to string format for agent processing
                input_grid_string = str(input_grid.tolist())
                output_grid_string = str(output_grid.tolist())
                
                # Get analysis from analyzer agent
                example_analysis = self.analyzer_agent.analyze(input_grid_string, output_grid_string)
                individual_analyses.append(f"Example {example_index + 1}: {example_analysis}")
            
            # Combine all individual analyses
            combined_analysis = "\n".join(individual_analyses)
            
            self.logger.info("Successfully completed analysis of all training examples")
            return combined_analysis
                
        except Exception as analysis_error:
            error_message = f"Training example analysis failed: {str(analysis_error)}"
            self.logger.error(error_message)
            return ""
    
    def _synthesize_transformation_programs(self, 
                                          input_grid: np.ndarray,
                                          output_grid: np.ndarray,
                                          analysis_summary: str,
                                          maximum_attempts: int) -> List[str]:
        """Generate candidate transformation programs using the synthesizer agent.
        
        Args:
            input_grid: Primary input grid for synthesis.
            output_grid: Expected output grid for synthesis.
            analysis_summary: Analysis summary from the analyzer agent.
            maximum_attempts: Maximum number of synthesis attempts.
            
        Returns:
            List of candidate program strings.
        """
        candidate_programs = []
        
        try:
            for attempt_number in range(maximum_attempts):
                self.logger.debug(f"Synthesis attempt {attempt_number + 1}/{maximum_attempts}")
                
                # Convert grids to string format for synthesizer
                input_grid_string = str(input_grid.tolist())
                output_grid_string = str(output_grid.tolist())
                
                # Generate program using synthesizer agent
                generated_program = self.synthesizer_agent.synthesize(
                    input_grid_string, output_grid_string, analysis_summary
                )
                
                if generated_program and generated_program not in candidate_programs:
                    candidate_programs.append(generated_program)
                    self.logger.debug(f"Generated unique program: {generated_program}")
                
            self.logger.info(f"Generated {len(candidate_programs)} unique candidate programs")
                
        except Exception as synthesis_error:
            error_message = f"Program synthesis failed: {str(synthesis_error)}"
            self.logger.error(error_message)
        
        return candidate_programs
    
    def load_puzzle_from_json_file(self, puzzle_file_path: str) -> Dict:
        """Load and solve ARC puzzle from JSON file.
        
        Args:
            puzzle_file_path: Path to ARC puzzle JSON file.
            
        Returns:
            Solution results dictionary.
            
        Raises:
            FileNotFoundError: If the puzzle file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        try:
            with open(puzzle_file_path, 'r') as file_handle:
                puzzle_data = json.load(file_handle)
            
            # Extract training examples
            training_input_grids = [np.array(example['input']) for example in puzzle_data['train']]
            training_output_grids = [np.array(example['output']) for example in puzzle_data['train']]
            
            # Extract test input
            test_input_grid = np.array(puzzle_data['test'][0]['input'])
            
            return self.solve_single_puzzle(
                training_input_grids, training_output_grids, test_input_grid
            )
            
        except FileNotFoundError:
            error_message = f"Puzzle file not found: {puzzle_file_path}"
            self.logger.error(error_message)
            raise
        except json.JSONDecodeError as json_error:
            error_message = f"Invalid JSON in puzzle file {puzzle_file_path}: {str(json_error)}"
            self.logger.error(error_message)
            raise
        except Exception as load_error:
            error_message = f"Failed to load puzzle from {puzzle_file_path}: {str(load_error)}"
            self.logger.error(error_message)
            return {
                'success': False,
                'error_messages': [error_message]
            }


def create_command_line_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="ARC Puzzle Solver Pipeline")
    parser.add_argument(
        "--puzzle", "-p", 
        type=str, 
        help="Single puzzle JSON file to solve"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--max-attempts", 
        type=int, 
        default=3, 
        help="Maximum synthesis attempts per puzzle (default: 3)"
    )
    
    return parser


def format_solution_results_for_display(solution_results: Dict) -> str:
    """Format solution results for human-readable display.
    
    Args:
        solution_results: Dictionary containing solution results.
        
    Returns:
        Formatted string representation of the results.
    """
    if solution_results['success']:
        status_message = "✅ Puzzle solved successfully!"
        analysis_info = f"Analysis: {solution_results.get('analysis_summary', 'N/A')}"
        programs_info = f"Generated {len(solution_results.get('synthesis_attempts', []))} programs"
        return f"{status_message}\n{analysis_info}\n{programs_info}"
    else:
        status_message = "❌ Failed to solve puzzle"
        error_info = "Errors: " + "; ".join(solution_results.get('error_messages', ['Unknown error']))
        return f"{status_message}\n{error_info}"


def main():
    """Main entry point for the ARC solver pipeline command-line interface."""
    parser = create_command_line_parser()
    command_line_arguments = parser.parse_args()
    
    if not command_line_arguments.puzzle:
        parser.error("Must specify a puzzle file with --puzzle")
    
    # Initialize pipeline with appropriate logging
    pipeline_logger = logging.getLogger(__name__)
    arc_pipeline = ARCSolverPipeline(verbose=command_line_arguments.verbose, logger=pipeline_logger)
    
    try:
        print(f"Solving puzzle: {command_line_arguments.puzzle}")
        
        solution_results = arc_pipeline.load_puzzle_from_json_file(command_line_arguments.puzzle)
        
        # Display formatted results
        formatted_results = format_solution_results_for_display(solution_results)
        print(formatted_results)
    
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as main_error:
        print(f"❌ Error: {str(main_error)}")
        if command_line_arguments.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()