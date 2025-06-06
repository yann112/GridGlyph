from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import numpy as np
from enum import Enum, auto
from core.llm import OpenRouterClient, LLMClient


class ARCStrategy(Enum):
    """Defined strategies the orchestrator can employ"""
    PATTERN_FOCUS = auto()
    TRANSFORMATION_FOCUS = auto()
    OBJECT_FOCUS = auto()
    SPATIAL_FOCUS = auto()
    REFINEMENT = auto()


class ARCProblemInput(BaseModel):
    input_grid: List[List[int]] = Field(..., description="The input ARC grid as 2D list of integers")
    output_grid: List[List[int]] = Field(..., description="The target output ARC grid as 2D list of integers")
    max_iterations: Optional[int] = Field(3, description="Maximum refinement iterations to perform")


class IterationContext:
    """Enhanced context tracking that works with existing tools"""
    def __init__(self):
        self.iteration_history = []
        self.best_solution = None
        self.best_confidence = 0.0
        self.failed_approaches = []
        self.insights = []
        self._current_strategy = None
    
    def add_iteration(self, iteration_num: int, analysis: str, solution: str, 
                     confidence: float, success: bool):
        record = {
            "iteration": iteration_num,
            "strategy": self._current_strategy,
            "analysis": analysis,
            "solution": solution,
            "confidence": confidence,
            "success": success
        }
        self.iteration_history.append(record)
        
        if success and confidence > self.best_confidence:
            self.best_solution = solution
            self.best_confidence = confidence
    
    def set_current_strategy(self, strategy: str):
        """Set the strategy for the current iteration"""
        self._current_strategy = strategy
    
    def add_failed_approach(self, reason: str):
        self.failed_approaches.append({
            "strategy": self._current_strategy,
            "reason": reason,
            "iteration": len(self.iteration_history)
        })
    
    def add_insight(self, insight: str):
        self.insights.append(insight)
    
    def get_summary(self) -> str:
        """Generate summary compatible with existing tools"""
        summary = f"Completed {len(self.iteration_history)} iterations.\n"
        if self.best_solution:
            summary += f"Best solution confidence: {self.best_confidence:.2f}\n"
        if self.failed_approaches:
            summary += f"Failed approaches: {len(self.failed_approaches)}\n"
        if self.insights:
            summary += f"Key insights: {len(self.insights)}\n"
        return summary


class ARCProblemOrchestrator:
    """Orchestrator that works with existing tools while improving strategy handling"""
    
    name: str = "arc_problem_orchestrator"
    description: str = "Coordinates pattern analysis and program synthesis for ARC problems"
    args_schema = ARCProblemInput
    
    def __init__(self, llm: LLMClient = None, analyze_tool=None, synth_tool=None):
        self.llm = llm or OpenRouterClient()
        self.analyze_tool = analyze_tool
        self.synth_tool = synth_tool
        self._strategy_cycle = [
            "pattern_focus",
            "transformation_focus",
            "object_focus",
            "spatial_focus"
        ]
        
    def solve(self, input_grid: List[List[int]], 
              output_grid: List[List[int]], 
              max_iterations: int = 3) -> Dict[str, Any]:
        """Main solving interface that works with existing tools"""
        
        context = IterationContext()
        input_np = np.array(input_grid)
        output_np = np.array(output_grid)
        
        # Initial analysis using existing tool interface
        try:
            initial_analysis = self._get_initial_analysis(input_np, output_np)
            context.add_insight("Initial analysis completed")
        except Exception as e:
            return {"error": f"Initial analysis failed: {str(e)}"}
        
        # Iterative refinement loop
        for iteration in range(max_iterations):
            try:
                # Determine strategy (rotating through options)
                strategy = self._determine_strategy(iteration, context)
                context.set_current_strategy(strategy)
                
                # Execute iteration using existing tool interfaces
                result = self._execute_iteration(
                    iteration,
                    strategy,
                    input_np,
                    output_np,
                    context,
                    initial_analysis
                )
                
                # Update context
                context.add_iteration(
                    iteration,
                    result["analysis"],
                    result["solution"],
                    result["confidence"],
                    result["success"]
                )
                
                # Early termination check
                if result.get("should_stop", False):
                    break
                    
            except Exception as e:
                context.add_failed_approach(str(e))
                continue
        
        return self._compile_final_result(context, input_grid, output_grid)
    
    def _get_initial_analysis(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """Get initial analysis using existing tool interface"""
        if self.analyze_tool:
            return self.analyze_tool._run(
                input_grid=input_grid.tolist(),
                output_grid=output_grid.tolist(),
                prompt_hint=None
            )
        else:
            prompt = f"""
            Analyze this ARC puzzle transformation:
            
            Input Grid:
            {self._format_grid(input_grid)}
            
            Output Grid:
            {self._format_grid(output_grid)}
            
            Identify key patterns, transformations, and relationships.
            """
            return self.llm(prompt)
    
    def _determine_strategy(self, iteration: int, context: IterationContext) -> str:
        """Determine strategy while maintaining compatibility"""
        # Simple rotating strategy for compatibility
        return self._strategy_cycle[iteration % len(self._strategy_cycle)]
    
    def _execute_iteration(self, iteration: int, strategy: str, input_grid: np.ndarray,
                         output_grid: np.ndarray, context: IterationContext,
                         initial_analysis: str) -> Dict[str, Any]:
        """Execute iteration using existing tool interfaces"""
        
        # Get analysis (pass strategy as hint to maintain compatibility)
        analysis = self._get_enhanced_analysis(
            input_grid, output_grid, strategy, context
        )
        
        # Generate solution using existing tool interface
        solution_result = self._generate_solution(
            strategy, analysis, input_grid, output_grid, context
        )
        
        # Evaluate solution
        evaluation = self._evaluate_solution(
            solution_result.get("program", ""),
            input_grid,
            output_grid,
            solution_result.get("score", 0.0)
        )
        
        return {
            "analysis": analysis,
            "solution": solution_result.get("program", ""),
            "confidence": evaluation["confidence"],
            "success": evaluation["success"],
            "should_stop": evaluation["confidence"] > 0.8 or iteration >= 2,
            "reason": evaluation["reason"]
        }
    
    def _get_enhanced_analysis(self, input_grid: np.ndarray, output_grid: np.ndarray,
                             strategy: str, context: IterationContext) -> str:
        """Get analysis using existing tool interface"""
        if self.analyze_tool:
            return self.analyze_tool._run(
                input_grid=input_grid.tolist(),
                output_grid=output_grid.tolist(),
                prompt_hint=f"Focus on: {strategy}"
            )
        else:
            prompt = f"""
            Re-analyze with focus on {strategy}:
            
            Input Grid:
            {self._format_grid(input_grid)}
            
            Output Grid:
            {self._format_grid(output_grid)}
            
            Previous attempts: {context.get_summary()}
            """
            return self.llm(prompt)
    
    def _generate_solution(self, strategy: str, analysis: str,
                         input_grid: np.ndarray, output_grid: np.ndarray,
                         context: IterationContext) -> Dict[str, Any]:
        """Generate solution using existing tool interface without passing strategy"""
        if self.synth_tool:
            # Only pass the parameters that the existing tool accepts
            return self.synth_tool._run(
                input_grid=input_grid.tolist(),
                output_grid=output_grid.tolist(),
                analysis_summary=f"Strategy: {strategy}\n{analysis}"  # Embed strategy in analysis
            )
        else:
            prompt = f"""
            Generate solution using {strategy} approach:
            
            Analysis:
            {analysis}
            """
            return {
                "program": self.llm(prompt),
                "score": 0.7  # Default confidence
            }
    
    def _evaluate_solution(self, solution: str, input_grid: np.ndarray,
                          output_grid: np.ndarray, score: float) -> Dict[str, Any]:
        """Evaluate solution while maintaining compatibility"""
        confidence = min(1.0, score)
        success = confidence > 0.6
        
        if confidence > 0.8:
            reason = "Excellent match"
        elif confidence > 0.6:
            reason = "Good match"
        else:
            reason = "Poor match"
            
        return {
            "confidence": confidence,
            "success": success,
            "reason": reason
        }
    
    def _compile_final_result(self, context: IterationContext, 
                             input_grid: List[List[int]], 
                             output_grid: List[List[int]]) -> Dict[str, Any]:
        """Compile final result compatible with existing expectations"""
        return {
            "solution": context.best_solution or "No solution found",
            "confidence": context.best_confidence,
            "iterations_completed": len(context.iteration_history),
            "insights": context.insights,
            "failed_approaches": context.failed_approaches,
            "iteration_details": [
                {
                    "iteration": i["iteration"],
                    "strategy": i["strategy"],
                    "confidence": i["confidence"],
                    "success": i["success"]
                }
                for i in context.iteration_history
            ],
            "summary": context.get_summary(),
            "input_grid": input_grid,
            "output_grid": output_grid
        }
    
    def _format_grid(self, grid: np.ndarray) -> str:
        """Format grid for display"""
        return "\n".join(" ".join(str(int(cell)) for cell in row) for row in grid)
    
    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")