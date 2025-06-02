class SynthesizeAgent:
    def __init__(self, llm, synthesizer):
        self.llm = llm
        self.synthesizer = synthesizer

    def generate_program_candidates(self, input_grid, output_grid, analysis_summary):
        """Generates multiple DSL program candidates via LLM."""
        prompt = f"""
        Given these grids and analysis, suggest 3-5 DSL programs that could transform:
        
        Input Grid: {input_grid}
        Output Grid: {output_grid}
        Analysis: {analysis_summary}
        
        Return ONLY valid DSL commands, one per line. Example:
        repeat_grid(identity(), 3, 3)
        flip_h(identity())
        """
        return self.llm(prompt).strip().split('\n')

    def synthesize(self, input_grid, output_grid, analysis_summary):
        """Full pipeline: Generate candidates → Validate → Return best program."""
        candidates = self.generate_program_candidates(input_grid, output_grid, analysis_summary)

        return candidates