class SynthesizeAgent:
    def __init__(self, llm):
        self.llm = llm

    def synthesize(self, input_grid, output_grid, analysis_summary):
        prompt = f"""
            You are the Synthesize Agent for the ARC challenge. Your task is to generate a DSL program that transforms the input grid into the output grid, using the given analysis.

            ## Input Grid:
            {input_grid}

            ## Output Grid:
            {output_grid}

            ## Analysis Summary:
            {analysis_summary}

            Propose a transformation program in DSL. If unsure, make a reasonable attempt.
            """
        return self.llm(prompt)
