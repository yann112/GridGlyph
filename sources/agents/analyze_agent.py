class AnalyzeAgent:
    def __init__(self, llm):
        self.llm = llm

    def analyze(self, input_grid, output_grid, hint=None):
        prompt = f"""
            You are the Analyze Agent for the ARC challenge. Your job is to compare the input and output grid and describe the main transformations needed.

            ## Input Grid:
            {input_grid}

            ## Output Grid:
            {output_grid}
            """
        if hint:
            prompt += f"\nAdditional instructions or hints:\n{hint}\n"

        prompt += """List differences, transformations (e.g., move, recolor, duplicate), symmetries, or patterns you detect. Use clear language."""

        return self.llm(prompt)

