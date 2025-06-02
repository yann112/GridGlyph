# Quick-Start Two-Agent LangChain System Roadmap

## Step 1: Define Clear Agent Roles & Interfaces

### Analyze Agent
- **Input:** ARC input/output grids
- **Tasks:**
  - Extract key features from grids (colors, shapes, differences, symmetries)
  - Summarize transformation hints (e.g., “color swap”, “fill region”, “shift pattern”)
- **Output:** Structured analysis report or feature summary for Synthesize Agent

### Synthesize Agent
- **Input:** Analysis report + ARC input/output grids
- **Tasks:**
  - Propose candidate programs based on hints
  - Run program executions (via your program executor tool)
  - Validate outputs, compare to target output
  - Provide success/failure feedback
- **Output:** Best candidate program or retry requests

---

## Step 2: Wrap Existing Components as Tools

- **GridAnalyzerTool:** Wrap your feature extractor and difference analyzer for Analyze Agent
- **ProgramSynthesizerTool:** Wrap your program synthesizer for Synthesize Agent
- **ProgramExecutorTool:** Wrap your DSL interpreter for executing candidate programs

> Tools should have clean APIs with input validation, error handling, and logging.

---

## Step 3: Build Minimal Prompts

### Analyze Agent Prompt

