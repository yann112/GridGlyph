Orchestrator Workflow v2

    First Analysis

        Ask analyzer to study input → output grids (initial patterns)

    Initial Synthesis

        Ask synthesizer to generate programs + score them

    Iteration Loop (if score not perfect)

        Compare: Ask analyzer to find differences (best output vs target)

        Select Strategy: Choose refinement approach based on:

            Error type from analyzer

            Previous strategy performance

        Refine: Ask synthesizer to extend programs using selected strategy

        Repeat until max iterations or perfect match

Greedy Strategy (Default Approach)

What It Does

    Makes incremental fixes without backtracking

    Each iteration:

        Identifies exact differences (e.g., "tiles need flipping")

        Builds minimal patch for those errors

Why We Start With It

    Solves most ARC problems with simple steps

    No memory needed → lightweight

    Naturally handles:

        Tiling + modifications

        Local transformations

        Layered patterns

Key Principles

    Analyzer = "The Detective" (finds patterns/differences)

    Synthesizer = "The Worker" (builds programs using current strategy)

    Strategies = "Toolbox" (greedy/backtracking/etc.)

    Orchestrator = "Manager" (routes tasks, never makes decisions)

✅ Strengths

    Minimal core workflow

    Analyzer guides everything

    Synthesizer stays dumb

    Easy to debug

⚠️ Watch Out For

    Start with only greedy strategy

    Add more strategies only when:

        You see specific failure cases

        Greedy gets stuck on 20% of problems

Evolution Path

    Perfect the greedy flow

    Add strategy registry (but keep greedy default)

    Introduce new strategies only for proven needs

This keeps your system laser-focused while leaving the door open for future flexibility. The greedy strategy alone will likely solve most problems cleanly.