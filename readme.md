GridGlyph

GridGlyph is a deterministic system designed to solve visual logic puzzles by finding the mathematical rules that govern them.
Project Overview

The system operates by translating visual transformations into a custom Domain Specific Language (DSL). Instead of relying on a generative model to "guess" what an output grid looks like, we use a small language model to suggest a formal logic program which is then executed and verified by code.
How it Works
1. The DSL (Domain Specific Language)

We have a library of Python functions that perform specific geometric and logical actions, such as:

    Movements and Translations

    Symmetries and Flips

    Rotations

    Object Filtering

Each of these functions is mapped to a unique symbolic character, or sigil. These sigils allow the language model to communicate complex transformations using a very compact string of tokens.
2. Dataset Generation

We build our training data from scratch. Starting from a "ground truth" set of core logical rules, we use our own generators to produce thousands of unique puzzle instances. By varying colors, sizes, and noise while keeping the underlying rule constant, we train the model to focus on the invariant logic rather than the specific pixels.
3. The Execution Loop

    Proposal: The language model analyzes a set of input/output examples and proposes a rule expressed in sigils.

    Execution: The system interprets these sigils and runs the corresponding Python functions on the input grid.

    Validation: The resulting grid is compared to the target. Because the process is symbolic, the result is binary: the rule is either 100% mathematically correct, or it is rejected.

Design Philosophy

    Deterministic: We eliminate hallucination by requiring the AI to provide a verifiable program rather than a raw image.

    Frugal: By using a compact DSL and a small language model, the system is designed to be efficient and specialized for industrial-grade logic tasks.

    Simple: The architecture prioritizes a robust, usable tool over unnecessary complexity. Usage decides future complexity, not assumptions.