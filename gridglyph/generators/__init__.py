"""
Generators Package
------------------
Responsible for the neuro-symbolic generation of datasets by mutating 
Ground Truth DSL rules and validating them through the Symbolic Interpreter.
"""

# Absolute import for maximum reliability
from gridglyph.generators.build import build_dataset

# We alias build_dataset to build for a more pragmatic/minimalist API
build = build_dataset

__all__ = ["build", "build_dataset"]