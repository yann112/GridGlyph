# Features Analysis Module Overview

This module provides abstract interfaces and concrete implementations
to analyze ARC puzzle input/output grids by extracting various
types of features.

---

## Abstract Base Classes

### `AbstractProblemAnalyzer`
- Base class for all analyzers.
- Defines abstract method:
  - `analyze(input_grid: np.ndarray, output_grid: np.ndarray) -> Dict`
- Purpose: Extract features or patterns from given grids.

### `AbstractFeatureExtractor` (inherits `AbstractProblemAnalyzer`)
- Defines additional abstract methods:
  - `get_shape_features(input_grid, output_grid) -> Dict`
  - `get_value_features(input_grid, output_grid) -> Dict`
- Purpose: Extract shape and value features from grids.

### `AbstractPatternDetector` (inherits `AbstractProblemAnalyzer`)
- Defines abstract methods:
  - `detect_repetition(grid) -> Dict`
  - `detect_symmetry(grid) -> Dict`
- Purpose: Detect repeating patterns and symmetries.

---

## Concrete Implementations

### `BasicShapeAnalyzer` (inherits `AbstractFeatureExtractor`)
- Implements:
  - `analyze()`: Returns shape and density features.
  - `get_shape_features()`: Extracts grid size, aspect ratio, total cells.
  - `get_value_features()`: Calculates foreground ratio based on background color.

### `BasicPatternDetector` (inherits `AbstractPatternDetector`)
- Implements:
  - `analyze()`: Returns repetition and symmetry patterns.
  - `detect_repetition()`: Checks horizontal and vertical repetition.
  - `detect_symmetry()`: Checks horizontal, vertical, and diagonal symmetry.

### `ComponentFeatureExtractor` (inherits `AbstractFeatureExtractor`)
- Implements:
  - `analyze()`: Returns component stats for input and output grids.
  - `get_shape_features()`: Extracts number of connected components,
    average area, width, and height.
  - `get_value_features()`: Returns empty dict (not used).

### `RelationalFeatureExtractor` (inherits `AbstractFeatureExtractor`)
- Implements:
  - `analyze()`: Returns relational shape, color, and density features.
  - `get_relational_shape_features()`: Computes ratios of output to input shape.
  - `get_relational_color_features()`: Compares color sets and intersections.
  - `get_relational_density_features()`: Compares foreground density changes.

---

## Utilities

- `get_background_color(grid: np.ndarray) -> int`  
  Returns the most frequent color in the grid (background color).

- `extract_components(grid: np.ndarray, background_color: int) -> List[Dict]`  
  Extracts connected components as bounding boxes, areas, masks, centroids.

---

## Usage Notes

- The `ProblemAnalyzer` class coordinates all analyzers,
  accumulating features into an `AnalysisResult`.
- Features extracted include shape, pattern, component stats,
  and relational comparisons between input and output grids.
- This modular design enables easy extension by adding new
  concrete analyzers implementing the base interfaces.

---

## Summary

| Class                     | Role/Responsibility                             |
|---------------------------|------------------------------------------------|
| `AbstractProblemAnalyzer` | Base interface for feature/pattern analyzers   |
| `AbstractFeatureExtractor`| Base interface for shape and value extractors  |
| `AbstractPatternDetector` | Base interface for pattern detectors            |
| `BasicShapeAnalyzer`      | Extracts basic grid shape and density features  |
| `BasicPatternDetector`    | Detects repetitions and symmetries               |
| `ComponentFeatureExtractor`| Extracts connected components stats             |
| `RelationalFeatureExtractor`| Extracts relational shape, color, and density features |

---
