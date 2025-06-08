from abc import ABC, abstractmethod
import logging
import numpy as np
import cv2
from typing import Dict, List


# --- Utilities ---
def get_background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def extract_components(grid: np.ndarray, background_color: int):
    mask = (grid != background_color).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    components = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        component_mask = np.zeros_like(mask)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=cv2.FILLED)
        centroid = (x + w // 2, y + h // 2)

        components.append({
            "bbox": (x, y, w, h),
            "area": area,
            "mask": component_mask.astype(bool),
            "centroid": centroid
        })
    return components


# --- Abstract base classes ---
class AbstractProblemAnalyzer(ABC):
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        pass


class AbstractFeatureExtractor(AbstractProblemAnalyzer):
    @abstractmethod
    def get_shape_features(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        pass

    @abstractmethod
    def get_value_features(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        pass


class AbstractPatternDetector(AbstractProblemAnalyzer):
    @abstractmethod
    def detect_repetition(self, grid: np.ndarray) -> Dict:
        pass

    @abstractmethod
    def detect_symmetry(self, grid: np.ndarray) -> Dict:
        pass


# --- Concrete analyzers ---
class BasicShapeAnalyzer(AbstractFeatureExtractor):
    def analyze(self, input_grid, output_grid):
        return {
            'shapefeature': self.get_shape_features(input_grid, output_grid),
            'densityfeature': self.get_value_features(input_grid, output_grid)
        }

    def get_shape_features(self, input_grid, output_grid):
        return {
            'rows': input_grid.shape[0],
            'cols': input_grid.shape[1],
            'aspect_ratio': input_grid.shape[1] / input_grid.shape[0] if input_grid.shape[0] != 0 else 0,
            'total_cells': input_grid.size
        }

    def get_value_features(self, input_grid, output_grid):
        bg_color = get_background_color(input_grid)
        fg_count = np.sum(input_grid != bg_color)
        fg_ratio = fg_count / input_grid.size if input_grid.size else 0
        return {
            'foreground_ratio': fg_ratio
        }


class BasicPatternDetector(AbstractPatternDetector):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.min_repeats = 2

    def analyze(self, input_grid, output_grid):
        return {
            'patterns': {
                'repetition': self.detect_repetition(output_grid),
                'symmetry': self.detect_symmetry(output_grid)
            }
        }

    def detect_repetition(self, grid):
        return {
            'horizontal': self._check_horizontal_repetition(grid),
            'vertical': self._check_vertical_repetition(grid)
        }

    def _check_horizontal_repetition(self, grid):
        if grid.shape[1] < self.min_repeats:
            return False
        pattern_width = grid.shape[1] // self.min_repeats
        first_block = grid[:, :pattern_width]
        for i in range(1, self.min_repeats):
            block = grid[:, i*pattern_width:(i+1)*pattern_width]
            if not np.array_equal(first_block, block):
                return False
        return True

    def _check_vertical_repetition(self, grid):
        if grid.shape[0] < self.min_repeats:
            return False
        pattern_height = grid.shape[0] // self.min_repeats
        first_block = grid[:pattern_height, :]
        for i in range(1, self.min_repeats):
            block = grid[i*pattern_height:(i+1)*pattern_height, :]
            if not np.array_equal(first_block, block):
                return False
        return True

    def detect_symmetry(self, grid):
        return {
            'horizontal': np.array_equal(grid, grid[::-1, :]),
            'vertical': np.array_equal(grid, grid[:, ::-1]),
            'diagonal': np.array_equal(grid, grid.T)
        }


class ComponentFeatureExtractor(AbstractFeatureExtractor):
    def analyze(self, input_grid, output_grid):
        return {
            'componentstats': {
                'input': self.get_shape_features(input_grid, None),
                'output': self.get_shape_features(output_grid, None)
            }
        }

    def get_shape_features(self, grid, _):
        bg_color = get_background_color(grid)
        components = extract_components(grid, bg_color)

        if not components:
            return {
                'num_components': 0,
                'avg_area': 0,
                'avg_width': 0,
                'avg_height': 0
            }

        areas = [c['area'] for c in components]
        widths = [c['bbox'][2] for c in components]
        heights = [c['bbox'][3] for c in components]

        return {
            'num_components': len(components),
            'avg_area': float(np.mean(areas)),
            'avg_width': float(np.mean(widths)),
            'avg_height': float(np.mean(heights))
        }

    def get_value_features(self, input_grid, output_grid):
        return {}

class RelationalFeatureExtractor(AbstractFeatureExtractor):
    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        return {
            'relational_shape': self.get_relational_shape_features(input_grid, output_grid),
            'relational_color': self.get_relational_color_features(input_grid, output_grid),
            'relational_density': self.get_relational_density_features(input_grid, output_grid),
        }

    def get_shape_features(self, input_grid, output_grid):
        # You can simply call your relational shape features here
        return self.get_relational_shape_features(input_grid, output_grid)

    def get_value_features(self, input_grid, output_grid):
        # Return combined color and density relational features for the abstract method
        color_features = self.get_relational_color_features(input_grid, output_grid)
        density_features = self.get_relational_density_features(input_grid, output_grid)
        # Merge dicts and return
        return {**color_features, **density_features}

    def get_relational_shape_features(self, input_grid, output_grid):
        row_ratio = output_grid.shape[0] / input_grid.shape[0] if input_grid.shape[0] != 0 else 0
        col_ratio = output_grid.shape[1] / input_grid.shape[1] if input_grid.shape[1] != 0 else 0
        return {
            'row_ratio': row_ratio,
            'col_ratio': col_ratio,
            'area_ratio': (output_grid.shape[0] * output_grid.shape[1]) / (input_grid.shape[0] * input_grid.shape[1]) if input_grid.size else 0
        }

    def get_relational_color_features(self, input_grid, output_grid):
        input_colors = set(np.unique(input_grid))
        output_colors = set(np.unique(output_grid))
        is_subset = output_colors.issubset(input_colors)
        is_superset = output_colors.issuperset(input_colors) and output_colors != input_colors
        intersection_size = len(input_colors.intersection(output_colors))
        return {
            'output_subset_of_input': is_subset,
            'output_superset_of_input': is_superset,
            'color_intersection_size': intersection_size,
            'input_color_count': len(input_colors),
            'output_color_count': len(output_colors)
        }

    def get_relational_density_features(self, input_grid, output_grid):
        input_bg = get_background_color(input_grid)
        output_bg = get_background_color(output_grid)
        input_foreground = np.sum(input_grid != input_bg)
        output_foreground = np.sum(output_grid != output_bg)
        input_size = input_grid.size
        output_size = output_grid.size
        input_density = input_foreground / input_size if input_size else 0
        output_density = output_foreground / output_size if output_size else 0
        ratio = output_density / input_density if input_density else 0
        return {
            'foreground_ratio_change': ratio
        }


# --- Result container ---
class AnalysisResult:
    def __init__(self):
        self.features = {
            'shapefeature': {},
            'densityfeature': {},
            'patterns': {},
            'componentstats': {},
            'relational_shape': {},
            'relational_color': {},
            'relational_density': {}
        }


    def add_features(self, feature_type: str, features: Dict):
        if feature_type not in self.features:
            self.features[feature_type] = {}
        self.features[feature_type].update(features)

    def to_dict(self) -> Dict:
        return {
            'features': self.features,
        }


# --- Coordinator ---
class ProblemAnalyzer:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.analyzers = [
            BasicShapeAnalyzer(logger),
            BasicPatternDetector(logger),
            ComponentFeatureExtractor(logger),
            RelationalFeatureExtractor(logger),
        ]
        self.result = AnalysisResult()

    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> AnalysisResult:
        try:
            for analyzer in self.analyzers:
                self.logger.debug(f"Running analyzer: {analyzer.__class__.__name__}")
                features = analyzer.analyze(input_grid, output_grid)
                for k, v in features.items():
                    self.result.add_features(k, v)


            return self.result
        except Exception as e:
            self.logger.error(f"Error during problem analysis: {e}")
            raise
