import numpy as np
import logging
import pytest

from features_analysis import ProblemAnalyzer


@pytest.fixture
def logger() -> logging.Logger:
    logging.basicConfig(level=logging.DEBUG)
    return logging.getLogger(__name__)


def test_relational_features_on_repeated_grid(logger: logging.Logger) -> None:
    input_grid = np.array([
        [7, 9],
        [4, 3]
    ])

    output_grid = np.array([
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3],
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3],
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3]
    ])

    analyzer = ProblemAnalyzer(logger=logger)
    result = analyzer.analyze(input_grid, output_grid)
    features = result.to_dict()["features"]

    relational_shape = features.get("relational_shape", {})
    relational_color = features.get("relational_color", {})
    relational_density = features.get("relational_density", {})

    # Test relational shape features (size ratios)
    assert relational_shape["row_ratio"] == pytest.approx(3.0)
    assert relational_shape["col_ratio"] == pytest.approx(3.0)
    assert relational_shape["area_ratio"] == pytest.approx(9.0)

    # Test relational color features (color set relations)
    assert relational_color["output_subset_of_input"] is True  # output uses only colors from input
    assert relational_color["output_superset_of_input"] is False  # output does not have extra colors
    assert relational_color["color_intersection_size"] == 4
    assert relational_color["input_color_count"] == 4
    assert relational_color["output_color_count"] == 4

    # Test relational density features (foreground pixel ratio change)
    assert relational_density["foreground_ratio_change"] == pytest.approx(1.0)

    # Suggested transform should be 'RepeatGrid' due to size increase
    suggested_names = [name for name, _ in result.to_dict()["suggested_transforms"]]
    assert "RepeatGrid" in suggested_names
