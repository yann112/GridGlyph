# tests/test_feature_collector.py

import numpy as np
from agents.agents_utils import MultiGridFeatureCollector
from core.features_analysis import ProblemAnalyzer

# tests/conftest.py or inside your test file

import pytest


@pytest.fixture
def sample_arc_data():
    return {
        'train': [
            {'input': [[7, 9], [4, 3]],
             'output': [
                 [7, 9, 7, 9, 7, 9],
                 [4, 3, 4, 3, 4, 3],
                 [9, 7, 9, 7, 9, 7],
                 [3, 4, 3, 4, 3, 4],
                 [7, 9, 7, 9, 7, 9],
                 [4, 3, 4, 3, 4, 3]
             ]},
            {'input': [[8, 6], [6, 4]],
             'output': [
                 [8, 6, 8, 6, 8, 6],
                 [6, 4, 6, 4, 6, 4],
                 [6, 8, 6, 8, 6, 8],
                 [4, 6, 4, 6, 4, 6],
                 [8, 6, 8, 6, 8, 6],
                 [6, 4, 6, 4, 6, 4]
             ]},
        ],
        'test': [{'input': [[3, 2], [7, 8]]}]
    }
    

def test_multi_grid_feature_collector_smoke(sample_arc_data):
    """
    Simple smoke test for MultiGridFeatureCollector.
    
    Checks basic structure of output without deep validation.
    """
    # Arrange: Use real analyzer
    collector = MultiGridFeatureCollector(analyzer=ProblemAnalyzer())

    # Act: Extract features from task
    feature_data = collector.extract_features_from_task(sample_arc_data)

    # Assert: Top level structure
    assert isinstance(feature_data, dict), "Must return dictionary"
