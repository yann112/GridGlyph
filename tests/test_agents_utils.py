# tests/test_feature_collector.py

import numpy as np
from agents.agents_utils import MultiGridFeatureCollector, SymbolicGridMapper
from core.features_analysis import ProblemAnalyzer

# tests/conftest.py or inside your test file

import pytest
from fixtures import train_data_0

# @pytest.fixture
# def sample_arc_data():
#     return {
#         'train': [
#             {'input': [[7, 9], [4, 3]],
#              'output': [
#                  [7, 9, 7, 9, 7, 9],
#                  [4, 3, 4, 3, 4, 3],
#                  [9, 7, 9, 7, 9, 7],
#                  [3, 4, 3, 4, 3, 4],
#                  [7, 9, 7, 9, 7, 9],
#                  [4, 3, 4, 3, 4, 3]
#              ]},
#             {'input': [[8, 6], [6, 4]],
#              'output': [
#                  [8, 6, 8, 6, 8, 6],
#                  [6, 4, 6, 4, 6, 4],
#                  [6, 8, 6, 8, 6, 8],
#                  [4, 6, 4, 6, 4, 6],
#                  [8, 6, 8, 6, 8, 6],
#                  [6, 4, 6, 4, 6, 4]
#              ]},
#         ],
#         'test': [{'input': [[3, 2], [7, 8]]}]
#     }
    

def test_multi_grid_feature_collector_smoke(train_data_0):
    """
    Simple smoke test for MultiGridFeatureCollector.
    
    Checks basic structure of output without deep validation.
    """
    # Arrange: Use real analyzer
    collector = MultiGridFeatureCollector(analyzer=ProblemAnalyzer())

    # Act: Extract features from task
    feature_data = collector.extract_features_from_task(train_data_0)

    # Assert: Top level structure
    assert isinstance(feature_data, dict), "Must return dictionary"


def test_symbolic_mapper_smoke(train_data_0):
    """
    Simple smoke test for MultiGridFeatureCollector.
    
    Checks basic structure of output without deep validation.
    """
    # Instantiate mapper
    mapper = SymbolicGridMapper(
    )

    # Get symbolic variants
    symbolic_variants = mapper.generate_variants(
        train_data_0,
        symbol_set_ids=["runic", "box_drawing"]
        )

    formated_str = mapper.format_variants_list(symbolic_variants)
    assert type(formated_str) is str

def test_symbolic_mapper_n_variants_single_inputs(train_data_0):
    """
    Simple smoke test for MultiGridFeatureCollector.
    
    Checks basic structure of output without deep validation.
    """
    mapper = SymbolicGridMapper()

    input_grid = [[7, 9], [4, 3]]
    output_grid = [
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3],
        [9, 7, 9, 7, 9, 7],
        [3, 4, 3, 4, 3, 4],
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3]
    ]

    variants = mapper.generate_n_variants(input_grid, output_grid, n=30)

    prompt = mapper.format_variants_list(variants, include_variant_headers=False)
    assert type(prompt) is str