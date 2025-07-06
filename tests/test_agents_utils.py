# tests/test_feature_collector.py

import numpy as np
from agents.agents_utils import MultiGridFeatureCollector, SymbolicGridMapper
from core.features_analysis import ProblemAnalyzer

# tests/conftest.py or inside your test file

import pytest
from fixtures import train_data_0


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
        symbol_set_ids=["katakana_final_refined_1", "katakana_final_refined_2"]
        )

    formated_str = mapper.format_variants_list(symbolic_variants)
    assert type(formated_str) is str


def test_symbolic_mapper_katana_bk(train_data_0):
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
        symbol_set_ids=["katakana_bk"]
        )

    formated_str = mapper.format_variants_list(symbolic_variants)
    assert type(formated_str) is str
    
def test_symbolic_mapper_n_variants_single_inputs(train_data_0):
    """
    Simple smoke test for MultiGridFeatureCollector.
    
    Checks basic structure of output without deep validation.
    """
    mapper = SymbolicGridMapper()


    variants = mapper.generate_n_variants(train_data_0, n=30)

    prompt = mapper.format_variants_list(variants, include_variant_headers=False)
    assert type(prompt) is str