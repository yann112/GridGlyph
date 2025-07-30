# tests/test_synthetic_grids.py

import numpy as np
import pytest


from utils.synthetic_grids import (
        generate_single_random_grid,
        create_base_grid,
        generate_random_shape_grid,
        )
    

class TestSyntheticGrids:

    def test_generate_single_random_grid_basic(self):
        """Smoke test: generate_single_random_grid should produce a NumPy array."""
        original_shape = (2, 3)
        grid = generate_single_random_grid(original_shape, max_dim=5, num_range=3)
        assert isinstance(grid, np.ndarray)
        assert grid.ndim == 2 
        assert grid.size > 0  

    def test_generate_single_random_grid_dimensions(self):
        """Smoke test: Check if generated grid dimensions are within max_dim."""
        original_shape = (1, 1)
        max_dim = 4
        grid = generate_single_random_grid(original_shape, max_dim=max_dim, num_range=2)
        rows, cols = grid.shape
        assert 1 <= rows <= max_dim + 1
        assert 1 <= cols <= max_dim + 1

    def test_generate_single_random_grid_values(self):
        """Smoke test: Check if generated grid values are within num_range."""
        original_shape = (3, 3)
        num_range = 5
        grid = generate_single_random_grid(original_shape, max_dim=10, num_range=num_range)
        unique_values = np.unique(grid)
        assert np.all(unique_values >= 1)
        assert np.all(unique_values <= num_range)


    def test_create_base_grid(self):
        """Test: create_base_grid creates a grid filled with background_value."""
        height, width = 5, 7
        background_value = 9
        grid = create_base_grid(height, width, background_value)
        assert isinstance(grid, np.ndarray)
        assert grid.shape == (height, width)
        assert np.all(grid == background_value)

    def test_create_base_grid_default_background(self):
        """Test: create_base_grid uses 0 as default background_value."""
        height, width = 3, 4
        grid = create_base_grid(height, width)
        assert isinstance(grid, np.ndarray)
        assert grid.shape == (height, width)
        assert np.all(grid == 0)


    def test_generate_random_shape_grid_basic(self):
        """Smoke test: generate_random_shape_grid should produce a NumPy array."""
        grid = generate_random_shape_grid(min_dim=5, max_dim=10, value_range=(1, 5), num_shapes=3)
        assert isinstance(grid, np.ndarray)
        assert grid.ndim == 2 
        assert grid.size > 0 

    def test_generate_random_shape_grid_dimensions(self):
        """Smoke test: Check if generated shape grid dimensions are within specified range."""
        min_dim, max_dim = 4, 8
        grid = generate_random_shape_grid(min_dim=min_dim, max_dim=max_dim, num_shapes=2)
        rows, cols = grid.shape
        assert min_dim <= rows <= max_dim
        assert min_dim <= cols <= max_dim

    def test_generate_random_shape_grid_values(self):
        """Smoke test: Check if generated shape grid values are plausible."""
        value_range = (2, 8)
        background_value = 0
        grid = generate_random_shape_grid(
            min_dim=5, max_dim=7,
            value_range=value_range,
            num_shapes=5,
            background_value=background_value
        )
        unique_values = np.unique(grid)
        assert background_value in unique_values
        for val in unique_values:
            if val != background_value:
                 assert value_range[0] <= val <= value_range[1]
