import random
import numpy as np
from .primitives import create_safe_base_grid, add_random_shapes

def mutate_geometric_rule(sigil: str, base_grid_shape=(10, 10)):
    """Generic mutator for rotations/flips."""
    h, w = base_grid_shape
    grid = create_safe_base_grid(w, h)
    grid = add_random_shapes(grid, num_shapes=random.randint(2, 5))
    return grid

def mutate_swap_value_rule(sigil: str, base_grid_shape=(10, 10)):
    """Specific mutator for value swapping (⤨)."""
    grid = mutate_geometric_rule(None, base_grid_shape)
    # Ensure at least two colors exist to swap
    unique_colors = np.unique(grid)
    if len(unique_colors) < 2:
        grid[0, 0] = 1
        grid[0, 1] = 2
    return grid