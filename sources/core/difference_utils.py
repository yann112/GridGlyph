import logging
import numpy as np


def compute_difference_mask(input_grid: np.ndarray, output_grid: np.ndarray, logger: logging.Logger = None) -> np.ndarray:
    """Computes a mask indicating cells that differ between input and output grids.

    Args:
        input_grid (np.ndarray): Input grid as a 2D NumPy array.
        output_grid (np.ndarray): Output grid as a 2D NumPy array.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        np.ndarray: Boolean mask where True indicates differing cells.

    Raises:
        ValueError: If the shapes of input and output grids do not match.
    """
    logger = logger or logging.getLogger(__name__)

    if input_grid.shape != output_grid.shape:
        logger.debug("Grid sizes differ: input %s, output %s", input_grid.shape, output_grid.shape)
        raise ValueError("Input and output grids must have the same shape to compute a difference mask.")

    mask = input_grid != output_grid
    logger.debug("Computed difference mask:\n%s", mask.astype(int))
    return mask
