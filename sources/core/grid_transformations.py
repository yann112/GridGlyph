import numpy as np
import logging
from typing import List


class GridTransformer:
    """Provides basic transformation operations on a 2D grid.

    Attributes:
        logger (logging.Logger): Logger instance for internal logging.
    """

    def __init__(self, logger: logging.Logger = None):
        """Initializes the GridTransformer.

        Args:
            logger (logging.Logger, optional): Logger instance. Defaults to module-level logger.
        """
        self.logger = logger or logging.getLogger(__name__)

    def flip_grid_horizontally(self, input_grid: np.ndarray) -> np.ndarray:
        """Flips the input grid horizontally (left to right).

        Args:
            input_grid (np.ndarray): A 2D NumPy array representing the grid.

        Returns:
            np.ndarray: Horizontally flipped grid.
        """
        self.logger.debug("Flipping grid horizontally.")
        return np.fliplr(input_grid)

    def flip_grid_vertically(self, input_grid: np.ndarray) -> np.ndarray:
        """Flips the input grid vertically (top to bottom).

        Args:
            input_grid (np.ndarray): A 2D NumPy array representing the grid.

        Returns:
            np.ndarray: Vertically flipped grid.
        """
        self.logger.debug("Flipping grid vertically.")
        return np.flipud(input_grid)

    def concatenate_grids_horizontally(self, list_of_grids: List[np.ndarray]) -> np.ndarray:
        """Concatenates a list of grids horizontally (side by side).

        Args:
            list_of_grids (List[np.ndarray]): A list of 2D NumPy arrays.

        Returns:
            np.ndarray: Concatenated grid.
        """
        self.logger.debug("Concatenating grids horizontally.")
        return np.hstack(list_of_grids)

    def concatenate_grids_vertically(self, list_of_grids: List[np.ndarray]) -> np.ndarray:
        """Concatenates a list of grids vertically (stacked).

        Args:
            list_of_grids (List[np.ndarray]): A list of 2D NumPy arrays.

        Returns:
            np.ndarray: Concatenated grid.
        """
        self.logger.debug("Concatenating grids vertically.")
        return np.vstack(list_of_grids)

    def repeat_grid(self, input_grid: np.ndarray, number_of_vertical_repeats: int, number_of_horizontal_repeats: int) -> np.ndarray:
        """Repeats the input grid in a tiled manner.

        Args:
            input_grid (np.ndarray): The 2D grid to be repeated.
            number_of_vertical_repeats (int): Number of times to repeat vertically.
            number_of_horizontal_repeats (int): Number of times to repeat horizontally.

        Returns:
            np.ndarray: Repeated grid.
        """
        self.logger.debug(
            "Repeating grid %d times vertically and %d times horizontally.",
            number_of_vertical_repeats, number_of_horizontal_repeats
        )
        return np.tile(input_grid, (number_of_vertical_repeats, number_of_horizontal_repeats))
