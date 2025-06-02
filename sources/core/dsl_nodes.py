from abc import ABC, abstractmethod
import logging
import numpy as np


class AbstractTransformationCommand(ABC):
    """Abstract base class for all DSL transformation commands."""

    def __init__(self, logger: logging.Logger = None):
        """
        Args:
            logger (logging.Logger, optional): Logger instance. Defaults to module-level logger.
        """
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        """Executes the transformation on the input grid.

        Args:
            input_grid (np.ndarray): A 2D NumPy grid.

        Returns:
            np.ndarray: Transformed output grid.
        """
        pass


class RepeatGrid(AbstractTransformationCommand):
    """Repeats the input grid vertically and horizontally."""

    def __init__(
        self,
        inner_command: AbstractTransformationCommand,
        vertical_repeats: int,
        horizontal_repeats: int,
        logger: logging.Logger = None,
    ):
        super().__init__(logger)
        self.inner_command = inner_command
        self.vertical_repeats = vertical_repeats
        self.horizontal_repeats = horizontal_repeats

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug("Executing RepeatGrid (%d x %d)", self.vertical_repeats, self.horizontal_repeats)
        grid_to_repeat = self.inner_command.execute(input_grid)
        return np.tile(grid_to_repeat, (self.vertical_repeats, self.horizontal_repeats))


class Identity(AbstractTransformationCommand):
    """A no-op transformation."""

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug("Executing Identity (no transformation).")
        return input_grid.copy()


class FlipGridHorizontally(AbstractTransformationCommand):
    """Flips the input grid horizontally."""

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug("Executing FlipGridHorizontally.")
        return np.fliplr(input_grid)


class FlipGridVertically(AbstractTransformationCommand):
    """Flips the input grid vertically."""

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug("Executing FlipGridVertically.")
        return np.flipud(input_grid)
