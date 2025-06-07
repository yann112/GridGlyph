from abc import ABC, abstractmethod
import logging
import numpy as np


class AbstractTransformationCommand(ABC):
    """
    Base class for all transformation commands.
    Subclasses can define synthesis rules for the synthesis engine.
    """
    synthesis_rules = {
        "type": "atomic",  # Can be "atomic" or "combinator"
        "requires_inner": False,
        "parameter_ranges": {}
    }

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def describe(cls)-> str:
        """Returns a human-readable description of the transformation"""
        return "Unknown transformation"

class MapColors(AbstractTransformationCommand):

    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False,
        "parameter_ranges": {
            "mapping": {1: 2, 3: 4}
        }
    }

    def __init__(self, mapping: dict, logger: logging.Logger = None):
        super().__init__(logger)
        self.mapping = {int(k) if isinstance(k, str) else k: v for k, v in mapping.items()}

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        output_grid = input_grid.copy()
        for old_color, new_color in self.mapping.items():
            output_grid[output_grid == old_color] = new_color
        return output_grid

    @classmethod
    def describe(cls)-> str:
        return """
            Replaces colors in the grid based on a mapping dictionary.
            e.g., {1: 9, 2: 8} replaces all 1s with 9s and 2s with 8s.
        """

class Identity(AbstractTransformationCommand):
    synthesis_rules = {"type": "atomic"}

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug("Executing Identity (no transformation).")
        return input_grid.copy()

    @classmethod
    def describe(cls)-> str:
        return "Returns the input grid unchanged."


class RepeatGrid(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "requires_inner": True,
        "parameter_ranges": {
            "vertical_repeats": (1, 4),
            "horizontal_repeats": (1, 4)
        }
    }

    def __init__(
        self,
        inner_command: AbstractTransformationCommand,
        vertical_repeats: int,
        horizontal_repeats: int,
        logger: logging.Logger = None
    ):
        super().__init__(logger)
        self.inner_command = inner_command
        self.vertical_repeats = vertical_repeats
        self.horizontal_repeats = horizontal_repeats

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug("Executing RepeatGrid (%d x %d)", self.vertical_repeats, self.horizontal_repeats)
        grid_to_repeat = self.inner_command.execute(input_grid)
        return np.tile(grid_to_repeat, (self.vertical_repeats, self.horizontal_repeats))

    @classmethod
    def describe(cls)-> str:
        return """
            Applies an inner transformation to the input grid, then tiles the result 
            in a grid pattern. Creates a larger grid by repeating the transformed 
            result vertically and horizontally.
            e.g., vertical_repeats=2, horizontal_repeats=3 creates a 2x3 arrangement 
            of the transformed grid.
        """


class FlipGridHorizontally(AbstractTransformationCommand):
    synthesis_rules = {"type": "atomic"}

    def __init__(self, logger: logging.Logger = None):
        super().__init__(logger)

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug("Executing FlipGridHorizontally.")
        return np.fliplr(input_grid)

    @classmethod
    def describe(cls)-> str:
        return "Mirrors the grid along the vertical axis (left becomes right)."


class FlipGridVertically(AbstractTransformationCommand):
    synthesis_rules = {"type": "atomic"}

    def __init__(self, logger: logging.Logger = None):
        super().__init__(logger)

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug("Executing FlipGridVertically.")
        return np.flipud(input_grid)

    @classmethod
    def describe(cls)-> str:
        return "Mirrors the grid along the horizontal axis (top becomes bottom)."


class Alternate(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "arity": 2,
        "parameter_ranges": {}
    }
    def __init__(
        self,
        first: AbstractTransformationCommand,
        second: AbstractTransformationCommand,
        logger: logging.Logger = None
    ):
        super().__init__(logger)
        self.first = first
        self.second = second

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        result = []
        for i, row in enumerate(input_grid):
            if i % 2 == 0:
                result.append(self.first.execute(np.array([row]))[0])
            else:
                result.append(self.second.execute(np.array([row]))[0])
        return np.array(result)

    @classmethod
    def describe(cls)-> str:
        return f"Applies alternating transformations to each row - even rows use the first transformation, odd rows use the second transformation."