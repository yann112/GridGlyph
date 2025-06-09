from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import List, Optional


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
        return """
        Returns the input grid unchanged.
        Useful for nested operations but should not be used alone when input and output differ.
        """


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
    def describe(cls) -> str:
        return """
            Applies an inner transformation to the input grid, then tiles the transformed 
            result cell-wise (like numpy.tile) to create a larger grid. The input grid 
            is repeated vertically and horizontally at the cell level, not as a single block.

            Parameters:
            - vertical_repeats: Number of times to repeat the grid vertically (e.g., 3 repeats a 2x2 grid into 6x2).
            - horizontal_repeats: Number of times to repeat the grid horizontally (e.g., 3 repeats a 2x2 grid into 2x6).

            Example:
            Input: 
                [[A, B], 
                [C, D]]
            Command: RepeatGrid(identity, vertical_repeats=2, horizontal_repeats=3)
            Output: 
                [[A, B, A, B, A, B], 
                [C, D, C, D, C, D], 
                [A, B, A, B, A, B], 
                [C, D, C, D, C, D]]
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

class SwapRowsOrColumns(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False,
        "parameter_ranges": {
            "row_swap": (0, 9),  # Assuming max 10 rows
            "col_swap": (0, 9),   # Assuming max 10 columns
            "swap_type": ["rows", "columns", "both"]
        }
    }

    def __init__(
        self,
        row_swap: tuple = None,
        col_swap: tuple = None,
        swap_type: str = "rows",
        logger: logging.Logger = None
    ):
        """
        Initialize the swap command.
        
        Args:
            row_swap: Tuple of two row indices to swap (e.g., (0, 1))
            col_swap: Tuple of two column indices to swap (e.g., (0, 1))
            swap_type: One of "rows", "columns", or "both"
            logger: Optional logger
        """
        super().__init__(logger)
        self.row_swap = row_swap
        self.col_swap = col_swap
        self.swap_type = swap_type
        
        if swap_type not in ["rows", "columns", "both"]:
            raise ValueError("swap_type must be 'rows', 'columns', or 'both'")

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        output_grid = input_grid.copy()
        
        if self.swap_type in ["rows", "both"] and self.row_swap is not None:
            if len(output_grid) > max(self.row_swap):
                output_grid[list(self.row_swap)] = output_grid[list(reversed(self.row_swap))]
        
        if self.swap_type in ["columns", "both"] and self.col_swap is not None:
            if output_grid.shape[1] > max(self.col_swap):
                output_grid[:, list(self.col_swap)] = output_grid[:, list(reversed(self.col_swap))]
        
        return output_grid

    @classmethod
    def describe(cls) -> str:
        return """
            Swaps rows or columns in the grid.
            
            Parameters:
            - row_swap: Tuple of two row indices to swap (e.g., (0, 1) swaps first and second rows)
            - col_swap: Tuple of two column indices to swap (e.g., (0, 1) swaps first and second columns)
            - swap_type: One of "rows", "columns", or "both" to determine what to swap
            
            Example 1 (row swap):
            Input: [[1, 2],    Output: [[3, 4],
                    [3, 4]]            [1, 2]]
            
            Example 2 (column swap):
            Input: [[1, 2],    Output: [[2, 1],
                    [3, 4]]            [4, 3]]
            
            Example 3 (both):
            Input: [[1, 2, 3],    Output: [[6, 5, 4],
                    [4, 5, 6]]            [3, 2, 1]]
        """


class ReverseRow(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False,
        "parameter_ranges": {}
    }

    def __init__(self, logger: logging.Logger = None):
        super().__init__(logger)

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug("Executing ReverseRow on all rows.")
        return input_grid[:, ::-1]  # Reverses each row

    @classmethod
    def describe(cls) -> str:
        return """
        Reverses the elements in every row of the grid.
        Useful as an inner command with ApplyToRow to reverse only specific rows.
        Example:
        Input: [[1, 2, 3], [4, 5, 6]]
        Command: ReverseRow()
        Output: [[3, 2, 1], [6, 5, 4]]
        """


class ApplyToRow(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "requires_inner": True,
        "parameter_ranges": {
            "row_index": (0, 9)  # Assuming max 10 rows
        }
    }

    def __init__(self, inner_command: AbstractTransformationCommand, row_index: int, logger: logging.Logger = None):
        super().__init__(logger)
        self.inner_command = inner_command
        self.row_index = row_index

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        output_grid = input_grid.copy()
        row_to_transform = output_grid[self.row_index].reshape(1, -1)
        transformed_row = self.inner_command.execute(row_to_transform)
        output_grid[self.row_index] = transformed_row.flatten()
        return output_grid

    @classmethod
    def describe(cls) -> str:
        return """
            Applies a given transformation to a specific row in the grid.
            
            Use this to target transformations like FlipGridHorizontally, ReverseRow, etc. to one row only.
            
            Parameters:
            - inner_command: The transformation command to apply to the row.
            - row_index: The index of the row to transform.

            Example:
            Input: [[1, 2, 3],
                    [4, 5, 6]]
            Command: ApplyToRow(ReverseRow(), row_index=1)
            Output: [[1, 2, 3],
                    [6, 5, 4]]
        """
    

class CustomPattern(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False,
        "parameter_ranges": {}
    }

    def __init__(self, pattern_func: callable, logger: logging.Logger = None):
        super().__init__(logger)
        self.pattern_func = pattern_func

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        return self.pattern_func(input_grid)

    @classmethod
    def describe(cls) -> str:
        return """
            Applies a custom pattern to the grid based on a user-defined function.

            Parameters:
            - pattern_func: A function that takes a grid and returns a transformed grid.

            Example:
            Input: [[1, 2, 3],
                    [4, 5, 6]]
            Command: CustomPattern(lambda grid: grid * 2)
            Output: [[2, 4, 6],
                     [8, 10, 12]]
        """

class ConditionalTransform(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "requires_inner": True,
        "parameter_ranges": {}
    }

    def __init__(self, inner_command: AbstractTransformationCommand, condition_func: callable, logger: logging.Logger = None):
        super().__init__(logger)
        self.inner_command = inner_command
        self.condition_func = condition_func

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        if self.condition_func(input_grid):
            return self.inner_command.execute(input_grid)
        else:
            return input_grid.copy()

    @classmethod
    def describe(cls) -> str:
        return """
            Applies a transformation conditionally based on a user-defined condition.

            Parameters:
            - inner_command: The transformation command to apply if the condition is met.
            - condition_func: A function that takes a grid and returns a boolean.

            Example:
            Input: [[1, 2, 3],
                    [4, 5, 6]]
            Command: ConditionalTransform(FlipGridHorizontally(), lambda grid: grid[0, 0] == 1)
            Output: [[3, 2, 1],
                     [6, 5, 4]]
        """
    
class MaskCombinator(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "requires_inner": True,
        "parameter_ranges": {
            "mask_func": "lambda grid: np.ones_like(grid, dtype=bool)" # Default mask selects all elements
        }
    }

    def __init__(self, inner_command: AbstractTransformationCommand, mask_func: callable, logger: logging.Logger = None):
        super().__init__(logger)
        self.inner_command = inner_command
        self.mask_func = mask_func

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        output_grid = input_grid.copy()
        mask = self.mask_func(input_grid)
        # Apply the inner command only to the elements selected by the mask
        transformed_grid = self.inner_command.execute(input_grid)
        output_grid[mask] = transformed_grid[mask]
        return output_grid

    @classmethod
    def describe(cls) -> str:
        return """
            Applies a transformation to specific elements of the grid based on a boolean mask.
            
            Use this to selectively apply operations like ReverseRow or FlipGridHorizontally to certain rows/columns.
            
            Parameters:
            - inner_command: The transformation command to apply to selected elements.
            - mask_func: A function that takes a grid and returns a boolean mask.

            Example:
            Input: [[1, 2, 3],
                    [4, 5, 6]]
            Command: MaskCombinator(ReverseRow(), lambda grid: np.array([[False, True, False], [True, False, True]]))
            Output: [[1, 2, 3],
                    [6, 5, 4]]
        """
 
class ShiftRowOrColumn:
    synthesis_rules = {
            "type": "atomic",
            "parameter_ranges": {
                "row_index": [0, 9],
                "col_index": [0, 9],
                "shift_amount": [-5, 5],
                "wrap": [True, False]
            }
        }

    def __init__(self, row_index=None, col_index=None, shift_amount=1, wrap=True):
        self.row_index = row_index
        self.col_index = col_index
        self.shift_amount = shift_amount
        self.wrap = wrap

    @classmethod
    def describe(cls) -> str:
        return """
        Shifts a specified row or column by a given amount. Optionally wraps around or pads with zeros.
        
        Parameters:
        - row_index: index of the row to shift (or None for column)
        - col_index: index of the column to shift (or None for row)
        - shift_amount: number of positions to shift (positive or negative)
        - wrap: whether to wrap around (True) or pad with zeros (False)
        """

    def execute(self, grid: np.ndarray) -> np.ndarray:
        grid = grid.copy()
        if self.row_index is not None:
            row = grid[self.row_index]
            if self.wrap:
                row = np.roll(row, self.shift_amount)
            else:
                if self.shift_amount > 0:
                    row = np.concatenate([np.zeros(self.shift_amount, dtype=row.dtype), row[:-self.shift_amount]])
                else:
                    row = np.concatenate([row[-self.shift_amount:], np.zeros(-self.shift_amount, dtype=row.dtype)])
            grid[self.row_index] = row
        elif self.col_index is not None:
            col = grid[:, self.col_index]
            if self.wrap:
                col = np.roll(col, self.shift_amount)
            else:
                if self.shift_amount > 0:
                    col = np.concatenate([np.zeros(self.shift_amount, dtype=col.dtype), col[:-self.shift_amount]])
                else:
                    col = np.concatenate([col[-self.shift_amount:], np.zeros(-self.shift_amount, dtype=col.dtype)])
            grid[:, self.col_index] = col
        return grid.tolist()
    
class Sequence(AbstractTransformationCommand):
    """
    Applies a sequence of transformation commands in order.
    
    This combinator allows multiple atomic or nested operations to be applied sequentially.
    Useful for expressing complex transformations that require multiple steps.
    """

    synthesis_rules = {
        "type": "combinator",
        "requires_inner": True,
        "parameter_ranges": {
            "commands": []  # Will be filled dynamically by synthesis engine
        }
    }

    def __init__(
        self,
        commands: List[AbstractTransformationCommand],
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.commands = commands

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        """
        Applies each command in sequence to the input grid.
        
        Args:
            input_grid (np.ndarray): The starting 2D grid.
            
        Returns:
            np.ndarray: Grid after applying all transformations in order.
        """
        grid = input_grid.copy()
        for cmd in self.commands:
            try:
                grid = cmd.execute(grid)
            except Exception as e:
                self.logger.error(f"Error executing {cmd.__class__.__name__}: {str(e)}")
                raise
        return grid

    @classmethod
    def describe(cls) -> str:

        return (
            "Applies a sequence of commands in order. "
            "Useful for combining multiple small transformations into one full solution. "
            "Example: tile grid → flip row 2 → flip row 3"
        )