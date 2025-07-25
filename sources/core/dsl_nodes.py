from abc import ABC, abstractmethod
import logging
import numpy as np
import cv2
from collections import deque
from typing import List, Optional, Dict, Any, Tuple, Union, Iterator
from assets.symbols import ROM_VAL_MAP , INT_VAL_MAP


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

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        """
        Yields all direct AbstractTransformationCommand children of this command.
        By default, a command is considered a 'leaf' node and has no children.
        Composite commands (e.g., Sequence, ConditionalTransform) MUST override
        this method to yield their actual child command instances.
        """
        yield from () # Default: yield nothing (an empty iterator)

    def set_executor_context(self, executor: Any): # 'DSLExecutor' forward ref
        """
        Allows the Executor to inject itself (or a relevant context) into the command.
        This is useful for commands needing access to shared state like variables or the logger.
        The context is propagated to children via get_children_commands().
        """
        self._executor_context = executor
        for child_cmd in self.get_children_commands():
            if hasattr(child_cmd, 'set_executor_context'):
                child_cmd.set_executor_context(executor)
                
class MapNumbers(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False,
        "parameter_ranges": {
            "mapping": {1: 2, 3: 4}  # Example mapping; actual values vary per use case
        }
    }

    def __init__(self, mapping: dict, logger: logging.Logger = None):
        super().__init__(logger)
        # Ensure keys are consistently typed (e.g., int) even if strings were passed
        self.mapping = {int(k) if isinstance(k, str) else k: v for k, v in mapping.items()}

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        conditions = []
        choices = []

        for old_value, new_value in self.mapping.items():
            conditions.append(input_grid == old_value)
            choices.append(new_value)

        output_grid = np.select(conditions, choices, default=input_grid)

        return output_grid

    @classmethod
    def describe(cls) -> str:
        return """
            Replaces specified numeric values in the grid based on a dictionary mapping.
            
            This operation is useful for changing specific numbers throughout the grid.
            For example, {1: 9, 2: 8} will replace all 1s with 9s and all 2s with 8s.

            Note: In ARC puzzles, these numbers often represent colors, but this operation treats them as generic integers.
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
            "vertical_repeats": (1, 30),
            "horizontal_repeats": (1, 30)
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
        
    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.inner_command
        
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

    def __init__(self, argument_command: Optional[AbstractTransformationCommand] = None, logger: logging.Logger = None):
        super().__init__(logger)
        self.argument_command = argument_command

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        if self.argument_command:
            grid_to_flip = self.argument_command.execute(input_grid)
        else:
            grid_to_flip = input_grid
        return np.fliplr(grid_to_flip)

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        if self.argument_command:
            yield self.argument_command

    @classmethod
    def describe(cls) -> str:
        return "Mirrors the grid along the vertical axis (left becomes right)."

class FlipGridVertically(AbstractTransformationCommand):
     synthesis_rules = {"type": "atomic"}

     def __init__(self, argument_command: Optional[AbstractTransformationCommand] = None, logger: logging.Logger = None):
         super().__init__(logger)
         self.argument_command = argument_command # Add this line

     def execute(self, input_grid: np.ndarray) -> np.ndarray:
         self.logger.debug("Executing FlipGridVertically.")
         # Modify this part to use argument_command if present
         if self.argument_command:
             grid_to_flip = self.argument_command.execute(input_grid)
         else:
             grid_to_flip = input_grid
         return np.flipud(grid_to_flip)

     def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
         # Add this method to yield the child command if it exists
         if self.argument_command:
             yield self.argument_command

     @classmethod
     def describe(cls)-> str:
         return "Mirrors the grid along the horizontal axis (top becomes bottom)."


class RotateGridClockwise(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False,
        "parameter_ranges": {
            "num_quarter_rotations": (1, 4)
        }
    }

    def __init__(
        self,
        num_quarter_rotations: int,
        argument_command: Optional[AbstractTransformationCommand] = None,
        logger: logging.Logger = None
    ):
        super().__init__(logger)
        if not 1 <= num_quarter_rotations <= 4:
            raise ValueError("Number of quarter rotations must be between 1 and 4.")
        self.num_quarter_rotations = num_quarter_rotations
        self.argument_command = argument_command

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        if self.argument_command:
            grid_to_rotate = self.argument_command.execute(input_grid)
        else:
            grid_to_rotate = input_grid
        
        k_for_rot90 = (4 - self.num_quarter_rotations) % 4
        
        return np.rot90(grid_to_rotate, k=k_for_rot90)

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        if self.argument_command:
            yield self.argument_command

    @classmethod
    def describe(cls) -> str:
        return """
            Rotates the grid clockwise by a specified number of quarter turns (90 degrees per turn).
            
            Can optionally apply the rotation to the result of an inner command.
            
            Parameters:
            - num_quarter_rotations: Integer (1-4) representing the number of 90-degree clockwise rotations.
            - argument_command: Optional inner command whose output will be rotated. If None, rotates the input grid.

            Example:
            Input: [[1, 2],
                    [3, 4]]
            Command: ↻(I)  # Rotate 90 degrees clockwise
            Output: [[3, 1],
                     [4, 2]]
            
            Command: ↻(II) # Rotate 180 degrees clockwise
            Output: [[4, 3],
                     [2, 1]]

            Command: ↻(I, ↔) # Apply Flip Horizontally, then rotate 90 degrees clockwise
            Input: [[1, 2],
                    [3, 4]]
            Intermediate (↔): [[2, 1],
                               [4, 3]]
            Output (↻(I)): [[4, 2],
                            [3, 1]]
        """
        
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

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.first
        yield self.second
        
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
            "row_swap": (0, 30),  # Assuming max 10 rows
            "col_swap": (0, 30),   # Assuming max 10 columns
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
            Row and column indices are zero-based.
            
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

class ApplyToRow(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "requires_inner": True,
        "parameter_ranges": {
            "row_index": (0, 30)  # Assuming max 10 rows
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

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.inner_command

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
    
class ApplyToColumn(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "requires_inner": True,
        "parameter_ranges": {
            "col_index": (0, 30)
        }
    }

    def __init__(self, inner_command: AbstractTransformationCommand, col_index: int, logger: logging.Logger = None):
        super().__init__(logger)
        self.inner_command = inner_command
        self.col_index = col_index

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        output_grid = input_grid.copy()
        
        if not (0 <= self.col_index < output_grid.shape[1]):
            self.logger.warning(
                f"Column index {self.col_index} out of bounds for grid with {output_grid.shape[1]} columns. "
                "Returning original grid."
            )
            return input_grid
            
        col_to_transform = output_grid[:, self.col_index].reshape(-1, 1)
        
        transformed_col_grid = self.inner_command.execute(col_to_transform)
        
        if transformed_col_grid.ndim == 1:
            transformed_col_grid = transformed_col_grid.reshape(-1, 1)
        elif transformed_col_grid.shape[1] != 1:
            self.logger.error(
                f"Inner command returned a grid with {transformed_col_grid.shape[1]} columns "
                f"when only 1 was expected for column transformation. Attempting to use the first column."
            )
            transformed_col_grid = transformed_col_grid[:, 0].reshape(-1, 1)

        output_grid[:, self.col_index] = transformed_col_grid.flatten()

        return output_grid

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.inner_command

    @classmethod
    def describe(cls) -> str:
        return """
            Applies a given transformation to a specific column in the grid.
            
            This allows inner commands to operate on a single column, which is passed as a (Height, 1) grid.
            
            Parameters:
            - inner_command: The transformation command to apply to the column.
            - col_index: The zero-based index of the column to transform.

            Example:
            Input: [[1, 2],
                    [3, 4],
                    [5, 6]]
            Command: ↓(I, FlipGridVertically())  # Apply FlipGridVertically to the first column
            Output: [[5, 2],
                     [3, 4],
                     [1, 6]]
        """
class ConditionalTransform(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "requires_inner": True, # This might need to be adjusted based on true_command being present
        "parameter_ranges": {}
    }

    def __init__(
        self,
        # This will be the command executed if the condition is True
        true_command: AbstractTransformationCommand,
        # This will be the command whose 'execute' method returns a boolean
        condition_command: AbstractTransformationCommand, # Assuming boolean commands inherit this.
                                                         # If you have AbstractBooleanCommand, use that.
        # Optional: a command to execute if the condition is False
        false_command: Optional[AbstractTransformationCommand] = None,
        logger: logging.Logger = None
    ):
        super().__init__(logger)
        self.true_command = true_command
        self.condition_command = condition_command
        self.false_command = false_command
        self.logger.debug(
            f"ConditionalTransform initialized: "
            f"condition='{condition_command.describe()}', "
            f"true_branch='{true_command.describe()}', "
            f"false_branch='{false_command.describe() if false_command else 'None'}'"
        )

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.true_command
        yield self.condition_command
        if self.false_command:
            yield self.false_command

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
            condition_met = False
            try:
                condition_result = self.condition_command.execute(input_grid)
                
                if isinstance(condition_result, bool):
                    condition_met = condition_result
                elif isinstance(condition_result, np.ndarray):
                    if condition_result.shape == (1, 1):
                        condition_met = bool(condition_result.item())
                    else:
                        self.logger.error(
                            f"ConditionalTransform: Condition command '{self.condition_command.describe()}' "
                            f"returned a NumPy array of unexpected shape: {condition_result.shape}. "
                            f"Expected (1,1) for boolean interpretation. Defaulting to False."
                        )
                        condition_met = False
                else:
                    self.logger.error(
                        f"ConditionalTransform: Condition command '{self.condition_command.describe()}' "
                        f"did not return a boolean or a 1x1 NumPy array. Got type: {type(condition_result)}. "
                        f"Defaulting to False for condition evaluation."
                    )
                    condition_met = False
            except Exception as e:
                self.logger.error(f"Error executing condition command '{self.condition_command.describe()}': {e}")
                condition_met = False 

            if condition_met:
                self.logger.debug(f"ConditionalTransform: Condition met. Executing true command: {self.true_command.describe()}")
                return self.true_command.execute(input_grid)
            elif self.false_command:
                self.logger.debug(f"ConditionalTransform: Condition not met. Executing false command: {self.false_command.describe()}")
                return self.false_command.execute(input_grid)
            else:
                self.logger.debug("ConditionalTransform: Condition not met and no false command. Returning input grid.")
                return input_grid.copy() 

    @classmethod
    def describe(cls) -> str:
        return """
            Applies a 'true_command' if a 'condition_command' evaluates to True, 
            otherwise applies an optional 'false_command' or returns the input grid.

            Parameters:
            - true_command: The transformation command to apply if the condition is met.
            - condition_command: A command that takes a grid and returns a boolean (e.g., a check/equality command).
            - false_command (optional): The transformation command to apply if the condition is NOT met.

            Example (conceptual, syntax depends on actual command definitions):
            Input: [[1, 2], [3, 4]]
            Command: ¿C(FlipGridHorizontally(), CheckGridHasColor(0))
            If grid has color 0: FlipGridHorizontally()
            Else: Return original grid

            Command: ¿C(Rotate90Degrees(), CheckGridSize(2,2), FlipGridVertically())
            If grid is 2x2: Rotate90Degrees()
            Else: FlipGridVertically()
        """
    
class MaskCombinator(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "arity": 3,
        "requires_inner": True,
        "parameter_ranges": {}
    }

    def __init__(self, 
                 inner_command: AbstractTransformationCommand, 
                 mask_command: AbstractTransformationCommand, 
                 false_value_command: AbstractTransformationCommand, 
                 logger: logging.Logger = None):
        super().__init__(logger)
        self.inner_command = inner_command
        self.mask_command = mask_command 
        self.false_value_command = false_value_command 

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.inner_command
        yield self.mask_command
        yield self.false_value_command
        
    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug(f"Executing MaskCombinator on {input_grid.shape}")
        
        transformed_grid = self.inner_command.execute(input_grid)
        mask = self.mask_command.execute(input_grid) 
        
        false_value_result = self.false_value_command.execute(input_grid)

        false_value_for_where = None
        if isinstance(false_value_result, (int, float)):
            false_value_for_where = false_value_result
        elif isinstance(false_value_result, np.ndarray) and false_value_result.ndim == 0:
            false_value_for_where = false_value_result.item()
        elif isinstance(false_value_result, np.ndarray) and false_value_result.shape == (1,1):
            false_value_for_where = false_value_result[0,0]
        elif isinstance(false_value_result, np.ndarray):
            false_value_for_where = false_value_result
        else:
            raise TypeError(f"MaskCombinator received unexpected type for false_value_command result: {type(false_value_result)}")

        if not (mask.shape == transformed_grid.shape):
            self.logger.warning(f"Shape mismatch: Mask {mask.shape}, Transformed {transformed_grid.shape}. Both must match for np.where.")
            raise ValueError(f"Shape mismatch in MaskCombinator: Mask {mask.shape}, Transformed {transformed_grid.shape}. Both must match.")
            
        result = np.where(mask, transformed_grid, false_value_for_where) 
        return result

    @classmethod
    def describe(cls) -> str:
        return """
            Applies a transformation to an input grid and then masks the result,
            allowing for a generic false_value grid.
            
            Parameters:
            - inner_command: The command to apply to the input grid (provides the 'true_value' grid).
            - mask_command: A command that produces a boolean mask (grid) of the same shape.
            - false_value_command: A command that produces the grid to use where the mask is 0 (false).
        """
 
class ShiftRowOrColumn(AbstractTransformationCommand):
    synthesis_rules = {
            "type": "atomic",
            "parameter_ranges": {
                "row_index": [0, 30],
                "col_index": [0, 30],
                "shift_amount": [-15, 15],
                "wrap": [True, False]
            }
        }

    def __init__(self, row_index=None, col_index=None, shift_amount=1, wrap=True, logger: logging.Logger = None): # Add logger param
        super().__init__(logger) # <-- ADD THIS LINE
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
        return grid
    
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
    
    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield from self.commands
        
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
        
class CreateSolidColorGrid(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False,
        "parameter_ranges": {
            "rows": (1, 30),
            "cols": (1, 30),
            "fill_color": (0, 9)
        }
    }

    def __init__(self,
                 rows: Union[int, AbstractTransformationCommand],
                 cols: Union[int, AbstractTransformationCommand],
                 fill_color: int,
                 logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.rows_arg = rows 
        self.cols_arg = cols 
        self.fill_color = fill_color

    def execute(self, input_grid: np.ndarray = None) -> np.ndarray:
        """
        Creates a new grid of specified dimensions filled with a single color.
        Resolves row/col commands if present.
        """
        self.logger.debug(f"Executing CreateSolidColorGrid: Resolving dimensions...")

        resolved_rows = self.rows_arg.execute(input_grid) \
                        if isinstance(self.rows_arg, AbstractTransformationCommand) \
                        else self.rows_arg

        resolved_cols = self.cols_arg.execute(input_grid) \
                        if isinstance(self.cols_arg, AbstractTransformationCommand) \
                        else self.cols_arg

        if not (isinstance(resolved_rows, int) and resolved_rows >= 0):
            self.logger.error(f"CreateSolidColorGrid received invalid resolved row dimension: {resolved_rows} (type {type(resolved_rows)})")
            raise ValueError(f"CreateSolidColorGrid expects non-negative integer for rows, got {resolved_rows}")
        if not (isinstance(resolved_cols, int) and resolved_cols >= 0):
            self.logger.error(f"CreateSolidColorGrid received invalid resolved col dimension: {resolved_cols} (type {type(resolved_cols)})")
            raise ValueError(f"CreateSolidColorGrid expects non-negative integer for cols, got {resolved_cols}")

        self.logger.debug(f"Resolved dimensions: {resolved_rows}x{resolved_cols} with color {self.fill_color}")
        return np.full((resolved_rows, resolved_cols), self.fill_color, dtype=int)

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        """Yields any nested command arguments for context propagation."""
        if isinstance(self.rows_arg, AbstractTransformationCommand):
            yield self.rows_arg
        if isinstance(self.cols_arg, AbstractTransformationCommand):
            yield self.cols_arg

    @classmethod
    def describe(cls) -> str:
        return """
            Creates a new grid of specified dimensions filled with a single color.
            Dimensions can be fixed integers or the output of other commands (e.g., GetGridHeight).
            Parameters:
            - rows: The number of rows for the new grid, or a command yielding an integer.
            - cols: The number of columns for the new grid, or a command yielding an integer.
            - fill_color: The color value to fill the grid with.
        """
        
    
class ScaleGrid(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False,
        "parameter_ranges": {
            "scale_factor": (1, 5)
        }
    }

    def __init__(self, scale_factor: int, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        if scale_factor < 1:
            raise ValueError("Scale factor must be at least 1.")
        self.scale_factor = scale_factor

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        """
        Scales up the input grid by repeating each pixel by the given factor.
        Uses Kronecker product for pixel-wise scaling.
        """
        self.logger.debug(f"Executing ScaleGrid: Scaling by {self.scale_factor}x")
        if input_grid is None:
            raise ValueError("ScaleGrid requires an input grid.")

        scaled_grid = np.kron(input_grid, np.ones((self.scale_factor, self.scale_factor), dtype=int))
        return scaled_grid

    @classmethod
    def describe(cls) -> str:
        return """
            Scales up the input grid by repeating each pixel by a given integer factor.
            This operation preserves the geometry of objects by enlarging individual pixels.
            Parameter:
            - scale_factor: The integer factor by which to scale the grid (e.g., 2 for 2x).
        """



class FilterGridByColor(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "grid_op",
        "requires_inner": False,
        "parameter_ranges": {"target_color": None}
    }

    def __init__(self, target_color: int, logger: logging.Logger = None):
        super().__init__(logger=logger)
        self.target_color = target_color

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        output_grid = np.zeros_like(input_grid, dtype=int)
        output_grid[input_grid == self.target_color] = self.target_color
        return output_grid

    @classmethod
    def describe(cls) -> str:
        return "Filters the input grid, keeping only cells of a specified color and setting all other cells to 0 (empty)."

    def __str__(self) -> str:
        # Use INT_VAL_MAP for direct lookup to get the Roman symbol
        # Added .get() with a fallback in case a number outside the map range somehow appears
        return f"◎({INT_VAL_MAP.get(self.target_color, str(self.target_color))})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "FilterGridByColor",
            "params": {"target_color": self.target_color}
        }
        

class ExtractBoundingBox(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False,
        "parameter_ranges": {}
    }

    def __init__(self, argument_command: Optional[AbstractTransformationCommand] = None, logger: Optional[logging.Logger] = None): # ADDED arg_command
        super().__init__(logger)
        self.argument_command = argument_command # ADDED

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug("Executing ExtractBoundingBox")
        # MODIFIED: Use argument_command if provided
        if self.argument_command:
            grid_to_process = self.argument_command.execute(input_grid)
        else:
            grid_to_process = input_grid

        if grid_to_process is None: # Changed from input_grid to grid_to_process
            raise ValueError("ExtractBoundingBox requires an input grid.")

        non_background_coords = np.where(grid_to_process != 0) # Changed from input_grid

        if non_background_coords[0].size == 0:
            self.logger.debug("Input grid is entirely background, returning 1x1 empty grid.")
            return np.array([[0]], dtype=int)

        min_row, max_row = non_background_coords[0].min(), non_background_coords[0].max()
        min_col, max_col = non_background_coords[1].min(), non_background_coords[1].max()

        bounding_box = grid_to_process[min_row : max_row + 1, min_col : max_col + 1] # Changed from input_grid
        return bounding_box

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']: # ADDED
        if self.argument_command:
            yield self.argument_command

    @classmethod
    def describe(cls) -> str:
        return """
            Extracts the smallest rectangular subgrid that encompasses all non-background (non-zero) pixels.
            If the input grid contains only background pixels (0s), it returns a 1x1 grid with 0.
            Can apply to the current input grid or the result of an argument command.
        """


class GetConstant(AbstractTransformationCommand):
    def __init__(self, value: int, logger: logging.Logger = None):
        super().__init__(logger)
        self.value = value

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        return self.value

    @classmethod
    def describe(cls) -> str:
        return "Create a 1x1 grid with a specific constant value."
    

class FlattenGrid(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False, # This should likely be True now if it can take an arg_command
        "parameter_ranges": {},
    }

    def __init__(self, argument_command: Optional[AbstractTransformationCommand] = None, logger: logging.Logger = None): # ADDED arg_command
        super().__init__(logger) # Pass logger to super()
        self.argument_command = argument_command # ADDED


    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        if self.argument_command:
            grid_to_flatten = self.argument_command.execute(input_grid)
        else:
            grid_to_flatten = input_grid

        if grid_to_flatten.ndim != 2: # Changed from grid to grid_to_flatten
            raise ValueError(f"FlattenGrid expects a 2D grid, but got {grid_to_flatten.ndim} dimensions.")
        return grid_to_flatten.flatten() # Changed from grid

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']: # ADDED
        if self.argument_command:
            yield self.argument_command

    @classmethod
    def describe(cls) -> str:
        return """
        Flattens a 2D grid into a 1D array.
        Can apply to the current input grid or the result of an argument command.
        """
    
class GetElement(AbstractTransformationCommand):
    """
    Extracts a single element (color value) from the grid at the specified row and column.
    Returns a 1x1 grid containing that element.
    """
    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False,
        "parameter_ranges": {
            "row_index": "dynamic_row_index", # Indicates it depends on input grid dimensions
            "col_index": "dynamic_col_index"  # Indicates it depends on input grid dimensions
        }
    }

    def __init__(self, row_index: int, col_index: int, logger: logging.Logger = None):
        super().__init__(logger=logger)
        self.row_index = row_index
        self.col_index = col_index

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug(f"Executing GetElement for indices ({self.row_index}, {self.col_index}) "
                         f"on grid shape: {input_grid.shape}")
        rows, cols = input_grid.shape
        if not (0 <= self.row_index < rows and 0 <= self.col_index < cols):
            self.logger.error(f"Element indices ({self.row_index}, {self.col_index}) "
                              f"out of bounds for grid of shape {input_grid.shape}")
            raise IndexError(f"Element indices ({self.row_index}, {self.col_index}) out of bounds for grid of shape {input_grid.shape}")
        
        # Return a 1x1 grid containing the element
        return np.array([[input_grid[self.row_index, self.col_index]]], dtype=int)

    @classmethod
    def describe(cls) -> str:
        """Returns a human-readable description of the transformation type."""
        return "Extract Single Element"


class CompareEquality(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "requires_inner": True,
        "parameter_ranges": {}
    }

    def __init__(self, command1: Union[AbstractTransformationCommand, int],
                 command2: Union[AbstractTransformationCommand, int],
                 logger: logging.Logger = None):
        super().__init__(logger=logger)
        self.command1 = command1
        self.command2 = command2
        self.logger.debug(f"Initialized CompareEquality with command1: {self.command1} and command2: {self.command2}")

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        if isinstance(self.command1, AbstractTransformationCommand):
            yield self.command1
        if isinstance(self.command2, AbstractTransformationCommand):
            yield self.command2

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug(f"Executing CompareEquality command.")
        try:
            val1_resolved = self.command1.execute(input_grid) \
                            if isinstance(self.command1, AbstractTransformationCommand) \
                            else self.command1
            
            val2_resolved = self.command2.execute(input_grid) \
                            if isinstance(self.command2, AbstractTransformationCommand) \
                            else self.command2

            if isinstance(val1_resolved, np.ndarray) and val1_resolved.shape == (1,1):
                val1 = int(val1_resolved[0,0])
            elif isinstance(val1_resolved, int):
                val1 = val1_resolved
            else:
                self.logger.error(f"CompareEquality: Unexpected type for command1 result: {type(val1_resolved)}")
                raise TypeError(f"CompareEquality: Expected int or 1x1 np.ndarray for command1, got {type(val1_resolved)}.")

            if isinstance(val2_resolved, np.ndarray) and val2_resolved.shape == (1,1):
                val2 = int(val2_resolved[0,0])
            elif isinstance(val2_resolved, int):
                val2 = val2_resolved
            else:
                self.logger.error(f"CompareEquality: Unexpected type for command2 result: {type(val2_resolved)}")
                raise TypeError(f"CompareEquality: Expected int or 1x1 np.ndarray for command2, got {type(val2_resolved)}.")
            
            are_equal = (val1 == val2)
            
            self.logger.info(f"Comparison of {str(self.command1)} (resolved to: {val1}) and {str(self.command2)} (resolved to: {val2}) resulted in: {are_equal}")
            return np.array([[1 if are_equal else 0]], dtype=int)
        except Exception as e:
            self.logger.error(f"Error executing CompareEquality command: {e}", exc_info=True)
            raise

    @classmethod
    def describe(cls) -> str:
        return "Compares two inputs for scalar equality."
    
class CompareGridEquality(AbstractTransformationCommand):
    """
    Compares two inputs for equality as grids.
    Inputs can be commands (producing grids).
    Returns a 1x1 grid containing 1 if the grids are equal, otherwise returns 0.
    """
    synthesis_rules = {
        "type": "combinator",
        "requires_inner": True,
        "parameter_ranges": {}
    }

    def __init__(self, command1: AbstractTransformationCommand,
                 command2: AbstractTransformationCommand,
                 logger: logging.Logger = None):
        super().__init__(logger=logger)
        self.command1 = command1
        self.command2 = command2
        self.logger.debug(f"Initialized CompareGridEquality with command1: {self.command1} and command2: {self.command2}")
    
    def get_children_commands(self) -> List[AbstractTransformationCommand]:
        children = []
        if self.command1:
            children.append(self.command1)
        if self.command2:
            children.append(self.command2)
        
        self.logger.debug(f"CompareGridEquality.get_children_commands called. ID: {id(self)}. Returning children IDs: {[id(c) for c in children]}")
        
        return children
    
    def _resolve_grid_arg(self, arg: AbstractTransformationCommand, input_grid: np.ndarray) -> np.ndarray:
        """
        Helper method to execute a command and ensure it returns a grid.
        """
        if isinstance(arg, AbstractTransformationCommand):
            result_grid = arg.execute(input_grid)
            if not isinstance(result_grid, np.ndarray):
                self.logger.error(f"Nested command '{str(arg)}' for grid comparison returned non-grid type: {type(result_grid)}.")
                raise TypeError("Grid comparison argument must resolve to a NumPy array (grid).")
            return result_grid
        else:
            self.logger.error(f"Argument '{arg}' provided to CompareGridEquality is not a command. Expected a command that produces a grid.")
            raise TypeError("CompareGridEquality expects arguments that are commands producing grids.")


    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        """
        Compares the resolved grid outputs of command1 and command2 using np.array_equal.
        Returns a 1x1 grid with 1 if equal, 0 if not equal.
        """
        self.logger.debug(f"Executing CompareGridEquality command.")
        try:
            grid_val1 = self._resolve_grid_arg(self.command1, input_grid)
            grid_val2 = self._resolve_grid_arg(self.command2, input_grid)

            are_equal = np.array_equal(grid_val1, grid_val2)
            
            self.logger.info(f"Comparison of {str(self.command1)} (resolved to grid of shape: {grid_val1.shape}) and {str(self.command2)} (resolved to grid of shape: {grid_val2.shape}) resulted in: {are_equal}")
            return np.array([[1 if are_equal else 0]], dtype=int)
        except Exception as e:
            self.logger.error(f"Error executing CompareGridEquality command: {e}", exc_info=True)
            raise

    @classmethod
    def describe(cls) -> str:
        return "Compares two inputs for grid equality."
    
class IfElseCondition(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "requires_inner": True,
        "parameter_ranges": {}
    }

    def __init__(self, condition: Any, 
                 true_branch: AbstractTransformationCommand,
                 false_branch: AbstractTransformationCommand,
                 logger: logging.Logger = None):
        super().__init__(logger=logger)
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.logger.debug(f"Initialized IfElseCondition: condition={self.condition}, true_branch={self.true_branch}, false_branch={self.false_branch}")

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug(f"Executing IfElseCondition.")
        try:
            if isinstance(self.condition, AbstractTransformationCommand):
                condition_result_grid = self.condition.execute(input_grid)
                if condition_result_grid.shape != (1, 1):
                    self.logger.error(f"IfElseCondition: condition command returned a grid of shape {condition_result_grid.shape}, expected (1,1) for scalar comparison.")
                    raise ValueError("IfElseCondition condition must resolve to a single scalar value (1x1 grid).")
                condition_value = condition_result_grid.item()
            else:
                condition_value = self.condition
                if not isinstance(condition_value, int) or (condition_value != 0 and condition_value != 1):
                     self.logger.warning(f"IfElseCondition: literal condition value is {condition_value}, expected 0 or 1.")

            if condition_value == 1:
                self.logger.debug(f"IfElseCondition: condition is TRUE. Executing true branch.")
                return self.true_branch.execute(input_grid)
            else:
                self.logger.debug(f"IfElseCondition: condition is FALSE. Executing false branch.")
                return self.false_branch.execute(input_grid)
        except Exception as e:
            self.logger.error(f"Error executing IfElseCondition: {e}", exc_info=True)
            raise

    @classmethod
    def describe(cls) -> str:
        return "Executes one of two commands based on a condition."
    

class BlockGridBuilder(AbstractTransformationCommand): # Renamed class
    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False,
        "parameter_ranges": {
            "block_rows": (1, 30),
            "block_cols": (1, 30),
            "block_pixel_size": (1, 10)
        }
    }

    def __init__(self, 
                 block_rows: int, 
                 block_cols: int, 
                 pattern_matrix: np.ndarray,
                 block_pixel_size: int = 1, 
                 logger: logging.Logger = None):
        super().__init__(logger)
        self.block_rows = block_rows
        self.block_cols = block_cols
        self.pattern_matrix = pattern_matrix 
        self.block_pixel_size = block_pixel_size

        if self.pattern_matrix.shape != (self.block_rows, self.block_cols):
            self.logger.error(f"BlockGridBuilder: pattern_matrix shape {self.pattern_matrix.shape} does not match declared block_rows {self.block_rows} and block_cols {self.block_cols}.")
            raise ValueError("Pattern matrix dimensions must match block_rows and block_cols.")
        
        self.logger.debug(f"Initialized BlockGridBuilder: {self.block_rows}x{self.block_cols} blocks, pixel size {self.block_pixel_size} with pattern:\n{self.pattern_matrix}")

    def execute(self, input_grid: np.ndarray = None) -> np.ndarray:
        self.logger.debug("Executing BlockGridBuilder.")
        
        total_rows = self.block_rows * self.block_pixel_size
        total_cols = self.block_cols * self.block_pixel_size

        output_grid = np.zeros((total_rows, total_cols), dtype=int) 

        for r_block in range(self.block_rows):
            for c_block in range(self.block_cols):
                block_color = self.pattern_matrix[r_block, c_block]
                
                if block_color != 0: 
                    start_row = r_block * self.block_pixel_size
                    end_row = start_row + self.block_pixel_size
                    start_col = c_block * self.block_pixel_size
                    end_col = start_col + self.block_pixel_size
                    output_grid[start_row:end_row, start_col:end_col] = block_color
        
        self.logger.debug(f"BlockGridBuilder generated grid of shape: {output_grid.shape}")
        return output_grid

    @classmethod
    def describe(cls) -> str:
        return """
            Generates a colored grid based on a specified pattern of symbolic color values.
            Each 'block' in the pattern is expanded to a given pixel size (e.g., 1x1, 3x3 pixels).
            Parameters:
            - block_rows: The number of rows in the block pattern (Roman numeral).
            - block_cols: The number of columns in the block pattern (Roman numeral).
            - pattern_list_str: A string representing the 2D pattern using symbolic colors
              (e.g., '[[II,∅,I],[∅,V,∅]]' for a 2x3 pattern with colors 2,0,1 / 0,5,0).
            - block_pixel_size: The number of pixels each block in the pattern expands to (e.g., 1 for 1x1).
        """
        
        
class MatchPattern(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "requires_inner": False,
        "parameter_ranges": {"grid_to_evaluate_cmd": None, "cases": [], "default_action_cmd": None}
    }

    def __init__(self, 
                 grid_to_evaluate_cmd: AbstractTransformationCommand,
                 cases: List[Tuple[AbstractTransformationCommand, AbstractTransformationCommand]], 
                 default_action_cmd: AbstractTransformationCommand,
                 logger: logging.Logger = None):
        super().__init__(logger=logger)
        self.grid_to_evaluate_cmd = grid_to_evaluate_cmd
        self.default_action_cmd = default_action_cmd
        self.cases = cases 
        
        self.logger.debug(f"Initialized MatchPattern: grid_to_eval={self.grid_to_evaluate_cmd}, num_cases={len(self.cases)}, default_action={self.default_action_cmd}")

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        if self.grid_to_evaluate_cmd:
            yield self.grid_to_evaluate_cmd
        
        for case_pattern_cmd, action_cmd in self.cases:
            if case_pattern_cmd:
                yield case_pattern_cmd 
            if action_cmd:
                yield action_cmd 

        if self.default_action_cmd:
            yield self.default_action_cmd

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        grid_to_match = self.grid_to_evaluate_cmd.execute(input_grid)
        
        self.logger.debug(f"MatchPattern: Grid to evaluate (shape: {grid_to_match.shape}):\n{grid_to_match}")

        for case_pattern_cmd, action_cmd in self.cases:
            try:
                pattern_grid_to_compare = case_pattern_cmd.execute(input_grid)
            except Exception as e:
                self.logger.error(f"MatchPattern: Error executing case pattern command '{case_pattern_cmd.__class__.__name__}': {e}", exc_info=True)
                continue 

            self.logger.debug(f"MatchPattern: Comparing with resolved case pattern (shape: {pattern_grid_to_compare.shape}):\n{pattern_grid_to_compare}")
            
            if grid_to_match.shape == pattern_grid_to_compare.shape and np.array_equal(grid_to_match, pattern_grid_to_compare):
                self.logger.info(f"MatchPattern: Pattern matched. Executing action: {action_cmd.__class__.__name__}")
                return action_cmd.execute(input_grid)

        self.logger.info(f"MatchPattern: No pattern matched. Executing default action: {self.default_action_cmd.__class__.__name__}")
        return self.default_action_cmd.execute(input_grid)


    @classmethod
    def describe(cls) -> str:
        return """
        Performs conditional execution based on matching an extracted subgrid against predefined patterns.
        It first evaluates a command to obtain a target grid. Then, it iterates through a list of cases,
        each consisting of a specific pattern (represented by a command that produces a grid) and an action to perform.
        If the target grid exactly matches the grid produced by a pattern command, the corresponding action is executed.
        If no pattern matches, a default action is executed.
        """
        
class InputGridReference(AbstractTransformationCommand):
    synthesis_rules = {"type": "atomic"} 

    def __init__(self, logger: logging.Logger = None):
        super().__init__(logger)
        self._initial_puzzle_input: Optional[np.ndarray] = None # Placeholder
        self.logger.debug(f"InputGridReference.__init__ called. ID: {id(self)}")

    def set_initial_puzzle_input(self, initial_grid: np.ndarray):
        """Sets the actual initial puzzle input from the Executor."""
        self.logger.debug(f"InputGridReference.set_initial_puzzle_input called. ID: {id(self)}. Setting input.")
        if not isinstance(initial_grid, np.ndarray):
            raise TypeError("Initial grid must be a NumPy array.")
        self._initial_puzzle_input = initial_grid
        self.logger.debug(f"InputGridReference.set_initial_puzzle_input called. ID: {id(self)}. Initial input set.")

    def execute(self, current_grid: np.ndarray) -> np.ndarray:
        """
        Returns the stored initial puzzle input grid.
        The current_grid argument is ignored for this command.
        """
        self.logger.debug(f"InputGridReference.execute called. ID: {id(self)}. _initial_puzzle_input is None: {self._initial_puzzle_input is None}")
        if self._initial_puzzle_input is None:
            raise ValueError(
                "InputGridReference's initial puzzle input was not set! "
                "Ensure the Executor initialized the command tree."
            )
        if self.logger:
            self.logger.debug(f"Executing InputGridReference: returning initial grid of shape {self._initial_puzzle_input.shape}")
        return self._initial_puzzle_input.copy()

    @classmethod
    def describe(cls) -> str:
        return "Returns the initial puzzle input grid."

    def set_executor_context(self, executor: Any):
        super().set_executor_context(executor)
        
        
class GetExternalBackgroundMask(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False,
        "parameter_ranges": {
            "background_color": (0, 9)
        }
    }

    def __init__(self, background_color: int, logger: logging.Logger = None):
        super().__init__(logger)
        self.background_color = background_color

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        rows, cols = input_grid.shape
        padded_rows, padded_cols = rows + 2, cols + 2

        binary_padded_input = np.zeros((padded_rows, padded_cols), dtype=np.uint8)
        
        binary_padded_input[1:rows+1, 1:cols+1][input_grid == self.background_color] = 1
        binary_padded_input[0, :] = 1
        binary_padded_input[padded_rows-1, :] = 1
        binary_padded_input[:, 0] = 1
        binary_padded_input[:, padded_cols-1] = 1
        
        binary_for_cv = (binary_padded_input * 255).astype(np.uint8)

        # Hardcoded 4-connectivity based on ARC puzzle common rules
        cv_connectivity = 4 

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_for_cv, 
            connectivity=cv_connectivity, 
            ltype=cv2.CV_32S
        )
        
        external_background_label = labels[0, 0]
        
        external_background_mask_padded = (labels == external_background_label)
        
        unpadded_mask = external_background_mask_padded[1:rows+1, 1:cols+1]
        
        return unpadded_mask.astype(input_grid.dtype)

    @classmethod
    def describe(cls) -> str:
        return """
            Generates a binary mask identifying the external background of the grid using OpenCV's connected components.
            Pixels connected to the grid's border (including through an added padding layer) that match the
            `background_color` will be marked (as 1s). Connectivity is always 4-connected (up, down, left, right).
            Internal regions of the `background_color` (holes) that are not connected to the external border
            will NOT be marked.

            Parameters:
            - background_color: The integer color value considered as the primary background.

            Output: A binary grid (0s and 1s) of the same shape as the input grid,
                    where 1s represent the external background.
        """
    
def _to_binary(grid: np.ndarray) -> np.ndarray:
    """Converts a grid to a binary mask where 0 is False, non-zero is True (1)."""
    return (grid != 0).astype(int)

class MaskNot(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "arity": 1,
        "requires_inner": False,
        "parameter_ranges": {}
    }

    def __init__(self, mask_cmd: AbstractTransformationCommand, logger: logging.Logger = None):
        super().__init__(logger)
        self.mask_cmd = mask_cmd

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.mask_cmd

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        mask = self.mask_cmd.execute(input_grid)

        binary_mask = (mask != 0).astype(int)

        return (1 - binary_mask).astype(int)

    @classmethod
    def describe(cls) -> str:
        return """
            Performs an element-wise logical NOT operation on an input mask.
            Any non-zero value in the input mask is treated as True (1), and 0 as False (0).
            The output grid will have 1 where the input mask is False (0), and 0 where the input mask is True (non-zero).
            The output is always a binary (0s and 1s) grid.
        """

class MaskOr(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "arity": 2,
        "requires_inner": False,
        "parameter_ranges": {}
    }

    def __init__(self, mask_cmd1: AbstractTransformationCommand, mask_cmd2: AbstractTransformationCommand, logger: logging.Logger = None):
        super().__init__(logger)
        self.mask_cmd1 = mask_cmd1
        self.mask_cmd2 = mask_cmd2

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.mask_cmd1
        yield self.mask_cmd2

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        mask1 = self.mask_cmd1.execute(input_grid)
        mask2 = self.mask_cmd2.execute(input_grid)

        binary_mask1 = (mask1 != 0).astype(int)
        binary_mask2 = (mask2 != 0).astype(int)

        if not (binary_mask1.shape == binary_mask2.shape):
            self.logger.warning(f"Shape mismatch: Mask1 {binary_mask1.shape}, Mask2 {binary_mask2.shape}. Both must match for element-wise OR.")
            raise ValueError("Shape mismatch in MaskOr: Masks must have same shape.")

        return ((binary_mask1 + binary_mask2) > 0).astype(int)

    @classmethod
    def describe(cls) -> str:
        return """
            Performs an element-wise logical OR operation on two input masks.
            Any non-zero value in the input masks is treated as True (1), and 0 as False (0).
            Both mask commands must produce grids of the same shape.
            The output grid will have 1 if either mask (after conversion to binary) has 1, and 0 otherwise.
            The output is always a binary (0s and 1s) grid.
        """

class MaskAnd(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "combinator",
        "arity": 2, 
        "requires_inner": False,
        "parameter_ranges": {}
    }

    def __init__(self, mask_cmd1: AbstractTransformationCommand, mask_cmd2: AbstractTransformationCommand, logger: logging.Logger = None):
        super().__init__(logger)
        self.mask_cmd1 = mask_cmd1
        self.mask_cmd2 = mask_cmd2

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.mask_cmd1
        yield self.mask_cmd2

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        mask1 = self.mask_cmd1.execute(input_grid)
        mask2 = self.mask_cmd2.execute(input_grid)

        binary_mask1 = (mask1 != 0).astype(int)
        binary_mask2 = (mask2 != 0).astype(int)

        if not (binary_mask1.shape == binary_mask2.shape):
            self.logger.warning(f"Shape mismatch: Mask1 {binary_mask1.shape}, Mask2 {binary_mask2.shape}. Both must match for element-wise AND.")
            raise ValueError("Shape mismatch in MaskAnd: Masks must have same shape.")

        return (binary_mask1 * binary_mask2).astype(int)

    @classmethod
    def describe(cls) -> str:
        return """
            Performs an element-wise logical AND operation on two input masks.
            Any non-zero value in the input masks is treated as True (1), and 0 as False (0).
            Both mask commands must produce grids of the same shape.
            The output grid will have 1 where both masks (after conversion to binary) have 1, and 0 otherwise.
            The output is always a binary (0s and 1s) grid.
        """
        
class Binarize(AbstractTransformationCommand):

    synthesis_rules = {
        "type": "transformation", 
        "arity": 1, 
        "requires_inner": False,
        "parameter_ranges": {}
    }

    def __init__(self, cmd: AbstractTransformationCommand, logger: logging.Logger = None):
        super().__init__(logger)
        self.cmd = cmd 

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.cmd

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        grid_to_binarize = self.cmd.execute(input_grid)

        return (grid_to_binarize != 0).astype(int)

    @classmethod
    def describe(cls) -> str:
        return """
            Converts an input grid into a binary mask (0s and 1s).
            Any non-zero value in the input grid is converted to 1 (True),
            and any 0 value is converted to 0 (False).
            This is useful for explicitly creating a mask from any grid.
        """
        
class LocatePattern(AbstractTransformationCommand):
    """
    Locates instances of a specified pattern within a grid and returns a binary mask
    where matching regions are marked with 1s.
    The wildcard symbol '?' in the pattern matches any color.
    """
    
    _WILDCARD_VALUE = -1 

    synthesis_rules = {
        "type": "combinator",
        "arity": 2,
        "parameter_ranges": {}
    }

    def __init__(
        self,
        grid_to_search_cmd: AbstractTransformationCommand,
        pattern_to_find_cmd: AbstractTransformationCommand,
        logger: logging.Logger = None
    ):
        super().__init__(logger)
        self.grid_to_search_cmd = grid_to_search_cmd
        self.pattern_to_find_cmd = pattern_to_find_cmd

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.grid_to_search_cmd
        yield self.pattern_to_find_cmd

    def _is_match(self, subgrid: np.ndarray, pattern_grid: np.ndarray) -> bool:
        """
        Helper method to check if a subgrid matches the pattern_grid,
        considering WILDCARD_VALUE as a wildcard.
        """

        if subgrid.shape != pattern_grid.shape:
            return False

        # Iterate element-wise to compare
        for r in range(pattern_grid.shape[0]):
            for c in range(pattern_grid.shape[1]):
                pattern_val = pattern_grid[r, c]
                subgrid_val = subgrid[r, c]

                if pattern_val == self._WILDCARD_VALUE:
                    continue
                elif pattern_val != subgrid_val:
                    return False
        return True # All cells matched

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug("Executing LocatePattern command.")
        
        main_grid = self.grid_to_search_cmd.execute(input_grid)
        pattern_grid = self.pattern_to_find_cmd.execute(input_grid) 

        mh, mw = main_grid.shape
        ph, pw = pattern_grid.shape

        if ph > mh or pw > mw:
            self.logger.debug("Pattern is larger than the main grid. No matches possible.")
            return np.zeros_like(main_grid, dtype=int)
        
        if ph == 0 or pw == 0:
            self.logger.debug("Pattern grid is empty. No matches possible.")
            return np.zeros_like(main_grid, dtype=int)

        result_mask = np.zeros_like(main_grid, dtype=int)

        for r_main in range(mh - ph + 1):
            for c_main in range(mw - pw + 1):
                subgrid = main_grid[r_main : r_main + ph, c_main : c_main + pw]
                
                if self._is_match(subgrid, pattern_grid):
                    self.logger.debug(f"Pattern match found at ({r_main}, {c_main}).")
                    result_mask[r_main : r_main + ph, c_main : c_main + pw] = 1

        self.logger.debug("LocatePattern execution complete.")
        return result_mask

    @classmethod
    def describe(cls) -> str:
        return """
            Locates instances of a specified pattern within a grid and returns a binary mask.
            The mask contains 1s where the pattern is found and 0s elsewhere.
            If multiple matches are found, their masks are merged (logical OR).

            Parameters:
            - grid_to_search_cmd: A command that evaluates to the grid to search within.
            - pattern_to_find_cmd: A command that evaluates to the pattern grid to find.
              The '?' symbol in the pattern (which should be translated to -1 internally)
              acts as a wildcard, matching any color.

            Example:
            Input Grid (main_grid):
            [[0,0,0,0,0],
             [0,1,1,1,0],
             [0,1,2,1,0],
             [0,1,1,1,0],
             [0,0,0,0,0]]

            Pattern (from pattern_to_find_cmd, e.g., ▦(III,III,[[?,I,?],[I,?,I],[?,I,?]])):
            [[ -1, 1, -1],
             [  1,-1,  1],
             [ -1, 1, -1]]  (where -1 is the internal WILDCARD_VALUE for '?')

            Command: ⌖(⌂, ▦(III,III,[[?,I,?],[I,?,I],[?,I,?]]))

            Output Mask:
            [[0,0,0,0,0],
             [0,1,1,1,0],
             [0,1,1,1,0],
             [0,1,1,1,0],
             [0,0,0,0,0]]
        """
    
    
class SliceGrid(AbstractTransformationCommand):
    def __init__(self,
                 row_start: int,
                 col_start: int,
                 row_end: int,
                 col_end: int,
                 logger=None):
        super().__init__(logger)
        self.row_start = row_start
        self.col_start = col_start
        self.row_end = row_end
        self.col_end = col_end

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        actual_row_start = self.row_start - 1
        actual_col_start = self.col_start - 1
        actual_row_end = self.row_end 
        actual_col_end = self.col_end 

        return input_grid[actual_row_start:actual_row_end, actual_col_start:actual_col_end]

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield from ()

    @classmethod
    def describe(cls) -> str:
        return "Slices a rectangular sub-grid from the input grid based on 1-indexed start and end row/column indices."
    
class FillRegion(AbstractTransformationCommand):
    def __init__(
        self,
        target_grid_command: AbstractTransformationCommand,
        fill_value: int,
        row_start: int,
        col_start: int,
        row_end: int,
        col_end: int,
        logger=None
    ):
        super().__init__(logger)
        self.target_grid_command = target_grid_command
        self.fill_value = fill_value
        self.row_start = row_start
        self.col_start = col_start
        self.row_end = row_end
        self.col_end = col_end

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        target_grid = self.target_grid_command.execute(input_grid)
        
        result_grid = np.copy(target_grid)

        np_row_start = self.row_start - 1
        np_col_start = self.col_start - 1
        np_row_end = self.row_end
        np_col_end = self.col_end

        grid_height, grid_width = result_grid.shape
        if not (1 <= self.row_start <= grid_height and 1 <= self.col_start <= grid_width and
                1 <= self.row_end <= grid_height and 1 <= self.col_end <= grid_width and
                self.row_start <= self.row_end and self.col_start <= self.col_end):
            self.logger.warning(
                f"FillRegion coordinates out of bounds or invalid: "
                f"Grid({grid_height}x{grid_width}), "
                f"Region([{self.row_start},{self.col_start}] to [{self.row_end},{self.col_end}]). "
                f"Returning original grid."
            )
            return target_grid

        result_grid[np_row_start:np_row_end, np_col_start:np_col_end] = self.fill_value
        return result_grid

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.target_grid_command
        
    @classmethod
    def describe(cls) -> str:
        return "FillRegion(target_grid, fill_value, row_start, col_start, row_end, col_end): Fills a rectangular region of target_grid."

    synthesis_rules = {
        "type": "transformation",
        "arity": 6,
        "allowed_input_types": {
            "target_grid_command": "grid",
            "fill_value": "scalar",
            "row_start": "scalar",
            "col_start": "scalar",
            "row_end": "scalar",
            "col_end": "scalar"
        }
    }
    

class AddGridToCanvas(AbstractTransformationCommand):
    """
    Adds a source_grid to a target_grid (canvas) at a specified (row, col) offset.
    The source_grid is cropped if it extends beyond the bounds of the target_grid.
    The left corner of the source_grid is used as the reference point for placement.
    
    Modified Behavior: Zeros in the source_grid do NOT overwrite values in the target_grid.
    They act as a transparent background. Only non-zero values are copied over.
    """

    def __init__(self,
                 target_grid_command: AbstractTransformationCommand,
                 source_grid_command: AbstractTransformationCommand,
                 row_offset: int,
                 col_offset: int,
                 logger=None
                 ):
        super().__init__(logger)
        self.target_grid_command = target_grid_command
        self.source_grid_command = source_grid_command
        self.row_offset = row_offset
        self.col_offset = col_offset

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        target_grid = self.target_grid_command.execute(input_grid)
        source_grid = self.source_grid_command.execute(input_grid)

        if not isinstance(target_grid, np.ndarray) or not isinstance(source_grid, np.ndarray):
            raise ValueError("AddGridToCanvas expects both target_grid and source_grid to be numpy arrays.")

        target_rows, target_cols = target_grid.shape
        source_rows, source_cols = source_grid.shape

        start_row_target = max(0, self.row_offset)
        end_row_target = min(target_rows, self.row_offset + source_rows)
        start_col_target = max(0, self.col_offset)
        end_col_target = min(target_cols, self.col_offset + source_cols)

        start_row_source = max(0, -self.row_offset)
        end_row_source = min(source_rows, target_rows - self.row_offset)
        start_col_source = max(0, -self.col_offset)
        end_col_source = min(source_cols, target_cols - self.col_offset)

        if start_row_target >= end_row_target or start_col_target >= end_col_target:
            if self.logger:
                self.logger.warning(
                    f"Attempted to add a grid completely out of bounds at ({self.row_offset}, {self.col_offset}). "
                    f"Target grid: {target_grid.shape}, Source grid: {source_grid.shape}. Returning original target grid."
                )
            return target_grid.copy() 

        result_grid = target_grid.copy()

        for r_src_idx in range(end_row_source - start_row_source):
            for c_src_idx in range(end_col_source - start_col_source):
                r_src = start_row_source + r_src_idx
                c_src = start_col_source + c_src_idx

                val_from_source = source_grid[r_src, c_src]

                if val_from_source != 0:
                    r_target = start_row_target + r_src_idx
                    c_target = start_col_target + c_src_idx
                    result_grid[r_target, c_target] = val_from_source

        return result_grid

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.target_grid_command
        yield self.source_grid_command

    @classmethod
    def describe(cls) -> str:
        return "Adds a source grid to a target grid at an offset, with non-zero values overwriting."

    synthesis_rules = {
        "output_type": "grid",
        "input_type": "grid",
        "parameters": [
            {"name": "target_grid_command", "type": "grid_command"},
            {"name": "source_grid_command", "type": "grid_command"},
            {"name": "row_offset", "type": "int"},
            {"name": "col_offset", "type": "int"},
        ],
    }
    
    
class GetConnectedComponent(AbstractTransformationCommand):
    def __init__(self, row_index: int, col_index: int, logger=None):
        super().__init__(logger)
        self.row_index = row_index
        self.col_index = col_index

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        np_row = self.row_index - 1
        np_col = self.col_index - 1

        if not (0 <= np_row < input_grid.shape[0] and 0 <= np_col < input_grid.shape[1]):
            return np.zeros_like(input_grid)

        start_pixel_value = input_grid[np_row, np_col]

        if start_pixel_value == 0:
            return np.zeros_like(input_grid)

        component_grid = self._find_connected_component(
            grid=input_grid,
            start_row=np_row,
            start_col=np_col,
            target_value=start_pixel_value
        )
        return component_grid

    def _find_connected_component(self, grid: np.ndarray, start_row: int, start_col: int, target_value: int) -> np.ndarray:
        rows, cols = grid.shape
        component_grid = np.zeros_like(grid, dtype=grid.dtype)
        visited = np.zeros_like(grid, dtype=bool)

        q = deque([(start_row, start_col)])
        visited[start_row, start_col] = True
        component_grid[start_row, start_col] = target_value

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while q:
            r, c = q.popleft()

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and
                    grid[nr, nc] == target_value and not visited[nr, nc]):
                    visited[nr, nc] = True
                    component_grid[nr, nc] = target_value
                    q.append((nr, nc))

        return component_grid

    def get_children_commands(self) -> Iterator[AbstractTransformationCommand]:
        return iter([])

    @classmethod
    def describe(cls) -> str:
        return "Extracts a 4-way connected component from a grid based on a starting position."

    @property
    def synthesis_rules(self) -> dict:
        return {
            "output_type": "grid",
            "args": {
                "row_index": {"type": "integer", "range": (1, 10)},
                "col_index": {"type": "integer", "range": (1, 10)}
            }
        }
        
class GetGridHeight(AbstractTransformationCommand):
    """
    DSL Command: │(grid_source_command)
    Returns the height (number of rows) of the grid produced by grid_source_command.
    Outputs an integer.
    """
    def __init__(self,target_grid_command: AbstractTransformationCommand, logger: logging.Logger=None,):
        super().__init__(logger)
        self.target_grid_command = target_grid_command
        self.logger.debug(f"Initialized GetGridHeight with source: {target_grid_command.describe()}")

    def execute(self, input_grid: np.ndarray) -> int:
        self.logger.debug(f"Executing GetGridHeight on grid_source_command.")
        grid_to_measure = self.target_grid_command.execute(input_grid)
        if not isinstance(grid_to_measure, np.ndarray):
            self.logger.error(f"GetGridHeight expects a grid, but received {type(grid_to_measure)}")
            raise TypeError(f"GetGridHeight expects a grid command output, but received {type(grid_to_measure)}")
        height = grid_to_measure.shape[0]
        self.logger.debug(f"Calculated height: {height}")
        return  height

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.target_grid_command

    @classmethod
    def describe(cls) -> str:
        return "Returns the height of the grid as an integer."

    synthesis_rules = {
            "output_type": "int_value",
            "args": [
                {"name": "target_grid_command", "type": "grid_command"}
            ]
        }

class GetGridWidth(AbstractTransformationCommand):
    """
    DSL Command: ─(grid_source_command)
    Returns the width (number of columns) of the grid produced by grid_source_command.
    Outputs an integer.
    """
    def __init__(self, target_grid_command: AbstractTransformationCommand , logger: logging.Logger=None,):
        super().__init__(logger)
        self.target_grid_command = target_grid_command
        self.logger.debug(f"Initialized GetGridWidth with source: {target_grid_command.describe()}")

    def execute(self, input_grid: np.ndarray) -> int:
        self.logger.debug(f"Executing GetGridWidth on grid_source_command.")
        grid_to_measure = self.target_grid_command.execute(input_grid)
        if not isinstance(grid_to_measure, np.ndarray):
            self.logger.error(f"GetGridWidth expects a grid, but received {type(grid_to_measure)}")
            raise TypeError(f"GetGridWidth expects a grid command output, but received {type(grid_to_measure)}")
        width = grid_to_measure.shape[1]
        self.logger.debug(f"Calculated width: {width}")
        return  width

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.target_grid_command

    @classmethod
    def describe(cls) -> str:
        return "Returns the width of the grid as an integer."


    synthesis_rules = {
            "output_type": "int_value",
            "args": [
                {"name": "target_grid_command", "type": "grid_command"}
            ]
        }