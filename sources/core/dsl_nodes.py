from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union, Iterator
from assets.symbols import ROM_VAL_MAP 

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


class ReverseRow(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False,
        "parameter_ranges": {}
    }

    def __init__(self, argument_command: Optional[AbstractTransformationCommand] = None, logger: logging.Logger = None):
        super().__init__(logger)
        self.argument_command = argument_command 

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug("Executing ReverseRow on all rows.")
        if self.argument_command:
            grid_to_reverse = self.argument_command.execute(input_grid)
        else:
            grid_to_reverse = input_grid
        return grid_to_reverse[:, ::-1]

    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        if self.argument_command:
            yield self.argument_command
            
    @classmethod
    def describe(cls) -> str:
        return """
        Reverses the elements in every row of the grid.
        Can apply to the current input grid or the result of an argument command.
        Example:
        Input: [[1, 2, 3], [4, 5, 6]]
        Command: ReverseRow() or ReverseRow(InputGridReference())
        Output: [[3, 2, 1], [6, 5, 4]]
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
        "arity": 3, # <--- IMPORTANT CHANGE: Now takes three arguments
        "requires_inner": True,
        "parameter_ranges": {}
    }

    def __init__(self, 
                 inner_command: AbstractTransformationCommand, 
                 mask_command: AbstractTransformationCommand, 
                 false_value_command: AbstractTransformationCommand, # <--- NEW PARAMETER
                 logger: logging.Logger = None):
        super().__init__(logger)
        self.inner_command = inner_command
        self.mask_command = mask_command 
        self.false_value_command = false_value_command # <--- NEW: Store the command


    def get_children_commands(self) -> Iterator['AbstractTransformationCommand']:
        yield self.inner_command

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug(f"Executing MaskCombinator on {input_grid.shape}")
        
        # 1. Get the 'true_value' grid
        transformed_grid = self.inner_command.execute(input_grid)
        
        # 2. Get the 'mask' grid
        mask = self.mask_command.execute(input_grid) 
        
        # 3. Get the 'false_value' grid <--- NEW: Execute the third command
        false_value_grid = self.false_value_command.execute(input_grid)

        # Ensure all three grids have matching shapes
        if not (mask.shape == transformed_grid.shape == false_value_grid.shape):
            self.logger.warning(f"Shape mismatch: Mask {mask.shape}, Transformed {transformed_grid.shape}, False Value {false_value_grid.shape}. All must match for np.where.")
            raise ValueError(f"Shape mismatch in MaskCombinator: Mask {mask.shape}, Transformed {transformed_grid.shape}, False Value {false_value_grid.shape}. All must match.")
            
        # Perform the masking using the generic false_value_grid
        result = np.where(mask, transformed_grid, false_value_grid) 
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

    def __init__(self, rows: int, cols: int, fill_color: int, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.rows = rows
        self.cols = cols
        self.fill_color = fill_color

    def execute(self, input_grid: np.ndarray = None) -> np.ndarray:
        """
        Creates a new grid of specified dimensions filled with a single color.
        Note: input_grid is ignored as this command creates a new grid.
        """
        self.logger.debug(f"Executing CreateSolidColorGrid: {self.rows}x{self.cols} with color {self.fill_color}")
        return np.full((self.rows, self.cols), self.fill_color, dtype=int)

    @classmethod
    def describe(cls) -> str:
        return """
            Creates a new grid of specified dimensions filled with a single color.
            Parameters:
            - rows: The number of rows for the new grid.
            - cols: The number of columns for the new grid.
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
        
class ExtractBoundingBox(AbstractTransformationCommand):
    synthesis_rules = {
        "type": "atomic",
        "requires_inner": False, # This should likely be True now if it can take an arg_command
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
    """
    Compares two inputs for equality as single numerical values.
    Inputs can be commands (producing 1x1 grids) or direct integer literals.
    Returns a 1x1 grid containing 1 if the values are equal, otherwise returns 0.
    """
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

    def _get_scalar_from_arg(self, arg: Union[AbstractTransformationCommand, int], input_grid: np.ndarray) -> int:
        """
        Helper method to resolve an argument to a single integer scalar.
        If it's a command, execute it and extract the single value from its 1x1 grid output.
        If it's a literal int, return the int itself.
        Raises ValueError if a command produces a multi-element grid.
        """
        if isinstance(arg, AbstractTransformationCommand):
            result_grid = arg.execute(input_grid)
            if result_grid.shape == (1, 1): # Ensure it's a 1x1 grid
                return result_grid.item() # Extract the single integer value
            else:
                self.logger.error(f"Nested command '{str(arg)}' for scalar comparison returned grid of shape {result_grid.shape}, expected (1,1).")
                raise ValueError("Scalar comparison argument must resolve to a single value.")
        else: # It's already a literal integer
            return arg

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        """
        Compares the resolved scalar values of command1 and command2.
        Returns a 1x1 grid with 1 if equal, 0 if not equal.
        """
        self.logger.debug(f"Executing CompareEquality command.")
        try:
            # Resolve both arguments to single scalar integers
            scalar_val1 = self._get_scalar_from_arg(self.command1, input_grid)
            scalar_val2 = self._get_scalar_from_arg(self.command2, input_grid)

            are_equal = (scalar_val1 == scalar_val2) # Direct Python scalar comparison
            
            self.logger.info(f"Comparison of {str(self.command1)} (resolved to: {scalar_val1}) and {str(self.command2)} (resolved to: {scalar_val2}) resulted in: {are_equal}")
            return np.array([[1 if are_equal else 0]], dtype=int)
        except Exception as e:
            self.logger.error(f"Error executing CompareEquality command: {e}", exc_info=True)
            raise

    @classmethod
    def describe(cls) -> str:
        return "Compares two inputs for scalar equality."
    
class GetConstant(AbstractTransformationCommand):
    def __init__(self, value: int, logger: logging.Logger = None):
        super().__init__(logger)
        self.value = value

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        output_grid = np.array([[self.value]], dtype=int)
        return output_grid

    @classmethod
    def describe(cls) -> str:
        return "Create a 1x1 grid with a specific constant value."
    
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

    # MODIFIED: Ensure parameter names match the 'nested_commands' keys from SYMBOL_RULES
    def __init__(self, condition: Any,           # This will be the parsed condition command/literal
                 true_branch: AbstractTransformationCommand, # This will be the parsed true_branch command
                 false_branch: AbstractTransformationCommand, # This will be the parsed false_branch command
                 logger: logging.Logger = None):
        super().__init__(logger=logger)
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.logger.debug(f"Initialized IfElseCondition: condition={self.condition}, true_branch={self.true_branch}, false_branch={self.false_branch}")

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        self.logger.debug(f"Executing IfElseCondition.")
        try:
            # The condition should resolve to a scalar (0 or 1)
            if isinstance(self.condition, AbstractTransformationCommand):
                condition_result_grid = self.condition.execute(input_grid)
                if condition_result_grid.shape != (1, 1):
                    self.logger.error(f"IfElseCondition: condition command returned a grid of shape {condition_result_grid.shape}, expected (1,1) for scalar comparison.")
                    raise ValueError("IfElseCondition condition must resolve to a single scalar value (1x1 grid).")
                condition_value = condition_result_grid.item()
            else: # If it's a literal integer (e.g., from ↱I)
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
    

class BlockPatternMask(AbstractTransformationCommand):
    """
    Generates a boolean mask grid based on a specified pattern of True/False blocks.
    The pattern is expanded into a full mask where each 'block' in the pattern
    corresponds to a fixed pixel size (e.g., 3x3).
    """
    synthesis_rules = {
        "type": "atomic", # It generates a mask, doesn't transform the input grid directly
        "requires_inner": False,
        "parameter_ranges": {
            "block_rows": (1, 30), # Number of block rows in the pattern
            "block_cols": (1, 30)  # Number of block columns in the pattern
        }
    }

    # Define the pixel size of each block in the mask.
    # Based on the puzzle, each logical block (True/False in pattern) is 3x3 pixels.
    BLOCK_PIXEL_SIZE = 3 

    def __init__(self, block_rows: int, block_cols: int, pattern_matrix: np.ndarray, logger: logging.Logger = None):
        super().__init__(logger)
        self.block_rows = block_rows
        self.block_cols = block_cols
        self.pattern_matrix = pattern_matrix # This is already a boolean NumPy array from parsing

        # Basic validation
        if self.pattern_matrix.shape != (self.block_rows, self.block_cols):
            self.logger.error(f"BlockPatternMask: pattern_matrix shape {self.pattern_matrix.shape} does not match declared block_rows {self.block_rows} and block_cols {self.block_cols}.")
            raise ValueError("Pattern matrix dimensions must match block_rows and block_cols.")
        
        self.logger.debug(f"Initialized BlockPatternMask: {self.block_rows}x{self.block_cols} blocks with pattern:\n{self.pattern_matrix}")

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        """
        Generates the full boolean mask grid.
        The input_grid is only used to infer the final output shape if needed,
        but the mask is primarily generated from its own parameters.
        """
        self.logger.debug("Executing BlockPatternMask.")
        
        # Calculate the total pixel dimensions of the mask
        total_rows = self.block_rows * self.BLOCK_PIXEL_SIZE
        total_cols = self.block_cols * self.BLOCK_PIXEL_SIZE

        # Create an empty mask of the final size, filled with False
        full_mask = np.full((total_rows, total_cols), False, dtype=bool)

        # Populate the full mask based on the pattern_matrix
        for r_block in range(self.block_rows):
            for c_block in range(self.block_cols):
                if self.pattern_matrix[r_block, c_block]:
                    # If the pattern block is True, fill the corresponding pixel area with True
                    start_row = r_block * self.BLOCK_PIXEL_SIZE
                    end_row = start_row + self.BLOCK_PIXEL_SIZE
                    start_col = c_block * self.BLOCK_PIXEL_SIZE
                    end_col = start_col + self.BLOCK_PIXEL_SIZE
                    full_mask[start_row:end_row, start_col:end_col] = True
        
        self.logger.debug(f"BlockPatternMask generated mask of shape: {full_mask.shape}")
        return full_mask

    @classmethod
    def describe(cls) -> str:
        return """
            Generates a boolean mask grid based on a specified pattern of True ('I') and False ('∅') blocks.
            Each 'block' in the pattern is expanded to a fixed pixel size (e.g., 3x3 pixels).
            Parameters:
            - block_rows: The number of rows in the block pattern.
            - block_cols: The number of columns in the block pattern.
            - pattern_str: A string representing the 2D pattern (e.g., "I∅I;∅I∅").
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