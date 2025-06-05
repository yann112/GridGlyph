from typing import Dict, Type
from core.dsl_nodes import AbstractTransformationCommand, Identity, RepeatGrid, FlipGridHorizontally, FlipGridVertically

class TransformationFactory:
    """Factory class to create transformation command objects from strings."""

    # Mapping from operation names to their corresponding classes
    OPERATION_MAP: Dict[str, Type[AbstractTransformationCommand]] = {
        'repeat_grid': RepeatGrid,
        'identity': Identity,
        'flip_h': FlipGridHorizontally,
        'flip_v': FlipGridVertically
    }

    @classmethod
    def create_operation(cls, operation_name: str, *args, **kwargs) -> AbstractTransformationCommand:
        """Creates an instance of the specified transformation command.

        Args:
            operation_name (str): The name of the operation to create.
            *args: Positional arguments to pass to the operation constructor.
            **kwargs: Keyword arguments to pass to the operation constructor.

        Returns:
            AbstractTransformationCommand: An instance of the specified transformation command.

        Raises:
            ValueError: If the operation name is not recognized.
        """
        if operation_name in cls.OPERATION_MAP:
            return cls.OPERATION_MAP[operation_name](*args, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation_name}")
