from typing import Dict, Type
from core.dsl_nodes import (
    AbstractTransformationCommand,
    Identity,
    RepeatGrid,
    FlipGridHorizontally,
    FlipGridVertically,
    Alternate,
    MapColors
    )

class TransformationFactory:
    OPERATION_MAP = {
        'identity': Identity,
        'repeat_grid': RepeatGrid,
        'flip_h': FlipGridHorizontally,
        'flip_v': FlipGridVertically,
        'alternate': Alternate,
        'map_colors': MapColors
    }

    @classmethod
    def get_class(cls, operation_name: str) -> Type[AbstractTransformationCommand]:
        if operation_name not in cls.OPERATION_MAP:
            raise ValueError(f"Unknown operation: {operation_name}")
        return cls.OPERATION_MAP[operation_name]

    @classmethod
    def create_operation(cls, operation_name: str, *args, **kwargs):
        operation_class = cls.get_class(operation_name)
        return operation_class(*args, **kwargs)