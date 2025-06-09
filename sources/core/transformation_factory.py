from typing import Dict, Type
from core.dsl_nodes import *

class TransformationFactory:
    OPERATION_MAP = {
        'identity': Identity,
        'repeat_grid': RepeatGrid,
        'flip_h': FlipGridHorizontally,
        'flip_v': FlipGridVertically,
        'alternate': Alternate,
        'map_colors': MapColors,
        'swap_rows_or_columns': SwapRowsOrColumns,
        'reverse_row': ReverseRow,
        'apply_to_row': ApplyToRow,
        'custom_pattern': CustomPattern,
        'conditional_transform': ConditionalTransform,
        'mask_combinator': MaskCombinator,
        'shift_row_or_column': ShiftRowOrColumn,
        'sequence': Sequence,
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