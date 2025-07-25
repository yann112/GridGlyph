from typing import Dict, Type
from core.dsl_nodes import *

class TransformationFactory:
    OPERATION_MAP = { 
        "shift_row": ShiftRowOrColumn,
        "shift_column": ShiftRowOrColumn,
        'identity': Identity,
        'repeat_grid': RepeatGrid,
        'flip_h': FlipGridHorizontally,
        'rotate_grid_clockwise': RotateGridClockwise,
        'flip_v': FlipGridVertically,
        'alternate': Alternate,
        'map_numbers': MapNumbers,
        'swap_rows': SwapRowsOrColumns,
        'swap_columns': SwapRowsOrColumns,
        'apply_to_row': ApplyToRow,
        'apply_to_column': ApplyToColumn, 
        'conditional_transform': ConditionalTransform,
        'mask_combinator': MaskCombinator,
        'shift_row_or_column': ShiftRowOrColumn,
        'sequence': Sequence,
        'create_solid_color_grid': CreateSolidColorGrid,
        'create_solid_color_grid_commands': CreateSolidColorGrid,  # Use command instead of building from roman number
        'repeat_grid_horizontal': lambda **kwargs: RepeatGrid(
                inner_command=Identity(),
                horizontal_repeats=kwargs.get("horizontal_repeats"),
                vertical_repeats=1
            ),
        'repeat_grid_vertical': lambda **kwargs: RepeatGrid(
                inner_command=Identity(),
                horizontal_repeats=1,
                vertical_repeats=kwargs.get("vertical_repeats")
            ),
        'scale_grid': ScaleGrid,
        'extract_bounding_box': ExtractBoundingBox,
        'flatten_grid': FlattenGrid ,
        'if_else_condition': IfElseCondition,
        'get_element': GetElement,
        'compare_equality': CompareEquality,
        'get_constant': GetConstant,
        'compare_grid_equality': CompareGridEquality,
        'block_grid_builder': BlockGridBuilder,
        'match_pattern': MatchPattern,
        'input_grid_reference': InputGridReference,
        'filter_grid_by_color':FilterGridByColor,
        'get_external_background_mask': GetExternalBackgroundMask,
        'mask_and': MaskAnd,
        'mask_or': MaskOr,
        'mask_not': MaskNot,
        'binarize': Binarize,
        'locate_pattern': LocatePattern,
        'slice_grid': SliceGrid,
        'fill_region': FillRegion,
        'add_grid_to_canvas': AddGridToCanvas,
        "get_connected_component": GetConnectedComponent,
        "get_grid_height": GetGridHeight,
        "get_grid_width": GetGridWidth,
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