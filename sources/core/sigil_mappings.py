SIGIL_TO_COMMAND = {
    # Atomic transformations
    "Ⳁ": "identity",              # Identity
    "⇒": "map_numbers",           # MapNumbers
    "↔": "flip_h",                # FlipGridHorizontally
    "↕": "flip_v",                # FlipGridVertically
    "⇄": "swap_rows_or_columns",  # SwapRowsOrColumns
    "↢": "reverse_row",           # ReverseRow
    "⮁": "shift_row_or_column",   # ShiftRowOrColumn
    
    # Combinators
    "◨": "repeat_grid",           # RepeatGrid (horizontal)
    "⬒": "repeat_grid",          # RepeatGrid (vertical)
    "⧖": "alternate",            # Alternate
    "→": "apply_to_row",          # ApplyToRow
    "⟁": "conditional_transform", # ConditionalTransform
    "⌺": "mask_combinator",       # MaskCombinator
    "⟹": "sequence"               # Sequence
}