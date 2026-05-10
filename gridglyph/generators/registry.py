from gridglyph.generators.mutators import (
    mutate_flip_rule, mutate_swap_row_rule, mutate_swap_col_rule,
    mutate_swap_value_rule, mutate_extract_value_rule, mutate_identity_rule,
    mutate_extract_background_rule, mutate_extract_value_occurrences_rule,
    mutate_get_connected_component_rule, mutate_crop_rule
)

MUTATION_FUNCTIONS_MAP = {
    '↔': (mutate_flip_rule, 2),
    '↕': (mutate_flip_rule, 2),
    '⇅': (mutate_swap_row_rule, 2),
    '⇄': (mutate_swap_col_rule, 2),
    '⇒': (mutate_swap_value_rule, 2),
    '⊡': (mutate_extract_value_rule, 3),
    '⌂': (mutate_identity_rule, 3),
    '⏚': (mutate_extract_background_rule, 2),
    '◎': (mutate_extract_value_occurrences_rule, 2),
    '⚇': (mutate_get_connected_component_rule, 3),
    '✂': (mutate_crop_rule, 2),
    '⤨': (lambda item, num_variants, **kwargs: [], 2)
}