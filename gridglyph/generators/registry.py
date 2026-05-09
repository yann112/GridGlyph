from .mutators import mutate_geometric_rule, mutate_swap_value_rule

MUTATOR_REGISTRY = {
    '↻': mutate_geometric_rule,
    '↔': mutate_geometric_rule,
    '↕': mutate_geometric_rule,
    '⤨': mutate_swap_value_rule,
}

def get_mutator(sigil: str):
    return MUTATOR_REGISTRY.get(sigil, mutate_geometric_rule)