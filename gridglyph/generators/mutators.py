import numpy as np
import random
import re
from gridglyph.generators.primitives import int_to_roman, generate_single_random_grid, generate_random_shape_grid

def mutate_flip_rule(item, num_variants=3, max_dim=30, num_range=9, **kwargs):
    mutated_items = []
    original_grid = np.array(item['input_grid'])
    for _ in range(num_variants - 1):
        new_grid = generate_single_random_grid(original_grid.shape, max_dim, num_range)
        mutated_items.append({'input_grid': new_grid.tolist(), 'dsl_rule': item['dsl_rule']})
    return mutated_items

def mutate_swap_rule(item, num_variants=3, max_dim=30, num_range=9, is_row_swap=True):
    mutated_items = []
    pattern = r'⇅\((?P<idx1>[IVX]+),(?P<idx2>[IVX]+)\)' if is_row_swap else r'⇄\((?P<idx1>[IVX]+),(?P<idx2>[IVX]+)\)'
    sigil = '⇅' if is_row_swap else '⇄'
    match = re.match(pattern, item['dsl_rule'])
    if not match: return []
    
    for _ in range(num_variants - 1):
        candidate_grid = generate_single_random_grid(np.array(item['input_grid']).shape, max_dim, num_range)
        dim_size = candidate_grid.shape[0] if is_row_swap else candidate_grid.shape[1]
        if dim_size >= 2:
            idx1, idx2 = random.sample(range(1, dim_size + 1), 2)
            new_rule = f"{sigil}({int_to_roman(idx1)},{int_to_roman(idx2)})"
            mutated_items.append({'input_grid': candidate_grid.tolist(), 'dsl_rule': new_rule})
    return mutated_items

def mutate_swap_row_rule(item, **kwargs): return mutate_swap_rule(item, is_row_swap=True, **kwargs)
def mutate_swap_col_rule(item, **kwargs): return mutate_swap_rule(item, is_row_swap=False, **kwargs)

def mutate_swap_value_rule(item, num_variants=3, max_dim=30, num_range=9, **kwargs):
    mutated_items = []
    for _ in range(num_variants - 1):
        candidate_grid = generate_single_random_grid(np.array(item['input_grid']).shape, max_dim, num_range)
        v1, v2 = random.randint(0, num_range), random.randint(0, num_range)
        if v1 not in candidate_grid and candidate_grid.size > 0:
            candidate_grid.flat[random.randint(0, candidate_grid.size-1)] = v1
        mutated_items.append({
            'input_grid': candidate_grid.tolist(), 
            'dsl_rule': f"⇒({int_to_roman(v1)}, {int_to_roman(v2)})"
        })
    return mutated_items

def mutate_extract_value_rule(item, num_variants=3, max_dim=30, num_range=9, **kwargs):
    mutated_items = []
    for _ in range(num_variants - 1):
        grid = generate_single_random_grid(np.array(item['input_grid']).shape, max_dim, num_range)
        r, c = random.randint(1, grid.shape[0]), random.randint(1, grid.shape[1])
        mutated_items.append({'input_grid': grid.tolist(), 'dsl_rule': f"⊡({int_to_roman(r)},{int_to_roman(c)})"})
    return mutated_items

def mutate_identity_rule(item, num_variants=3, **kwargs):
    return [{'input_grid': generate_single_random_grid(np.array(item['input_grid']).shape).tolist(), 'dsl_rule': '⌂'} for _ in range(num_variants-1)]

def mutate_extract_background_rule(item, num_variants=2, max_dim=30, num_range=9, **kwargs):
    mutated_items = []
    for _ in range(num_variants - 1):
        v = random.randint(0, num_range)
        grid = generate_random_shape_grid(min_dim=3, max_dim=max_dim, background_value=v)
        mutated_items.append({'input_grid': grid.tolist(), 'dsl_rule': f"⏚({int_to_roman(v)})"})
    return mutated_items

def mutate_extract_value_occurrences_rule(item, num_variants=5, max_dim=20, num_range=9, **kwargs):
    mutated_items = []
    for _ in range(num_variants - 1):
        grid = generate_single_random_grid(np.array(item['input_grid']).shape, max_dim, num_range)
        v = random.randint(0, num_range)
        if v not in grid: grid.flat[0] = v
        mutated_items.append({'input_grid': grid.tolist(), 'dsl_rule': f"◎({int_to_roman(v)})"})
    return mutated_items

def mutate_get_connected_component_rule(item, num_variants=5, max_dim=20, num_range=9, **kwargs):
    mutated_items = []
    for _ in range(num_variants - 1):
        grid = generate_random_shape_grid(min_dim=5, max_dim=max_dim, background_value=0)
        nonzero = np.argwhere(grid != 0)
        r, c = nonzero[random.randint(0, len(nonzero)-1)] if len(nonzero) > 0 else (0,0)
        mutated_items.append({'input_grid': grid.tolist(), 'dsl_rule': f"⚇({int_to_roman(r+1)},{int_to_roman(c+1)})"})
    return mutated_items

def mutate_crop_rule(item, num_variants=5, max_dim=20, num_range=9, **kwargs):
    mutated_items = []
    for _ in range(num_variants - 1):
        grid = generate_single_random_grid(np.array(item['input_grid']).shape, max_dim, num_range)
        rows, cols = grid.shape
        if rows < 2 or cols < 2: continue
        r1, c1 = random.randint(1, rows-1), random.randint(1, cols-1)
        r2, c2 = random.randint(r1, rows), random.randint(c1, cols)
        mutated_items.append({'input_grid': grid.tolist(), 'dsl_rule': f"✂({int_to_roman(r1)},{int_to_roman(c1)},{int_to_roman(r2)},{int_to_roman(c2)})"})
    return mutated_items