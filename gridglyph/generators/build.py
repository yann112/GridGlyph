import json
import re
import numpy as np
from tqdm import tqdm
from itertools import groupby

from gridglyph.core.dsl_symbolic_executor import DSLExecutor
from gridglyph.core.dsl_symbolic_interpreter import SymbolicRuleParser, SYMBOL_RULES
from gridglyph.core.ground_truth import TEST_CASES  # Use internal ground truth
from gridglyph.generators.registry import MUTATION_FUNCTIONS_MAP

# --- Atomic Logic Patterns (Cached) ---
ATOMIC_PATTERNS = []
CONDITIONALLY_ATOMIC_PATTERNS_CHECKERS = {}
NON_ATOMIC_PATTERNS = []

for rule_key, rule_definition in SYMBOL_RULES.items():
    sigil = rule_definition.get("sigil")
    pattern = rule_definition.get("pattern")
    nested_commands = rule_definition.get("nested_commands")

    if sigil and not pattern and not nested_commands:
        ATOMIC_PATTERNS.append(re.compile(rf"^{re.escape(sigil)}$"))
        continue
    if not pattern: continue
    try: compiled_pattern = re.compile(pattern)
    except re.error: continue

    if nested_commands is None or nested_commands == {}:
        ATOMIC_PATTERNS.append(compiled_pattern)
    elif rule_key in ["flip_h", "flip_v", "flatten_grid", "extract_bounding_box", "reverse_row"]:
        def make_checker(rule_key_inner):
            return lambda match: (match.group("arg_content") if "arg_content" in match.groupdict() else None) in [None, "⌂"]
        CONDITIONALLY_ATOMIC_PATTERNS_CHECKERS[compiled_pattern] = make_checker(rule_key)
    else:
        NON_ATOMIC_PATTERNS.append(compiled_pattern)

def is_atomic_rule(rule_str: str) -> bool:
    rule_str = rule_str.strip()
    if rule_str.startswith('◫('): return False
    for p in NON_ATOMIC_PATTERNS:
        if p.match(rule_str): return False
    for p in ATOMIC_PATTERNS:
        if p.match(rule_str): return True
    for p, checker in CONDITIONALLY_ATOMIC_PATTERNS_CHECKERS.items():
        m = p.match(rule_str)
        if m and checker(m): return True
    return False

def build_dataset(output_path: str, test_cases=None, multiplier: int = 1):
    """
    Builds the dataset by applying mutations to ground truth cases.
    
    Args:
        output_path: Path to save the .jsonl file.
        test_cases: List of (rule, input_grid, output_grid). Defaults to internal TEST_CASES.
        multiplier: Scaling factor for mutation volume (num_variants * multiplier).
    """
    if test_cases is None:
        test_cases = TEST_CASES

    parser = SymbolicRuleParser()
    atomic_data = []

    # 1. Filter Ground Truth
    for rule_str, input_grid_np, _ in test_cases:
        if is_atomic_rule(rule_str) and input_grid_np is not None:
            atomic_data.append({
                "input_grid": input_grid_np.tolist(), 
                "dsl_rule": rule_str
            })
    
    # Sort for grouping (usually by sigil/first char)
    atomic_data.sort(key=lambda x: x['dsl_rule'][0])
    grouped = {k: list(g) for k, g in groupby(atomic_data, key=lambda x: x['dsl_rule'][0])}
    
    total_written = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        # Progress bar by sigil
        pbar = tqdm(grouped.items(), desc="Generating Atomic Dataset")
        for sigil, items in pbar:
            mutate_func, base_num_variants = MUTATION_FUNCTIONS_MAP.get(sigil, (None, None))
            if not mutate_func:
                continue

            # Apply the multiplier to the base volume
            effective_variants = base_num_variants * multiplier

            for pair in items:
                # Batch = Ground Truth + Mutated Variants
                batch = [pair] + mutate_func(pair, num_variants=effective_variants)
                
                for item in batch:
                    try:
                        # Re-execute every sample to ensure validity
                        executor = DSLExecutor(
                            parser.parse_rule(item['dsl_rule']), 
                            np.array(item['input_grid'])
                        )
                        item["output_grid"] = executor.execute_program().tolist()
                        
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        total_written += 1
                    except Exception:
                        continue
            
            pbar.set_postfix({"samples": total_written})

    print(f"\n✅ Generation complete: {total_written} samples saved to {output_path}")
    