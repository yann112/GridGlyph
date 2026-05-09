import json
import random
from generators.registry import get_mutator

def build(num_samples: int, output_file: str):
    dataset = []
    sigils = ['↻', '↔', '↕', '⤨'] # Start simple
    
    for i in range(num_samples):
        sigil = random.choice(sigils)
        mutator = get_mutator(sigil)
        
        # 1. Generate Input
        input_grid = mutator(sigil)
        
        # 2. Logic to call your Core Executor would go here
        # For now, we placeholder the output
        sample = {
            "input": input_grid.tolist(),
            "output": "EXECUTOR_RESULT_HERE", 
            "sigil": sigil
        }
        dataset.append(sample)
        
    with open(output_file, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')
    print(f"Successfully generated {num_samples} samples to {output_file}")

if __name__ == "__main__":
    # Add argparse logic here
    build(10, "test_dataset.jsonl")