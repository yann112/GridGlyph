import pytest
import json
import numpy as np
from gridglyph.generators.build import build_dataset
from gridglyph.core.dsl_symbolic_executor import DSLExecutor
from gridglyph.core.dsl_symbolic_interpreter import SymbolicRuleParser

# Match the structure used in the notebook/build.py
MOCK_TEST_CASES = [
    ("↔", np.eye(3, dtype=np.uint8), None),
    ("⇅(I,II)", np.array([[1, 1], [0, 0]], dtype=np.uint8), None),
]

def test_build_dataset_output_validity(tmp_path):
    output_file = tmp_path / "test_data.jsonl"
    parser = SymbolicRuleParser()
    
    # Run the build using keyword arguments to match the new signature:
    # build_dataset(output_path, test_cases=None, multiplier=1)
    build_dataset(
        output_path=str(output_file), 
        test_cases=MOCK_TEST_CASES,
        multiplier=1
    )
    
    assert output_file.exists()
    
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        assert len(lines) > 0
        
        for line in lines:
            sample = json.loads(line)
            
            # 1. Structural Check
            assert "dsl_rule" in sample
            assert "input_grid" in sample
            assert "output_grid" in sample
            
            # 2. Logic Verification
            cmd = parser.parse_rule(sample["dsl_rule"])
            # Ensure we compare with consistent dtypes (int32 is safe for ARC)
            executor = DSLExecutor(cmd, np.array(sample["input_grid"], dtype=np.int32))
            expected_output = executor.execute_program()
            
            np.testing.assert_array_equal(
                np.array(sample["output_grid"], dtype=np.int32), 
                expected_output,
                err_msg=f"Logic mismatch for rule {sample['dsl_rule']}"
            )