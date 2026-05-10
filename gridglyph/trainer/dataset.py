import torch
from torch.utils.data import IterableDataset
from  datasets import load_dataset
from gridglyph.trainer.generator import GridAlchemist

class GridGlyphDataset(IterableDataset):
    def __init__(self, tokenizer, repo_id="yannumber1/gridglyph-atomic-seeds", max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load seeds from Hugging Face Hub (streaming=True for efficiency)
        dataset = load_dataset(repo_id, split="train", streaming=True)
        self.seeds = [sample for sample in dataset]
        
        # Initialize the Alchemist with the loaded seeds
        self.alchemist = GridAlchemist(self.seeds)

    def __iter__(self):
        while True:
            sample = self.alchemist.get_sample()
            
            # Format into a structured prompt
            prompt = self._format_prompt(sample)
            
            # Tokenization
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # For CausalLM, labels are the same as input_ids
            # We want to learn the whole sequence (Rule + Input -> Output)
            yield {
                "input_ids": tokenized["input_ids"].squeeze(),
                "attention_mask": tokenized["attention_mask"].squeeze(),
                "labels": tokenized["input_ids"].squeeze()
            }

    def _format_prompt(self, sample):
        """
        Formats the sample into a clear instruction for Qwen.
        Uses a consistent template to help the model distinguish between 
        the rule, the input grid, and the expected result.
        """
        rule = sample["dsl_rule"]
        in_grid = sample["input_grid"]
        out_grid = sample["output_grid"]

        return (
            f"<|im_start|>system\nYou are a symbolic logic solver for the ARC prize.<|im_end|>\n"
            f"<|im_start|>user\nApply the rule '{rule}' to the following grid:\n{in_grid}<|im_end|>\n"
            f"<|im_start|>assistant\n{out_grid}<|im_end|>"
        )