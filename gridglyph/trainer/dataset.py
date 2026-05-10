import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from gridglyph.trainer.generator import GridAlchemist

class GridGlyphDataset(IterableDataset):
    def __init__(self, tokenizer, repo_id="yannumber1/gridglyph-atomic-seeds", max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load seeds
        dataset = load_dataset(repo_id, split="train", streaming=True)
        self.seeds = [sample for sample in dataset]
        
        # INJECTION: Pass the tokenizer into the alchemist
        self.alchemist = GridAlchemist(self.seeds, self.tokenizer)

    def __iter__(self):
            while True:
                sample = self.alchemist.get_sample()
                prompt = self._format_prompt(sample)
                
                tokenized = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # CRITICAL for Inductive Training:
                # We must mask the User/System prompt in the labels so the model 
                # only earns 'points' for predicting the Rule, not the input.
                input_ids = tokenized["input_ids"].squeeze()
                labels = input_ids.clone()
                
                # Optional: Mask everything before the assistant response
                # Search for the assistant token ID in your tokenizer
                assistant_start_token = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
                # (Logic to find index and set labels[:index] = -100 would go here)

                yield {
                    "input_ids": input_ids,
                    "attention_mask": tokenized["attention_mask"].squeeze(),
                    "labels": labels
                }

    def _format_prompt(self, sample):
            """
            Input: input_grid + output_grid
            Task: Identify the dsl_rule
            """
            import json
            # ensure_ascii=False is critical to keep the Kanji as single characters in the string
            in_grid = json.dumps(sample["input_grid"], ensure_ascii=False)
            out_grid = json.dumps(sample["output_grid"], ensure_ascii=False)
            rule = sample["dsl_rule"]

            return (
                f"<|im_start|>system\nYou are a symbolic logic architect. Given a transformation, identify the underlying DSL rule.<|im_end|>\n"
                f"<|im_start|>user\nInput Grid:\n{in_grid}\n\nOutput Grid:\n{out_grid}\n\nWhat is the rule?<|im_end|>\n"
                f"<|im_start|>assistant\n{rule}<|im_end|>"
            )