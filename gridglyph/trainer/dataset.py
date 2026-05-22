import json
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from gridglyph.trainer.generator import GridAlchemist

class GridGlyphDataset(IterableDataset):
    def __init__(self, tokenizer, repo_id="yannumber1/gridglyph-atomic-seeds", max_length=1024):
        self.tokenizer = tokenizer
        self.repo_id = repo_id
        self.max_length = max_length
        
        # Initialisation du flux
        self._reset_iterator()
        
        self.assistant_start_tokens = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def _reset_iterator(self):
        """Réinitialise le flux de données."""
        ds = load_dataset(self.repo_id, split="train", streaming=True)
        self.dataset_iterator = iter(ds)
        # On passe le nouvel itérateur à l'alchimiste
        self.alchemist = GridAlchemist(self.dataset_iterator, self.tokenizer)

    def __iter__(self):
        while True:
            try:
                sample = self.alchemist.get_sample()
            except StopIteration:
                # Réinitialisation transparente du flux quand il est épuisé
                self._reset_iterator()
                continue
            except Exception:
                # Sécurité pour les autres erreurs potentielles
                continue
                
            prompt = self._format_prompt(sample)
            
            tokenized = self.tokenizer(
                prompt,
                truncation=False,
                return_tensors="pt"
            )
            
            input_ids = tokenized["input_ids"].squeeze()
            attention_mask = tokenized["attention_mask"].squeeze()
            
            if len(input_ids) > self.max_length:
                continue 
            
            labels = input_ids.clone()
            assistant_idx = self._find_subsequence(input_ids, self.assistant_start_tokens)
            
            if assistant_idx != -1:
                rule_start_idx = assistant_idx + len(self.assistant_start_tokens)
                labels[:rule_start_idx] = -100
            else:
                continue

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

    def _format_prompt(self, sample):
        in_grid = json.dumps(sample["input_grid"], ensure_ascii=False)
        out_grid = json.dumps(sample["output_grid"], ensure_ascii=False)
        rule = str(sample["dsl_rule"]).replace(" ", "")

        return (
            f"<|im_start|>user\n{in_grid}\n{out_grid}<|im_end|>\n"
            f"<|im_start|>assistant\n{rule}<|im_end|>"
        )

    def _find_subsequence(self, tensor, sequence):
        seq_len = len(sequence)
        for i in range(len(tensor) - seq_len + 1):
            if tensor[i:i + seq_len].tolist() == sequence:
                return i
        return -1