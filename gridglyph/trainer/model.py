import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from gridglyph.generators.registry import MUTATION_FUNCTIONS_MAP
from gridglyph.generators.primitives import INT_TO_ROMAN_MAP

class GridGlyphModel:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", r=64, use_4bit=True):
        self.model_id = model_id
        
        if use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, 
                bnb_4bit_use_double_quant=True,
            )
            device_map = {"": 0} 
        else:
            bnb_config = None
            device_map = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        mandatory_sigils = list(MUTATION_FUNCTIONS_MAP.keys())
        structural_separators = [",", "[", "]", "(", ")"]
        roman_tokens = list(set(INT_TO_ROMAN_MAP.values()))
        roman_tokens = roman_tokens + [f"-{r}" for r in roman_tokens if r != "∅"]
        
        symbols_to_verify = mandatory_sigils + structural_separators + roman_tokens
        tokens_to_add = []
        for symbol in symbols_to_verify:
            encoded = self.tokenizer.encode(symbol, add_special_tokens=False)
            if len(encoded) != 1 or encoded[0] == self.tokenizer.unk_token_id or symbol in structural_separators:
                tokens_to_add.append(symbol)
                
        if tokens_to_add:
            self.tokenizer.add_tokens(tokens_to_add)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if use_4bit else torch.float32 
        )

        if tokens_to_add:
            self.model.resize_token_embeddings(len(self.tokenizer))

        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=r, 
            lora_alpha=r * 2, 
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        
        if tokens_to_add:
            self.model.get_input_embeddings().weight.requires_grad = True
            if hasattr(self.model, "get_output_embeddings") and self.model.get_output_embeddings() is not None:
                self.model.get_output_embeddings().weight.requires_grad = True

        self.model.print_trainable_parameters()

    def save_adapter(self, output_path):
            self.tokenizer.save_pretrained(output_path)
            self.model.save_pretrained(output_path)
            
            config_path = os.path.join(output_path, "tokenizer_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r+") as f:
                    data = json.load(f)
                    data["tokenizer_class"] = "Qwen2Tokenizer"
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()