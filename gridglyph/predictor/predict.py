import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class Predictor:
    def __init__(self, model_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 1. Chargement du modèle de base directement en float16
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", 
            torch_dtype=torch.float16
        ).to(self.device)
        
        base_model.resize_token_embeddings(len(self.tokenizer))
        
        # 2. Chargement de l'adaptateur
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        # 3. CONVERSION TOTALE : on force chaque paramètre et buffer en Half immédiatement
        self.model = self.model.to(self.device).half()
        self.model.eval()

    def predict(self, input_grid, output_grid):
        messages = [
            {"role": "user", "content": f"{json.dumps(input_grid)}\n{json.dumps(output_grid)}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 4. TOKENISATION : On laisse en Float32, mais on caste le masque et les embeddings
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        # On caste le masque en Half explicitement pour correspondre aux attentes du modèle
        attention_mask = inputs["attention_mask"].to(self.device).half()
        
        with torch.no_grad():
            # Inférence en contexte Half
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        generated_ids = output_ids[0][input_ids.shape[1]:].cpu()
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip().replace(" ", "")