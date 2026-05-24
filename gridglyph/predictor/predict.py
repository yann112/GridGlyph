import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class Predictor:
    def __init__(self, model_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Chargement du modèle de base
        # On ne force pas le dtype ici pour laisser le modèle charger nativement
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            device_map=None
        )
        
        # Resize du vocabulaire AVANT le chargement PEFT
        base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Chargement de l'adaptateur
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        # RUPTURE : Forçage explicite sur le modèle complet APRES chargement
        # On s'assure que le cast est propagé sur TOUT le modèle
        self.model = self.model.to(self.device).float()
        self.model.eval()

    def predict(self, input_grid, output_grid):
        messages = [
            {"role": "user", "content": f"{json.dumps(input_grid)}\n{json.dumps(output_grid)}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Utilisation d'un contexte de précision explicite pour l'inférence
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False): # Désactivation de l'autocast
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip().replace(" ", "")