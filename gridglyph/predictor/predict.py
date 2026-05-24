import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class Predictor:
    def __init__(self, model_path):
        # 0. Définition du device cible (le plus rapide)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 1. Chargement du tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 2. Chargement du modèle de base
        base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            torch_dtype=torch.float16,
            device_map={"": self.device} # Forçage strict sur le GPU 0
        )
        
        # 3. Synchronisation géométrique et typage
        model.resize_token_embeddings(len(self.tokenizer))
        
        # Correction sécurisée du lm_head avec alignement de type
        vocab_size = len(self.tokenizer)
        if model.lm_head.out_features != vocab_size:
            model.lm_head = torch.nn.Linear(model.config.hidden_size, vocab_size, bias=False).to(self.device).half()
        
        # 4. Chargement de l'adaptateur LoRA
        self.model = PeftModel.from_pretrained(model, model_path).to(self.device)
        self.model.eval()

    def predict(self, input_grid, output_grid):
        messages = [
            {"role": "user", "content": f"{json.dumps(input_grid)}\n{json.dumps(output_grid)}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenisation avec cast explicite en .half() pour correspondre au modèle
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inférence avec typage forcé
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extraction
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip().replace(" ", "")