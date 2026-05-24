import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class Predictor:
    def __init__(self, model_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Chargement en float16 natif
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", 
            torch_dtype=torch.float16
        ).to(self.device)
        
        base_model.resize_token_embeddings(len(self.tokenizer))
        
        self.model = PeftModel.from_pretrained(base_model, model_path).to(self.device).half()
        self.model.eval()

    def predict(self, input_grid, output_grid):
        messages = [
            {"role": "user", "content": f"{json.dumps(input_grid)}\n{json.dumps(output_grid)}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # RUPTURE : On envoie tout sur le GPU, PUIS on caste tout en half.
        # Les input_ids restent des LongTensor, c'est toléré par Qwen.
        # Le masque d'attention doit être forcé en Half.
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device).half()
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # On récupère les IDs en CPU pour le décodage
        generated_ids = output_ids[0][input_ids.shape[1]:].cpu()
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip().replace(" ", "")