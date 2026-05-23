import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class Predictor:
    def __init__(self, model_path):
        # 1. Charger le tokenizer local (qui contient tes 52 tokens additionnels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 2. Charger le modèle de base
        base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        
        # 3. Synchronisation indispensable : 
        # On ajuste la taille des embeddings du modèle de base pour qu'elle corresponde 
        # exactement à celle de ton entraînement (151712 au lieu de 151936)
        model.resize_token_embeddings(len(self.tokenizer))
        
        # 4. Chargement de l'adaptateur LoRA une fois la géométrie alignée
        self.model = PeftModel.from_pretrained(model, model_path)
        self.model.eval()

    def predict(self, input_grid, output_grid):
        # 1. Utiliser le chat_template officiel de Qwen
        messages = [
            {"role": "user", "content": f"{json.dumps(input_grid)}\n{json.dumps(output_grid)}"}
        ]
        
        # Le tokenizer transforme les rôles en VRAIS jetons spéciaux
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # 2. Génération
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=10,
                do_sample=False,
                # On interdit au modèle de recracher le prompt
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 3. Extraction : on ne garde que ce qui suit le prompt
        # La longueur des inputs nous permet de sauter exactement ce que le modèle a reçu
        input_length = inputs.input_ids.shape[1]
        generated_tokens = output_ids[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip().replace(" ", "")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./gridglyph_outputs", help="Path to the model directory")
    parser.add_argument("--input", required=True, help="Input grid as Kanji string")
    parser.add_argument("--output", required=True, help="Output grid as Kanji string")
    args = parser.parse_args()
    
    predictor = Predictor(args.model)
    dsl_command = predictor.predict(args.input, args.output)
    print(dsl_command)

if __name__ == "__main__":
    main()