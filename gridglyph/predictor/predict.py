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
        # Format identique à l'entraînement
        in_grid = json.dumps(input_grid, ensure_ascii=False)
        out_grid = json.dumps(output_grid, ensure_ascii=False)
        prompt = f"<|im_start|>user\n{in_grid}\n{out_grid}<|im_end|>\n<|im_start|>assistant\n"
        
        # 1. Tokenisation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 2. Génération forcée
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=5, 
                do_sample=False,
                # On utilise le token de fin pour stopper la "récitation"
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            )
        
        # 3. RUPTURE : On ne décode que la partie générée (les nouveaux tokens)
        # output_ids contient [prompt_ids + generation_ids]
        generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
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