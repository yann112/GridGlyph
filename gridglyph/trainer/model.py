import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


from gridglyph.generators.registry import MUTATION_FUNCTIONS_MAP
from gridglyph.generators.primitives import INT_TO_ROMAN_MAP

class GridGlyphModel:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", r=64, use_4bit=True):
        self.model_id = model_id
        
        # 1. Hardware Adaptation Logic (Frugal GPU Path)
        if use_4bit and torch.cuda.is_available():
            print(f"💎 Loading in 4-bit mode (Frugal GPU Path)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, 
                bnb_4bit_use_double_quant=True,
            )
            device_map = {"": 0} 
        else:
            print(f"🐢 Loading in standard mode (CPU/Non-Quantized Path)")
            bnb_config = None
            device_map = "auto"

        # 2. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # --- CONFIGURATION DES TOKENS ATOMIQUES DU DSL ---
        mandatory_sigils = list(MUTATION_FUNCTIONS_MAP.keys())
        
        # Séparateurs structurels absolus à isoler pour empêcher toute fusion parasite (ex: ,I)
        structural_separators = [",", "[", "]", "(", ")"]
        
        # Extraction de tes patterns de chiffres romains (Chaines littérales standards)
        roman_tokens = list(INT_TO_ROMAN_MAP.values())
        
        all_roman_strings = set(roman_tokens)
        roman_tokens = list(all_roman_strings) + [f"-{r}" for r in all_roman_strings if r != "∅"]
        
        # Collecte globale incluant désormais explicitement les séparateurs syntaxiques
        symbols_to_verify = mandatory_sigils + structural_separators + roman_tokens
        
        tokens_to_add = []
        for symbol in symbols_to_verify:
            encoded = self.tokenizer.encode(symbol, add_special_tokens=False)
            
            # On force l'injection si :
            # - Le symbole n'est pas de longueur 1 ou est inconnu (unk)
            # - OU s'il fait partie des séparateurs pour briser les fusions BPE natives de Qwen
            if len(encoded) != 1 or encoded[0] == self.tokenizer.unk_token_id or symbol in structural_separators:
                tokens_to_add.append(symbol)
                
        if tokens_to_add:
            print(f"⚙️ Extension du vocabulaire : {len(tokens_to_add)} tokens injectés de force.")
            # En les ajoutant ici, HuggingFace interdit au tokenizer de les fusionner avec d'autres caractères
            self.tokenizer.add_tokens(tokens_to_add)
        
        # 3. Load Base Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if use_4bit else torch.float32 
        )

        # Redimensionnement impératif de la matrice d'embeddings du modèle de base
        if tokens_to_add:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Préparation du modèle quantifié pour l'entraînement k-bit (fige les couches de base)
        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # 4. LoRA Configuration (Ciblage exhaustif pour capturer les abstractions spatiales)
        lora_config = LoraConfig(
            r=r, 
            lora_alpha=r * 2, 
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Encapsulation LoRA (PEFT applique ici un gel massif de tous les paramètres non-LoRA)
        self.model = get_peft_model(self.model, lora_config)
        
        # Réactivation explicite des gradients sur les couches d'embeddings APRÈS l'encapsulation LoRA
        if tokens_to_add:
            # Couche d'entrée (input tokens -> embeddings)
            self.model.get_input_embeddings().weight.requires_grad = True
            
            # Couche de sortie (embeddings -> distribution de probabilités des tokens)
            if hasattr(self.model, "get_output_embeddings") and self.model.get_output_embeddings() is not None:
                self.model.get_output_embeddings().weight.requires_grad = True

        self.model.print_trainable_parameters()

    def save_adapter(self, output_path):
        """Sauvegarde l'adaptateur LoRA léger et le tokenizer synchronisé."""
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)