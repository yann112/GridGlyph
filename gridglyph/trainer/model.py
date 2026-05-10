import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class GridGlyphModel:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", r=16, use_4bit=True):
        """
        The 'Brain' of the operation. 
        Adaptive to hardware: uses 4-bit quantization if a GPU is available,
        otherwise falls back to standard loading for CPU testing.
        """
        self.model_id = model_id
        
        # 1. Hardware Adaptation Logic
        if use_4bit:
            print(f"💎 Loading in 4-bit mode (Frugal GPU Path)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            device_map = "auto"
        else:
            print(f"🐢 Loading in standard mode (CPU Test Path)")
            bnb_config = None
            device_map = "cpu"

        # 2. Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )

        # 3. LoRA Configuration
        # If we are in 4-bit, we need to prepare the model for k-bit training
        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=r, 
            lora_alpha=32,
            # Targeted layers for Qwen attention and MLP
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        
        # Print parameter count to confirm LoRA efficiency
        self.model.print_trainable_parameters()

    def save_adapter(self, output_path):
        """
        Saves only the lightweight LoRA adapters.
        Essential for your 'simple tool' philosophy—distributing 
        MBs of logic instead of GBs of weights.
        """
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)