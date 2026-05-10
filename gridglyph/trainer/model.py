import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class GridGlyphModel:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", r=64, use_4bit=True):
        self.model_id = model_id
        
        # 1. Hardware Adaptation Logic
        if use_4bit and torch.cuda.is_available():
            print(f"💎 Loading in 4-bit mode (Frugal GPU Path)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                # Use float16 compute for T4 compatibility
                bnb_4bit_compute_dtype=torch.float16, 
                bnb_4bit_use_double_quant=True,
            )
            # Use specific device map to support the 'os.environ' pinning from the loop
            device_map = {"": 0} 
        else:
            print(f"🐢 Loading in standard mode (CPU/Non-Quantized Path)")
            bnb_config = None
            device_map = "auto"

        # 2. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Ensure right-side padding for Causal LM stability
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 3. Load Base Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            # Essential for training stability
            torch_dtype=torch.float16 if use_4bit else torch.float32 
        )

        # 4. LoRA Configuration
        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # INDUSTRIAL ADJUSTMENT: alpha = 2 * r
        lora_config = LoraConfig(
            r=r, 
            lora_alpha=r * 2, 
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def save_adapter(self, output_path):
        """Saves only the lightweight adapters (~100MB for r=64)."""
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)