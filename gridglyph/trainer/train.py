import os
import torch
import yaml
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from .model import GridGlyphModel
from .dataset import GridGlyphDataset

def load_final_config(override_path="config.yaml"):
    """
    Industrial configuration loader:
    Ensures 'Frugal' defaults are maintained while allowing 
    experiment-specific overrides (like higher LoRA rank or new paths).
    """
    base_path = os.path.dirname(__file__)
    default_path = os.path.join(base_path, "default_config.yaml")
    
    if not os.path.exists(default_path):
        # Fallback to current directory if not found in package
        default_path = "default_config.yaml"

    with open(default_path, "r") as f:
        config = yaml.safe_load(f)
    
    if os.path.exists(override_path):
        print(f"🔧 Applying local overrides from {override_path}")
        with open(override_path, "r") as f:
            overrides = yaml.safe_load(f)
            
        if overrides:
            for key, value in overrides.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value
    return config

def start_fine_tuning(config_override="config.yaml"):
    cfg = load_final_config(config_override)
    
    # 1. Hardware Guard: Force single GPU to prevent Kaggle multi-T4 desync errors
    if torch.cuda.device_count() > 1:
        print("💡 Multiple GPUs detected. Pinning to Device 0 for stability.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 2. The Brain: Optimized for Qwen-0.5B + LoRA Rank 64
    gg_model = GridGlyphModel(
        model_id=cfg['model']['name'], 
        r=cfg['model']['lora_r'],
        use_4bit=cfg['model']['use_4bit']
    )
    
    # 3. The Bridge: Inductive Dataset (Grids -> Rule) 
    # Tokenizer injection happens here for the Alchemist's safe Kanji filtering
    train_dataset = GridGlyphDataset(
        tokenizer=gg_model.tokenizer,
        repo_id=cfg['repo']['seeds']
    )

    # 4. Training Engine: Configured for 4-bit / FP16 T4 Workload
    use_gpu = torch.cuda.is_available() and cfg['model']['use_4bit']
    
    training_args = TrainingArguments(
            output_dir="./gridglyph_outputs",
            per_device_train_batch_size=cfg['training']['batch_size'],
            gradient_accumulation_steps=cfg['training']['grad_accum'],
            learning_rate=cfg['training']['learning_rate'],
            max_steps=cfg['training']['max_steps'],
            fp16=cfg['training']['fp16'] and use_gpu,
            # Gradient checkpointing est vital pour LoRA r=64 sur T4
            gradient_checkpointing=True,
            use_cpu=not use_gpu,
            
            # --- Stratégie de Push & Sauvegarde ---
            push_to_hub=cfg['training']['push_to_hub'],
            hub_model_id=cfg['repo']['output'],
            hub_strategy=cfg['training'].get('hub_strategy', "every_save"), 
            save_steps=cfg['training'].get('save_steps', 500),
            save_total_limit=cfg['training'].get('save_total_limit', 2),
            
            logging_steps=10,
            report_to="none"
        )

    # Causal LM Collator (mlm=False)
    # The model learns to predict the DSL Rule at the end of the prompt
    trainer = Trainer(
        model=gg_model.model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(gg_model.tokenizer, mlm=False),
    )

    # Final cleanup before ignition
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"🛰 Engine Ready: {cfg['model']['name']}")
    print(f"🛰 Target: Inductive Spatial Reasoning (10x10 Support)")
    
    trainer.train()
    
    if cfg['training']['push_to_hub']:
        print("🚀 Pushing optimized adapter and tokenizer to Hub...")
        gg_model.tokenizer.push_to_hub(cfg['repo']['output'])
        trainer.push_to_hub()