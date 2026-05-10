import os
import yaml
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from .model import GridGlyphModel
from .dataset import GridGlyphDataset

def load_final_config(override_path="config.yaml"):
    # 1. Get path to the default config inside the package
    base_path = os.path.dirname(__file__)
    default_path = os.path.join(base_path, "default_config.yaml")
    
    with open(default_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 2. Override with local config if it exists
    if os.path.exists(override_path):
        print(f"🔧 Applying local overrides from {override_path}")
        with open(override_path, "r") as f:
            overrides = yaml.safe_load(f)
            # Deep merge (simplified for this structure)
            for key in overrides:
                if isinstance(overrides[key], dict) and key in config:
                    config[key].update(overrides[key])
                else:
                    config[key] = overrides[key]
    return config

def start_fine_tuning(config_override="config.yaml"):
    cfg = load_final_config(config_override)
    
    # 1. The Brain (using config)
    # Note: You'll need to update GridGlyphModel to accept use_4bit
    gg_model = GridGlyphModel(
        model_id=cfg['model']['name'], 
        r=cfg['model']['lora_r'],
        use_4bit=cfg['model']['use_4bit']
    )
    
    # 2. The Bridge
    train_dataset = GridGlyphDataset(
        tokenizer=gg_model.tokenizer,
        repo_id=cfg['repo']['seeds']
    )

    # 3. Training Logic
    # We dynamically set device/fp16 based on use_4bit (our proxy for 'Is there a GPU?')
    use_gpu = cfg['model']['use_4bit'] 
    
    training_args = TrainingArguments(
        output_dir="./gridglyph_outputs",
        per_device_train_batch_size=cfg['training']['batch_size'],
        gradient_accumulation_steps=cfg['training']['grad_accum'],
        learning_rate=cfg['training']['learning_rate'],
        max_steps=cfg['training']['max_steps'],
        fp16=cfg['training']['fp16'] and use_gpu,
        use_cpu=not use_gpu,
        push_to_hub=cfg['training']['push_to_hub'],
        hub_model_id=cfg['repo']['output'],
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=gg_model.model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(gg_model.tokenizer, mlm=False),
    )

    print(f"🛰 Training on {'GPU' if use_gpu else 'CPU'}...")
    trainer.train()
    
    if cfg['training']['push_to_hub']:
        trainer.push_to_hub()