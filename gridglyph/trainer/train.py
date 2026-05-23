import os
import torch
import yaml
import argparse
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from .model import GridGlyphModel
from .dataset import GridGlyphDataset

def load_final_config(override_path="config.yaml"):
    base_path = os.path.dirname(__file__)
    default_path = os.path.join(base_path, "default_config.yaml")
    if not os.path.exists(default_path):
        default_path = "default_config.yaml"

    with open(default_path, "r") as f:
        config = yaml.safe_load(f)
    
    if os.path.exists(override_path):
        with open(override_path, "r") as f:
            overrides = yaml.safe_load(f)
        if overrides:
            for key, value in overrides.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value
    return config

def start_fine_tuning():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args, _ = parser.parse_known_args()
    
    cfg = load_final_config(args.config)
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gg_model = GridGlyphModel(
        model_id=cfg['model']['name'], 
        r=cfg['model']['lora_r'],
        use_4bit=cfg['model']['use_4bit']
    )
    
    train_dataset = GridGlyphDataset(
        tokenizer=gg_model.tokenizer,
        repo_id=cfg['repo']['seeds']
    )

    use_gpu = torch.cuda.is_available()
    
    training_args = TrainingArguments(
        output_dir=cfg['training'].get('output_dir', "./gridglyph_outputs"),
        per_device_train_batch_size=cfg['training']['batch_size'],
        gradient_accumulation_steps=cfg['training']['grad_accum'],
        learning_rate=cfg['training']['learning_rate'],
        max_steps=cfg['training']['max_steps'],
        fp16=cfg['training']['fp16'] and use_gpu,
        gradient_checkpointing=True,
        use_cpu=not use_gpu,
        push_to_hub=cfg['training']['push_to_hub'],
        hub_model_id=cfg['repo']['output'],
        hub_strategy="every_save", 
        save_steps=cfg['training'].get('save_steps', 500),
        save_total_limit=2,
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=gg_model.model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(gg_model.tokenizer, mlm=False),
    )

    if use_gpu:
        torch.cuda.empty_cache()

    trainer.train()
    
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    gg_model.model.config.vocab_size = len(gg_model.tokenizer)
    
    gg_model.tokenizer.save_pretrained(output_dir)
    gg_model.model.config.save_pretrained(output_dir)
    gg_model.model.save_pretrained(output_dir)

    if cfg['training']['push_to_hub']:
        trainer.push_to_hub(commit_message="Train: Add DSL tokenizer and LoRA adapter")
        gg_model.tokenizer.push_to_hub(cfg['repo']['output'])

if __name__ == "__main__":
    start_fine_tuning()