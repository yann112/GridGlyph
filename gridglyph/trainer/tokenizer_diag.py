import json
import argparse
import sys
import torch
from model import GridGlyphModel
from gridglyph.trainer.dataset import GridGlyphDataset 

def run_tokenizer_diagnostic(jsonl_path, max_samples=20):
    print("🚀 Initialisation du modèle...")
    gg_model = GridGlyphModel(use_4bit=False)
    tokenizer = gg_model.tokenizer

    print(f"📦 Chargement des graines locales depuis : {jsonl_path}")
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            local_seeds = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"❌ Impossible de lire le fichier local : {e}")
        sys.exit(1)

    print("🔧 Injection dans le GridGlyphDataset de production...")
    try:
        dataset = GridGlyphDataset(tokenizer=tokenizer)
        dataset.seeds = local_seeds
        dataset.alchemist.seeds = local_seeds  
        dataset_iter = iter(dataset)
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation du dataset : {e}")
        sys.exit(1)

    print(f"🔍 Analyse dynamique de max {min(max_samples, len(local_seeds))} échantillons...")
    
    for idx in range(1, min(max_samples, len(local_seeds)) + 1):
        batch = next(dataset_iter)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        masked_tokens_ids = input_ids[labels == -100].tolist()
        loss_tokens_ids = input_ids[labels != -100].tolist()
        
        context_text = tokenizer.decode(masked_tokens_ids)
        loss_text = tokenizer.decode(loss_tokens_ids)
        
        syntax_split = [tokenizer.decode([t]) for t in loss_tokens_ids]
        masked_syntax_split = [tokenizer.decode([t]) for t in masked_tokens_ids]

        is_loss_clean = loss_text.endswith("<|im_end|>") or "<|im_end|>" in loss_text
        has_byte_fallback = any(len(t) > 1 and "\\" in repr(t) for t in (syntax_split + masked_syntax_split))

        print(f"\n================ ÉCHANTILLON LOCAL N°{idx} ================")
        print("📖 CONTEXTE ENTRÉE (Masqué pour la Loss, labels == -100) :")
        print(context_text)
        print("-" * 40)
        print("🎯 CIBLE APPRISE (Loss active, calcul des gradients) :")
        print(loss_text)
        print("-" * 40)
        print(f"📊 Métriques :")
        print(f"  • Longueur totale : {len(input_ids)} tokens")
        print(f"  • Tokens masqués  : {len(masked_tokens_ids)}")
        print(f"  • Tokens calculés : {len(loss_tokens_ids)}")
        print(f"  • Découpage règle : {syntax_split}")
        
        if labels.eq(-100).all():
            print("🎯 Alignment Loss Masking : ❌ CRITIQUE : Tout est masqué.")
        elif not is_loss_clean:
            print("🎯 Alignment Loss Masking : ⚠️ ALERTE : Fin de séquence sale.")
        else:
            print("🎯 Alignment Loss Masking : ✅ Alignement parfait.")

        if has_byte_fallback:
            print("🧩 Intégrité Vocabulaire : ❌ Fragmentation détectée.")
        else:
            print("🧩 Intégrité Vocabulaire : ✅ Atomicité OK.")
        print("==========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic d'alignement via le Dataset de production GridGlyph")
    parser.add_argument(
        "--path", 
        type=str, 
        required=True, 
        help="Chemin vers le fichier de graines au format JSON Lines (.jsonl)"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=20, 
        help="Nombre maximum d'échantillons à analyser"
    )
    
    args = parser.parse_args()
    run_tokenizer_diagnostic(jsonl_path=args.path, max_samples=args.samples)