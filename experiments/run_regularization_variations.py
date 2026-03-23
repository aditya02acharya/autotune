"""
Regularization Variations Experiment

Tests different regularization combinations:
1. Lower dropout (0.05)
2. No label smoothing
3. Lower weight decay (0.01)
4. Lower LoRA+ ratio (8x)
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from guardrails.utils.data import PromptInjectionDataLoader, DatasetConfig


def compute_metrics(preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        nll_loss = F.cross_entropy(logits, targets, reduction='none')
        smooth_loss = -F.log_softmax(logits, dim=-1).mean(dim=-1)
        return ((1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss).mean()


def run_variation(name, config, train_data, val_data, test_data, device):
    """Run a single variation with specified config."""
    print(f"\n{'='*60}")
    print(f"VARIATION: {name}")
    print(f"{'='*60}")
    print(f"Config: {config}")

    # Load model
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        quantization_config=quantization_config, device_map="auto"
    )

    # LoRA config
    ALL_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=config["rank"],
        lora_alpha=config["rank"] * 2,
        lora_dropout=config["dropout"],
        target_modules=ALL_LAYERS,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=config["use_dora"],
    )
    model = get_peft_model(model, lora_config)

    # Classifier
    hidden_size = model.config.hidden_size
    classifier = nn.Sequential(
        nn.Dropout(config["classifier_dropout"]),
        nn.Linear(hidden_size, 2)
    ).to(device)

    # Tokenize
    def tokenize(texts, labels):
        prompts = [f"Classify if this prompt is an injection attack or benign.\n\nPrompt: {t[:500]}\n\nAnswer:" for t in texts]
        enc = tokenizer(prompts, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor(labels)

    train_ds = torch.utils.data.TensorDataset(*tokenize(train_data["text"], train_data["label"]))
    val_ds = torch.utils.data.TensorDataset(*tokenize(val_data["text"], val_data["label"]))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    # LoRA+ optimizer
    lora_A, lora_B, other = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'lora_A' in n:
            lora_A.append(p)
        elif 'lora_B' in n:
            lora_B.append(p)
        else:
            other.append(p)

    base_lr = config["lr"]
    optimizer = torch.optim.AdamW([
        {"params": lora_A, "lr": base_lr, "weight_decay": config["weight_decay"]},
        {"params": lora_B, "lr": base_lr * config["lora_plus_ratio"], "weight_decay": config["weight_decay"]},
        {"params": other, "lr": base_lr, "weight_decay": config["weight_decay"]},
        {"params": classifier.parameters(), "lr": base_lr, "weight_decay": config["weight_decay"]},
    ])

    criterion = LabelSmoothingLoss(smoothing=config["label_smoothing"])

    best_f1 = 0
    best_metrics = None
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        classifier.train()
        total_loss = 0
        all_preds, all_labels = [], []

        n_batches = len(train_loader)
        report_at = [n_batches // 2]

        for i, (ids, mask, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)

            with torch.no_grad():
                out = model.model(ids, attention_mask=mask, output_hidden_states=True)
                h = out.hidden_states[-1]
            last_idx = mask.sum(1) - 1
            last_h = h[torch.arange(ids.shape[0], device=device), last_idx]

            logits = classifier(last_h.float())
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if i in report_at:
                print(f"  E{epoch} 50%: loss={loss.item():.4f}")

        train_m = compute_metrics(all_preds, all_labels)

        # Validation
        model.eval()
        classifier.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for ids, mask, labels in val_loader:
                ids, mask = ids.to(device), mask.to(device)
                out = model.model(ids, attention_mask=mask, output_hidden_states=True)
                h = out.hidden_states[-1]
                last_idx = mask.sum(1) - 1
                last_h = h[torch.arange(ids.shape[0], device=device), last_idx]
                logits = classifier(last_h.float())
                val_preds.extend(logits.argmax(-1).cpu().numpy())
                val_labels.extend(labels.numpy())

        val_m = compute_metrics(val_preds, val_labels)
        print(f"E{epoch}: Train F1={train_m['f1']:.4f} | Val F1={val_m['f1']:.4f}")

        if val_m['f1'] > best_f1:
            best_f1 = val_m['f1']
            best_metrics = val_m.copy()
            best_metrics['epoch'] = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test evaluation
    test_ds = torch.utils.data.TensorDataset(*tokenize(test_data["text"], test_data["label"]))
    test_loader = DataLoader(test_ds, batch_size=8)

    model.eval()
    classifier.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for ids, mask, labels in test_loader:
            ids, mask = ids.to(device), mask.to(device)
            out = model.model(ids, attention_mask=mask, output_hidden_states=True)
            h = out.hidden_states[-1]
            last_idx = mask.sum(1) - 1
            last_h = h[torch.arange(ids.shape[0], device=device), last_idx]
            logits = classifier(last_h.float())
            test_preds.extend(logits.argmax(-1).cpu().numpy())
            test_labels.extend(labels.numpy())

    test_m = compute_metrics(test_preds, test_labels)

    print(f"\n>>> {name} <<<")
    print(f"Best Val F1: {best_f1:.4f} (Epoch {best_metrics['epoch']})")
    print(f"Test F1: {test_m['f1']:.4f} | P={test_m['precision']:.4f} R={test_m['recall']:.4f}")

    # Clean up
    del model, classifier, optimizer
    torch.cuda.empty_cache()

    return {
        "name": name,
        "config": config,
        "best_val_f1": best_f1,
        "best_epoch": best_metrics['epoch'],
        "test_metrics": test_m,
    }


def main():
    print("=" * 70)
    print("REGULARIZATION VARIATIONS EXPERIMENT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    config = DatasetConfig(max_samples=10000, balance_classes=True)
    loader = PromptInjectionDataLoader(config=config)
    dataset = loader.load_or_create_dataset(force_recreate=False)
    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]
    print(f"Data: train={len(train_data['text'])} val={len(val_data['text'])} test={len(test_data['text'])}")

    # Variations to test
    variations = [
        {
            "name": "low_dropout",
            "rank": 16, "dropout": 0.05, "classifier_dropout": 0.1,
            "weight_decay": 0.05, "label_smoothing": 0.1, "lora_plus_ratio": 16,
            "use_dora": True, "lr": 2e-4, "epochs": 3, "patience": 2
        },
        {
            "name": "no_label_smooth",
            "rank": 16, "dropout": 0.15, "classifier_dropout": 0.2,
            "weight_decay": 0.05, "label_smoothing": 0.0, "lora_plus_ratio": 16,
            "use_dora": True, "lr": 2e-4, "epochs": 3, "patience": 2
        },
        {
            "name": "low_weight_decay",
            "rank": 16, "dropout": 0.15, "classifier_dropout": 0.2,
            "weight_decay": 0.01, "label_smoothing": 0.1, "lora_plus_ratio": 16,
            "use_dora": True, "lr": 2e-4, "epochs": 3, "patience": 2
        },
        {
            "name": "low_lora_plus",
            "rank": 16, "dropout": 0.15, "classifier_dropout": 0.2,
            "weight_decay": 0.05, "label_smoothing": 0.1, "lora_plus_ratio": 8,
            "use_dora": True, "lr": 2e-4, "epochs": 3, "patience": 2
        },
    ]

    results = []
    for var in variations:
        result = run_variation(
            var["name"], var,
            train_data, val_data, test_data, device
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Variation':<20} {'Val F1':>10} {'Test F1':>10} {'Epoch':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<20} {r['best_val_f1']:>10.4f} {r['test_metrics']['f1']:>10.4f} {r['best_epoch']:>8}")

    # Best result
    best = max(results, key=lambda x: x['test_metrics']['f1'])
    print(f"\n>>> BEST: {best['name']} with Test F1={best['test_metrics']['f1']:.4f} <<<")

    # Save results
    with open("data/adapters/regularized_lora/variations_results.json", "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, f, indent=2)

    print("\nSaved to data/adapters/regularized_lora/variations_results.json")


if __name__ == "__main__":
    main()
