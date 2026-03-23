"""
Sample Efficiency Test - Lightweight Version

Tests how few examples are needed for F1 >= 0.87.
Uses smaller test sizes and efficient memory management.
"""

import sys
import time
import json
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

from guardrails.utils.data import PromptInjectionDataLoader, DatasetConfig


def compute_metrics(preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


def train_and_evaluate(model, tokenizer, classifier_head, device, train_data, test_data, n_samples, epochs=3):
    """Train with n_samples and evaluate."""
    print(f"\n{'='*50}")
    print(f"Training with {n_samples} samples")
    print(f"{'='*50}")

    # Subsample
    indices = np.random.choice(len(train_data["text"]), min(n_samples, len(train_data["text"])), replace=False)
    texts = [train_data["text"][i] for i in indices]
    labels = [train_data["label"][i] for i in indices]

    print(f"Samples: {len(texts)}, Injection ratio: {sum(labels)/len(labels):.2f}")

    # Tokenize
    prompts = [f"Classify if this prompt is an injection attack or benign.\n\nPrompt: {t[:500]}\n\nAnswer:" for t in texts]
    encoded = tokenizer(prompts, max_length=256, truncation=True, padding="max_length", return_tensors="pt")

    # Reset classifier
    hidden_size = model.config.hidden_size
    classifier_head = nn.Linear(hidden_size, 2).to(device)
    optimizer = torch.optim.AdamW(classifier_head.parameters(), lr=1e-3)

    # Training
    model.train()
    classifier_head.train()
    batch_size = min(8, len(texts))

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        all_preds, all_labels = [], []

        for i in range(0, len(texts), batch_size):
            optimizer.zero_grad()

            batch_ids = encoded["input_ids"][i:i+batch_size].to(device)
            batch_mask = encoded["attention_mask"][i:i+batch_size].to(device)
            batch_labels = torch.tensor(labels[i:i+batch_size]).to(device)

            with torch.no_grad():
                outputs = model.model(input_ids=batch_ids, attention_mask=batch_mask, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]

            bsize = batch_ids.shape[0]
            last_idx = batch_mask.sum(dim=1) - 1
            last_hidden = hidden[torch.arange(bsize, device=device), last_idx]

            logits = classifier_head(last_hidden.float())
            loss = nn.CrossEntropyLoss()(logits, batch_labels)

            loss.backward()
            optimizer.step()

            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels[i:i+batch_size])

        train_f1 = f1_score(all_labels, all_preds)
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Train F1={train_f1:.4f}")

    train_time = time.time() - start_time
    print(f"Training time: {train_time:.1f}s")

    # Evaluation on subset of test data
    model.eval()
    classifier_head.eval()

    test_texts = test_data["text"][:1000]  # Use 1000 samples for faster eval
    test_labels = np.array(test_data["label"][:1000])

    all_preds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_texts), 16), desc="Evaluating"):
            batch_texts = test_texts[i:i+16]
            prompts = [f"Classify if this prompt is an injection attack or benign.\n\nPrompt: {t[:500]}\n\nAnswer:" for t in batch_texts]
            encoded = tokenizer(prompts, max_length=256, truncation=True, padding="max_length", return_tensors="pt")

            outputs = model.model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                output_hidden_states=True
            )
            hidden = outputs.hidden_states[-1]

            bsize = encoded["input_ids"].shape[0]
            last_idx = encoded["attention_mask"].sum(dim=1) - 1
            last_hidden = hidden[torch.arange(bsize, device=device), last_idx]

            logits = classifier_head(last_hidden.float())
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())

    metrics = compute_metrics(all_preds, test_labels)

    print(f"\nTest Results ({len(test_texts)} samples):")
    print(f"   F1:        {metrics['f1']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")

    target_met = "[PASS]" if metrics['f1'] >= 0.87 else "[FAIL]"
    print(f"   Target F1 >= 0.87: {target_met}")

    # Cleanup
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return metrics, train_time, classifier_head


def main():
    print("="*60)
    print("Sample Efficiency Experiment")
    print("="*60)

    # Load data
    print("\n1. Loading data...")
    config = DatasetConfig(max_samples=10000, balance_classes=True)
    loader = PromptInjectionDataLoader(config=config)
    dataset = loader.load_or_create_dataset(force_recreate=False)

    train_data = dataset["train"]
    test_data = dataset["test"]

    print(f"   Train: {len(train_data['text'])} samples")
    print(f"   Test: {len(test_data['text'])} samples")

    # Setup model
    print("\n2. Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    hidden_size = model.config.hidden_size
    classifier_head = nn.Linear(hidden_size, 2).to(device)

    # Test sample sizes
    sample_sizes = [50, 100, 200, 500, 1000, 2000]
    results = []

    print("\n3. Testing sample efficiency...")

    for n in sample_sizes:
        metrics, train_time, classifier_head = train_and_evaluate(
            model, tokenizer, classifier_head, device,
            train_data, test_data, n, epochs=3
        )

        results.append({
            "n_samples": n,
            "f1": metrics['f1'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "train_time_s": train_time,
            "target_met": metrics['f1'] >= 0.87
        })

        if metrics['f1'] >= 0.87:
            print(f"\n[TARGET] Achieved with {n} samples!")

    # Summary
    print(f"\n{'='*60}")
    print("Sample Efficiency Summary")
    print(f"{'='*60}")
    print(f"{'Samples':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Time':<10} {'Target'}")
    print("-"*60)

    min_samples = None
    for r in results:
        target = "[PASS]" if r['target_met'] else "[FAIL]"
        print(f"{r['n_samples']:<10} {r['f1']:<10.4f} {r['precision']:<12.4f} {r['recall']:<10.4f} {r['train_time_s']:<10.1f}s {target}")

        if min_samples is None and r['target_met']:
            min_samples = r['n_samples']

    if min_samples:
        print(f"\n[RESULT] Minimum samples for F1 >= 0.87: {min_samples}")
    else:
        print(f"\n[WARN] Target not achieved with {sample_sizes[-1]} samples")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "sample_efficiency",
        "results": results,
        "min_samples_for_f1_87": min_samples,
    }

    with open("data/experiment_progress.tsv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tsample_efficiency\tcompleted\t{json.dumps(output)}\n")

    with open("data/adapters/prompt_injection/sample_efficiency_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
