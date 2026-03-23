"""
Test Sample Efficiency - Adaptation Spectrum Experiment

How few examples do we need to achieve F1 >= 0.87?

Tests the adaptation spectrum:
- Zero-shot: Use base model without adapter
- Few-shot (50 examples): Quick fine-tuning
- Few-shot (100 examples): META-LoRA style
- Few-shot (500 examples): Medium training
- Full (3.5K examples): Full LoRA fine-tuning
"""

import sys
import time
import json
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


class SampleEfficiencyTrainer:
    """Test how few samples are needed for good performance."""

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.model = None
        self.tokenizer = None
        self.classifier_head = None

    def setup_model(self, use_4bit=True):
        """Load model with optional quantization."""
        print(f"\nLoading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_4bit and self.device.type == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
            )

        # Apply LoRA
        print("Applying LoRA adapters...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)

        # Classification head
        hidden_size = self.model.config.hidden_size
        self.classifier_head = nn.Linear(hidden_size, 2).to(self.device)
        print(f"Classifier head: {hidden_size} -> 2")

    def tokenize_data(self, texts, labels, max_length=256):
        """Tokenize texts."""
        prompts = [
            f"Classify if this prompt is an injection attack or benign.\n\nPrompt: {text[:500]}\n\nAnswer:"
            for text in texts
        ]

        encodings = self.tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels),
        }

    def train_with_samples(self, train_data, n_samples, epochs=3, batch_size=8, lr=1e-3):
        """Train with a specific number of samples."""
        print(f"\n{'='*60}")
        print(f"Training with {n_samples} samples")
        print(f"{'='*60}")

        # Subsample training data
        if n_samples < len(train_data["text"]):
            indices = np.random.choice(
                len(train_data["text"]),
                n_samples,
                replace=False
            )
            texts = [train_data["text"][i] for i in indices]
            labels = [train_data["label"][i] for i in indices]
        else:
            texts = train_data["text"]
            labels = train_data["label"]

        print(f"Actual samples: {len(texts)}")
        print(f"Class distribution: {sum(labels)}/{len(labels)} injection")

        # Tokenize
        encoded = self.tokenize_data(texts, labels)

        dataset = torch.utils.data.TensorDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
            encoded["labels"],
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=True
        )

        # Reset classifier head for fair comparison
        hidden_size = self.model.config.hidden_size
        self.classifier_head = nn.Linear(hidden_size, 2).to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(self.classifier_head.parameters(), lr=lr)

        # Training loop
        self.model.train()
        self.classifier_head.train()

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            total_loss = 0
            all_preds = []
            all_labels = []

            progress = tqdm(dataloader, desc=f"Epoch {epoch}")
            for input_ids, attention_mask, labels in progress:
                optimizer.zero_grad()

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                # Get hidden states
                with torch.no_grad():
                    outputs = self.model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    hidden_states = outputs.hidden_states[-1]

                # Get last token representation
                batch_size_curr = input_ids.shape[0]
                last_token_idx = attention_mask.sum(dim=1) - 1
                last_hidden = hidden_states[torch.arange(batch_size_curr, device=self.device), last_token_idx]

                # Classify
                logits = self.classifier_head(last_hidden.float())
                loss = nn.CrossEntropyLoss()(logits, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            metrics = compute_metrics(all_preds, all_labels)
            print(f"Epoch {epoch}: Loss={total_loss/len(dataloader):.4f}, F1={metrics['f1']:.4f}")

        training_time = time.time() - start_time
        print(f"Training time: {training_time:.1f}s")

        return training_time

    def evaluate(self, test_data, batch_size=16):
        """Evaluate on test set."""
        self.model.eval()
        self.classifier_head.eval()

        all_preds = []

        with torch.no_grad():
            for i in tqdm(range(0, len(test_data["text"]), batch_size), desc="Evaluating"):
                batch_texts = test_data["text"][i:i+batch_size]
                batch_labels = test_data["label"][i:i+batch_size]

                encoded = self.tokenize_data(batch_texts, batch_labels)

                outputs = self.model.model(
                    input_ids=encoded["input_ids"].to(self.device),
                    attention_mask=encoded["attention_mask"].to(self.device),
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]

                batch_size_curr = encoded["input_ids"].shape[0]
                last_token_idx = encoded["attention_mask"].sum(dim=1) - 1
                last_hidden = hidden_states[torch.arange(batch_size_curr, device=self.device), last_token_idx]

                logits = self.classifier_head(last_hidden.float())
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)

        test_labels = np.array(test_data["label"])
        return compute_metrics(all_preds, test_labels)


def main():
    print("="*70)
    print("Experiment 7: Sample Efficiency - Adaptation Spectrum")
    print("="*70)

    # Load data
    print("\n1. Loading dataset...")
    config = DatasetConfig(max_samples=10000, balance_classes=True)
    loader = PromptInjectionDataLoader(config=config)
    dataset = loader.load_or_create_dataset(force_recreate=False)

    train_data = dataset["train"]
    test_data = dataset["test"]

    print(f"   Train: {len(train_data)} samples")
    print(f"   Test: {len(test_data)} samples")

    # Create trainer
    print("\n2. Setting up model...")
    trainer = SampleEfficiencyTrainer()
    trainer.setup_model(use_4bit=True)

    # Test different sample sizes
    sample_sizes = [50, 100, 500, 1000, 5000]
    results = []

    for n_samples in sample_sizes:
        print(f"\n3. Testing with {n_samples} samples...")

        # Train
        training_time = trainer.train_with_samples(
            train_data,
            n_samples=n_samples,
            epochs=3,
            batch_size=8,
            lr=1e-3,
        )

        # Evaluate
        metrics = trainer.evaluate(test_data)

        print(f"\nResults for {n_samples} samples:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   F1:        {metrics['f1']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   Train time: {training_time:.1f}s")

        results.append({
            "n_samples": n_samples,
            "metrics": metrics,
            "training_time_s": training_time,
        })

        # Check if target achieved
        if metrics['f1'] >= 0.87:
            print(f"   ✅ Target F1 >= 0.87 achieved with {n_samples} samples!")
        else:
            print(f"   ❌ Target F1 >= 0.87 not yet achieved (need more data)")

    # Summary
    print(f"\n{'='*70}")
    print("Sample Efficiency Summary")
    print(f"{'='*70}")
    print(f"{'Samples':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Time':<10} {'Target'}")
    print("-"*60)
    for r in results:
        target = "✅" if r['metrics']['f1'] >= 0.87 else "❌"
        print(f"{r['n_samples']:<10} {r['metrics']['f1']:<10.4f} {r['metrics']['precision']:<12.4f} {r['metrics']['recall']:<10.4f} {r['training_time_s']:<10.1f}s {target}")

    # Find minimum samples for target
    min_samples = None
    for r in results:
        if r['metrics']['f1'] >= 0.87:
            min_samples = r['n_samples']
            break

    if min_samples:
        print(f"\n🎯 Minimum samples for F1 >= 0.87: {min_samples}")
    else:
        print(f"\n⚠️ Target F1 >= 0.87 not achieved with {sample_sizes[-1]} samples")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "sample_efficiency",
        "results": results,
        "min_samples_for_target": min_samples,
    }

    with open("data/experiment_progress.tsv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tsample_efficiency\tcompleted\t{json.dumps(output)}\n")

    with open("data/adapters/prompt_injection/sample_efficiency_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print("Experiment 7 Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
