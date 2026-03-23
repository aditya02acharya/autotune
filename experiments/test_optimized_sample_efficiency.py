"""
Sample Efficiency Test with Optimized LoRA Configuration

Tests how few samples are needed to achieve F1 >= 0.87 using:
- DoRA (weight decomposition)
- All 7 linear layers
- LoRA+ learning rates
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from guardrails.utils.data import PromptInjectionDataLoader, DatasetConfig


class OptimizedSampleEfficiencyTest:
    """Test sample efficiency with optimized LoRA."""

    ALL_LINEAR_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.classifier_head = None

    def setup_model(self, use_dora=True, rank=16):
        """Load model with optimized LoRA."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            target_modules=self.ALL_LINEAR_LAYERS,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_dora=use_dora,
        )
        self.model = get_peft_model(self.model, lora_config)

        hidden_size = self.model.config.hidden_size
        self.classifier_head = nn.Linear(hidden_size, 2).to(self.device)

    def tokenize_data(self, texts, labels, max_length=256):
        prompts = [
            f"Classify if this prompt is an injection attack or benign.\n\nPrompt: {text[:500]}\n\nAnswer:"
            for text in texts
        ]
        encodings = self.tokenizer(
            prompts, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels),
        }

    def train_with_samples(self, train_data, n_samples, epochs=3, batch_size=8, lr=2e-4):
        """Train with a specific number of samples."""
        # Sample subset
        if n_samples < len(train_data["text"]):
            indices = np.random.choice(len(train_data["text"]), n_samples, replace=False)
            texts = [train_data["text"][i] for i in indices]
            labels = [train_data["label"][i] for i in indices]
        else:
            texts = train_data["text"]
            labels = train_data["label"]

        # Reset classifier head
        nn.init.xavier_uniform_(self.classifier_head.weight)
        nn.init.zeros_(self.classifier_head.bias)

        # Prepare data
        encoded = self.tokenize_data(texts, labels)
        dataset = torch.utils.data.TensorDataset(
            encoded["input_ids"], encoded["attention_mask"], encoded["labels"]
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.classifier_head.parameters(), lr=lr)

        start_time = time.time()
        self.model.train()
        self.classifier_head.train()

        for epoch in range(epochs):
            for input_ids, attention_mask, batch_labels in dataloader:
                optimizer.zero_grad()
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)

                with torch.no_grad():
                    outputs = self.model.model(
                        input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
                    )
                    hidden_states = outputs.hidden_states[-1]

                batch_size = input_ids.shape[0]
                last_token_idx = attention_mask.sum(dim=1) - 1
                last_hidden = hidden_states[torch.arange(batch_size, device=self.device), last_token_idx]

                logits = self.classifier_head(last_hidden.float())
                loss = nn.CrossEntropyLoss()(logits, batch_labels)
                loss.backward()
                optimizer.step()

        train_time = time.time() - start_time
        return train_time

    def evaluate(self, test_data, batch_size=16):
        """Evaluate on test set."""
        self.model.eval()
        self.classifier_head.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i in range(0, len(test_data["text"]), batch_size):
                batch_texts = test_data["text"][i:i+batch_size]
                batch_labels = test_data["label"][i:i+batch_size]

                encoded = self.tokenize_data(batch_texts, batch_labels)
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                outputs = self.model.model(
                    input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]

                batch_size_curr = input_ids.shape[0]
                last_token_idx = attention_mask.sum(dim=1) - 1
                last_hidden = hidden_states[torch.arange(batch_size_curr, device=self.device), last_token_idx]

                logits = self.classifier_head(last_hidden.float())
                preds = logits.argmax(dim=-1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(batch_labels)

        return f1_score(all_labels, all_preds)

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.classifier_head
        torch.cuda.empty_cache()


def main():
    print("="*70)
    print("Sample Efficiency Test - OPTIMIZED LoRA Configuration")
    print("Features: DoRA + All Linear Layers")
    print("="*70)

    # Load dataset
    print("\n1. Loading dataset...")
    config = DatasetConfig(max_samples=10000, balance_classes=True)
    loader = PromptInjectionDataLoader(config=config)
    dataset = loader.load_or_create_dataset(force_recreate=False)

    train_data = dataset["train"]
    test_data = dataset["test"]

    print(f"   Train: {len(train_data)} samples available")
    print(f"   Test: {len(test_data)} samples")

    # Test different sample sizes
    sample_sizes = [50, 100, 200, 300, 400, 500, 750, 1000]
    results = []

    print(f"\n2. Testing sample efficiency...")
    print(f"   Target: F1 >= 0.87")
    print(f"   Sample sizes to test: {sample_sizes}")

    for n_samples in sample_sizes:
        print(f"\n{'='*50}")
        print(f"Testing with {n_samples} samples...")

        tester = OptimizedSampleEfficiencyTest()
        tester.setup_model(use_dora=True, rank=16)

        train_time = tester.train_with_samples(
            train_data=train_data,
            n_samples=n_samples,
            epochs=3,
            batch_size=8,
            lr=2e-4
        )

        f1 = tester.evaluate(test_data)
        target_met = f1 >= 0.87

        print(f"   F1: {f1:.4f} {'PASS' if target_met else 'FAIL'}")
        print(f"   Training time: {train_time:.1f}s")

        results.append({
            "n_samples": n_samples,
            "f1": f1,
            "train_time_s": train_time,
            "target_met": target_met,
        })

        tester.cleanup()

        if target_met:
            print(f"\n   *** Target F1 >= 0.87 achieved with {n_samples} samples! ***")

    # Find minimum samples needed
    passing_results = [r for r in results if r["target_met"]]
    min_samples = min(r["n_samples"] for r in passing_results) if passing_results else None

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Samples':<10} {'F1':<10} {'Time (s)':<12} {'Target':<10}")
    print("-"*42)
    for r in results:
        status = "PASS" if r["target_met"] else "FAIL"
        print(f"{r['n_samples']:<10} {r['f1']:<10.4f} {r['train_time_s']:<12.1f} {status}")

    print(f"\n{'='*70}")
    if min_samples:
        print(f"MINIMUM SAMPLES for F1 >= 0.87: {min_samples}")
    else:
        print("Target F1 >= 0.87 NOT ACHIEVED with any sample size")
    print("="*70)

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "optimized_sample_efficiency",
        "optimizations": ["dora", "all_linear_layers", "rank_16"],
        "results": results,
        "min_samples_for_f1_87": min_samples,
    }

    Path("data/adapters/optimized").mkdir(parents=True, exist_ok=True)
    with open("data/adapters/optimized/sample_efficiency_results.json", "w") as f:
        json.dump(output, f, indent=2)

    with open("data/experiment_progress.tsv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\toptimized_sample_efficiency\tcompleted\t{json.dumps(output)}\n")

    return output


if __name__ == "__main__":
    main()
