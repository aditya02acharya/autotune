"""
Sample Efficiency Test with Regularization Fixes

Based on updated research: Fix the overfitting at 1,000 samples
- Reduce epochs (early stopping)
- Add weight decay (0.01-0.1)
- Add dropout (0.1-0.2) to classification head
- Use early stopping on validation split

Target: 500 samples should jump from 0.84 to 0.87-0.89
"""

import sys
import time
import json
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

from guardrails.utils.data import PromptInjectionDataLoader, DatasetConfig


class RegularizedClassifierHead(nn.Module):
    """Classifier head with dropout for regularization."""

    def __init__(self, hidden_size, num_classes=2, dropout=0.15):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))


class RegularizedSampleEfficiencyTest:
    """Test sample efficiency with regularization fixes."""

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.classifier_head = None

    def setup_model(self, dropout=0.15):
        """Load model with standard LoRA config."""
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

        # Standard LoRA config (we're training classifier head only)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)

        hidden_size = self.model.config.hidden_size
        self.classifier_head = RegularizedClassifierHead(hidden_size, dropout=dropout).to(self.device)

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

    def compute_metrics(self, preds, labels):
        return {
            "f1": f1_score(labels, preds),
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
        }

    def train_with_samples(
        self,
        train_data,
        val_data,
        n_samples,
        max_epochs=10,
        batch_size=8,
        lr=1e-3,
        weight_decay=0.05,
        early_stopping_patience=3,
    ):
        """Train with regularization and early stopping."""
        # Sample subset
        if n_samples < len(train_data["text"]):
            indices = np.random.choice(len(train_data["text"]), n_samples, replace=False)
            texts = [train_data["text"][i] for i in indices]
            labels = [train_data["label"][i] for i in indices]
        else:
            texts = train_data["text"]
            labels = train_data["label"]

        # Reset classifier head
        self.classifier_head.apply(self._init_weights)

        # Prepare data
        encoded = self.tokenize_data(texts, labels)
        dataset = torch.utils.data.TensorDataset(
            encoded["input_ids"], encoded["attention_mask"], encoded["labels"]
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Prepare validation data (use 20% of available data)
        val_encoded = self.tokenize_data(val_data["text"][:1000], val_data["label"][:1000])
        val_dataset = torch.utils.data.TensorDataset(
            val_encoded["input_ids"], val_encoded["attention_mask"], val_encoded["labels"]
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.classifier_head.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        start_time = time.time()
        best_f1 = 0
        best_state = None
        patience_counter = 0

        self.model.train()
        self.classifier_head.train()

        for epoch in range(max_epochs):
            # Training
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

            # Validation
            val_f1 = self._evaluate_f1(val_loader)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = copy.deepcopy(self.classifier_head.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                break

        # Restore best model
        if best_state is not None:
            self.classifier_head.load_state_dict(best_state)

        train_time = time.time() - start_time
        return train_time, best_f1, epoch + 1

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _evaluate_f1(self, dataloader):
        """Quick F1 evaluation."""
        self.model.eval()
        self.classifier_head.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, batch_labels in dataloader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                outputs = self.model.model(
                    input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]

                batch_size = input_ids.shape[0]
                last_token_idx = attention_mask.sum(dim=1) - 1
                last_hidden = hidden_states[torch.arange(batch_size, device=self.device), last_token_idx]

                logits = self.classifier_head(last_hidden.float())
                preds = logits.argmax(dim=-1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(batch_labels.numpy())

        return f1_score(all_labels, all_preds)

    def evaluate_full(self, test_data, batch_size=16):
        """Full evaluation on test set."""
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

        return self.compute_metrics(all_preds, all_labels)

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.classifier_head
        torch.cuda.empty_cache()


def main():
    print("="*70)
    print("Sample Efficiency Test - REGULARIZATION FIXES")
    print("Features: Dropout + Weight Decay + Early Stopping")
    print("Target: Fix overfitting, achieve F1 >= 0.87 with <500 samples")
    print("="*70)

    # Load dataset
    print("\n1. Loading dataset...")
    config = DatasetConfig(max_samples=10000, balance_classes=True)
    loader = PromptInjectionDataLoader(config=config)
    dataset = loader.load_or_create_dataset(force_recreate=False)

    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    print(f"   Train: {len(train_data)} samples available")
    print(f"   Val: {len(val_data)} samples")
    print(f"   Test: {len(test_data)} samples")

    # Test different sample sizes (focus on lower range)
    sample_sizes = [100, 200, 300, 400, 500, 750, 1000]
    results = []

    print(f"\n2. Testing with regularization...")
    print(f"   - Dropout: 0.15")
    print(f"   - Weight decay: 0.05")
    print(f"   - Early stopping patience: 3")
    print(f"   - Max epochs: 10")

    for n_samples in sample_sizes:
        print(f"\n{'='*50}")
        print(f"Testing with {n_samples} samples...")

        tester = RegularizedSampleEfficiencyTest()
        tester.setup_model(dropout=0.15)

        train_time, val_f1, epochs_used = tester.train_with_samples(
            train_data=train_data,
            val_data=val_data,
            n_samples=n_samples,
            max_epochs=10,
            batch_size=8,
            lr=1e-3,
            weight_decay=0.05,
            early_stopping_patience=3,
        )

        test_metrics = tester.evaluate_full(test_data)
        target_met = test_metrics["f1"] >= 0.87

        print(f"   Val F1: {val_f1:.4f}")
        print(f"   Test F1: {test_metrics['f1']:.4f} {'PASS' if target_met else 'FAIL'}")
        print(f"   Test Precision: {test_metrics['precision']:.4f}")
        print(f"   Test Recall: {test_metrics['recall']:.4f}")
        print(f"   Epochs used: {epochs_used}")
        print(f"   Training time: {train_time:.1f}s")

        results.append({
            "n_samples": n_samples,
            "val_f1": val_f1,
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "epochs_used": epochs_used,
            "train_time_s": train_time,
            "target_met": target_met,
        })

        tester.cleanup()

        if target_met and n_samples <= 500:
            print(f"\n   *** Target F1 >= 0.87 achieved with only {n_samples} samples! ***")

    # Find minimum samples needed
    passing_results = [r for r in results if r["target_met"]]
    min_samples = min(r["n_samples"] for r in passing_results) if passing_results else None

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY - Regularized Training")
    print("="*70)
    print(f"{'Samples':<10} {'Val F1':<10} {'Test F1':<10} {'Prec':<10} {'Recall':<10} {'Epochs':<8} {'Status':<8}")
    print("-"*66)
    for r in results:
        status = "PASS" if r["target_met"] else "FAIL"
        print(f"{r['n_samples']:<10} {r['val_f1']:<10.4f} {r['test_f1']:<10.4f} {r['test_precision']:<10.4f} {r['test_recall']:<10.4f} {r['epochs_used']:<8} {status}")

    print(f"\n{'='*70}")
    if min_samples:
        print(f"MINIMUM SAMPLES for F1 >= 0.87: {min_samples}")
        improvement = 2000 - min_samples
        print(f"IMPROVEMENT: {improvement} fewer samples ({improvement/2000*100:.0f}% reduction)")
    else:
        print("Target F1 >= 0.87 NOT ACHIEVED with any sample size")
    print("="*70)

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "regularized_sample_efficiency",
        "regularization": {
            "dropout": 0.15,
            "weight_decay": 0.05,
            "early_stopping_patience": 3,
            "max_epochs": 10,
        },
        "results": results,
        "min_samples_for_f1_87": min_samples,
    }

    Path("data/adapters/regularized").mkdir(parents=True, exist_ok=True)
    with open("data/adapters/regularized/sample_efficiency_results.json", "w") as f:
        json.dump(output, f, indent=2)

    with open("data/experiment_progress.tsv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tregularized_sample_efficiency\tcompleted\t{json.dumps(output)}\n")

    # Update experiment ledger
    update_ledger(results, min_samples)

    return output


def update_ledger(results, min_samples):
    """Update experiment ledger with findings."""
    ledger_path = Path("data/experiment_ledger.md")

    with open(ledger_path, "r") as f:
        content = f.read()

    # Add new experiment entry
    new_entry = f"""
### Experiment 8: Regularized Sample Efficiency
- **Time**: ~{time.strftime('%H:%M')}
- **Status**: Completed
- **Question**: Can regularization fix the overfitting at 1000 samples?
- **Config**:
  - Dropout: 0.15
  - Weight decay: 0.05
  - Early stopping patience: 3
  - Max epochs: 10
- **Results**:
"""

    for r in results:
        new_entry += f"  | {r['n_samples']:4d} samples | F1: {r['test_f1']:.4f} | Prec: {r['test_precision']:.4f} | Recall: {r['test_recall']:.4f} | Epochs: {r['epochs_used']:2d} | {'PASS' if r['target_met'] else 'FAIL'} |\n"

    new_entry += f"""- **Learning**:
  - Minimum samples for F1 >= 0.87: {min_samples if min_samples else 'Not achieved'}
  - Regularization {'helped' if min_samples and min_samples < 2000 else 'did not significantly help'}
"""

    # Insert after the last experiment entry
    if "### Experiment 8:" in content:
        # Already updated
        pass
    else:
        # Add before "Key Insights" section
        content = content.replace("---\n\n## Key Insights", new_entry + "\n---\n\n## Key Insights")

        with open(ledger_path, "w") as f:
            f.write(content)


if __name__ == "__main__":
    main()
