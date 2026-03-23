"""
Experiment 9: Synthetic + Real Data Combination

Hypothesis: Adding synthetic adversarial data improves F1 by exposing the model
to more diverse attack patterns.

Target: F1 >= 0.95 (up from current best 0.938)
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from guardrails.utils.data import PromptInjectionDataLoader, DatasetConfig


def compute_metrics(preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


class SyntheticCombinedTrainer:
    """Trainer with optimized LoRA + synthetic data augmentation."""

    ALL_LINEAR_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", output_dir="data/adapters/synthetic_combined"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        self.model = None
        self.tokenizer = None
        self.classifier_head = None

    def setup_model(self, use_dora=True, rank=16):
        """Load model with DoRA and all linear layers."""
        print(f"\nLoading model: {self.model_name}")

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
        self.model.print_trainable_parameters()

        hidden_size = self.model.config.hidden_size
        self.classifier_head = nn.Linear(hidden_size, 2).to(self.device)
        print(f"Classifier head: {hidden_size} -> 2")

    def load_synthetic_data(self):
        """Load synthetic injection data."""
        synthetic_path = Path("data/synthetic/synthetic_injection_data.json")
        if not synthetic_path.exists():
            print("WARNING: Synthetic data not found!")
            return None

        with open(synthetic_path) as f:
            data = json.load(f)

        print(f"Loaded {len(data)} synthetic samples")
        labels = [s['label'] for s in data]
        print(f"  - Injection: {sum(labels)}")
        print(f"  - Benign: {len(labels) - sum(labels)}")
        return data

    def combine_data(self, real_data, synthetic_data, synthetic_ratio=0.3):
        """Combine real and synthetic data with specified ratio."""
        # Extract real data
        real_texts = list(real_data["text"])
        real_labels = list(real_data["label"])

        # Sample synthetic data (to avoid overwhelming)
        import random
        random.seed(42)

        # Balance synthetic: we have 6000 injection, 500 benign
        # Sample injection cases
        synthetic_injection = [s for s in synthetic_data if s['label'] == 1]
        synthetic_benign = [s for s in synthetic_data if s['label'] == 0]

        n_synthetic = int(len(real_texts) * synthetic_ratio)
        n_injection = int(n_synthetic * 0.9)  # 90% injection (attack-focused)
        n_benign = n_synthetic - n_injection

        sampled_injection = random.sample(synthetic_injection, min(n_injection, len(synthetic_injection)))
        sampled_benign = random.sample(synthetic_benign, min(n_benign, len(synthetic_benign)))

        combined_texts = real_texts + [s['text'] for s in sampled_injection] + [s['text'] for s in sampled_benign]
        combined_labels = real_labels + [1] * len(sampled_injection) + [0] * len(sampled_benign)

        print(f"Combined dataset: {len(combined_texts)} samples")
        print(f"  - Real: {len(real_texts)}")
        print(f"  - Synthetic injection: {len(sampled_injection)}")
        print(f"  - Synthetic benign: {len(sampled_benign)}")

        return {"text": combined_texts, "label": combined_labels}

    def tokenize_data(self, texts, labels, max_length=256):
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

    def train_epoch(self, dataloader, optimizer, epoch):
        self.model.train()
        self.classifier_head.train()

        total_loss = 0
        all_preds = []
        all_labels = []

        for i, (input_ids, attention_mask, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]

            batch_size = input_ids.shape[0]
            last_token_idx = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[torch.arange(batch_size, device=self.device), last_token_idx]

            logits = self.classifier_head(last_hidden.float())
            loss = nn.CrossEntropyLoss()(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # Strategic status: every 25% of epoch
            if (i + 1) % max(1, len(dataloader) // 4) == 0:
                pct = 100 * (i + 1) // len(dataloader)
                print(f"  Epoch {epoch}: {pct}% - loss={loss.item():.3f}", flush=True)

        return compute_metrics(all_preds, all_labels)

    def evaluate(self, dataloader, silent=False):
        self.model.eval()
        self.classifier_head.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, labels in dataloader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]

                batch_size = input_ids.shape[0]
                last_token_idx = attention_mask.sum(dim=1) - 1
                last_hidden = hidden_states[torch.arange(batch_size, device=self.device), last_token_idx]

                logits = self.classifier_head(last_hidden.float())
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        if not silent:
            print(f"  Eval complete: {len(all_preds)} samples")
        return compute_metrics(all_preds, all_labels)

    def train(self, train_data, val_data, epochs=3, batch_size=8, lr=2e-4):
        train_encoded = self.tokenize_data(train_data["text"], train_data["label"])
        val_encoded = self.tokenize_data(val_data["text"], val_data["label"])

        train_dataset = torch.utils.data.TensorDataset(
            train_encoded["input_ids"],
            train_encoded["attention_mask"],
            train_encoded["labels"],
        )
        val_dataset = torch.utils.data.TensorDataset(
            val_encoded["input_ids"],
            val_encoded["attention_mask"],
            val_encoded["labels"],
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(self.classifier_head.parameters(), lr=lr)

        best_f1 = 0
        best_metrics = None
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            print(f"\n[Epoch {epoch}/{epochs}]")

            train_metrics = self.train_epoch(train_loader, optimizer, epoch)
            print(f"  Train: F1={train_metrics['f1']:.4f}")

            val_metrics = self.evaluate(val_loader)
            print(f"  Val:   F1={val_metrics['f1']:.4f} P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f}")

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_metrics = val_metrics.copy()
                best_epoch = epoch
                self.save(epoch)

        print(f"\nBest Val F1: {best_f1:.4f} (Epoch {best_epoch})")
        return best_metrics, best_epoch

    def save(self, epoch):
        save_dir = self.output_dir / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir / "adapter")
        torch.save(self.classifier_head.state_dict(), save_dir / "classifier_head.pt")
        print(f"Saved to {save_dir}")

    def predict(self, texts, batch_size=16):
        self.model.eval()
        self.classifier_head.eval()
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                encoded = self.tokenize_data(batch, [0]*len(batch))

                outputs = self.model.model(
                    input_ids=encoded["input_ids"].to(self.device),
                    attention_mask=encoded["attention_mask"].to(self.device),
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]

                bs = encoded["input_ids"].shape[0]
                last_idx = encoded["attention_mask"].sum(dim=1) - 1
                last_hidden = hidden_states[torch.arange(bs, device=self.device), last_idx]

                logits = self.classifier_head(last_hidden.float())
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())

        return np.array(all_preds)


def main():
    print("\n" + "="*50)
    print("EXP 9: Synthetic + Real Data | Target: F1>=0.95")
    print("="*50)

    # Load real data
    config = DatasetConfig(max_samples=10000, balance_classes=True)
    loader = PromptInjectionDataLoader(config=config)
    dataset = loader.load_or_create_dataset(force_recreate=False)
    print(f"Real data: train={len(dataset['train']['text'])}")

    # Load synthetic data
    trainer = SyntheticCombinedTrainer()
    synthetic_data = trainer.load_synthetic_data()

    # Combine train data
    combined_train = trainer.combine_data(
        dataset["train"],
        synthetic_data,
        synthetic_ratio=0.3
    )
    val_data = dataset["validation"]
    test_data = dataset["test"]

    # Setup model
    print("\n[Loading model...]")
    trainer.setup_model(use_dora=True, rank=16)

    # Train
    start_time = time.time()
    best_metrics, best_epoch = trainer.train(
        combined_train, val_data,
        epochs=3, batch_size=8, lr=2e-4
    )
    train_time = time.time() - start_time

    # Test
    print("\n[Testing...]")
    test_texts = test_data["text"]
    test_labels = np.array(test_data["label"])

    # Load best checkpoint
    best_dir = trainer.output_dir / f"epoch_{best_epoch}"
    trainer.model = trainer.model.from_pretrained(trainer.model, best_dir / "adapter")
    trainer.classifier_head.load_state_dict(torch.load(best_dir / "classifier_head.pt"))

    preds = trainer.predict(test_texts)
    test_metrics = compute_metrics(preds, test_labels)

    print(f"\n{'='*50}")
    print(f"RESULTS | Time: {train_time:.0f}s")
    print(f"{'='*50}")
    print(f"Val F1:   {best_metrics['f1']:.4f}")
    print(f"Test F1:  {test_metrics['f1']:.4f}")
    print(f"Test P/R: {test_metrics['precision']:.4f}/{test_metrics['recall']:.4f}")

    target_met = test_metrics['f1'] >= 0.95
    print(f"\n>>> Target F1>=0.95: {'PASS' if target_met else 'FAIL'} <<<")

    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "synthetic_combined",
        "target_f1": 0.95,
        "target_met": target_met,
        "config": {
            "synthetic_ratio": 0.3,
            "synthetic_samples": len(combined_train['text']) - len(dataset['train']['text']),
            "use_dora": True,
            "all_linear_layers": True,
            "rank": 16,
            "epochs": 3,
        },
        "best_val_f1": best_metrics['f1'],
        "test_metrics": test_metrics,
        "train_time_s": train_time,
    }

    with open(trainer.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Log to progress
    with open("data/experiment_progress.tsv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tsynthetic_combined\tcompleted\t{json.dumps(results)}\n")

    return results


if __name__ == "__main__":
    main()
