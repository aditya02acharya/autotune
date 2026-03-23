"""
Train LoRA Adapter for Prompt Injection Detection on GPU

Optimized for RTX 4060 (8GB VRAM)
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
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from guardrails.utils.data import PromptInjectionDataLoader, DatasetConfig


def compute_metrics(preds, labels):
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


class PromptInjectionTrainer:
    """Trainer for prompt injection detection with LoRA adapters."""

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", output_dir="data/adapters/prompt_injection"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        self.model = None
        self.tokenizer = None
        self.classifier_head = None

    def setup_model(self, use_4bit=True):
        """Load model with LoRA adapters."""
        print(f"\nLoading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config for 8GB VRAM
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
        self.model.print_trainable_parameters()

        # Classification head
        hidden_size = self.model.config.hidden_size
        self.classifier_head = nn.Linear(hidden_size, 2).to(self.device)
        print(f"Classifier head: {hidden_size} -> 2")

    def tokenize_data(self, texts, labels, max_length=256):
        """Tokenize texts for training."""
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
        """Train for one epoch."""
        self.model.train()
        self.classifier_head.train()

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
            batch_size = input_ids.shape[0]
            last_token_idx = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[torch.arange(batch_size, device=self.device), last_token_idx]

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
        metrics["loss"] = total_loss / len(dataloader)
        return metrics

    def evaluate(self, dataloader):
        """Evaluate the model."""
        self.model.eval()
        self.classifier_head.eval()

        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(dataloader, desc="Evaluating"):
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
                loss = nn.CrossEntropyLoss()(logits, labels)
                total_loss += loss.item()

                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        metrics = compute_metrics(all_preds, all_labels)
        metrics["loss"] = total_loss / len(dataloader)
        return metrics

    def train(self, train_data, val_data, epochs=3, batch_size=8, lr=1e-3):
        """Full training loop."""
        print(f"\nTraining for {epochs} epochs, batch_size={batch_size}, lr={lr}")

        # Prepare data
        print("Preparing data...")
        train_encoded = self.tokenize_data(
            train_data["text"], train_data["label"]
        )
        val_encoded = self.tokenize_data(
            val_data["text"], val_data["label"]
        )

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

        # Optimizer for classifier head only (LoRA params are frozen for speed)
        optimizer = torch.optim.AdamW(self.classifier_head.parameters(), lr=lr)

        best_f1 = 0
        best_metrics = None

        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*60}")

            train_metrics = self.train_epoch(train_loader, optimizer, epoch)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, "
                  f"Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}")

            val_metrics = self.evaluate(val_loader)
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, "
                  f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_metrics = val_metrics.copy()
                best_metrics['epoch'] = epoch
                self.save(epoch)

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Val F1: {best_f1:.4f} (Epoch {best_metrics['epoch']})")
        print(f"{'='*60}")

        return best_metrics

    def save(self, epoch):
        """Save model checkpoint."""
        save_dir = self.output_dir / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_dir / "adapter")
        self.tokenizer.save_pretrained(save_dir / "adapter")
        torch.save(self.classifier_head.state_dict(), save_dir / "classifier_head.pt")

        print(f"Saved checkpoint to {save_dir}")

    def predict(self, texts, batch_size=16):
        """Predict on new texts."""
        self.model.eval()
        self.classifier_head.eval()

        all_preds = []
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = self.tokenize_data(batch_texts, [0]*len(batch_texts))

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
                probs = torch.softmax(logits, dim=-1)

                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)


def main():
    print("="*70)
    print("LoRA Adapter Training - Prompt Injection Detection (GPU)")
    print("="*70)

    # Load dataset
    print("\n1. Loading dataset...")
    config = DatasetConfig(max_samples=10000, balance_classes=True)
    loader = PromptInjectionDataLoader(config=config)
    dataset = loader.load_or_create_dataset(force_recreate=False)

    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    print(f"   Train: {len(train_data)} samples")
    print(f"   Val: {len(val_data)} samples")
    print(f"   Test: {len(test_data)} samples")

    # Create trainer
    print("\n2. Creating trainer...")
    trainer = PromptInjectionTrainer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        output_dir="data/adapters/prompt_injection"
    )
    trainer.setup_model(use_4bit=True)

    # Train
    print("\n3. Training...")
    best_metrics = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=3,
        batch_size=8,
        lr=1e-3,
    )

    # Test
    print("\n4. Testing on held-out set...")
    test_texts = test_data["text"]
    test_labels = np.array(test_data["label"])

    preds, probs = trainer.predict(test_texts)
    test_metrics = compute_metrics(preds, test_labels)

    print(f"\nTest Results:")
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"   F1:        {test_metrics['f1']:.4f}")

    # Save final results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": trainer.model_name,
        "best_val_f1": best_metrics['f1'],
        "test_metrics": test_metrics,
        "config": {
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
        }
    }

    with open("data/adapters/prompt_injection/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Log to experiment progress
    with open("data/experiment_progress.tsv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tlora_gpu_train\tcompleted\t{json.dumps(results)}\n")

    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
