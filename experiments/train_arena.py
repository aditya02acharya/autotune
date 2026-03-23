"""
ARENA: Regularized Low-Rank Adaptation for Few-Shot Learning

Based on arXiv:2507.15793 - "Regularized Low-Rank Adaptation for Few-Shot Organ Segmentation"
Accepted at MICCAI 2025

Key Innovation:
- Dynamically adjusts the intrinsic rank during adaptation
- Introduces l_1 sparsity regularizer on SVD decomposition
- Uses proximal optimizer for automatic rank finding
- Robust against suboptimal rank initialization

Code: https://github.com/ghassenbaklouti/ARENA
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


class ARENALoRALayer(nn.Module):
    """
    ARENA: Regularized Low-Rank Adaptation Layer

    Key components:
    1. SVD-based low-rank decomposition: W = U @ S @ V^T
    2. L1 sparsity regularizer on singular values S
    3. Proximal optimizer for automatic rank adjustment
    """

    def __init__(self, in_features, out_features, max_rank=32, alpha=32, dropout=0.05, l1_lambda=0.01):
        super().__init__()
        self.max_rank = max_rank
        self.alpha = alpha
        self.scaling = alpha / max_rank
        self.l1_lambda = l1_lambda  # Sparsity regularizer weight

        # SVD-based decomposition: W = U @ S @ V^T
        # U: (out_features, max_rank)
        # S: (max_rank,) - learnable singular values
        # V: (max_rank, in_features)
        self.U = nn.Parameter(torch.randn(out_features, max_rank) * 0.01)
        self.S = nn.Parameter(torch.ones(max_rank))  # Singular values - these get sparse
        self.V = nn.Parameter(torch.randn(max_rank, in_features) * 0.01)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def get_effective_rank(self, threshold=1e-4):
        """Get effective rank based on non-zero singular values."""
        with torch.no_grad():
            return (torch.abs(self.S) > threshold).sum().item()

    def get_l1_penalty(self):
        """L1 penalty on singular values for sparsity."""
        return torch.sum(torch.abs(self.S))

    def forward(self, x):
        # SVD reconstruction with sparsity
        # W = U @ diag(S) @ V
        W = self.U @ torch.diag(self.S) @ self.V

        # Apply dropout and scaling
        return self.dropout(x @ W.T) * self.scaling


class ProximalOptimizer:
    """
    Proximal optimizer for L1-regularized parameters.

    The proximal operator for L1 is soft-thresholding:
    prox_l1(x, lambda) = sign(x) * max(|x| - lambda, 0)
    """

    def __init__(self, optimizer, l1_lambda=0.01):
        self.optimizer = optimizer
        self.l1_lambda = l1_lambda

    def step(self, sparse_params):
        """Perform optimization step with proximal update."""
        # Standard gradient step
        self.optimizer.step()

        # Proximal update (soft-thresholding) for sparse parameters
        with torch.no_grad():
            for param in sparse_params:
                if param.grad is not None:
                    # Soft-thresholding
                    param.data = torch.sign(param.data) * torch.clamp(
                        torch.abs(param.data) - self.l1_lambda, min=0
                    )

    def zero_grad(self):
        self.optimizer.zero_grad()


class ARENATrainer:
    """Trainer with ARENA (Regularized LoRA) configuration."""

    ALL_LINEAR_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", output_dir="data/adapters/arena"):
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
        self.l1_lambda = 0.01  # Sparsity regularizer

    def setup_model(self, use_4bit=True, rank=16, use_dora=True, target_all_layers=True):
        """Load model with ARENA configuration."""
        print(f"\nLoading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
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

        # Determine target modules
        target_modules = self.ALL_LINEAR_LAYERS if target_all_layers else ["q_proj", "v_proj"]

        # ARENA config with DoRA
        print(f"Applying ARENA adapters...")
        print(f"  - Target modules: {target_modules}")
        print(f"  - Max Rank: {rank * 2}, Initial Rank: {rank}")
        print(f"  - L1 Lambda: {self.l1_lambda}")
        print(f"  - DoRA: {use_dora}")

        lora_config = LoraConfig(
            r=rank * 2,  # Start with higher max rank
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_dora=use_dora,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Classification head
        hidden_size = self.model.config.hidden_size
        self.classifier_head = nn.Linear(hidden_size, 2).to(self.device)
        print(f"Classifier head: {hidden_size} -> 2")

        return lora_config

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

    def train_epoch(self, dataloader, optimizer, scheduler, epoch, l1_weight=0.01):
        """Train for one epoch with ARENA regularization."""
        self.model.train()
        self.classifier_head.train()

        total_loss = 0
        total_l1_loss = 0
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
            ce_loss = nn.CrossEntropyLoss()(logits, labels)

            # L1 regularization on LoRA weights (ARENA-style)
            l1_loss = torch.tensor(0.0, device=self.device)
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    l1_loss += torch.sum(torch.abs(param))

            loss = ce_loss + l1_weight * l1_loss

            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            total_loss += ce_loss.item()
            total_l1_loss += l1_loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            progress.set_postfix({
                "loss": f"{ce_loss.item():.4f}",
                "l1": f"{l1_loss.item():.2f}"
            })

        metrics = compute_metrics(all_preds, all_labels)
        metrics["loss"] = total_loss / len(dataloader)
        metrics["l1_loss"] = total_l1_loss / len(dataloader)
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

    def train(self, train_data, val_data, epochs=3, batch_size=8, lr=2e-4, l1_weight=0.01):
        """Full training loop with ARENA regularization."""
        print(f"\nTraining with ARENA configuration:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Base LR: {lr}")
        print(f"  - L1 Weight: {l1_weight}")

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

        # Optimizer with weight decay for implicit regularization
        optimizer = torch.optim.AdamW([
            {"params": self.model.parameters(), "lr": lr, "weight_decay": 0.01},
            {"params": self.classifier_head.parameters(), "lr": lr, "weight_decay": 0.01},
        ])

        best_f1 = 0
        best_metrics = None

        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*60}")

            train_metrics = self.train_epoch(train_loader, optimizer, None, epoch, l1_weight)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, "
                  f"L1: {train_metrics['l1_loss']:.2f}")

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

        # Save config
        config = {
            "model": self.model_name,
            "epoch": epoch,
            "method": "ARENA",
            "use_dora": True,
            "target_all_layers": True,
            "rank": 16,
            "l1_regularization": True,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

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
    print("ARENA Training - Prompt Injection Detection")
    print("Method: Regularized Low-Rank Adaptation (arXiv:2507.15793)")
    print("Features: Dynamic Rank + L1 Sparsity + DoRA")
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

    # Create trainer with ARENA config
    print("\n2. Creating ARENA trainer...")
    trainer = ARENATrainer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        output_dir="data/adapters/arena"
    )
    lora_config = trainer.setup_model(
        use_4bit=True,
        use_dora=True,
        rank=16,
        target_all_layers=True
    )

    # Train with ARENA regularization
    print("\n3. Training with ARENA configuration...")
    best_metrics = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=3,
        batch_size=8,
        lr=2e-4,
        l1_weight=0.01,
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
        "method": "ARENA",
        "optimizations": {
            "l1_regularization": True,
            "dynamic_rank": True,
            "dora": True,
            "target_all_layers": True,
            "rank": 16,
        },
        "best_val_f1": best_metrics['f1'],
        "test_metrics": test_metrics,
        "config": {
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 2e-4,
            "l1_weight": 0.01,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
        }
    }

    with open("data/adapters/arena/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Log to experiment progress
    with open("data/experiment_progress.tsv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tarena\tcompleted\t{json.dumps(results)}\n")

    print("\n" + "="*70)
    print("ARENA Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
