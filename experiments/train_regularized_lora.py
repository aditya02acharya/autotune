"""
Regularized LoRA Training for Prompt Injection Detection

Key regularizations:
1. DoRA (Weight Decomposition)
2. LoRA+ (16x higher LR for B matrices)
3. Higher dropout (0.15)
4. Higher weight decay (0.05)
5. Label smoothing (0.1)
6. Gradient clipping (1.0)
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


class RegularizedClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes=2, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))


class RegularizedLoRATrainer:
    ALL_LINEAR_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", output_dir="data/adapters/regularized_lora", force_cuda=True):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        if force_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        self.model = None
        self.tokenizer = None
        self.classifier_head = None

    def setup_model(self, use_4bit=True, rank=16, use_dora=True, lora_dropout=0.15):
        print(f"\nLoading model: {self.model_name}")
        print(f"Target device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device.type == "cuda":
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
            # CPU fallback - no quantization
            print("WARNING: Running on CPU - training will be slower")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )

        print(f"Applying LoRA adapters...")
        print(f"  - Target modules: {self.ALL_LINEAR_LAYERS}")
        print(f"  - Rank: {rank}, Alpha: {rank * 2}")
        print(f"  - Dropout: {lora_dropout}")
        print(f"  - DoRA: {use_dora}")

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=lora_dropout,
            target_modules=self.ALL_LINEAR_LAYERS,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_dora=use_dora,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        hidden_size = self.model.config.hidden_size
        self.classifier_head = RegularizedClassifier(hidden_size, dropout=0.2).to(self.device)
        print(f"Classifier: {hidden_size} -> 2 (dropout=0.2)")

        return lora_config

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

    def get_lora_plus_param_groups(self, base_lr, lora_plus_lr_ratio=16, weight_decay=0.05):
        """Create parameter groups with LoRA+: B matrices get higher LR."""
        lora_A_params = []
        lora_B_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'lora_A' in name:
                lora_A_params.append(param)
            elif 'lora_B' in name:
                lora_B_params.append(param)
            else:
                other_params.append(param)

        print(f"  - LoRA A params: {sum(p.numel() for p in lora_A_params):,}")
        print(f"  - LoRA B params: {sum(p.numel() for p in lora_B_params):,} (LR: {base_lr * lora_plus_lr_ratio:.6f})")
        print(f"  - Other params: {sum(p.numel() for p in other_params):,}")

        return [
            {"params": lora_A_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": lora_B_params, "lr": base_lr * lora_plus_lr_ratio, "weight_decay": weight_decay},  # LoRA+
            {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": self.classifier_head.parameters(), "lr": base_lr, "weight_decay": weight_decay},
        ]

    def train_epoch(self, dataloader, optimizer, epoch, criterion, max_grad_norm=1.0):
        self.model.train()
        self.classifier_head.train()

        total_loss = 0
        all_preds = []
        all_labels = []

        progress = tqdm(dataloader, desc=f"Epoch {epoch}", disable=True)
        n_batches = len(dataloader)
        report_at = [n_batches // 4, n_batches // 2, 3 * n_batches // 4]
        for i, (input_ids, attention_mask, labels) in enumerate(progress):
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
            loss = criterion(logits, labels)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.classifier_head.parameters(), max_grad_norm)

            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # Periodic status (25%, 50%, 75%)
            if i in report_at:
                print(f"  [{100*(i+1)//n_batches}%] loss={loss.item():.4f}", flush=True)

        metrics = compute_metrics(all_preds, all_labels)
        metrics["loss"] = total_loss / len(dataloader)
        return metrics

    def evaluate(self, dataloader):
        self.model.eval()
        self.classifier_head.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, labels in dataloader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

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
                all_labels.extend(labels.numpy())

        return compute_metrics(all_preds, all_labels)

    def train(self, train_data, val_data, epochs=5, batch_size=8, lr=2e-4,
              lora_plus_ratio=16, weight_decay=0.05, label_smoothing=0.1, patience=2):
        print(f"\nTraining with regularized settings:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Base LR: {lr}")
        print(f"  - LoRA+ ratio: {lora_plus_ratio}x for B matrices")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - Label smoothing: {label_smoothing}")
        print(f"  - Early stopping patience: {patience}")

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

        # LoRA+ optimizer with separate LRs
        print("\nSetting up LoRA+ optimizer:")
        param_groups = self.get_lora_plus_param_groups(lr, lora_plus_ratio, weight_decay)
        optimizer = torch.optim.AdamW(param_groups)

        # Label smoothing loss
        criterion = LabelSmoothingLoss(smoothing=label_smoothing)

        best_f1 = 0
        best_metrics = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*60}")

            train_metrics = self.train_epoch(train_loader, optimizer, epoch, criterion)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")

            val_metrics = self.evaluate(val_loader)
            print(f"Val   - F1: {val_metrics['f1']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_metrics = val_metrics.copy()
                best_metrics['epoch'] = epoch
                patience_counter = 0
                self.save(epoch)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Val F1: {best_f1:.4f} (Epoch {best_metrics['epoch']})")
        print(f"{'='*60}")

        return best_metrics

    def save(self, epoch):
        save_dir = self.output_dir / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_dir / "adapter")
        self.tokenizer.save_pretrained(save_dir / "adapter")
        torch.save(self.classifier_head.state_dict(), save_dir / "classifier_head.pt")

        config = {
            "model": self.model_name,
            "epoch": epoch,
            "method": "Regularized LoRA",
            "regularization": {
                "dora": True,
                "lora_plus": True,
                "lora_plus_ratio": 16,
                "lora_dropout": 0.15,
                "weight_decay": 0.05,
                "label_smoothing": 0.1,
                "classifier_dropout": 0.2,
                "grad_clip": 1.0,
            },
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved checkpoint to {save_dir}")

    def predict(self, texts, batch_size=16):
        self.model.eval()
        self.classifier_head.eval()

        all_preds = []

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
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())

        return np.array(all_preds)


def main():
    print("="*70)
    print("Regularized LoRA Training - Prompt Injection Detection")
    print("="*70)
    print("\nRegularization Stack:")
    print("  1. DoRA (Weight Decomposition)")
    print("  2. LoRA+ (16x LR for B matrices)")
    print("  3. Higher dropout (0.15)")
    print("  4. Higher weight decay (0.05)")
    print("  5. Label smoothing (0.1)")
    print("  6. Gradient clipping (1.0)")
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
    trainer = RegularizedLoRATrainer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        output_dir="data/adapters/regularized_lora"
    )
    trainer.setup_model(
        use_4bit=True,
        use_dora=True,
        rank=16,
        lora_dropout=0.15
    )

    # Train
    print("\n3. Training...")
    best_metrics = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=5,
        batch_size=8,
        lr=2e-4,
        lora_plus_ratio=16,
        weight_decay=0.05,
        label_smoothing=0.1,
        patience=2
    )

    # Test
    print("\n4. Testing on held-out set...")
    test_texts = test_data["text"]
    test_labels = np.array(test_data["label"])

    preds = trainer.predict(test_texts)
    test_metrics = compute_metrics(preds, test_labels)

    print(f"\nTest Results:")
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"   F1:        {test_metrics['f1']:.4f}")

    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": trainer.model_name,
        "method": "Regularized LoRA",
        "regularization": {
            "dora": True,
            "lora_plus": True,
            "lora_plus_ratio": 16,
            "lora_dropout": 0.15,
            "weight_decay": 0.05,
            "label_smoothing": 0.1,
            "classifier_dropout": 0.2,
            "grad_clip": 1.0,
        },
        "best_val_f1": best_metrics['f1'],
        "test_metrics": test_metrics,
        "config": {
            "epochs": 5,
            "batch_size": 8,
            "learning_rate": 2e-4,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
        }
    }

    with open("data/adapters/regularized_lora/results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("data/experiment_progress.tsv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tregularized_lora\tcompleted\t{json.dumps(results)}\n")

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
