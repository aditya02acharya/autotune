"""
Quick Training Experiment - Fast version

Uses smaller dataset and simplified training for rapid iteration.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from guardrails.utils.data import PromptInjectionDataLoader, DatasetConfig


class SimpleClassifier(nn.Module):
    """Simple classifier head on top of language model."""

    def __init__(self, model, hidden_size, num_labels=2):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get last token hidden state
        hidden_states = outputs.hidden_states[-1]

        # Get last non-padding token
        if attention_mask is not None:
            last_token_idx = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[torch.arange(hidden_states.shape[0]), last_token_idx]
        else:
            last_hidden = hidden_states[:, -1, :]

        # Classify
        logits = self.classifier(last_hidden)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


def main():
    print("=" * 60)
    print("Quick Training Experiment - Prompt Injection Detection")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load small dataset
    print("\n1. Loading dataset...")
    config = DatasetConfig(max_samples=2000, balance_classes=True)
    loader = PromptInjectionDataLoader(config=config)
    dataset = loader.load_or_create_dataset(force_recreate=False)

    train_data = dataset["train"].select(range(min(1000, len(dataset["train"]))))
    val_data = dataset["validation"].select(range(min(200, len(dataset["validation"]))))
    test_data = dataset["test"].select(range(min(200, len(dataset["test"]))))

    print(f"   Train: {len(train_data)} samples")
    print(f"   Val: {len(val_data)} samples")
    print(f"   Test: {len(test_data)} samples")

    # Load model
    print("\n2. Loading model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    # Apply LoRA
    print("   Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Create classifier
    classifier = SimpleClassifier(model, model.config.hidden_size)
    classifier.to(device)

    # Tokenize data
    print("\n3. Tokenizing data...")

    def tokenize(batch):
        prompts = [
            f"Classify the following text as 'injection' (malicious) or 'benign' (safe):\n\nText: {text}\n\nClassification:"
            for text in batch["text"]
        ]
        tokenized = tokenizer(
            prompts,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized["labels"] = torch.tensor(batch["label"])
        return tokenized

    train_processed = train_data.map(tokenize, batched=True, remove_columns=train_data.column_names)
    val_processed = val_data.map(tokenize, batched=True, remove_columns=val_data.column_names)

    # Create dataloaders
    def collate_fn(features):
        return {
            "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
            "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in features]),
            "labels": torch.tensor([f["labels"] for f in features]),
        }

    train_loader = DataLoader(train_processed, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_processed, batch_size=4, collate_fn=collate_fn)

    # Train
    print("\n4. Training...")
    optimizer = torch.optim.AdamW(classifier.classifier.parameters(), lr=2e-4)

    num_epochs = 3
    for epoch in range(num_epochs):
        classifier.train()
        epoch_loss = 0

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = classifier(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )

            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"   Epoch {epoch+1}, Step {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"   Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        # Evaluate
        classifier.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                outputs = classifier(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        print(f"   Val Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")

    # Test
    print("\n5. Testing...")
    test_processed = test_data.map(tokenize, batched=True, remove_columns=test_data.column_names)
    test_loader = DataLoader(test_processed, batch_size=4, collate_fn=collate_fn)

    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = classifier(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds)
    test_recall = recall_score(all_labels, all_preds)

    print(f"\n   Test Results:")
    print(f"   Accuracy:  {test_acc:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f}")
    print(f"   F1:        {test_f1:.4f}")

    # Save
    print("\n6. Saving model...")
    output_dir = Path("data/adapters/prompt_injection_quick")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    torch.save(classifier.classifier.state_dict(), output_dir / "classifier_head.pt")

    print(f"   Saved to: {output_dir}")

    # Log experiment
    with open("data/experiment_progress.tsv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tlora_quick_train\tcompleted\t{{\"accuracy\": {test_acc}, \"f1\": {test_f1}, \"precision\": {test_precision}, \"recall\": {test_recall}}}\n")

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
