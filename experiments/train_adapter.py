"""
Train LoRA Adapter for Prompt Injection Detection

This script trains a LoRA adapter using the processed dataset.
Target: F1 >= 0.87 on test set
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from guardrails.utils.data import PromptInjectionDataLoader
from guardrails.tier3.training import LoRATrainingConfig, LoRAClassifierTrainer


def main():
    print("=" * 60)
    print("Training LoRA Adapter for Prompt Injection Detection")
    print("=" * 60)

    # Load dataset
    print("\n1. Loading dataset...")
    loader = PromptInjectionDataLoader()
    dataset = loader.load_or_create_dataset(force_recreate=False)

    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    print(f"   Train: {len(train_data)} samples")
    print(f"   Val: {len(val_data)} samples")
    print(f"   Test: {len(test_data)} samples")

    # Training config - use Qwen2.5-0.5B-Instruct for speed
    config = LoRATrainingConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lora_r=16,
        lora_alpha=32,
        max_length=256,  # Shorter for speed
        output_dir="data/adapters/prompt_injection",
    )

    print(f"\n2. Training config:")
    print(f"   Model: {config.model_name}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   LoRA rank: {config.lora_r}")

    # Train
    print("\n3. Starting training...")
    trainer = LoRAClassifierTrainer(config)
    trainer.train(train_data, val_data)

    print("\n4. Training complete!")
    print(f"   Adapter saved to: {config.output_dir}")

    # Test
    print("\n5. Testing on held-out test set...")
    test_texts = test_data["text"][:100]
    test_labels = test_data["label"][:100]

    predictions = trainer.predict(test_texts)

    correct = sum(1 for p, l in zip(predictions, test_labels) if p["label"] == l)
    accuracy = correct / len(test_labels)

    # Calculate F1
    tp = sum(1 for p, l in zip(predictions, test_labels) if p["label"] == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, test_labels) if p["label"] == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, test_labels) if p["label"] == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n   Test Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1: {f1:.4f}")

    # Show some predictions
    print("\n6. Sample predictions:")
    for i in range(5):
        print(f"\n   Text: {test_texts[i][:80]}...")
        print(f"   True: {test_labels[i]}, Pred: {predictions[i]['prediction']} "
              f"(conf: {predictions[i]['confidence']:.3f})")


if __name__ == "__main__":
    main()
