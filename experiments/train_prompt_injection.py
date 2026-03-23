"""
Quick training script for prompt injection classifier.

Usage:
    python experiments/train_prompt_injection.py --model qwen-0.5b --epochs 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from guardrails.utils.data import PromptInjectionDataLoader, DatasetConfig
from guardrails.tier3.training import LoRATrainingConfig, LoRAClassifierTrainer


def main():
    parser = argparse.ArgumentParser(description="Train prompt injection classifier")
    parser.add_argument("--model", type=str, default="qwen-0.5b",
                       choices=["qwen-0.5b", "qwen-1.5b", "llama-3b"],
                       help="Model to use")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max samples for quick experiments")
    parser.add_argument("--output-dir", type=str, default="data/adapters/prompt_injection",
                       help="Output directory")
    args = parser.parse_args()

    # Model mapping
    model_map = {
        "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    }

    # Load dataset
    print("Loading dataset...")
    data_config = DatasetConfig(
        max_samples=args.max_samples,
        balance_classes=True,
    )
    loader = PromptInjectionDataLoader(config=data_config)
    dataset = loader.load_or_create_dataset(force_recreate=False)

    train_data = dataset["train"]
    eval_data = dataset["validation"]
    test_data = dataset["test"]

    print(f"Train: {len(train_data)}, Val: {len(eval_data)}, Test: {len(test_data)}")

    # Training config
    config = LoRATrainingConfig(
        model_name=model_map[args.model],
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    # Train
    trainer = LoRAClassifierTrainer(config)
    trainer.train(train_data, eval_data)

    # Test
    print("\nTesting on held-out test set...")
    test_texts = test_data["text"][:100]  # Sample for quick test
    test_labels = test_data["label"][:100]

    predictions = trainer.predict(test_texts)

    # Calculate metrics
    correct = sum(1 for p, l in zip(predictions, test_labels) if p["label"] == l)
    accuracy = correct / len(test_labels)

    print(f"Test Accuracy: {accuracy:.4f}")

    # Show some predictions
    print("\nSample predictions:")
    for i in range(5):
        print(f"\nText: {test_texts[i][:100]}...")
        print(f"True: {test_labels[i]}, Pred: {predictions[i]['prediction']} "
              f"(conf: {predictions[i]['confidence']:.3f})")


if __name__ == "__main__":
    main()
