"""
Quick Training Experiment for Prompt Injection Detection

Tests the LoRA training pipeline with a small model and dataset.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from guardrails.utils.data import PromptInjectionDataLoader, DatasetConfig
from guardrails.tier2.deterministic import Tier2Deterministic
from guardrails.tier2.neural_pre_screen import NeuralPreScreen


# Metrics tracking
PROGRESS_FILE = Path("data/experiment_progress.tsv")


def log_experiment(name: str, status: str, metrics: dict = None):
    """Log experiment progress to TSV file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics_str = json.dumps(metrics) if metrics else "{}"

    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{timestamp}\t{name}\t{status}\t{metrics_str}\n")

    print(f"[{timestamp}] {name}: {status}")


def test_deterministic_only(test_data):
    """Test deterministic filter only (Tier 2)."""
    print("\n=== Experiment: Deterministic Filter Only ===")
    start_time = time.time()

    det = Tier2Deterministic()

    correct = 0
    total = len(test_data)
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for sample in test_data:
        text = sample["text"]
        true_label = sample["label"]

        result = det.check(text)
        predicted_blocked = result["summary"]["blocked"]

        # Blocked = injection (1), Not blocked = benign (0)
        predicted_label = 1 if predicted_blocked else 0

        if predicted_label == true_label:
            correct += 1
            if predicted_label == 1:
                true_positives += 1
        else:
            if predicted_label == 1:
                false_positives += 1
            else:
                false_negatives += 1

    latency = time.time() - start_time
    accuracy = correct / total

    # Calculate F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "latency_s": latency,
        "samples": total,
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Latency: {latency:.2f}s ({latency*1000/total:.2f}ms/sample)")

    log_experiment("deterministic_only", "completed", metrics)
    return metrics


def test_neural_pre_screen(test_data, model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Test neural pre-screen (Tier 2 neural)."""
    print(f"\n=== Experiment: Neural Pre-Screen ({model}) ===")
    start_time = time.time()

    screen = NeuralPreScreen(model_name=model, threshold=0.5)

    correct = 0
    total = min(len(test_data), 500)  # Limit for speed
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i, sample in enumerate(test_data.select(range(total))):
        text = sample["text"]
        true_label = sample["label"]

        result = screen.check(text)
        predicted_label = 1 if result.score >= 0.5 else 0

        if predicted_label == true_label:
            correct += 1
            if predicted_label == 1:
                true_positives += 1
        else:
            if predicted_label == 1:
                false_positives += 1
            else:
                false_negatives += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total}...")

    latency = time.time() - start_time
    accuracy = correct / total

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "latency_s": latency,
        "samples": total,
        "model": model,
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Avg latency: {latency*1000/total:.2f}ms/sample")

    log_experiment("neural_pre_screen", "completed", metrics)
    return metrics


def test_combined(test_data):
    """Test deterministic + neural combined."""
    print("\n=== Experiment: Combined (Deterministic + Neural) ===")
    start_time = time.time()

    det = Tier2Deterministic()
    screen = NeuralPreScreen(threshold=0.5)

    total = min(len(test_data), 500)
    correct = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i, sample in enumerate(test_data.select(range(total))):
        text = sample["text"]
        true_label = sample["label"]

        # Run deterministic first
        det_result = det.check(text)

        if det_result["summary"]["blocked"]:
            predicted_label = 1
        else:
            # Fall back to neural
            neural_result = screen.check(text)
            predicted_label = 1 if neural_result.score >= 0.5 else 0

        if predicted_label == true_label:
            correct += 1
            if predicted_label == 1:
                true_positives += 1
        else:
            if predicted_label == 1:
                false_positives += 1
            else:
                false_negatives += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total}...")

    latency = time.time() - start_time
    accuracy = correct / total

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "latency_s": latency,
        "samples": total,
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Avg latency: {latency*1000/total:.2f}ms/sample")

    log_experiment("combined_det_neural", "completed", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Quick experiments")
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["det", "neural", "combined", "all"])
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")
    loader = PromptInjectionDataLoader()
    dataset = loader.load_or_create_dataset(force_recreate=False)

    test_data = dataset["test"]
    print(f"Test set: {len(test_data)} samples")

    # Run experiments
    if args.experiment in ["det", "all"]:
        test_deterministic_only(test_data)

    if args.experiment in ["neural", "all"]:
        test_neural_pre_screen(test_data, args.model)

    if args.experiment in ["combined", "all"]:
        test_combined(test_data)

    print("\n=== Experiments Complete ===")
    print(f"Progress logged to: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
