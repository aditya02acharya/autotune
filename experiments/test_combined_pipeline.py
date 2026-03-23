"""
Test Combined Deterministic + LoRA Pipeline

Experiment 6: Evaluate the full Tier 2 + Tier 3 pipeline
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from guardrails.tier2.deterministic import Tier2Deterministic, FilterResult
from guardrails.utils.data import PromptInjectionDataLoader, DatasetConfig


def compute_metrics(preds, labels):
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


class LoRAInference:
    """Inference with trained LoRA adapter."""

    def __init__(self, adapter_path: str, base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.adapter_path = Path(adapter_path)
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None
        self.classifier_head = None
        self._load_model()

    def _load_model(self):
        """Load model with LoRA adapter."""
        print(f"Loading base model: {self.base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path / "adapter",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        if self.device.type == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )

        # Load LoRA adapter
        print(f"Loading LoRA adapter from: {self.adapter_path / 'adapter'}")
        self.model = PeftModel.from_pretrained(base, self.adapter_path / "adapter")

        # Load classifier head
        hidden_size = self.model.config.hidden_size
        self.classifier_head = nn.Linear(hidden_size, 2).to(self.device)
        self.classifier_head.load_state_dict(
            torch.load(self.adapter_path / "classifier_head.pt", map_location=self.device)
        )
        print(f"Loaded classifier head: {hidden_size} -> 2")

    def predict(self, texts, batch_size=16):
        """Predict on texts."""
        self.model.eval()
        self.classifier_head.eval()

        all_preds = []
        all_probs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="LoRA inference"):
                batch_texts = texts[i:i+batch_size]

                # Tokenize
                prompts = [
                    f"Classify if this prompt is an injection attack or benign.\n\nPrompt: {text[:500]}\n\nAnswer:"
                    for text in batch_texts
                ]
                encoded = self.tokenizer(
                    prompts,
                    max_length=256,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                # Get hidden states
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]

                # Get last token representation
                batch_size_curr = input_ids.shape[0]
                last_token_idx = attention_mask.sum(dim=1) - 1
                last_hidden = hidden_states[torch.arange(batch_size_curr, device=self.device), last_token_idx]

                # Classify
                logits = self.classifier_head(last_hidden.float())
                probs = torch.softmax(logits, dim=-1)

                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)


class CombinedPipeline:
    """Combined Deterministic + LoRA pipeline."""

    def __init__(self, adapter_path: str, base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.deterministic = Tier2Deterministic()
        self.lora = LoRAInference(adapter_path, base_model)

        # Stats tracking
        self.stats = {
            "deterministic_blocks": 0,
            "lora_blocks": 0,
            "total": 0,
        }

    def predict(self, texts, batch_size=16):
        """Predict using combined pipeline."""
        self.stats = {"deterministic_blocks": 0, "lora_blocks": 0, "total": len(texts)}

        # First pass: deterministic filter
        det_results = []
        needs_lora = []

        for i, text in enumerate(texts):
            result = self.deterministic.check(text)
            if result["injection"].blocked:
                det_results.append((i, 1))  # Blocked = injection
                self.stats["deterministic_blocks"] += 1
            else:
                needs_lora.append(i)

        # Second pass: LoRA for texts that passed deterministic
        lora_preds = []
        lora_probs = []
        if needs_lora:
            texts_for_lora = [texts[i] for i in needs_lora]
            lora_preds, lora_probs = self.lora.predict(texts_for_lora, batch_size)

        # Combine results
        final_preds = [0] * len(texts)
        final_probs = [[1.0, 0.0]] * len(texts)  # Default: benign

        # Fill in deterministic blocks
        for idx, pred in det_results:
            final_preds[idx] = pred
            final_probs[idx] = [0.0, 1.0]  # High confidence injection

        # Fill in LoRA predictions
        for i, orig_idx in enumerate(needs_lora):
            final_preds[orig_idx] = lora_preds[i]
            final_probs[orig_idx] = lora_probs[i].tolist()
            if lora_preds[i] == 1:
                self.stats["lora_blocks"] += 1

        return np.array(final_preds), np.array(final_probs)

    def benchmark_latency(self, texts, n_samples=100):
        """Benchmark latency for the pipeline."""
        latencies = {
            "deterministic": [],
            "lora": [],
            "total": [],
        }

        sample_texts = texts[:n_samples]

        # Benchmark deterministic
        for text in sample_texts:
            start = time.perf_counter()
            _ = self.deterministic.check(text)
            latencies["deterministic"].append((time.perf_counter() - start) * 1000)

        # Benchmark LoRA (on texts that pass deterministic)
        texts_for_lora = []
        for text in sample_texts:
            result = self.deterministic.check(text)
            if not result["injection"].blocked:
                texts_for_lora.append(text)

        if texts_for_lora:
            # Measure per-sample latency
            for text in texts_for_lora[:50]:  # Limit to 50 for time
                start = time.perf_counter()
                _, _ = self.lora.predict([text], batch_size=1)
                latencies["lora"].append((time.perf_counter() - start) * 1000)

        # Calculate total latency (deterministic + conditional LoRA)
        det_p50 = np.percentile(latencies["deterministic"], 50)
        det_p95 = np.percentile(latencies["deterministic"], 95)

        if latencies["lora"]:
            lora_p50 = np.percentile(latencies["lora"], 50)
            lora_p95 = np.percentile(latencies["lora"], 95)
        else:
            lora_p50, lora_p95 = 0, 0

        # Estimate combined latency (det always runs, lora runs ~93% of time based on det recall)
        det_block_rate = self.stats["deterministic_blocks"] / max(1, self.stats["total"])
        combined_p50 = det_p50 + lora_p50 * (1 - det_block_rate)
        combined_p95 = det_p95 + lora_p95 * (1 - det_block_rate)

        return {
            "deterministic": {"p50_ms": det_p50, "p95_ms": det_p95, "mean_ms": np.mean(latencies["deterministic"])},
            "lora": {"p50_ms": lora_p50, "p95_ms": lora_p95, "mean_ms": np.mean(latencies["lora"]) if latencies["lora"] else 0},
            "combined": {"p50_ms": combined_p50, "p95_ms": combined_p95},
            "det_block_rate": det_block_rate,
        }


def main():
    print("="*70)
    print("Experiment 6: Combined Deterministic + LoRA Pipeline")
    print("="*70)

    # Load test data
    print("\n1. Loading test data...")
    config = DatasetConfig(max_samples=10000, balance_classes=True)
    loader = PromptInjectionDataLoader(config=config)
    dataset = loader.load_or_create_dataset(force_recreate=False)

    test_data = dataset["test"]
    test_texts = test_data["text"]
    test_labels = np.array(test_data["label"])

    print(f"   Test samples: {len(test_texts)}")

    # Create combined pipeline
    print("\n2. Creating combined pipeline...")
    adapter_path = "data/adapters/prompt_injection/epoch_3"
    pipeline = CombinedPipeline(adapter_path)

    # Run predictions
    print("\n3. Running predictions...")
    preds, probs = pipeline.predict(test_texts, batch_size=16)

    # Compute metrics
    metrics = compute_metrics(preds, test_labels)

    print(f"\n{'='*60}")
    print("RESULTS: Combined Deterministic + LoRA Pipeline")
    print(f"{'='*60}")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1:        {metrics['f1']:.4f}")
    print(f"\n   Pipeline Stats:")
    print(f"   - Deterministic blocks: {pipeline.stats['deterministic_blocks']} ({100*pipeline.stats['deterministic_blocks']/len(test_texts):.1f}%)")
    print(f"   - LoRA blocks:          {pipeline.stats['lora_blocks']} ({100*pipeline.stats['lora_blocks']/len(test_texts):.1f}%)")

    # Classification report
    print(f"\n   Classification Report:")
    print(classification_report(test_labels, preds, target_names=["Benign", "Injection"]))

    # Benchmark latency
    print("\n4. Benchmarking latency...")
    latency_results = pipeline.benchmark_latency(test_texts, n_samples=100)

    print(f"\n   Latency Benchmarks:")
    print(f"   Deterministic: p50={latency_results['deterministic']['p50_ms']:.1f}ms, p95={latency_results['deterministic']['p95_ms']:.1f}ms")
    print(f"   LoRA:          p50={latency_results['lora']['p50_ms']:.1f}ms, p95={latency_results['lora']['p95_ms']:.1f}ms")
    print(f"   Combined:      p50={latency_results['combined']['p50_ms']:.1f}ms, p95={latency_results['combined']['p95_ms']:.1f}ms")
    print(f"   (Det block rate: {100*latency_results['det_block_rate']:.1f}%)")

    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "combined_det_lora",
        "test_metrics": metrics,
        "pipeline_stats": pipeline.stats,
        "latency": latency_results,
        "config": {
            "adapter_path": adapter_path,
            "test_samples": len(test_texts),
        }
    }

    with open("data/experiment_progress.tsv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tcombined_det_lora\tcompleted\t{json.dumps(results)}\n")

    # Save detailed results
    with open("data/adapters/prompt_injection/combined_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("Experiment 6 Complete!")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    main()
