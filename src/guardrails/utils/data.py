"""
Data Processing Pipeline for Prompt Injection Detection

Processes raw data from multiple sources into training-ready format.
Supports train/val/test splits, class balancing, and data augmentation.
"""

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


@dataclass
class PromptInjectionSample:
    """A single prompt injection sample."""
    text: str
    label: int  # 0 = benign, 1 = injection
    source: str = "unknown"
    category: str = "uncategorized"
    metadata: dict = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""
    test_size: float = 0.15
    val_size: float = 0.1  # Of training after test split
    random_seed: int = 42
    max_samples: Optional[int] = None  # Limit samples for quick experiments
    balance_classes: bool = True
    min_text_length: int = 10
    max_text_length: int = 2048


class PromptInjectionDataLoader:
    """
    Loads and processes prompt injection data from multiple sources.

    Supported sources:
    - synapsecai: Synthetic prompt injections (252K train, 63K test)
    - mpdd: Multi-Prompt Detection Dataset (39K)
    - malignant: Malignant prompts (1.5K)
    - jailbreak_prompts: Jailbreak prompts (2K)
    - malicious_deepset: Malicious prompts
    - forbidden_questions: Forbidden questions
    - guychuk: Benign-malicious classification (464K)
    """

    def __init__(self, base_path: Path = None, config: DatasetConfig = None):
        self.base_path = base_path or Path(".")
        self.config = config or DatasetConfig()

    def load_synapsecai(self) -> list[PromptInjectionSample]:
        """Load SynapseCAI synthetic prompt injection dataset."""
        samples = []
        paths = {
            "train": self.base_path / "raw_data/synapsecai/synthetic-prompt-injections_train.parquet",
            "test": self.base_path / "raw_data/synapsecai/synthetic-prompt-injections_test.parquet",
        }

        for split, path in paths.items():
            if not path.exists():
                print(f"Warning: {path} not found, skipping")
                continue

            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                label_val = row["label"]
                # Handle both string and int labels
                if isinstance(label_val, str):
                    label = 1 if label_val == "1" else 0
                else:
                    label = int(label_val)

                samples.append(PromptInjectionSample(
                    text=str(row["text"]),
                    label=label,
                    source="synapsecai",
                    category=str(row.get("category", "uncategorized")),
                    metadata={"split": split}
                ))

        print(f"Loaded {len(samples)} samples from synapsecai")
        return samples

    def load_mpdd(self) -> list[PromptInjectionSample]:
        """Load MPDD (Multi-Prompt Detection Dataset)."""
        samples = []
        path = self.base_path / "raw_data/mpdd/MPDD.csv"

        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            return samples

        df = pd.read_csv(path)
        for _, row in df.iterrows():
            samples.append(PromptInjectionSample(
                text=str(row["Prompt"]),
                label=int(row["isMalicious"]),
                source="mpdd",
                category="mpdd",
                metadata={}
            ))

        print(f"Loaded {len(samples)} samples from mpdd")
        return samples

    def load_malignant(self) -> list[PromptInjectionSample]:
        """Load malignant dataset - contains category-classified prompts."""
        samples = []
        path = self.base_path / "raw_data/mpdd/malignant.csv"

        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            return samples

        df = pd.read_csv(path)
        for _, row in df.iterrows():
            # malignant has category-based classification
            # conversation = benign, others are malicious
            category = str(row.get("category", ""))
            base_class = str(row.get("base_class", ""))
            is_benign = category == "conversation" or base_class == "conversation"

            samples.append(PromptInjectionSample(
                text=str(row["text"]),
                label=0 if is_benign else 1,
                source="malignant",
                category=category,
                metadata={"base_class": base_class}
            ))

        print(f"Loaded {len(samples)} samples from malignant")
        return samples

    def load_jailbreak_prompts(self) -> list[PromptInjectionSample]:
        """Load jailbreak prompts - all are malicious."""
        samples = []
        path = self.base_path / "raw_data/mpdd/jailbreak_prompts.csv"

        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            return samples

        df = pd.read_csv(path)
        for _, row in df.iterrows():
            samples.append(PromptInjectionSample(
                text=str(row["Prompt"]),
                label=1,  # All jailbreak prompts are malicious
                source="jailbreak_prompts",
                category="jailbreak",
                metadata={}
            ))

        print(f"Loaded {len(samples)} samples from jailbreak_prompts")
        return samples

    def load_malicious_deepset(self) -> list[PromptInjectionSample]:
        """Load malicious deepset - all are malicious."""
        samples = []
        path = self.base_path / "raw_data/mpdd/malicous_deepset.csv"

        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            return samples

        df = pd.read_csv(path)
        for _, row in df.iterrows():
            samples.append(PromptInjectionSample(
                text=str(row["Prompt"]),
                label=1,  # All are malicious
                source="malicious_deepset",
                category="malicious",
                metadata={}
            ))

        print(f"Loaded {len(samples)} samples from malicious_deepset")
        return samples

    def load_forbidden_questions(self) -> list[PromptInjectionSample]:
        """Load forbidden questions - all are prompt injection/malicious."""
        samples = []
        path = self.base_path / "raw_data/mpdd/forbidden_question_set_df.csv"

        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            return samples

        df = pd.read_csv(path)
        # No binary label column - treat all as prompt injection
        for _, row in df.iterrows():
            prompt = row.get("Prompt", "")
            if pd.notna(prompt) and len(str(prompt).strip()) > 0:
                samples.append(PromptInjectionSample(
                    text=str(prompt),
                    label=1,  # Forbidden questions are prompt injection
                    source="forbidden_questions",
                    category="forbidden",
                    metadata={}
                ))

        print(f"Loaded {len(samples)} samples from forbidden_questions")
        return samples

    def load_predictionguard(self) -> list[PromptInjectionSample]:
        """Load predictionguard dataset - all are prompt injection/malicious."""
        samples = []
        path = self.base_path / "raw_data/mpdd/predictionguard_df.csv"

        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            return samples

        df = pd.read_csv(path)
        # No binary label column - treat all as prompt injection
        for _, row in df.iterrows():
            prompt = row.get("Prompt", "")
            if pd.notna(prompt) and len(str(prompt).strip()) > 0:
                samples.append(PromptInjectionSample(
                    text=str(prompt),
                    label=1,  # All are prompt injection
                    source="predictionguard",
                    category="predictionguard",
                    metadata={}
                ))

        print(f"Loaded {len(samples)} samples from predictionguard")
        return samples

    def load_regular_prompts(self) -> list[PromptInjectionSample]:
        """Load regular/benign prompts from TrustAIRLab dataset."""
        samples = []
        path = self.base_path / "raw_data/TrustAIRLab___in-the-wild-jailbreak-prompts/regular_2023_05_07/0.0.0/a10aab8eff1c73165a442d4464dce192bd28b9c5"

        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            return samples

        # Find arrow file
        arrow_files = [f for f in os.listdir(path) if f.endswith('.arrow')]
        if not arrow_files:
            return samples

        try:
            from datasets import Dataset
            ds = Dataset.from_file(os.path.join(path, arrow_files[0]))
            for item in ds:
                text = item.get("prompt", item.get("text", ""))
                if text:
                    samples.append(PromptInjectionSample(
                        text=str(text),
                        label=0,  # Regular prompts are benign
                        source="regular_prompts",
                        category="benign",
                        metadata={}
                    ))
            print(f"Loaded {len(samples)} samples from regular_prompts")
        except Exception as e:
            print(f"Error loading regular_prompts: {e}")

        return samples

    def load_guychuk(self) -> list[PromptInjectionSample]:
        """Load guychuk benign-malicious classification dataset."""
        samples = []
        path = self.base_path / "raw_data/guychuk___benign-malicious-prompt-classification/default/0.0.0/e13090643e6f018c4db239d82463a4fd9649d77c"

        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            return samples

        # Find the arrow file
        arrow_files = [f for f in os.listdir(path) if f.endswith('.arrow')]
        if not arrow_files:
            print(f"Warning: No arrow files in {path}")
            return samples

        try:
            from datasets import Dataset
            ds = Dataset.from_file(os.path.join(path, arrow_files[0]))
            for item in ds:
                samples.append(PromptInjectionSample(
                    text=str(item["prompt"]),
                    label=int(item["label"]),
                    source="guychuk",
                    category="guychuk",
                    metadata={}
                ))
            print(f"Loaded {len(samples)} samples from guychuk")
        except Exception as e:
            print(f"Error loading guychuk: {e}")

        return samples

    def load_all_sources(self) -> list[PromptInjectionSample]:
        """Load data from all available sources."""
        all_samples = []

        # Load all datasets
        all_samples.extend(self.load_synapsecai())
        all_samples.extend(self.load_mpdd())
        all_samples.extend(self.load_malignant())
        all_samples.extend(self.load_jailbreak_prompts())
        all_samples.extend(self.load_malicious_deepset())
        all_samples.extend(self.load_forbidden_questions())
        all_samples.extend(self.load_predictionguard())
        all_samples.extend(self.load_guychuk())
        all_samples.extend(self.load_regular_prompts())  # Add benign samples

        # Remove duplicates based on text
        seen_texts = set()
        unique_samples = []
        for sample in all_samples:
            if sample.text not in seen_texts:
                seen_texts.add(sample.text)
                unique_samples.append(sample)

        print(f"\nTotal unique samples: {len(unique_samples)} (removed {len(all_samples) - len(unique_samples)} duplicates)")

        return unique_samples

    def process_samples(
        self,
        samples: list[PromptInjectionSample],
    ) -> tuple[list[PromptInjectionSample], list[PromptInjectionSample], list[PromptInjectionSample]]:
        """
        Process samples into train/val/test splits.

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        # Filter by text length
        filtered = [
            s for s in samples
            if self.config.min_text_length <= len(s.text) <= self.config.max_text_length
        ]
        print(f"Filtered {len(samples) - len(filtered)} samples by text length")

        # Limit samples if configured
        if self.config.max_samples and len(filtered) > self.config.max_samples:
            random.seed(self.config.random_seed)
            filtered = random.sample(filtered, self.config.max_samples)
            print(f"Limited to {len(filtered)} samples")

        # Separate by label for stratified splitting
        benign = [s for s in filtered if s.label == 0]
        injection = [s for s in filtered if s.label == 1]

        print(f"\nClass distribution:")
        print(f"  Benign: {len(benign)}")
        print(f"  Injection: {len(injection)}")

        # Balance if needed
        if self.config.balance_classes:
            min_count = min(len(benign), len(injection))
            random.seed(self.config.random_seed)
            benign = random.sample(benign, min_count)
            injection = random.sample(injection, min_count)
            print(f"  Balanced to {min_count} per class")

        # Combine and split
        all_filtered = benign + injection
        random.seed(self.config.random_seed)
        random.shuffle(all_filtered)

        texts = [s.text for s in all_filtered]
        labels = [s.label for s in all_filtered]

        # First split: train vs (val + test)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels,
            test_size=self.config.test_size + self.config.val_size,
            random_state=self.config.random_seed,
            stratify=labels
        )

        # Second split: val vs test
        val_ratio = self.config.val_size / (self.config.test_size + self.config.val_size)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels,
            test_size=(1 - val_ratio),
            random_state=self.config.random_seed,
            stratify=temp_labels
        )

        # Reconstruct sample objects
        text_to_sample = {(s.text, s.label): s for s in all_filtered}

        train_samples = [text_to_sample[(t, l)] for t, l in zip(train_texts, train_labels)]
        val_samples = [text_to_sample[(t, l)] for t, l in zip(val_texts, val_labels)]
        test_samples = [text_to_sample[(t, l)] for t, l in zip(test_texts, test_labels)]

        print(f"\nFinal splits:")
        print(f"  Train: {len(train_samples)}")
        print(f"  Val: {len(val_samples)}")
        print(f"  Test: {len(test_samples)}")

        return train_samples, val_samples, test_samples

    def to_huggingface_dataset(
        self,
        train_samples: list[PromptInjectionSample],
        val_samples: list[PromptInjectionSample],
        test_samples: list[PromptInjectionSample],
    ) -> DatasetDict:
        """Convert samples to HuggingFace DatasetDict for training."""

        def samples_to_dict(samples):
            return {
                "text": [s.text for s in samples],
                "label": [s.label for s in samples],
                "source": [s.source for s in samples],
                "category": [s.category for s in samples],
            }

        return DatasetDict({
            "train": Dataset.from_dict(samples_to_dict(train_samples)),
            "validation": Dataset.from_dict(samples_to_dict(val_samples)),
            "test": Dataset.from_dict(samples_to_dict(test_samples)),
        })

    def save_processed_data(
        self,
        dataset: DatasetDict,
        output_path: Path = None,
    ) -> None:
        """Save processed dataset to disk."""
        output_path = output_path or self.base_path / "data" / "processed"
        output_path.mkdir(parents=True, exist_ok=True)

        dataset.save_to_disk(str(output_path))
        print(f"Saved processed dataset to {output_path}")

        # Save stats
        train_labels = pd.Series(dataset["train"]["label"])
        stats = {
            "train_size": int(len(dataset["train"])),
            "val_size": int(len(dataset["validation"])),
            "test_size": int(len(dataset["test"])),
            "train_label_dist": {str(k): int(v) for k, v in train_labels.value_counts().items()},
            "config": {
                "test_size": self.config.test_size,
                "val_size": self.config.val_size,
                "random_seed": self.config.random_seed,
                "balance_classes": self.config.balance_classes,
            }
        }

        with open(output_path / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    def load_or_create_dataset(
        self,
        force_recreate: bool = False,
    ) -> DatasetDict:
        """Load existing processed dataset or create new one."""
        processed_path = self.base_path / "data" / "processed"

        if not force_recreate and (processed_path / "dataset_dict.json").exists():
            print(f"Loading existing dataset from {processed_path}")
            return DatasetDict.load_from_disk(str(processed_path))

        print("Creating new dataset from all sources...")
        samples = self.load_all_sources()
        train, val, test = self.process_samples(samples)
        dataset = self.to_huggingface_dataset(train, val, test)
        self.save_processed_data(dataset, processed_path)

        return dataset


def create_quick_dataset(
    max_samples: int = 10000,
    base_path: Path = None,
) -> DatasetDict:
    """Create a small dataset for quick experimentation."""
    config = DatasetConfig(
        max_samples=max_samples,
        balance_classes=True,
    )
    loader = PromptInjectionDataLoader(base_path=base_path, config=config)
    return loader.load_or_create_dataset(force_recreate=True)


if __name__ == "__main__":
    # Test the data loader
    loader = PromptInjectionDataLoader()
    dataset = loader.load_or_create_dataset(force_recreate=True)
    print(dataset)
    print(f"\nSample from train:")
    print(dataset["train"][0])
