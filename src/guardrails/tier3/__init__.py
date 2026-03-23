"""
Tier 3: LoRA Adapter Training Infrastructure

Implements Luna-2 style single-token classification with LoRA adapters
on shared SLM backbone. Supports Qwen-0.5B/1.5B and LLaMA-3.2-3B.
"""

__all__ = ["train_lora_classifier", "LoRATrainer", "LoRAConfig"]
