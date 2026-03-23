# Novel LoRA Methods Research Summary

**Date:** 2026-03-21
**Goal:** Find novel LoRA methods for fast domain/policy adaptation with >95% accuracy

---

## Research Findings

### TOP NOVEL LoRA METHODS (2025-2026)

| # | Method | Paper | Key Innovation | Venue | Expected Gain |
|---|--------|-------|----------------|-------|---------------|
| 1 | **H-LoRA** | arXiv:2603.02281 | Hilbert Transform for phase-structured reparameterization | Mar 2026 | **+5% accuracy** |
| 2 | **SR-LoRA** | arXiv:2507.00327 | Stable rank-guided layer-wise rank allocation | ICCV 2025 | Superior on domain gaps |
| 3 | **ARENA** | arXiv:2507.15793 | L1 sparsity regularizer for dynamic rank adjustment | MICCAI 2025 | Automatic rank finding |
| 4 | **MTA** | arXiv:2511.20072 | Meta-LoRA Bank + Adaptive LoRA Fusion + LoRA Stacking | Nov 2025 | SOTA on LaMP |
| 5 | **C-LoRA** | arXiv:2505.17773 | Contextual LoRA modules for uncertainty estimation | NeurIPS 2025 | Well-calibrated uncertainties |
| 6 | **RwF** | arXiv:2603.09576 | Energy-based associative retrieval layers (Modern Hopfield) | Mar 2026 | Great for few-shot continual |
| 7 | **DAC-LoRA** | arXiv:2509.20792 | Dynamic Adversarial Curriculum with TRADES loss | ICCV 2025 Workshop | Adversarial robustness |
| 8 | **TA-LoRA** | arXiv:2505.00009 | Task-Adaptive with fast-slow weights mechanism | May 2025 | Multi-task learning |
| 9 | **MoE LoRA** | arXiv:2511.11236 | Mixture-of-Experts with style-specific routing | Nov 2025 | Multi-style editing |
| 10 | **ICM-Fusion** | arXiv:2508.04153 | In-Context Meta-Optimized LoRA Fusion | Aug 2025 | Multi-task adaptation |

---

## Method Details

### 1. H-LoRA (Hilbert Transform LoRA) - **RECOMMENDED**
**Paper:** arXiv:2603.02281 - "Quantum-Inspired Fine-Tuning for Few-Shot AIGC Detection via Phase-Structured Reparameterization"

**Key Innovation:**
- Applies Hilbert transform within LoRA adapter for phase-structured reparameterization
- Phase-aware representations encode richer information across orthogonal amplitude-phase components
- Norm-constrained transformations stabilize optimization via inherent orthogonality

**Results:** +5% accuracy improvement over standard LoRA in few-shot settings

**Implementation:** `experiments/train_hlora.py`

---

### 2. SR-LoRA (Stable Rank-Guided LoRA)
**Paper:** arXiv:2507.00327 - "Beyond Low-Rank Tuning: Model Prior-Guided Rank Allocation"
**Venue:** ICCV 2025

**Key Innovation:**
- Uses stable rank of pre-trained weight matrices as natural prior for layer-wise rank allocation
- No search costs - leverages intrinsic dimensionality of weights
- Principled redistribution of ranks across layers

**Best For:** Few-shot tasks with significant domain gaps

---

### 3. ARENA (Regularized LoRA)
**Paper:** arXiv:2507.15793 - "Regularized Low-Rank Adaptation for Few-Shot Organ Segmentation"
**Venue:** MICCAI 2025

**Key Innovation:**
- Dynamically adjusts intrinsic rank during adaptation
- L1 sparsity regularizer on SVD decomposition
- Proximal optimizer for automatic rank finding
- Robust against suboptimal rank initialization

**Code:** https://github.com/ghassenbaklouti/ARENA

**Implementation:** `experiments/train_arena.py`

---

### 4. MTA (Merge-then-Adapt)
**Paper:** arXiv:2511.20072 - "MTA: A Merge-then-Adapt Framework for Personalized LLMs"

**Key Innovation:**
- Meta-LoRA Bank: Pre-trained meta-personalization traits
- Adaptive LoRA Fusion: Dynamic merging of relevant meta-LoRAs
- LoRA Stacking: Ultra-low-rank module for few-shot personalization

**Best For:** Few-shot personalization, eliminates user-specific storage

---

### 5. C-LoRA (Contextual LoRA)
**Paper:** arXiv:2505.17773 - "Contextual Low-Rank Adaptation for Uncertainty Estimation"
**Venue:** NeurIPS 2025

**Key Innovation:**
- Input-contextualized LoRA modules for dynamic uncertainty estimation
- Mitigates overfitting in few-shot regimes
- Well-calibrated uncertainties

**Best For:** Uncertainty-aware classification

---

### 6. RwF (Routing without Forgetting)
**Paper:** arXiv:2603.09576 - "Routing without Forgetting"

**Key Innovation:**
- Energy-based associative retrieval layers (Modern Hopfield Networks)
- Single-step associative retrieval via closed-form minimization
- Input-conditioned routing within each forward pass

**Best For:** Few-shot continual learning, online adaptation

---

## Current Project Configuration

**Base Model:** Qwen/Qwen2.5-0.5B-Instruct
**Current Methods:**
- DoRA (Weight Decomposition) - enabled
- LoRA+ (16x LR for B matrices) - enabled
- All 7 linear layers - enabled

**Dataset:**
- Train: 35,680 samples
- Val: 4,757 samples
- Test: 7,137 samples

---

## Experiment Files Created

| File | Method | Status |
|------|--------|--------|
| `experiments/train_hlora.py` | H-LoRA (Hilbert Transform) | RUNNING |
| `experiments/train_arena.py` | ARENA (Regularized LoRA) | READY |
| `experiments/train_lora_optimized.py` | DoRA + LoRA+ (baseline) | COMPLETED |

---

## TODO: Next Experiments

### Priority 1: Core Methods
- [ ] **H-LoRA** - Currently training, wait for results
- [ ] **ARENA** - Ready to run, compare with H-LoRA
- [ ] **SR-LoRA** - Implement stable rank-guided allocation

### Priority 2: Advanced Methods
- [ ] **MTA** - Implement Meta-LoRA Bank for few-shot
- [ ] **C-LoRA** - Add uncertainty estimation
- [ ] **RwF** - Implement for continual learning

### Priority 3: Combinations
- [ ] **H-LoRA + ARENA** - Combine Hilbert transform with dynamic rank
- [ ] **H-LoRA + SR-LoRA** - Phase structure + stable rank allocation
- [ ] **MTA + H-LoRA** - Meta-learning + phase structure

### Priority 4: Hyperparameter Search
- [ ] Rank search: [8, 16, 32, 64]
- [ ] Alpha search: [rank, 2*rank, 4*rank]
- [ ] L1 lambda search (ARENA): [0.001, 0.01, 0.1]
- [ ] Learning rate search: [1e-4, 2e-4, 5e-4]

---

## Expected Results

| Method | Target F1 | Target Accuracy |
|--------|-----------|-----------------|
| Baseline (DoRA) | 0.87 | 88% |
| H-LoRA | 0.92 | 93% |
| ARENA | 0.90 | 91% |
| H-LoRA + ARENA | 0.95 | **>95%** |

---

## References

1. H-LoRA: https://arxiv.org/abs/2603.02281
2. SR-LoRA: https://arxiv.org/abs/2507.00327
3. ARENA: https://arxiv.org/abs/2507.15793
4. MTA: https://arxiv.org/abs/2511.20072
5. C-LoRA: https://arxiv.org/abs/2505.17773
6. RwF: https://arxiv.org/abs/2603.09576
7. DAC-LoRA: https://arxiv.org/abs/2509.20792
8. TA-LoRA: https://arxiv.org/abs/2505.00009
9. MoE LoRA: https://arxiv.org/abs/2511.11236
10. ICM-Fusion: https://arxiv.org/abs/2508.04153
