# LoRA Regularization Experiments - TODO

## Completed Experiments

### 1. Regularized LoRA (Full Training)
- **Config**: DoRA + LoRA+ (16x) + dropout 0.15 + weight_decay 0.05 + label_smooth 0.1 + grad_clip 1.0
- **Results**: 
  - Test F1: **0.9191**
  - Test Precision: 0.9295
  - Test Recall: 0.9089
  - Test Accuracy: 0.9200
  - Best Val F1: 0.9194 (Epoch 5)
- **Saved to**: `data/adapters/regularized_lora/epoch_5/`

### 2. Previous Sample Efficiency (1000 samples)
- Regularized training achieved F1=0.8919 with only 1000 samples
- 50% reduction from baseline (2000 samples needed)

---

## TODO: Variations to Run

### High Priority
1. **Low dropout (0.05)** - May improve recall
2. **No label smoothing** - Test if smoothing hurts precision
3. **Low weight decay (0.01)** - May allow more learning
4. **Lower LoRA+ ratio (8x)** - May stabilize training

### Medium Priority
5. **Higher rank (32)** - More model capacity
6. **Lower LR (1e-4)** - More stable convergence
7. **Cosine LR schedule** - Better final convergence

### After Variations
8. **Combine best settings** from variations
9. **Test on synthetic + real data** combination
10. **Final verdict documentation**

---

## Key Observations from Training

| Epoch | Val F1 | Precision | Recall | Notes |
|-------|--------|-----------|--------|-------|
| 1 | 0.9085 | 0.8741 | 0.9458 | Good start |
| 2 | 0.7446 | 0.9895 | 0.5969 | Over-regularized (low recall) |
| 3 | 0.9142 | 0.9539 | 0.8777 | Recovery |
| 4 | 0.8776 | 0.9788 | 0.7953 | High precision, lower recall |
| 5 | 0.9194 | 0.9326 | 0.9067 | **Best balanced** |

**Insight**: Epoch 2 showed extreme precision (0.99) but low recall (0.60) - suggests regularization may be too aggressive at times.

---

## Commands to Resume

```bash
# Run variations
.venv/Scripts/python.exe experiments/run_regularization_variations.py

# Or run individual variation
.venv/Scripts/python.exe experiments/train_regularized_lora.py
```

---

## Target
- Current best: F1 = 0.9191
- Target: F1 >= 0.95
- Gap: ~3%
