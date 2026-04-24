# Self-Pruning Neural Network — CIFAR-10

Tredence AI Engineering Internship — Case Study Submission

---

## What This Is

A feedforward neural network that learns to prune its own weights during training using learnable gates and an L1 sparsity penalty. Built on CIFAR-10 image classification.

Every weight has a learnable gate between 0 and 1. An L1 penalty during training pushes most gates to zero, removing those weights entirely — no post-training pruning step needed.

---

## Files

| File | Description |
|------|-------------|
| `solution.py` | Single script with everything — PrunableLinear layer, network, training loop, evaluation |
| `report.md` | Results, analysis, and gate distribution plot |
| `results/` | Saved plots from training |

---

## How to Run

```bash
pip install -r requirements.txt
python solution.py
```

CIFAR-10 downloads automatically on first run. Results and plots are saved to `./results/`.

---

## Results

| Lambda | Test Accuracy | Sparsity |
|--------|:------------:|:--------:|
| 0.0001 | 55.14%       | 42.0%    |
| 0.001  | 50.84%       | 94.8%    |
| 0.01   | 41.65%       | 99.8%    |

See `report.md` for full analysis.
