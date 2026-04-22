# Self-Pruning Neural Network — CIFAR-10

A feed-forward neural network that **learns to prune itself during training** using learnable gates and an L1 sparsity penalty.

Built as part of the Tredence AI Engineering Internship case study.

---

## The Idea

Every weight in the network has a learnable **gate** — a scalar value between 0 and 1. During training, an L1 penalty pushes most gates toward zero, which effectively removes those weights from the network. The result is a sparse model that discards unnecessary connections on its own.

```
Total Loss = CrossEntropyLoss + λ × Σ(gate values)
```

Higher `λ` → more pruning → potentially lower accuracy. Three lambda values are compared to show this trade-off.

---

## Project Structure

```
self-pruning-network/
├── config.py       # all hyperparameters in one place
├── model.py        # PrunableLinear layer + SelfPruningNetwork
├── utils.py        # sparsity loss, metrics, plotting
├── train.py        # training loop + evaluation
├── report.md       # written analysis and results table
├── requirements.txt
└── results/        # plots saved here after training
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run (downloads CIFAR-10 automatically on first run)
python train.py
```

Training takes ~10-15 min on CPU per lambda value (3 runs total). GPU is faster.

---

## Output

After training, `results/` will contain gate distribution plots and training curves for each lambda. A bimodal gate distribution (big spike at 0 + cluster near 1) confirms the self-pruning is working.

---

## Key Concepts

**Why Sigmoid for gates?** Maps any real number to (0,1). Gate near 0 kills the weight; near 1 passes it through.

**Why L1 for sparsity?** Constant gradient pushes gates to exactly zero. L2 only shrinks them.

**What is `hard_prune()`?** Permanently zeros weights after training — for deployment of the sparse model.

See `report.md` for full analysis and results.
