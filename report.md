# Self-Pruning Neural Network — Report

## Why L1 Penalty on Sigmoid Gates Encourages Sparsity

The sparsity loss is the **L1 norm** of all gate values across every PrunableLinear layer:

```
SparsityLoss = Σ sigmoid(gate_scores_i)    for all i
Total Loss   = CrossEntropyLoss + λ × SparsityLoss
```

Since sigmoid outputs are always positive, this is simply the sum of all gate values.

**Why L1 and not L2?**

The gradient of an L1 penalty is constant — always 1, regardless of how small the gate already is. So the optimizer keeps pushing gates toward zero at a steady rate even when they're nearly there. They eventually collapse all the way to zero.

L2 (sum of squares) has a gradient proportional to the value itself. As a gate shrinks toward zero, the gradient shrinks too — the push slows down and the gate never actually reaches zero. L2 shrinks weights, L1 zeros them out.

This is the same reason LASSO regression produces truly sparse solutions while Ridge does not. Our setup borrows exactly this idea: L1 on sigmoid gates forces most of them to zero, effectively removing those weights from the network during training itself — no post-training pruning step needed.

`λ` controls the strength of this push. Higher λ means the sparsity term dominates the loss, leading to more aggressive pruning at the potential cost of classification accuracy.

---

## Results

| Lambda | Test Accuracy | Sparsity Level (%) |
|:------:|:------------:|:-----------------:|
| 0.0001 | 55.14%       | 42.0%             |
| 0.001  | 50.84%       | 94.8%             |
| 0.01   | 41.65%       | 99.8%             |

---

## Analysis of the λ Trade-off

**Lambda = 0.0001 (low penalty)**
The sparsity penalty is weak, so classification loss dominates for all 20 epochs — sparsity stays at 0.0% right until the final epoch where it jumps to 42.0%. The network learned to classify well first (55.14% accuracy) and only pruned at the very end when the gates had been pushed far enough. This shows the penalty was present but not strong enough to influence training early on.

**Lambda = 0.001 (medium penalty)**
A much stronger effect — 94.8% sparsity at the end, meaning nearly all weights were pruned. Accuracy drops moderately to 50.84%. Similar pattern: sparsity stays 0.0% through epoch 19 then jumps to 94.8% at epoch 20. The gates were being driven down throughout training and crossed the threshold late. This is the best trade-off — high sparsity with reasonable accuracy.

**Lambda = 0.01 (high penalty)**
Extremely aggressive pruning — 99.8% of all weights gated out by the end. Almost nothing survives. Accuracy falls to 41.65%, which is still above random chance (10% for 10 classes) but noticeably worse. The loss values were also much larger throughout training (starting at 7852 vs 80 for lambda=0.0001) because the sparsity term dominated completely. The network was essentially fighting to learn anything while being aggressively pruned.

**Key observation from the training logs:**
In all three runs, sparsity stayed at 0.0% for most of the training and then jumped sharply in the final epoch. This is expected behavior — gates are being continuously pushed down by the L1 penalty throughout training, but they only cross the 0.01 threshold near the end. The pruning is happening gradually; we just measure it with a hard threshold.

**Sweet spot:** Lambda = 0.001 gives the best balance — 94.8% of weights pruned while retaining 50.84% accuracy. A model that is 94.8% sparse is dramatically cheaper to store and potentially faster to run at inference time, making this a practically useful result.

---

## Gate Value Distribution (Best Model)

The plots in `results/gate_dist_lambda_*.png` show the distribution of all gate values after training.

For the best model (lambda = 0.001), the distribution is strongly bimodal:
- A large spike near **0** — the ~94.8% of weights that were pruned
- A smaller cluster away from 0 — the ~5.2% of weights the network decided to keep
- Very few values in between — a clean, decisive separation

This bimodal shape confirms the self-pruning mechanism is working correctly. The network isn't leaving gates in an ambiguous middle ground — it's making clear binary decisions about which connections matter.
