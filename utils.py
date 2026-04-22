import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List


def sparsity_loss(model: nn.Module) -> torch.Tensor:
    # L1 norm of all gate values across the network
    # L1 works here because its gradient is constant (always 1)
    # so it keeps pushing gates toward zero even when they're already small
    # L2 would slow down near zero and never fully zero things out
    from model import PrunableLinear

    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total = total + gates.sum()
    return total


def total_loss(classification_loss: torch.Tensor, model: nn.Module, lam: float) -> torch.Tensor:
    # total loss = cross entropy + lambda * sparsity
    # lambda controls how hard we push toward pruning
    return classification_loss + lam * sparsity_loss(model)


def compute_sparsity(model: nn.Module, threshold: float = 1e-2) -> float:
    # percentage of weights whose gate is below threshold
    # gate < 0.01 means that weight contributes almost nothing
    from model import PrunableLinear

    total_weights = 0
    pruned_weights = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach()
            total_weights += gates.numel()
            pruned_weights += (gates < threshold).sum().item()

    return 100.0 * pruned_weights / total_weights if total_weights > 0 else 0.0


def evaluate(model: nn.Module, loader, device: str) -> float:
    # returns test accuracy as a percentage
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    model.train()
    return 100.0 * correct / total


def plot_gate_distribution(model: nn.Module, lam: float, save_dir: str) -> None:
    # plot histogram of all gate values after training
    # good result = big spike at 0 (pruned) + smaller cluster near 1 (active)
    gates = model.get_all_gates().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gates, bins=80, color="#2563eb", edgecolor="white", linewidth=0.3)
    ax.set_title(f"Gate Value Distribution  (lambda = {lam})", fontsize=11)
    ax.set_xlabel("Gate Value", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.2, label="prune threshold")
    ax.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"gate_dist_lambda_{lam}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [plot saved] {path}")


def plot_training_curves(
    sparsity_history: List[float],
    loss_history: List[float],
    lam: float,
    save_dir: str
) -> None:
    # sparsity and loss side by side so you can see them evolve together
    epochs = list(range(1, len(sparsity_history) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, sparsity_history, color="#16a34a", linewidth=2)
    ax1.set_title(f"Sparsity Over Training (lambda = {lam})", fontsize=11)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Sparsity (%)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, loss_history, color="#dc2626", linewidth=2)
    ax2.set_title(f"Training Loss (lambda = {lam})", fontsize=11)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Total Loss")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"training_curves_lambda_{lam}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [plot saved] {path}")