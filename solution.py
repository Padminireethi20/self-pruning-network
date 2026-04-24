"""
Self-Pruning Neural Network on CIFAR-10
----------------------------------------
A feedforward network where each weight has a learnable gate.
During training, an L1 sparsity penalty pushes most gates toward zero,
effectively pruning those weights without any post-training step.

Run:
    python solution.py

Results and plots are saved to ./results/
"""

import os
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Hyperparameters — change these to experiment
# ---------------------------------------------------------------------------

INPUT_SIZE    = 3072        # CIFAR-10 images are 32x32x3, flattened = 3072
HIDDEN_SIZES  = [512, 256, 128]
NUM_CLASSES   = 10

EPOCHS        = 20
BATCH_SIZE    = 128
LEARNING_RATE = 1e-3

# three lambda values — low / medium / high pruning pressure
LAMBDA_VALUES   = [0.0001, 0.001, 0.01]

# gate below this value is considered pruned
PRUNE_THRESHOLD = 1e-2

DATA_DIR    = "./data"
RESULTS_DIR = "./results"
SEED        = 42


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Part 1: PrunableLinear Layer
# ---------------------------------------------------------------------------

class PrunableLinear(nn.Module):
    """
    Custom linear layer with a learnable gate for each weight.

    How it works:
      - gate_scores is a parameter tensor with the same shape as weight
      - sigmoid(gate_scores) produces gate values between 0 and 1
      - pruned_weights = weight * gates  (element-wise)
      - output = pruned_weights @ input + bias

    Gradients flow through both weight and gate_scores automatically
    because all operations (sigmoid, multiply, linear) are differentiable.
    The optimizer updates gate_scores just like any other parameter.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        # standard weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # gate_scores: one per weight, initialized to 0
        # sigmoid(0) = 0.5 so all gates start at neutral (half open)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # kaiming init for weights, same as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # step 1: turn gate_scores into values between 0 and 1
        gates = torch.sigmoid(self.gate_scores)

        # step 2: element-wise multiply — gate near 0 kills that weight
        pruned_weights = self.weight * gates

        # step 3: standard linear operation with masked weights
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Returns current gate values detached from the computation graph."""
        return torch.sigmoid(self.gate_scores).detach()

    def hard_prune(self, threshold: float = PRUNE_THRESHOLD) -> int:
        """
        Permanently zeros out weights whose gate is below threshold.
        Call this after training if you want to deploy the sparse model.
        Returns the number of weights pruned.
        """
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            mask  = gates < threshold
            self.weight[mask]      = 0.0
            self.gate_scores[mask] = -10.0  # force sigmoid to ~0
        return mask.sum().item()


# ---------------------------------------------------------------------------
# Network definition using PrunableLinear layers
# ---------------------------------------------------------------------------

class SelfPruningNetwork(nn.Module):
    """
    Feedforward network for CIFAR-10 built entirely from PrunableLinear layers.
    Every weight in the network can be gated to zero during training.

    Architecture: 3072 -> 512 -> 256 -> 128 -> 10
    """

    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super().__init__()

        layer_sizes = [input_size] + hidden_sizes + [num_classes]

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(PrunableLinear(layer_sizes[i], layer_sizes[i + 1]))
            # ReLU after every layer except the last
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten: (batch, 3, 32, 32) -> (batch, 3072)
        x = x.view(x.size(0), -1)
        return self.network(x)

    def get_all_gates(self) -> torch.Tensor:
        """Collects gate values from every PrunableLinear into one flat tensor."""
        all_gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().view(-1))
        return torch.cat(all_gates)


# ---------------------------------------------------------------------------
# Part 2: Sparsity Loss
# ---------------------------------------------------------------------------

def sparsity_loss(model: nn.Module) -> torch.Tensor:
    """
    L1 norm of all gate values across every PrunableLinear layer.

    Why L1 and not L2?
    L1 gradient is constant (always 1) so the optimizer keeps pushing
    gates toward zero at a steady rate even when they are already small.
    L2 gradient shrinks as the value shrinks, so gates approach zero
    but rarely reach it exactly. L1 is what causes gates to fully collapse.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total = total + gates.sum()
    return total


def compute_total_loss(cls_loss: torch.Tensor, model: nn.Module, lam: float) -> torch.Tensor:
    """
    Total Loss = CrossEntropyLoss + lambda * SparsityLoss

    lambda controls the trade-off:
    - low lambda  -> model focuses on accuracy, less pruning
    - high lambda -> aggressive pruning, accuracy may drop
    """
    return cls_loss + lam * sparsity_loss(model)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_sparsity(model: nn.Module) -> float:
    """
    Returns the percentage of weights whose gate is below PRUNE_THRESHOLD.
    High sparsity means the self-pruning mechanism worked well.
    """
    total_w  = 0
    pruned_w = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates     = torch.sigmoid(module.gate_scores).detach()
            total_w  += gates.numel()
            pruned_w += (gates < PRUNE_THRESHOLD).sum().item()
    return 100.0 * pruned_w / total_w if total_w > 0 else 0.0


def evaluate_accuracy(model: nn.Module, loader, device: str) -> float:
    """Returns test accuracy as a percentage."""
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            _, predicted   = outputs.max(1)
            correct       += predicted.eq(labels).sum().item()
            total         += labels.size(0)
    model.train()
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_dataloaders():
    """Downloads CIFAR-10 on first run and returns train/test loaders."""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std =(0.2470, 0.2435, 0.2616)
        )
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std =(0.2470, 0.2435, 0.2616)
        )
    ])

    train_set = datasets.CIFAR10(root=DATA_DIR, train=True,  download=True, transform=transform_train)
    test_set  = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_gate_distribution(model: nn.Module, lam: float) -> None:
    """
    Plots the distribution of all gate values after training.
    A successful result shows a bimodal distribution:
    - large spike near 0 (pruned weights)
    - smaller cluster near 1 (active weights the network kept)
    """
    gates = model.get_all_gates().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gates, bins=80, color="#2563eb", edgecolor="white", linewidth=0.3)
    ax.set_title(f"Gate Value Distribution  (lambda = {lam})", fontsize=11)
    ax.set_xlabel("Gate Value", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.2, label="prune threshold (0.01)")
    ax.legend()
    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"gate_dist_lambda_{lam}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [plot saved] {path}")


def plot_training_curves(sparsity_history: List[float], loss_history: List[float], lam: float) -> None:
    """Plots sparsity % and training loss side by side over epochs."""
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"training_curves_lambda_{lam}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [plot saved] {path}")


# ---------------------------------------------------------------------------
# Part 3: Training loop
# ---------------------------------------------------------------------------

def train_one_run(lam: float, train_loader, test_loader, device: str):
    """Full training run for a single lambda value."""
    log.info(f"{'='*55}")
    log.info(f"  Starting run  |  lambda = {lam}")
    log.info(f"{'='*55}")

    model = SelfPruningNetwork(
        input_size   = INPUT_SIZE,
        hidden_sizes = HIDDEN_SIZES,
        num_classes  = NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Adam updates both weight and gate_scores together
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    sparsity_history = []
    loss_history     = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        batches    = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs  = model(images)
            cls_loss = criterion(outputs, labels)

            # compute total loss with sparsity penalty
            loss = compute_total_loss(cls_loss, model, lam)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches    += 1

        avg_loss = epoch_loss / batches
        sparsity = compute_sparsity(model)

        sparsity_history.append(sparsity)
        loss_history.append(avg_loss)

        log.info(
            f"  Epoch {epoch:>2}/{EPOCHS}  |  "
            f"Loss: {avg_loss:.4f}  |  "
            f"Sparsity: {sparsity:.1f}%"
        )

    # final evaluation
    test_acc       = evaluate_accuracy(model, test_loader, device)
    final_sparsity = compute_sparsity(model)

    log.info(f"\n  Done  |  lambda={lam}  |  Test Accuracy: {test_acc:.2f}%  |  Sparsity: {final_sparsity:.1f}%\n")

    # save plots for this run
    plot_gate_distribution(model, lam)
    plot_training_curves(sparsity_history, loss_history, lam)

    return test_acc, final_sparsity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders()

    results = []
    for lam in LAMBDA_VALUES:
        test_acc, sparsity = train_one_run(lam, train_loader, test_loader, device)
        results.append({"lambda": lam, "test_accuracy": test_acc, "sparsity": sparsity})

    # summary table
    print("\n" + "=" * 52)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>15}")
    print("=" * 52)
    for r in results:
        print(f"  {r['lambda']:<12} {r['test_accuracy']:>14.2f}% {r['sparsity']:>14.1f}%")
    print("=" * 52)
    print("\nAll plots saved to ./results/\n")


if __name__ == "__main__":
    main()
