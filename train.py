"""
train.py — Trains the self-pruning network on CIFAR-10.
           Runs three experiments (low / medium / high lambda) and prints a results table.

Run:
    python train.py
"""

import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import config
from model import SelfPruningNetwork
from utils import total_loss, compute_sparsity, evaluate, plot_gate_distribution, plot_training_curves

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data loading ───────────────────────────────────────────────────────────────
def get_dataloaders():
    """
    Downloads CIFAR-10 on first run (saved to ./data).
    Standard normalization values for CIFAR-10 (mean and std per channel).
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root=config.data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=config.data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader


# ── Training loop ──────────────────────────────────────────────────────────────
def train_one_run(lam: float, train_loader, test_loader, device: str):
    """
    Full training run for a single lambda value.
    Returns: (test_accuracy, sparsity_level)
    """
    log.info(f"{'='*55}")
    log.info(f"  Starting run  |  lambda = {lam}")
    log.info(f"{'='*55}")

    model = SelfPruningNetwork(
        input_size=config.input_size,
        hidden_sizes=config.hidden_sizes,
        num_classes=config.num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    sparsity_history = []
    loss_history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            cls_loss = criterion(outputs, labels)

            # Total loss = classification loss + lambda * sparsity loss
            loss = total_loss(cls_loss, model, lam)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        sparsity = compute_sparsity(model, config.prune_threshold)

        sparsity_history.append(sparsity)
        loss_history.append(avg_loss)

        log.info(
            f"  Epoch {epoch:>2}/{config.epochs}  |  "
            f"Loss: {avg_loss:.4f}  |  "
            f"Sparsity: {sparsity:.1f}%"
        )

    # Final evaluation
    test_acc = evaluate(model, test_loader, device)
    final_sparsity = compute_sparsity(model, config.prune_threshold)

    log.info(f"\n  Done  |  lambda={lam}  |  Test Accuracy: {test_acc:.2f}%  |  Sparsity: {final_sparsity:.1f}%\n")

    # Save plots
    plot_gate_distribution(model, lam, config.results_dir)
    plot_training_curves(sparsity_history, loss_history, lam, config.results_dir)

    return test_acc, final_sparsity


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    set_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders()

    results = []

    for lam in config.lambda_values:
        test_acc, sparsity = train_one_run(lam, train_loader, test_loader, device)
        results.append({
            "lambda": lam,
            "test_accuracy": test_acc,
            "sparsity": sparsity
        })

    # Print summary table
    print("\n" + "=" * 52)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>15}")
    print("=" * 52)
    for r in results:
        print(f"  {r['lambda']:<12} {r['test_accuracy']:>14.2f}% {r['sparsity']:>14.1f}%")
    print("=" * 52)
    print("\nPlots saved to ./results/")
    print("Paste this table into report.md\n")


if __name__ == "__main__":
    main()
