from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    # dataset
    dataset: str = "CIFAR-10"
    num_classes: int = 10
    data_dir: str = "./data"

    # network shape - CIFAR-10 images are 32x32x3 = 3072 when flattened
    input_size: int = 3072
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])

    # training
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3

    # three lambda values to compare: low / medium / high pruning pressure
    lambda_values: List[float] = field(default_factory=lambda: [0.0001, 0.001, 0.01])

    # gate below this threshold = considered pruned
    prune_threshold: float = 1e-2

    seed: int = 42
    device: str = "cuda"
    results_dir: str = "./results"


config = TrainConfig()