import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PrunableLinear(nn.Module):
    # same as nn.Linear but each weight has a gate that can shut it off
    # gate_scores are learned just like weights - optimizer updates both
    # sigmoid(gate_score) gives a value between 0 and 1
    # if gate -> 0, that weight basically stops contributing

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # one gate score per weight, same shape
        # initialized to 0 so sigmoid(0) = 0.5, gates start neutral
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # same init as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # squash gate scores to (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # multiply gates with weights - near-zero gates kill the weight
        pruned_weights = self.weight * gates

        # standard linear op with the masked weights
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).detach()

    def hard_prune(self, threshold: float = 1e-2) -> int:
        # after training, permanently zero out weights with gate < threshold
        # useful if you want to deploy a smaller model
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            mask = gates < threshold
            self.weight[mask] = 0.0
            self.gate_scores[mask] = -10.0
        return mask.sum().item()


class SelfPruningNetwork(nn.Module):
    # feedforward network for CIFAR-10 built entirely from PrunableLinear layers
    # architecture: 3072 -> 512 -> 256 -> 128 -> 10

    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super().__init__()

        layer_sizes = [input_size] + hidden_sizes + [num_classes]

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(PrunableLinear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten image: (batch, 3, 32, 32) -> (batch, 3072)
        x = x.view(x.size(0), -1)
        return self.network(x)

    def get_all_gates(self) -> torch.Tensor:
        # collect all gate values into one flat tensor for plotting/analysis
        all_gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().view(-1))
        return torch.cat(all_gates)

    def get_prunable_layers(self) -> List[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]