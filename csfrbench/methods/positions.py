"""Neuron positions (Erb et al., 2025).

Every input and hidden unit has a position in the plane of its layer; the layers
are one unit apart. The weight of a hidden transition is not a parameter: it is
the inverse Euclidean distance between the two units. Every input and hidden
unit also has an inhibition value ``v``; the factor ``2 * sigmoid(10 v) - 1``,
close to +1 or -1, multiplies the output of the unit. The learned parameters
are the positions, the biases and the inhibition values. The readout is a
linear layer with free weights.

The source has no public implementation. Choices where the source is silent:
positions start uniformly in [0, 8)^2, inhibition values start standard normal,
biases start at zero, and the activation is ReLU.
"""
from __future__ import annotations

import torch
from torch import nn

from ..models import DenseLinear
from .base import Method, register

MASK_STEEPNESS = 10.0
DISTANCE_FLOOR = 1e-4
POSITION_SPREAD = 8.0


class SpatialMLP(nn.Module):
    def __init__(self, layer_sizes: list[int], generator: torch.Generator):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        self.out_dim = layer_sizes[-1]
        self.positions = nn.ParameterList(
            nn.Parameter(torch.rand(n, 2, generator=generator) * POSITION_SPREAD) for n in layer_sizes[:-1])
        torch.rand(layer_sizes[-1], 2, generator=generator)      # one unused draw: keeps the stream of the record
        self.biases = nn.ParameterList(nn.Parameter(torch.zeros(n)) for n in layer_sizes[1:-1])
        self.inhibitions = nn.ParameterList(
            nn.Parameter(torch.randn(n, generator=generator)) for n in layer_sizes[:-1])
        self.output_layer = DenseLinear(layer_sizes[-2], layer_sizes[-1], is_output=True)

    def weight(self, layer: int) -> torch.Tensor:
        p_in, p_out = self.positions[layer], self.positions[layer + 1]
        difference = p_out.unsqueeze(1) - p_in.unsqueeze(0)
        distance = (difference.pow(2).sum(-1) + 1.0).sqrt()
        return 1.0 / distance.clamp_min(DISTANCE_FLOOR)

    def sign(self, layer: int) -> torch.Tensor:
        return 2.0 * torch.sigmoid(MASK_STEEPNESS * self.inhibitions[layer]) - 1.0

    def forward(self, x):
        h = x * self.sign(0)
        for layer in range(len(self.layer_sizes) - 2):
            pre = torch.nn.functional.linear(h, self.weight(layer), self.biases[layer])
            h = torch.relu(pre) * self.sign(layer + 1)
        return self.output_layer(h).squeeze(-1)


@register("positions")
class NeuronPositions(Method):
    def build(self, cfg, target):
        sizes = [target.d] + [int(cfg["width"])] * int(cfg["hidden"]) + [1]
        generator = torch.Generator().manual_seed(int(cfg["auxiliary_seed"]))
        return SpatialMLP(sizes, generator)

    def effective_weights(self, model):
        """Signed matrices: the inverse distances times the factor of the sending unit."""
        last = len(model.layer_sizes) - 2
        matrices = [model.weight(i) * model.sign(i).unsqueeze(0) for i in range(last)]
        matrices.append(model.output_layer.weight * model.sign(last).unsqueeze(0))
        return tuple(m.detach() for m in matrices)

    def diagnostics(self, model):
        signs = torch.cat([model.sign(i).detach() for i in range(len(model.layer_sizes) - 1)])
        return {"inhibitory_fraction": float((signs < 0).float().mean())}
