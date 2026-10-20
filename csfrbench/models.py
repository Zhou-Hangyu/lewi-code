"""The common backbone: a ReLU multilayer perceptron.

Every method trains the same backbone within a setting: width 32 with four
hidden layers, or width 64 with eight hidden layers on the depth-5 tree. A
method changes the affine layers (gates, masks, penalties), not the shape.

Hidden layers use Kaiming normal initialization for ReLU (fan-in mode); the
readout uses the gain of a linear layer. Biases start at zero.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLinear(nn.Module):
    """An affine layer with the benchmark's initialization."""

    def __init__(self, in_features: int, out_features: int, *, is_output: bool):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.is_output = bool(is_output)
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear" if self.is_output else "relu")
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def effective_weight(self) -> torch.Tensor:
        """The weight matrix that acts at evaluation."""
        return self.weight


LayerFactory = Callable[[int, int, bool], nn.Module]


class ReLUMLP(nn.Module):
    """``hidden`` ReLU layers of equal width and a linear readout with one output."""

    def __init__(self, input_dim: int, width: int, hidden: int, *, layer_factory: LayerFactory):
        super().__init__()
        input_dim, width, hidden = int(input_dim), int(width), int(hidden)
        if input_dim <= 0 or width <= 0 or hidden < 1:
            raise ValueError("input_dim and width must be positive and hidden at least 1")
        self.out_dim = 1
        sizes = [input_dim] + [width] * hidden
        self.hidden_layers = nn.ModuleList(layer_factory(sizes[i], sizes[i + 1], False) for i in range(hidden))
        self.output_layer = layer_factory(sizes[-1], 1, True)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x).squeeze(-1)

    def linear_layers(self) -> tuple[nn.Module, ...]:
        """Every affine layer in forward order, the readout last."""
        return (*self.hidden_layers, self.output_layer)

    def hidden_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        """The output of every hidden layer after the ReLU."""
        states = []
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            states.append(x)
        return states

    def effective_weights(self) -> tuple[torch.Tensor, ...]:
        return tuple(layer.effective_weight() for layer in self.linear_layers())


class DenseMLP(ReLUMLP):
    """The fully connected backbone."""

    def __init__(self, input_dim: int, width: int, hidden: int):
        super().__init__(input_dim, width, hidden,
                         layer_factory=lambda i, o, is_output: DenseLinear(i, o, is_output=is_output))


def boundary_widths(model: ReLUMLP) -> tuple[int, ...]:
    """The number of units at every layer boundary, from the input to the output."""
    layers = model.linear_layers()
    return (int(layers[0].in_features), *(int(layer.out_features) for layer in layers))
