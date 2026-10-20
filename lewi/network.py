"""Neuron embeddings of a LeWi network, its initialization checks, and a ready-made LeWi multilayer perceptron."""
from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .layers import LeWiLinear

EMBEDDING_DIM = 3           # r, the dimension of the embedding space
EMBEDDING_STD = 0.5         # sigma_e, the standard deviation of the initial embeddings
MIN_OPEN_INPUT = 5          # open outgoing connections that every input has at the initialization


def derive_seed(*parts: object) -> int:
    """A seed below 2^63 - 1 from ordered keys (a keyed BLAKE2b hash; the derivation of the paper's experiments)."""
    hasher = hashlib.blake2b(digest_size=8, person=b"arityseed")
    for part in parts:
        if type(part) not in (int, str, float, tuple):
            raise TypeError(f"a seed key must be an int, a str, a float or a tuple; received {type(part).__qualname__}")
        type_name = type(part).__qualname__.encode("utf-8")
        value = repr(part).encode("utf-8")
        hasher.update(struct.pack(">I", len(type_name)))
        hasher.update(type_name)
        hasher.update(struct.pack(">Q", len(value)))
        hasher.update(value)
    return int.from_bytes(hasher.digest(), "big") % (2**63 - 1)


def init_embeddings(layers: Sequence[LeWiLinear], *, seed: int, code_dim: int = EMBEDDING_DIM, radius: float = 1.0,
                    min_open_input: int = MIN_OPEN_INPUT) -> tuple[nn.ParameterList, int, int]:
    """Draw the embeddings of every layer boundary of a chain of LeWi layers and bind them to the layers.

    ``layers`` are the affine layers in forward order: layer ``k`` reads boundary ``k`` and writes boundary
    ``k + 1``, so the receivers of one layer are the senders of the next. Every unit gets an embedding drawn from
    ``N(0, sigma_e^2 I_r)`` with ``sigma_e = 0.5``. The connection radius is ``R^2 = 2 r sigma_e^2`` times ``radius``,
    the same for every layer and not trained (``R^2 = 1.5`` at ``r = 3``).

    Two checks follow (rejection sampling; they use no information about the task and treat every input alike):

    1. a layer without an open connection has the embeddings of its receiving units drawn again until one opens;
    2. an input with fewer than ``min_open_input`` open outgoing connections has its embedding drawn again until it
       has that many.

    Returns the embeddings (one parameter per boundary; register them on the network so that the optimizer trains
    them), the number of layer redraws and the number of input redraws.
    """
    layers = list(layers)
    if not layers:
        raise ValueError("a LeWi network needs at least one layer")
    for previous, layer in zip(layers, layers[1:]):
        if previous.out_features != layer.in_features:
            raise ValueError("the layers do not form a chain: a layer's outputs must be the next layer's inputs")
    r, seed = int(code_dim), int(seed)
    if r < 1:
        raise ValueError("code_dim must be a positive integer")
    if radius <= 0:
        raise ValueError("radius must be positive")
    widths = (layers[0].in_features, *(layer.out_features for layer in layers))
    generator = torch.Generator().manual_seed(seed)
    embeddings = nn.ParameterList([nn.Parameter(torch.randn(int(w), r, generator=generator) * EMBEDDING_STD)
                                   for w in widths])
    squared_radius = 2.0 * r * EMBEDDING_STD ** 2 * float(radius)
    for boundary, layer in enumerate(layers):
        layer.bind(embeddings[boundary + 1], embeddings[boundary])
        with torch.no_grad():
            layer.gate_offset.fill_(squared_radius)
        layer.gate_scale.requires_grad_(False)
        layer.gate_offset.requires_grad_(False)

    layer_redraws = input_redraws = 0
    with torch.no_grad():
        for boundary, layer in enumerate(layers):                       # check 1: every layer has an open connection
            attempt = 0
            while not bool((layer.score() > 0).any()):
                attempt += 1
                key = ("output",) if boundary == len(layers) - 1 else ("hidden", boundary)
                g = torch.Generator().manual_seed(int(derive_seed(seed, *key, attempt)))
                codes = embeddings[boundary + 1]
                codes.copy_(torch.randn(codes.shape[0], r, generator=g) * EMBEDDING_STD)
                layer_redraws += 1
        update_masks(layers)
        first, inputs = layers[0], embeddings[0]                        # check 2: every input has open connections
        for i in range(inputs.shape[0]):
            attempt = 0
            while int((first.score()[:, i] > 0).sum()) < min_open_input:
                attempt += 1
                g = torch.Generator().manual_seed(int(derive_seed(seed, "input", i, attempt)))
                inputs[i].copy_(torch.randn(inputs.shape[1], generator=g) * EMBEDDING_STD)
                input_redraws += 1
        if input_redraws:
            update_masks(layers)
    return embeddings, layer_redraws, input_redraws


def _lewi_layers(model_or_layers) -> list[LeWiLinear]:
    if isinstance(model_or_layers, nn.Module):
        return [m for m in model_or_layers.modules() if isinstance(m, LeWiLinear)]
    return list(model_or_layers)


def update_masks(model_or_layers) -> None:
    """Recompute the binary mask of every LeWi layer from the current embeddings (after every optimizer step)."""
    for layer in _lewi_layers(model_or_layers):
        layer.update_mask()


def connection_density(model_or_layers) -> dict[str, float]:
    """The fraction of open connections: overall, of the input layer, of the hidden-to-hidden layers, of the readout."""
    masks = [layer.event_mask for layer in _lewi_layers(model_or_layers)]
    open_edges = [float(m.sum()) for m in masks]
    edges = [float(m.numel()) for m in masks]
    hidden = slice(1, len(masks) - 1)
    return {
        "density_overall": sum(open_edges) / sum(edges),
        "density_input": open_edges[0] / edges[0],
        "density_hidden": sum(open_edges[hidden]) / sum(edges[hidden]) if edges[hidden] else float("nan"),
        "density_readout": open_edges[-1] / edges[-1],
    }


class LeWiMLP(nn.Module):
    """A ReLU multilayer perceptron whose every affine layer, the readout included, is a LeWi layer.

    ``depth`` hidden layers of ``width`` units and a linear readout of ``out_features`` units. The output has shape
    ``(batch, out_features)``. ``seed`` fixes the embeddings; the weights follow PyTorch's global generator, as in
    any ``nn.Module``. Train the weights and the embeddings together with any optimizer and call
    :meth:`update_masks` after every optimizer step.
    """

    def __init__(self, in_features: int, width: int, depth: int, out_features: int = 1, *, seed: int = 0,
                 code_dim: int = EMBEDDING_DIM, radius: float = 1.0, min_open_input: int = MIN_OPEN_INPUT):
        super().__init__()
        if int(depth) < 1:
            raise ValueError("depth must be at least 1")
        sizes = [int(in_features)] + [int(width)] * int(depth) + [int(out_features)]
        self.layers = nn.ModuleList(LeWiLinear(sizes[k], sizes[k + 1], is_output=k == int(depth))
                                    for k in range(int(depth) + 1))
        self.embeddings, self.layer_redraws, self.input_redraws = init_embeddings(
            self.layers, seed=seed, code_dim=code_dim, radius=radius, min_open_input=min_open_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

    def update_masks(self) -> None:
        update_masks(self.layers)

    def connection_density(self) -> dict[str, float]:
        return connection_density(self.layers)

    def positions(self) -> list[torch.Tensor]:
        """The current embeddings of every layer boundary, from the inputs to the outputs (detached copies)."""
        return [e.detach().clone() for e in self.embeddings]
