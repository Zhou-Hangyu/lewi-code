"""BIMT: brain-inspired modular training (Liu, Gan and Tegmark, 2023).

The units of every layer have fixed coordinates in [0, 1]. The penalty is
``lam`` times the connection cost ``sum |w_ij| (|x_i - x_j| + l0) + l0 sum |b|``
over the hidden transitions. Every 200 steps, the 5 units of each layer with the
largest summed ``|w|`` are moved: a unit is exchanged with the unit whose
exchange lowers the connection cost most, when the exchange lowers the cost.
An exchange of two inputs is kept in a permutation that the forward pass
applies, so the function does not change.

Constants from the released regression experiment: ``l0 = 0.1``, 5 units per
layer, period 200. Differences from the source: one constant ``lam`` is searched
(the source schedules it by hand) and the activation is ReLU (the source uses SiLU).
"""
from __future__ import annotations

import torch

from ..models import DenseMLP
from .base import Method, register

L0 = 0.1
TOP_K = 5
SWAP_PERIOD = 200


def _coordinates(n: int, device=None) -> torch.Tensor:
    return torch.linspace(1.0 / (2 * n), 1.0 - 1.0 / (2 * n), n, device=device)


class BIMTMLP(DenseMLP):
    """The dense backbone with a persistent permutation of the input coordinates."""

    def __init__(self, input_dim: int, width: int, hidden: int):
        super().__init__(input_dim, width, hidden)
        self.register_buffer("bimt_in_perm", torch.arange(input_dim), persistent=True)
        self.register_buffer("bimt_out_perm", torch.arange(self.out_dim), persistent=True)

    def forward(self, x):
        return super().forward(x[:, self.bimt_in_perm])

    def hidden_states(self, x):
        return super().hidden_states(x[:, self.bimt_in_perm])


@register("bimt")
class BIMT(Method):
    hp_keys = frozenset({"lam"})

    def build(self, cfg, target):
        return BIMTMLP(target.d, cfg["width"], cfg["hidden"])

    def penalty(self, model, hp):
        lam = float(hp["lam"])
        if lam < 0:
            raise ValueError(f"lam must be non-negative; received {lam}")
        cost = torch.zeros((), device=next(model.parameters()).device)
        for layer in model.hidden_layers:
            x_in = _coordinates(layer.weight.shape[1], layer.weight.device)
            x_out = _coordinates(layer.weight.shape[0], layer.weight.device)
            distance = (x_out.unsqueeze(1) - x_in.unsqueeze(0)).abs()
            cost = cost + (layer.weight.abs() * (distance + L0)).sum()
            cost = cost + L0 * layer.bias.abs().sum()
        return lam * cost

    @torch.no_grad()
    def after_step(self, model, optimizer, *, step, total_steps):
        if step == 0 or step % SWAP_PERIOD:
            return
        layers = list(model.linear_layers())
        L = len(layers)
        for i in range(L + 1):                      # unit layer i: 0 = inputs, L = output
            w_in = layers[i - 1].weight if i >= 1 else None
            w_out = layers[i].weight if i < L else None
            n = w_in.shape[0] if w_in is not None else w_out.shape[1]
            if n < 2:
                continue
            score = torch.zeros(n, device=(w_in if w_in is not None else w_out).device)
            if w_in is not None:
                score += w_in.abs().sum(dim=1)
            if w_out is not None:
                score += w_out.abs().sum(dim=0)
            for j in torch.topk(score, min(TOP_K, n)).indices.tolist():
                M = self._slot_cost(layers, i, n)
                diagonal = M.diagonal()
                delta = (M[j, :] + M[:, j]) - (diagonal[j] + diagonal)
                delta[j] = 0.0
                k = int(torch.argmin(delta))
                if k != j and float(delta[k]) < 0.0:
                    self._swap(model, layers, i, j, k)

    @staticmethod
    def _slot_cost(layers, i, n) -> torch.Tensor:
        """``M[unit, slot]``: the distance-weighted cost of the unit's connections at the slot."""
        L = len(layers)
        w_in = layers[i - 1].weight if i >= 1 else None
        w_out = layers[i].weight if i < L else None
        device = (w_in if w_in is not None else w_out).device
        x_self = _coordinates(n, device)
        M = torch.zeros(n, n, device=device)
        if w_in is not None:
            x_prev = _coordinates(w_in.shape[1], device)
            M += w_in.abs() @ (x_self.unsqueeze(0) - x_prev.unsqueeze(1)).abs()
        if w_out is not None:
            x_next = _coordinates(w_out.shape[0], device)
            M += w_out.abs().t() @ (x_self.unsqueeze(0) - x_next.unsqueeze(1)).abs()
        return M

    @staticmethod
    @torch.no_grad()
    def _swap(model, layers, i, j, k):
        L = len(layers)
        if i == 0:
            w = layers[0].weight
            w[:, [j, k]] = w[:, [k, j]].clone()
            p = model.bimt_in_perm
            p[[j, k]] = p[[k, j]].clone()
        elif i == L:
            w, b = layers[-1].weight, layers[-1].bias
            w[[j, k]] = w[[k, j]].clone()
            b[[j, k]] = b[[k, j]].clone()
            p = model.bimt_out_perm
            p[[j, k]] = p[[k, j]].clone()
        else:
            w_in, w_out = layers[i - 1], layers[i]
            w_in.weight[[j, k]] = w_in.weight[[k, j]].clone()
            w_in.bias[[j, k]] = w_in.bias[[k, j]].clone()
            w_out.weight[:, [j, k]] = w_out.weight[:, [k, j]].clone()

    def effective_weights(self, model):
        """The weight matrices with the input permutation folded back, so column ``c`` reads input ``c``."""
        matrices = [layer.effective_weight() for layer in model.linear_layers()]
        inverse = torch.empty_like(model.bimt_in_perm)
        inverse[model.bimt_in_perm] = torch.arange(len(model.bimt_in_perm), device=model.bimt_in_perm.device)
        matrices[0] = matrices[0][:, inverse]
        return tuple(matrices)

    def diagnostics(self, model):
        identity = torch.arange(len(model.bimt_in_perm), device=model.bimt_in_perm.device)
        return {"inputs_relocated": float(int((model.bimt_in_perm != identity).sum()))}
