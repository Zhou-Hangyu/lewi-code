"""Fiedler regularization (Tam and Dunson, 2020).

The network is a graph with one node per unit and edge weights ``|w_ij|``. The
penalty is ``delta`` times the quadratic form ``sum |w_ij| (u_i - u_j)^2`` over
the hidden transitions, an upper bound of the Fiedler value of the graph
Laplacian. ``u`` is the Fiedler vector of the whole graph; it is computed at
the initialization and again every 100 steps.
"""
from __future__ import annotations

import torch

from ..models import DenseMLP
from .base import Method, register

REFRESH_PERIOD = 100


def _offsets(model) -> list[int]:
    layers = list(model.linear_layers())
    sizes = [layers[0].weight.shape[1]] + [layer.weight.shape[0] for layer in layers]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    return offsets


@torch.no_grad()
def fiedler_vector(model) -> torch.Tensor:
    """The Fiedler vector of the unnormalized Laplacian of the ``|W|`` graph."""
    layers = list(model.linear_layers())
    offsets = _offsets(model)
    n = offsets[-1]
    adjacency = torch.zeros(n, n)
    for i, layer in enumerate(layers):
        w = layer.weight.detach().abs().cpu()
        r0, r1 = offsets[i + 1], offsets[i + 2]
        c0, c1 = offsets[i], offsets[i + 1]
        adjacency[r0:r1, c0:c1] = w
        adjacency[c0:c1, r0:r1] = w.t()
    laplacian = -adjacency
    laplacian.diagonal().copy_(adjacency.sum(dim=1))
    _, vectors = torch.linalg.eigh(laplacian)
    return vectors[:, 1].contiguous()


@register("fiedler")
class Fiedler(Method):
    hp_keys = frozenset({"delta"})

    def build(self, cfg, target):
        model = DenseMLP(target.d, cfg["width"], cfg["hidden"])
        model.register_buffer("fiedler_u", fiedler_vector(model), persistent=True)
        return model

    def _quadratic_form(self, model, device):
        offsets = _offsets(model)
        u = model.fiedler_u
        total = torch.zeros((), device=device)
        for i, layer in enumerate(model.hidden_layers):
            u_in = u[offsets[i]:offsets[i + 1]].to(device)
            u_out = u[offsets[i + 1]:offsets[i + 2]].to(device)
            total = total + (layer.weight.abs() * (u_out.unsqueeze(1) - u_in.unsqueeze(0)) ** 2).sum()
        return total

    def penalty(self, model, hp):
        delta = float(hp["delta"])
        if delta < 0:
            raise ValueError(f"delta must be non-negative; received {delta}")
        return delta * self._quadratic_form(model, next(model.parameters()).device)

    def after_step(self, model, optimizer, *, step, total_steps):
        if step % REFRESH_PERIOD:
            return
        model.fiedler_u.copy_(fiedler_vector(model))

    def diagnostics(self, model):
        with torch.no_grad():
            return {"fiedler_bound": float(self._quadratic_form(model, next(model.parameters()).device))}
