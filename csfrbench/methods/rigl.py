"""RigL: dynamic sparse training (Evci et al., 2020).

The hidden transitions carry binary masks; biases and the readout stay dense.
The layer densities follow the Erdos-Renyi-Kernel rule for the searched global
``density``. At an update iteration (the first step, then every 100 steps until
75% of the training) the gradient step is undone and each layer drops the
fraction ``f(t) = 0.15 * (1 + cos(pi * t / t_end))`` of its active connections
with the smallest magnitude and grows the same number of inactive connections
with the largest gradient magnitude. A grown connection starts at weight 0 with
a zero optimizer state.

Differences from the source, which uses SGD with momentum: the optimizer is
Adam, as for every method, and the cosine learning-rate schedule also advances
on update iterations.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..models import DenseLinear, ReLUMLP
from .base import Method, register

DELTA_T = 100           # steps between mask updates
ALPHA = 0.3             # initial drop fraction
T_END_FRACTION = 0.75   # mask updates stop after this fraction of the training
TIE_NOISE_STD = 1e-5    # noise that breaks ties of the drop score


def erk_densities(shapes: list[tuple[int, int]], density: float) -> list[float]:
    """Layer densities under the Erdos-Renyi-Kernel rule for a global ``density``."""
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must lie in (0, 1]; received {density}")
    n_params = [o * i for (o, i) in shapes]
    dense_layers: set[int] = set()
    while True:
        divisor = 0.0
        rhs = 0.0
        raw = {}
        for index, (n, (o, i)) in enumerate(zip(n_params, shapes)):
            n_zeros = int(round(n * (1.0 - density)))
            if index in dense_layers:
                rhs -= n_zeros
            else:
                rhs += n - n_zeros
                raw[index] = (o + i) / (o * i)
                divisor += raw[index] * n
        eps = rhs / divisor
        overflow = [index for index, p in raw.items() if p * eps > 1.0]
        if not overflow:
            break
        max_p = max(raw[index] for index in overflow)
        dense_layers.update(index for index in overflow if raw[index] == max_p)
    return [1.0 if index in dense_layers else raw[index] * eps for index in range(len(shapes))]


class RigLLinear(DenseLinear):
    def __init__(self, in_features, out_features, *, is_output):
        super().__init__(in_features, out_features, is_output=is_output)
        self.register_buffer("mask", torch.ones_like(self.weight))
        self._dense_grad = None         # gradient of the loss with respect to the masked weight, training only

    def forward(self, x):
        w_eff = self.weight * self.mask
        if w_eff.requires_grad:
            w_eff.register_hook(self._capture)
        return F.linear(x, w_eff, self.bias)

    def _capture(self, grad):
        self._dense_grad = grad.detach()

    def effective_weight(self):
        return self.weight * self.mask


def _is_update_step(step: int, t_end: int) -> bool:
    return step == 1 or (step % DELTA_T == 0 and step <= t_end)


def _stash(model, optimizer) -> dict:
    return {
        "params": [p.detach().clone() for p in model.parameters()],
        "state": [({key: (value.clone() if torch.is_tensor(value) else value)
                    for key, value in optimizer.state[p].items()} if p in optimizer.state else None)
                  for p in model.parameters()],
    }


def _restore(model, optimizer, stash) -> None:
    """Undo the step that just ran."""
    with torch.no_grad():
        for p, saved in zip(model.parameters(), stash["params"]):
            p.copy_(saved)
    if stash["state"] == "initial":
        for p in model.parameters():
            if p in optimizer.state:
                optimizer.state[p] = {}
        return
    for p, saved in zip(model.parameters(), stash["state"]):
        if saved is None:
            if p in optimizer.state:
                optimizer.state[p] = {}
            continue
        live = optimizer.state[p]
        for key, value in saved.items():
            if torch.is_tensor(value) and torch.is_tensor(live.get(key)):
                live[key].copy_(value)
            else:
                live[key] = value


def _drop_and_grow(model, optimizer, *, fraction: float) -> None:
    for layer in model.hidden_layers:
        mask = layer.mask
        n_active = int(mask.sum().item())
        k = int(fraction * n_active)
        if k == 0:
            continue
        grad = layer._dense_grad
        if grad is None:
            continue
        weight = layer.weight.detach()
        score_drop = (weight * mask).abs().flatten()
        score_drop += torch.randn_like(score_drop) * TIE_NOISE_STD
        kept = torch.zeros_like(score_drop)
        kept[score_drop.topk(n_active - k).indices] = 1.0
        score_grow = grad.abs().flatten().clone()
        score_grow[kept.bool()] = float("-inf")
        grown = torch.zeros_like(kept)
        grown[score_grow.topk(k).indices] = 1.0
        new_mask = (kept + grown).view_as(mask)
        fresh = (grown.view_as(mask) > 0) & (mask == 0)
        with torch.no_grad():
            layer.weight[fresh] = 0.0
            state = optimizer.state.get(layer.weight)
            if state:
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        state[key][fresh] = 0.0
        mask.copy_(new_mask)
        layer._dense_grad = None


@register("rigl")
class RigL(Method):
    hp_keys = frozenset({"density"})

    def build(self, cfg, target):
        density = float(cfg["hp"]["density"])
        model = ReLUMLP(target.d, cfg["width"], cfg["hidden"],
                        layer_factory=lambda i, o, is_output: (DenseLinear(i, o, is_output=True) if is_output
                                                               else RigLLinear(i, o, is_output=False)))
        layers = list(model.hidden_layers)
        per_layer = erk_densities([(la.out_features, la.in_features) for la in layers], density)
        generator = torch.Generator().manual_seed(int(cfg["auxiliary_seed"]))
        for layer, layer_density in zip(layers, per_layer):
            n = layer.weight.numel()
            n_active = min(n, int(round(layer_density * n)))
            scores = torch.rand(n, generator=generator)
            mask = torch.zeros(n)
            mask[scores.topk(n_active).indices] = 1.0
            layer.mask.copy_(mask.view_as(layer.mask))
            layer.weight.data.mul_(layer.mask)
        model.rigl_density = density
        return model

    def penalty(self, model, hp):
        # No penalty. The call before the first step stores the initial state, so the
        # mask update at step 1 can undo that step.
        if getattr(model, "_rigl_prestep", None) is None and not getattr(model, "_rigl_started", False):
            model._rigl_prestep = {"params": [p.detach().clone() for p in model.parameters()], "state": "initial"}
        return None

    def after_step(self, model, optimizer, *, step, total_steps):
        model._rigl_started = True
        t_end = int(T_END_FRACTION * total_steps)
        if _is_update_step(step, t_end):
            stash = getattr(model, "_rigl_prestep", None)
            if stash is not None:
                _restore(model, optimizer, stash)
            fraction = 0.5 * ALPHA * (1.0 + math.cos(math.pi * min(step, t_end) / t_end))
            _drop_and_grow(model, optimizer, fraction=fraction)
        model._rigl_prestep = None
        if _is_update_step(step + 1, t_end):
            model._rigl_prestep = _stash(model, optimizer)

    def diagnostics(self, model):
        masks = [layer.mask for layer in model.hidden_layers]
        active = sum(int(m.sum().item()) for m in masks)
        total = sum(m.numel() for m in masks)
        return {"active_connections": float(active), "density_realized": active / total,
                "density_target": float(model.rigl_density)}
