"""L1 and L2 regularization of the weights.

The penalty is ``lam`` times the mean of ``|w|`` (L1) or of ``w^2`` (L2) over the
weights of the hidden transitions. Biases and the readout carry no penalty.
"""
from __future__ import annotations

import torch

from ..models import DenseMLP
from .base import Method, register


def _hidden_weights(model):
    return tuple(layer.weight for layer in model.hidden_layers)


class _WeightPenalty(Method):
    hp_keys = frozenset({"lam"})

    def build(self, cfg, target):
        return DenseMLP(target.d, cfg["width"], cfg["hidden"])

    @staticmethod
    def _magnitude(weight):
        raise NotImplementedError

    def penalty(self, model, hp):
        lam = float(hp["lam"])
        if lam < 0:
            raise ValueError(f"lam must be non-negative; received {lam}")
        weights = _hidden_weights(model)
        count = sum(w.numel() for w in weights)
        total = torch.stack([self._magnitude(w).sum() for w in weights]).sum()
        return lam * total / count


@register("l1")
class L1(_WeightPenalty):
    @staticmethod
    def _magnitude(weight):
        return weight.abs()


@register("l2")
class L2(_WeightPenalty):
    @staticmethod
    def _magnitude(weight):
        return weight.square()
