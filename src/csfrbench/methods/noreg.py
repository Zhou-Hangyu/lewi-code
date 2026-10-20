"""Unregularized dense-MLP benchmark arm."""

from __future__ import annotations

from ..registry import method
from .base import Method
from .nns import DenseMLP


@method("noreg")
class NoReg(Method):
    """Unregularized dense MLP baseline under the shared training protocol."""

    hp_keys = frozenset()

    def build(self, cfg, target):
        return DenseMLP(
            input_dim=int(target.d),
            width=int(cfg["width"]),
            num_hidden_layers=int(cfg["hidden"]),
        )

@method("dense")
class Dense(NoReg):
    """Canonical v1 name for the unregularized dense baseline (spec section 10)."""
