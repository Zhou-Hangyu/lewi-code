"""Validation slot -- `tree_fourier`: balanced tree of Q = 8 random-Fourier nodes
(§4.2, §5.4): generic nonseparable multivariate surfaces with realized intrinsic
dimensionality audited. sigma_omega moves bandwidth and guidance together
(entangled by design); guidance is measured, never designed -- analyses stratify
on realized Gamma_1. Frozen scope: k = 2 only; fourier is excluded from the
general k sweep, and any arity study must recalibrate Q jointly with k."""
from __future__ import annotations

from collections.abc import Mapping

from ..registry import task
from .graph import BalancedGraphFamily
from .laws import FourierLaw, _acceptance_is_catastrophic


def _check_frozen_k(cell: Mapping) -> None:
    if int(cell.get("k", 2)) != 2:
        raise ValueError(
            "tree_fourier is frozen at k = 2 (csfrbench-targets/freeze-1.0): "
            "fourier is excluded from the k sweep; an arity study must "
            "recalibrate Q jointly with k and re-audit realized intrinsic "
            "dimension (target-design.md §4.2)"
        )


@task("tree_fourier")
class TreeFourier(BalancedGraphFamily):
    """Do slot C's conclusions survive without the ANOVA scaffolding?"""

    cell_keys = ("k", "layer_profile", "sigma_omega", "acceptance")
    cell_defaults = {"k": 2, "layer_profile": (4, 2, 1), "sigma_omega": 0.85}

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        _check_frozen_k(cell)
        return super().check_cell(cell)

    def make_law(self, cfg: Mapping):
        _check_frozen_k(cfg)
        return FourierLaw(float(cfg.get("sigma_omega", 0.85)),
                          catastrophic_only=_acceptance_is_catastrophic(cfg))
