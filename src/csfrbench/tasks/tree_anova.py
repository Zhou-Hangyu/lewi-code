"""Slot C -- `tree_anova`: the primary family. Balanced k-ary anova tree (§5.1).
Frozen roles: plain (sigma_shape = 0) is the mechanistic/analytic primary;
shaped at sigma_shape = 1 is the empirical primary (a frozen setting, not a
difficulty axis). Defaults L = 3, profile (4, 2, 1), k = 2, rho = 0.5; the core
arity sweep is k in {2, 3}, with k = 4 an M1 stress condition (§5.1)."""
from __future__ import annotations

from collections.abc import Mapping

from ..registry import task
from .graph import BalancedGraphFamily
from .laws import anova_law_from_cell

ANOVA_INPUTS = ("uniform_pm1", "uniform", "normal")


@task("tree_anova")
class TreeAnova(BalancedGraphFamily):
    """Hierarchical discovery under designed guidance: per-node split (1-rho, rho)."""

    cell_keys = ("k", "layer_profile", "topology", "A", "rho", "sigma_shape")
    cell_defaults = {
        "k": 2, "layer_profile": (4, 2, 1), "topology": "balanced",
        "rho": 0.5, "sigma_shape": 0.0,
    }
    allowed_inputs = ANOVA_INPUTS

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        topology = str(cell.get("topology", "balanced"))
        if topology != "balanced":
            raise ValueError(f"unknown topology {topology!r}; expected balanced")
        return super().check_cell(cell)

    def make_law(self, cfg: Mapping):
        return anova_law_from_cell(cfg)

    def family_options(self, cfg: Mapping, law) -> dict:
        return {**law.describe(), "topology": "balanced"}
