"""Slot A -- `node_anova`: one anova node on k of d coordinates (§5.3).
The M1 sanity anchor: local fitting plus sparse-input discovery, no hierarchy."""
from __future__ import annotations

from collections.abc import Mapping

from ..registry import task
from .graph import BalancedGraphFamily
from .laws import anova_law_from_cell

ANOVA_INPUTS = ("uniform_pm1", "uniform", "normal")


@task("node_anova")
class NodeAnova(BalancedGraphFamily):
    """Single-node control drawn at the same knobs as every anova workload."""

    cell_keys = ("k", "rho", "sigma_shape")
    cell_defaults = {"k": 2, "rho": 0.5, "sigma_shape": 0.0}
    allowed_inputs = ANOVA_INPUTS

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        k = int(cell["k"])
        if k < 2:
            raise ValueError(f"node_anova requires k >= 2; received {k}")
        return k

    def make_law(self, cfg: Mapping):
        return anova_law_from_cell(cfg)

    def draw(self, cfg: Mapping, function_seed: int):
        return super().draw({**self.resolve_cell(cfg), "layer_profile": (1,)}, function_seed)
