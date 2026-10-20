"""Slot G -- `dense_anova`: one anova node over all A signal leaves (§5.3).
The decomposability-removal negative control -- never a matched tree baseline:
tree-vs-dense conflates decomposition with local dimensionality (the 8-ary node
is hard even with teacher inputs), so the licensed reading is "remove low-arity
decomposability and the target stops being learnable at these n", nothing more.
First-order mass stays 1-rho (equal to flat, NOT to the tree's (1-rho)^L)."""
from __future__ import annotations

from collections.abc import Mapping

from ..registry import task
from .graph import BalancedGraphFamily
from .laws import anova_law_from_cell

ANOVA_INPUTS = ("uniform_pm1", "uniform", "normal")


@task("dense_anova")
class DenseAnova(BalancedGraphFamily):
    """y = sqrt((1-rho)/A) sum_i s_i l_i + sqrt(rho) s' prod_i l_i, interaction at order A."""

    cell_keys = ("A", "rho", "sigma_shape")
    cell_defaults = {"A": 8, "rho": 0.5, "sigma_shape": 0.0}
    allowed_inputs = ANOVA_INPUTS

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        A = int(cell["A"])
        if A < 2:
            raise ValueError(f"dense_anova requires A >= 2; received {A}")
        return A

    def make_law(self, cfg: Mapping):
        return anova_law_from_cell(cfg)

    def family_options(self, cfg: Mapping, law) -> dict:
        return {**law.describe(), "A": int(cfg["A"])}

    def draw(self, cfg: Mapping, function_seed: int):
        cfg = self.resolve_cell(cfg)
        A = int(cfg["A"])
        return super().draw({**cfg, "k": A, "layer_profile": (1,)}, function_seed)
