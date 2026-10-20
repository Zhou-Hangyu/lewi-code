"""Slot B -- `flat_anova`: w disjoint anova pairs under an additive root (§5.3).
y = w^{-1/2} sum_j g_j(l_{2j-1}, l_{2j}); Gamma_1 = 1 - rho exactly. The core
suite runs the law-matched rho; the guidance-matched rho' = 1 - (1 - rho_tree)^L
cell is appendix material (§6.3, §10)."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..registry import task
from ..utils.seeds import derive
from .base import TaskFamily
from .graph import GraphSpec, draw_graph_target
from .laws import anova_law_from_cell

ANOVA_INPUTS = ("uniform_pm1", "uniform", "normal")


@task("flat_anova")
class FlatAnova(TaskFamily):
    """Discovery of multiple sparse interactions without hierarchy."""

    cell_keys = ("w", "rho", "sigma_shape")
    cell_defaults = {"w": 4, "rho": 0.5, "sigma_shape": 0.0}
    allowed_inputs = ANOVA_INPUTS

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        w = int(cell["w"])
        if w < 2:
            raise ValueError(f"flat_anova requires w >= 2 modules; received {w}")
        return 2 * w

    def draw(self, cfg: Mapping, function_seed: int):
        cfg = self.resolve_cell(cfg)
        family = self.registry_name
        d, w = int(cfg["d"]), int(cfg["w"])
        A = 2 * w
        if d < A:
            raise ValueError(f"d={d} < A={A}: target cannot fit in ambient space")
        law = anova_law_from_cell(cfg)

        structural_rng = np.random.default_rng(derive(family, int(function_seed), "structure"))
        embedding_rng = np.random.default_rng(
            derive(family, int(function_seed), int(d), "embedding")
        )
        active_raw = embedding_rng.choice(d, size=A, replace=False).reshape(w, 2)

        node_parents = [(2 * j, 2 * j + 1) for j in range(w)]
        weight = 1.0 / float(np.sqrt(w))
        graph = GraphSpec(A, node_parents, tuple(range(w)), (weight,) * w)

        return draw_graph_target(
            family=family,
            cfg=cfg,
            function_seed=function_seed,
            law=law,
            graph=graph,
            structural_rng=structural_rng,
            active_raw=active_raw,
            parents_attr=(np.arange(w, dtype=np.int64).reshape(1, w),),
            layer_profile=(w, 1),
            k=2,
            task_options={**law.describe(), "w": w},
            wiring={"mode": "flat_additive"},
            allowed_inputs=self.allowed_inputs,
        )
