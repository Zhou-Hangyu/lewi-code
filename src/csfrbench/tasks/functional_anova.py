"""CSFRBench v0.5 primary families: functional-ANOVA local functions with
sinusoidal factors.

The local function is the k-input functional-ANOVA construction

    g(z) = sqrt((1-rho)/k) sum_i a_i(z_i) + sqrt(rho) prod_i b_i(z_i)

whose one-dimensional factors are sinusoids with an interpretable frequency in
cycles, s(u) = sin(2*pi*(f*u/2 + phi)) with f ~ U[f_lo, f_hi] and
phi ~ U[0, 1), centered and standardized under the actual (induced) parent
distribution (laws.AnovaLaw with f_range). Sequential normalization: every
node output is affinely standardized on the target-specific calibration pool,
the stored transforms are part of the realized target, and the additive
readout is standardized the same way (v0.5 §6-7).

Families (registry ids):
  node_functional_anova   one k-input local function; input selection + fitting
  flat_functional_anova   w disjoint pairs under an additive readout
  tree_functional_anova   balanced binary tree; depth 3 primary, depth 4 via
                          layer_profile (8, 4, 2, 1) as depth/scale
                          generalization
  dag_functional_anova    the fixed 8-input reuse DAG (topologies.dag8_graph)
  rand_functional_anova   held-out random graphs, exactly 7 local functions
  dense_functional_anova  one 8-input local function; high-arity stress setting

The frequency interval f_range is a frozen design constant of the benchmark
version; until the authors freezes it, cells must state it explicitly.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..registry import task
from ..utils.seeds import derive
from .base import TaskFamily
from .graph import BalancedGraphFamily, GraphSpec, draw_graph_target
from .laws import functional_anova_law_from_cell
from .topologies import (
    DAG8_A,
    DAG8_NODE_PARENTS,
    DAG8_PROFILE,
    canonical_topology_hash,
    dag8_graph,
    graph_profile7,
    sample_random_graph7,
)

FANOVA_INPUTS = ("uniform_pm1", "uniform", "normal")


@task("node_functional_anova")
class NodeFunctionalAnova(BalancedGraphFamily):
    """Local fitting plus active-input selection, no hierarchy."""

    cell_keys = ("k", "rho", "f_range")
    cell_defaults = {"k": 2, "rho": 0.5}
    allowed_inputs = FANOVA_INPUTS

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        k = int(cell["k"])
        if k < 2:
            raise ValueError(f"node_functional_anova requires k >= 2; received {k}")
        return k

    def make_law(self, cfg: Mapping):
        return functional_anova_law_from_cell(cfg)

    def draw(self, cfg: Mapping, function_seed: int):
        return super().draw({**self.resolve_cell(cfg), "layer_profile": (1,)},
                            function_seed)


@task("dense_functional_anova")
class DenseFunctionalAnova(BalancedGraphFamily):
    """High-arity stress setting: one local function over all A inputs."""

    cell_keys = ("A", "rho", "f_range")
    cell_defaults = {"A": 8, "rho": 0.5}
    allowed_inputs = FANOVA_INPUTS

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        A = int(cell["A"])
        if A < 2:
            raise ValueError(f"dense_functional_anova requires A >= 2; received {A}")
        return A

    def make_law(self, cfg: Mapping):
        return functional_anova_law_from_cell(cfg)

    def family_options(self, cfg: Mapping, law) -> dict:
        return {**law.describe(), "A": int(cfg["A"])}

    def draw(self, cfg: Mapping, function_seed: int):
        cfg = self.resolve_cell(cfg)
        A = int(cfg["A"])
        return super().draw({**cfg, "k": A, "layer_profile": (1,)}, function_seed)


@task("tree_functional_anova")
class TreeFunctionalAnova(BalancedGraphFamily):
    """Balanced binary tree: the primary hierarchical workload (three
    nonlinear levels; public name "balanced tree, depth 4" - readout node
    counted, amendment 13), and depth/scale generalization at
    layer_profile (8, 4, 2, 1) (public depth 5)."""

    cell_keys = ("k", "layer_profile", "rho", "f_range")
    cell_defaults = {"k": 2, "layer_profile": (4, 2, 1), "rho": 0.5}
    allowed_inputs = FANOVA_INPUTS

    def make_law(self, cfg: Mapping):
        return functional_anova_law_from_cell(cfg)


@task("flat_functional_anova")
class FlatFunctionalAnova(TaskFamily):
    """w disjoint pairwise local functions under one additive readout."""

    cell_keys = ("w", "rho", "f_range")
    cell_defaults = {"w": 4, "rho": 0.5}
    allowed_inputs = FANOVA_INPUTS

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        w = int(cell["w"])
        if w < 2:
            raise ValueError(f"flat_functional_anova requires w >= 2 modules; received {w}")
        return 2 * w

    def draw(self, cfg: Mapping, function_seed: int):
        cfg = self.resolve_cell(cfg)
        family = self.registry_name
        d, w = int(cfg["d"]), int(cfg["w"])
        A = 2 * w
        if d < A:
            raise ValueError(f"d={d} < A={A}: target cannot fit in ambient space")
        law = functional_anova_law_from_cell(cfg)

        structural_rng = np.random.default_rng(derive(family, int(function_seed), "structure"))
        embedding_rng = np.random.default_rng(
            derive(family, int(function_seed), int(d), "embedding")
        )
        active_raw = embedding_rng.choice(d, size=A, replace=False).reshape(w, 2)

        node_parents = [(2 * j, 2 * j + 1) for j in range(w)]
        graph = GraphSpec(A, node_parents, tuple(range(w)), (1.0,) * w)

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


@task("dag_functional_anova")
class DagFunctionalAnova(TaskFamily):
    """The fixed 8-input reuse DAG: 7 local functions, a2/a3 each read twice,
    shared ancestry merged only at the standardized additive readout."""

    cell_keys = ("rho", "f_range")
    cell_defaults = {"rho": 0.5}
    allowed_inputs = FANOVA_INPUTS

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        return DAG8_A

    def draw(self, cfg: Mapping, function_seed: int):
        cfg = self.resolve_cell(cfg)
        family = self.registry_name
        d = int(cfg["d"])
        if d < DAG8_A:
            raise ValueError(f"d={d} < A={DAG8_A}: target cannot fit in ambient space")
        law = functional_anova_law_from_cell(cfg)

        structural_rng = np.random.default_rng(derive(family, int(function_seed), "structure"))
        embedding_rng = np.random.default_rng(
            derive(family, int(function_seed), int(d), "embedding")
        )
        active_raw = embedding_rng.choice(d, size=DAG8_A, replace=False)
        graph = dag8_graph()

        return draw_graph_target(
            family=family,
            cfg=cfg,
            function_seed=function_seed,
            law=law,
            graph=graph,
            structural_rng=structural_rng,
            active_raw=active_raw,
            parents_attr=tuple(np.asarray(p, dtype=np.int64) for p in DAG8_NODE_PARENTS),
            layer_profile=DAG8_PROFILE,
            k=2,
            task_options=law.describe(),
            wiring={"mode": "reuse_dag8"},
            allowed_inputs=self.allowed_inputs,
        )


@task("rand_functional_anova")
class RandFunctionalAnova(TaskFamily):
    """Held-out random graphs, exactly 7 local functions: topology
    generalization across computation-graph draws."""

    cell_keys = ("k", "rho", "f_range")
    cell_defaults = {"rho": 0.5}
    allowed_inputs = FANOVA_INPUTS

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        return DAG8_A

    def draw(self, cfg: Mapping, function_seed: int):
        cfg = self.resolve_cell(cfg)
        family = self.registry_name
        d = int(cfg["d"])
        if d < DAG8_A:
            raise ValueError(f"d={d} < A={DAG8_A}: target cannot fit in ambient space")
        if int(cfg.get("k", 2)) != 2:
            raise ValueError(f"{family} is frozen at k = 2")
        law = functional_anova_law_from_cell(cfg)

        structural_rng = np.random.default_rng(derive(family, int(function_seed), "structure"))
        embedding_rng = np.random.default_rng(
            derive(family, int(function_seed), int(d), "embedding")
        )
        graph = sample_random_graph7(structural_rng)
        active_raw = embedding_rng.choice(d, size=DAG8_A, replace=False)

        target = draw_graph_target(
            family=family,
            cfg=cfg,
            function_seed=function_seed,
            law=law,
            graph=graph,
            structural_rng=structural_rng,
            active_raw=active_raw,
            parents_attr=tuple(np.asarray(p, dtype=np.int64) for p in graph.node_parents),
            layer_profile=graph_profile7(graph),
            k=2,
            task_options=law.describe(),
            wiring={"mode": "random_layered_7"},
            allowed_inputs=self.allowed_inputs,
        )
        target.topology_hash = canonical_topology_hash(graph)
        return target
