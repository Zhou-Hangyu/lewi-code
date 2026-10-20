"""`dag_fourier`: the shared-intermediate topology under the fourier law.

Same fixed 6-leaf wiring as `dag_anova` (one level-1 intermediate feeds two
consumers; additive readout), with fourier local nodes. Completes the fourier
column of the candidate envelope where the reuse construct is distinct;
calibration is audited-MC, as for every fourier target.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..registry import task
from ..utils.seeds import derive
from .base import TaskFamily
from .dag_anova import (D4_NODE_PARENTS, D4_READOUT, D4_SHARED_SLOTS,
                        NODE_PARENTS, READOUT, SHARED_SLOTS)
from .graph import GraphSpec, draw_graph_target
from .laws import FourierLaw, _acceptance_is_catastrophic
from .topologies import DAG8_A, DAG8_NODE_PARENTS, DAG8_PROFILE, dag8_graph


@task("dag_fourier")
class DagFourier(TaskFamily):
    """Is a shared intermediate discovered and reused, without the ANOVA scaffolding?"""

    cell_keys = ("sigma_omega", "depth")
    cell_defaults = {"sigma_omega": 0.85, "depth": 3}

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        return 14 if int(cell.get("depth", 3)) == 4 else 6

    def draw(self, cfg: Mapping, function_seed: int):
        cfg = self.resolve_cell(cfg)
        family = self.registry_name
        depth = int(cfg.get("depth", 3))
        A = 14 if depth == 4 else 6
        parents = D4_NODE_PARENTS if depth == 4 else NODE_PARENTS
        readout = D4_READOUT if depth == 4 else READOUT
        d = int(cfg["d"])
        if d < A:
            raise ValueError(f"d={d} < A={A}: target cannot fit in ambient space")
        law = FourierLaw(float(cfg["sigma_omega"]))

        if depth == 4:
            structural_rng = np.random.default_rng(
                derive(family, int(function_seed), "structure", "d4"))
            embedding_rng = np.random.default_rng(
                derive(family, int(function_seed), int(d), "embedding", "d4"))
        else:
            structural_rng = np.random.default_rng(derive(family, int(function_seed), "structure"))
            embedding_rng = np.random.default_rng(
                derive(family, int(function_seed), int(d), "embedding")
            )
        active_raw = embedding_rng.choice(d, size=A, replace=False)

        graph = GraphSpec(A, parents, readout, (1.0, 1.0))

        return draw_graph_target(
            family=family,
            cfg=cfg,
            function_seed=function_seed,
            law=law,
            graph=graph,
            structural_rng=structural_rng,
            active_raw=active_raw,
            parents_attr=tuple(np.asarray(pp, dtype=np.int64) for pp in parents),
            layer_profile=(7, 4, 2, 1) if depth == 4 else (4, 2, 1),
            k=2,
            task_options=law.describe(),
            wiring={"mode": "shared_dag"},
            ties=D4_SHARED_SLOTS if depth == 4 else SHARED_SLOTS,
            allowed_inputs=self.allowed_inputs,
        )


@task("dag_fourier_v2")
class DagFourierV2(TaskFamily):
    """CSFRBench v0.5 fourier reuse DAG: the fixed 8-input reuse-DAG topology
    (topologies.dag8_graph) with random Fourier local functions -- checks that
    the reuse-topology conclusion is not specific to the primary construction.
    """

    cell_keys = ("sigma_omega", "acceptance")
    cell_defaults = {"sigma_omega": 0.85}

    @classmethod
    def check_cell(cls, cell) -> int:
        return DAG8_A

    def draw(self, cfg, function_seed: int):
        cfg = self.resolve_cell(cfg)
        family = self.registry_name
        d = int(cfg["d"])
        if d < DAG8_A:
            raise ValueError(f"d={d} < A={DAG8_A}: target cannot fit in ambient space")
        law = FourierLaw(float(cfg["sigma_omega"]),
                         catastrophic_only=_acceptance_is_catastrophic(cfg))

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
