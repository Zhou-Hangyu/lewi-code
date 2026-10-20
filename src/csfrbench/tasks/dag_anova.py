"""Slot E -- `dag_anova`: the shared-intermediate topology, additive readout (§5.2).

A DAG extends the tree by letting one intermediate computation feed several
downstream nodes, which introduces reuse. Fixed wiring on 6 leaves, where the
level-1 intermediate z_1 on the scope (l_1, l_2) feeds both consumers:

    z_1 = g_1(l_1, l_2)
    z_2 = g_2(l_3, l_4)       z_3 = g_3(l_5, l_6)
    z_4 = g_4(z_1, z_2)       z_5 = g_5(z_1, z_3)
    y = (z_4 + z_5) / sqrt(Var(z_4 + z_5))

Every nonlinear node keeps independent parents, so the parent-level ANOVA law
is exact at every node; only the linear readout merges shared ancestry. Plain
readout normalizer is analytic through the exact multilinear expansion; shaped
is audited-MC (§7)."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..registry import task
from ..utils.seeds import derive
from .base import TaskFamily
from .graph import GraphSpec, draw_graph_target
from .laws import anova_law_from_cell

ANOVA_INPUTS = ("uniform_pm1", "uniform", "normal")

# Leaves 0..5; node refs are 6 + node index. Slots 0 and 1 are the two
# consumer-facing references to the one shared intermediate z_1: the tie below
# gives both slots the same function object, so z_1 is computed once and read
# by g_4 and g_5.
NODE_PARENTS = (
    (0, 1),     # z1 as seen by z4
    (0, 1),     # z1 as seen by z5 (tied to the slot above)
    (2, 3),     # z2
    (4, 5),     # z3
    (6, 8),     # z4 = g4(z1, z2)
    (7, 9),     # z5 = g5(z1, z3)
)
READOUT = (4, 5)
SHARED_SLOTS = ((1, 0),)        # slot 1 carries slot 0's function object

# Depth-4 template (frozen-1.7 era validation/stress cell): the same single
# shared-intermediate motif one level deeper. 14 leaves; seven level-1 nodes
# with z1 shared by two level-2 consumers; two level-3 nodes; additive
# readout. Every nonlinear node keeps independent parents (scopes disjoint
# within each node); only the readout merges the shared ancestry -- the same
# invariants as depth 3, so this varies depth+reuse, not the reuse pattern.
#
#   L1: z1=g(l0,l1)*  z2=g(l2,l3)  z3=g(l4,l5)   z4=g(l6,l7)
#       z5=g(l8,l9)  z6=g(l10,l11) z7=g(l12,l13)     (* shared)
#   L2: z8=g(z1,z2)  z9=g(z1,z3)  z10=g(z4,z5)  z11=g(z6,z7)
#   L3: z12=g(z8,z10)  z13=g(z9,z11)
#   y  = (z12 + z13) / sqrt(Var)
D4_NODE_PARENTS = (
    (0, 1),      # s0: z1 as seen by z8
    (0, 1),      # s1: z1 as seen by z9 (tied to s0)
    (2, 3),      # s2: z2
    (4, 5),      # s3: z3
    (6, 7),      # s4: z4
    (8, 9),      # s5: z5
    (10, 11),    # s6: z6
    (12, 13),    # s7: z7
    (14, 16),    # s8: z8  = g(z1, z2)
    (15, 17),    # s9: z9  = g(z1, z3)
    (18, 19),    # s10: z10 = g(z4, z5)
    (20, 21),    # s11: z11 = g(z6, z7)
    (22, 24),    # s12: z12 = g(z8, z10)
    (23, 25),    # s13: z13 = g(z9, z11)
)
D4_READOUT = (12, 13)
D4_SHARED_SLOTS = ((1, 0),)


@task("dag_anova")
class DagAnova(TaskFamily):
    """Is a shared intermediate discovered and reused?"""

    cell_keys = ("rho", "sigma_shape", "depth")
    cell_defaults = {"rho": 0.5, "sigma_shape": 0.0, "depth": 3}
    allowed_inputs = ANOVA_INPUTS

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
        ties = D4_SHARED_SLOTS if depth == 4 else SHARED_SLOTS
        profile = (7, 4, 2, 1) if depth == 4 else (4, 2, 1)
        d = int(cfg["d"])
        if d < A:
            raise ValueError(f"d={d} < A={A}: target cannot fit in ambient space")
        law = anova_law_from_cell(cfg)

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
            parents_attr=tuple(np.asarray(p, dtype=np.int64) for p in parents),
            layer_profile=profile,
            k=2,
            task_options=law.describe(),
            wiring={"mode": "shared_dag"},
            ties=ties,
            allowed_inputs=self.allowed_inputs,
        )
