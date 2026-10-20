"""`rand_anova` / `rand_fourier`: random layered-DAG extension families.

Graph-generalization extension (graph-generalization.md; not part of the frozen
core suite). Each function seed draws a previously unseen computation graph
from a constrained distribution, then resamples local functions exactly as the
core families do. Frozen constraint envelope:

  * A = 8 active leaves, local arity k = 2, depth exactly 3;
  * 6-8 local-function nodes; connected; one standardized scalar readout;
  * every node's parents have disjoint leaf ancestry (the GraphSpec merge
    tripwire), so the parent-level ANOVA construction is exact at every node
    and only the additive readout merges shared ancestry;
  * never isomorphic to a frozen template: two or more base strands exclude
    the single-root complete tree, A = 8 excludes the 6-leaf dag, and exact
    depth 3 excludes node/flat/dense.

The sampler composes two phases from the structural stream (d never enters, so
the graph and functions are bit-identical across ambient dimensions):

  1. base forest -- partition the 8 leaves into 2-3 strands and grow a random
     binary tree over each (balanced, caterpillar, and mixed shapes arise);
  2. reuse taps -- 0-2 extra nodes, each reading one already-consumed ref (an
     intermediate or a leaf: fan-out / scope overlap) plus one ancestry-
     disjoint partner; every tap becomes an extra readout term. No two nodes
     may share an unordered parent pair (two functions of one scope collapse
     to one effective node when both are readout terms).

Calibration follows the section-7 policy unchanged: plain anova is exact
through the multilinear expansion, shaped anova and fourier are audited
deterministic MC on the fixed pool.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..registry import task
from ..utils.seeds import derive
from .base import TaskFamily
from .graph import GraphSpec, draw_graph_target
from .laws import FourierLaw, anova_law_from_cell

RAND_A = 8
RAND_K = 2
RAND_DEPTH = 3
RAND_NODES = (6, 8)             # inclusive bounds on local-function nodes
_MAX_DRAW_TRIES = 64
_MAX_GROUP_TRIES = 16

# Leaf partitions per strand count; each group must admit a depth<=3 binary tree.
_PARTITIONS = {2: ((2, 6), (3, 5), (4, 4)), 3: ((2, 2, 4), (2, 3, 3))}


class _Unit:
    """One wiring unit during sampling: a ref with its depth and leaf ancestry."""

    __slots__ = ("ref", "depth", "leaves")

    def __init__(self, ref: int, depth: int, leaves: frozenset):
        self.ref, self.depth, self.leaves = ref, depth, leaves


def _grow_strand(leaf_refs, rng, node_parents):
    """Random binary tree over one leaf group; returns its root unit or None."""
    for _ in range(_MAX_GROUP_TRIES):
        units = [_Unit(r, 0, frozenset((r,))) for r in leaf_refs]
        parents_added: list[tuple[int, int]] = []
        ok = True
        while len(units) > 1:
            pairs = [
                (i, j)
                for i in range(len(units))
                for j in range(i + 1, len(units))
                if max(units[i].depth, units[j].depth) + 1 <= RAND_DEPTH
            ]
            if not pairs:
                ok = False
                break
            i, j = pairs[rng.integers(len(pairs))]
            a, b = units[i], units[j]
            ref = RAND_A + len(node_parents) + len(parents_added)
            parents_added.append((a.ref, b.ref))
            merged = _Unit(ref, max(a.depth, b.depth) + 1, a.leaves | b.leaves)
            units = [u for idx, u in enumerate(units) if idx not in (i, j)]
            units.append(merged)
        if ok:
            node_parents.extend(parents_added)
            return units[0]
    return None


def _depth_of(ref: int, node_parents) -> int:
    if ref < RAND_A:
        return 0
    a, b = node_parents[ref - RAND_A]
    return 1 + max(_depth_of(a, node_parents), _depth_of(b, node_parents))


def _leaves_of(ref: int, node_parents) -> frozenset:
    if ref < RAND_A:
        return frozenset((ref,))
    a, b = node_parents[ref - RAND_A]
    return _leaves_of(a, node_parents) | _leaves_of(b, node_parents)


def sample_random_graph(rng: np.random.Generator) -> GraphSpec:
    """One constrained random layered DAG; deterministic in the rng stream."""
    for _ in range(_MAX_DRAW_TRIES):
        n_strands = int(rng.choice((2, 3)))
        sizes = _PARTITIONS[n_strands][rng.integers(len(_PARTITIONS[n_strands]))]
        leaf_order = rng.permutation(RAND_A)

        node_parents: list[tuple[int, int]] = []
        roots: list[_Unit] = []
        pos = 0
        failed = False
        for size in sizes:
            group = [int(r) for r in leaf_order[pos:pos + size]]
            pos += size
            root = _grow_strand(group, rng, node_parents)
            if root is None:
                failed = True
                break
            roots.append(root)
        if failed or max(u.depth for u in roots) != RAND_DEPTH:
            continue

        base_nodes = len(node_parents)          # 8 - n_strands
        lo, hi = RAND_NODES
        tap_options = [t for t in (0, 1, 2) if lo <= base_nodes + t <= hi]
        n_taps = int(rng.choice(tap_options))
        taps: list[_Unit] = []
        for _tap in range(n_taps):
            refs = list(range(RAND_A + len(node_parents)))
            sources = [
                r for r in refs
                if 1 <= _depth_of(r, node_parents) <= RAND_DEPTH - 1
            ]
            if rng.random() < 0.25:             # scope overlap through a leaf
                sources += [r for r in refs if r < RAND_A]
            x = int(sources[rng.integers(len(sources))])
            x_leaves = _leaves_of(x, node_parents)
            partners = [
                r for r in refs
                if not (_leaves_of(r, node_parents) & x_leaves)
                and max(_depth_of(x, node_parents), _depth_of(r, node_parents))
                + 1 <= RAND_DEPTH
            ]
            existing = {frozenset(p) for p in node_parents}
            partners = [r for r in partners if frozenset((x, r)) not in existing]
            if not partners:
                continue
            y = int(partners[rng.integers(len(partners))])
            ref = RAND_A + len(node_parents)
            node_parents.append((x, y) if rng.random() < 0.5 else (y, x))
            taps.append(_Unit(ref, _depth_of(ref, node_parents), frozenset()))

        readout = tuple(u.ref - RAND_A for u in roots + taps)
        if not (lo <= len(node_parents) <= hi) or len(readout) < 2:
            continue
        try:
            return GraphSpec(
                RAND_A, node_parents, readout, (1.0,) * len(readout)
            )
        except ValueError:                      # tripwire: reject and resample
            continue
    raise RuntimeError("random-graph sampler exhausted its retry budget")


def graph_profile(graph: GraphSpec) -> tuple[int, ...]:
    """Nodes per depth level plus the readout row (identity metadata)."""
    widths = [0] * RAND_DEPTH
    for v in range(graph.S):
        widths[_depth_of(RAND_A + v, graph.node_parents) - 1] += 1
    return (*widths, 1)


class _RandomGraphFamily(TaskFamily):
    """Shared draw for the random layered-DAG extension families."""

    allowed_inputs = ("uniform_pm1", "uniform", "normal")

    def make_law(self, cfg: Mapping):
        raise NotImplementedError

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        return RAND_A

    def draw(self, cfg: Mapping, function_seed: int):
        cfg = self.resolve_cell(cfg)
        family = self.registry_name
        d = int(cfg["d"])
        if d < RAND_A:
            raise ValueError(f"d={d} < A={RAND_A}: target cannot fit in ambient space")
        if int(cfg.get("k", RAND_K)) != RAND_K:
            raise ValueError(f"{family} is frozen at k = {RAND_K}")
        law = self.make_law(cfg)

        structural_rng = np.random.default_rng(derive(family, int(function_seed), "structure"))
        embedding_rng = np.random.default_rng(
            derive(family, int(function_seed), int(d), "embedding")
        )
        graph = sample_random_graph(structural_rng)
        active_raw = embedding_rng.choice(d, size=RAND_A, replace=False)

        return draw_graph_target(
            family=family,
            cfg=cfg,
            function_seed=function_seed,
            law=law,
            graph=graph,
            structural_rng=structural_rng,
            active_raw=active_raw,
            parents_attr=tuple(np.asarray(p, dtype=np.int64) for p in graph.node_parents),
            layer_profile=graph_profile(graph),
            k=RAND_K,
            task_options=law.describe(),
            wiring={"mode": "random_layered"},
            allowed_inputs=self.allowed_inputs,
        )


@task("rand_anova")
class RandAnova(_RandomGraphFamily):
    """Does hierarchical discovery survive on unseen computation graphs?"""

    cell_keys = ("rho", "sigma_shape")
    cell_defaults = {"rho": 0.5, "sigma_shape": 0.0}

    def make_law(self, cfg: Mapping):
        return anova_law_from_cell(cfg)


@task("rand_fourier")
class RandFourier(_RandomGraphFamily):
    """Non-ANOVA confirmation of the graph-generalization result."""

    cell_keys = ("sigma_omega",)
    cell_defaults = {"sigma_omega": 0.85}

    def make_law(self, cfg: Mapping):
        return FourierLaw(float(cfg.get("sigma_omega", 0.85)))
