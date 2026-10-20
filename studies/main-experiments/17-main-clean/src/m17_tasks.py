"""17-main-clean: the 16-input, all-active workload suite.

Registers two new task families and the presets the suite needs. Nothing here
changes an existing family, preset or record as run: the new families have
their own registry ids and the new presets have their own names, so every
08-main, 12-extra and 16-main-final target keeps its fingerprint.

The reuse DAG on 16 leaves (`dag16_functional_anova`) mirrors the v0.5
8-leaf DAG one level up. v0.5: 4 disjoint level-1 nodes, 3 sliding level-2
nodes that reuse the two middle level-1 nodes, and a 3-node standardized
additive readout. Here:

    L1  a1..a8   = g(x1,x2) ... g(x15,x16)          8 nodes, disjoint pairs
    L2  b1..b4   = g(a1,a2) g(a3,a4) g(a5,a6) g(a7,a8)   4 nodes, disjoint
    L3  c1..c3   = g(b1,b2) g(b2,b3) g(b3,b4)       3 nodes; b2, b3 reused
    y            = (c1 + c2 + c3) standardized

15 local functions on 16 active inputs, matching the 16-leaf balanced tree in
active-input count, local-function count and local arity. Every law node has
parents with disjoint leaf ancestries, so the parent-level ANOVA law is exact
at every node and only the linear readout merges shared ancestry - the same
invariant the v0.5 DAG carries.

The random graphs on 16 leaves (`rand16_functional_anova`) carry the
benchmark's held-out graph distribution (graph-generalization extension,
csfrbench.md section 17; the suite family `rand_functional_anova` pins 8
leaves, depth 3, exactly 7 nodes) to the 16-input regime with the same
two-phase construction and the constraint envelope scaled to the 16-leaf
tree:

    active leaves       16 (all of d = 16)                 was 8
    local arity         2                                  same
    depth               exactly 4, the depth of tree16     was 3
    local functions     exactly 15, as tree16 and dag16    was 7
    base forest         2, 3 or 4 strands over a random    was 2 or 3
                        leaf partition (each strand at
                        least 2 leaves), a random binary
                        tree per strand
    reuse taps          strands - 1 (1 to 3), each reading   was 1 or 2
                        one consumed intermediate (or a
                        leaf, probability 1/4) plus one
                        ancestry-disjoint partner
    readout             equal-weight sum of the strand      same rule
                        roots and the taps (3 to 7 terms)

Every other rule of the sampler is copied unchanged: the depth cap during
strand growth, rejection of a forest whose deepest root is not at the full
depth, the ban on a duplicated unordered parent pair, the merge tripwire
of GraphSpec, and the seed streams (`derive(family, seed, "structure")` for
the graph and the local functions, so the graph never depends on d;
`derive(family, seed, d, "embedding")` for the active coordinates, which at
d = 16 is a permutation of all sixteen). The graph a function seed carries
is identified by `canonical_topology_hash`, as in the benchmark.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from csfrbench.registry import task
from csfrbench.tasks.base import TaskFamily
from csfrbench.tasks.graph import GraphSpec, draw_graph_target
from csfrbench.tasks.laws import functional_anova_law_from_cell
from csfrbench.tasks.topologies import canonical_topology_hash
from csfrbench.utils.seeds import derive

FANOVA_INPUTS = ("uniform_pm1", "uniform", "normal")

DAG16_A = 16
DAG16_NODE_PARENTS = (
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),   # a1..a8
    (16, 17), (18, 19), (20, 21), (22, 23),                                  # b1..b4
    (24, 25), (25, 26), (26, 27),                                            # c1..c3; b2, b3 reused
)
DAG16_READOUT = (12, 13, 14)          # c1, c2, c3
DAG16_PROFILE = (8, 4, 3, 1)          # nodes per level plus the readout row


def dag16_graph() -> GraphSpec:
    return GraphSpec(DAG16_A, DAG16_NODE_PARENTS, DAG16_READOUT, (1.0,) * len(DAG16_READOUT))


@task("dag16_functional_anova")
class Dag16FunctionalAnova(TaskFamily):
    """The 16-input reuse DAG: 15 local functions, b2 and b3 each read twice,
    shared ancestry merged only at the standardized additive readout."""

    cell_keys = ("rho", "f_range")
    cell_defaults = {"rho": 0.5}
    allowed_inputs = FANOVA_INPUTS

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        return DAG16_A

    def draw(self, cfg: Mapping, function_seed: int):
        cfg = self.resolve_cell(cfg)
        family = self.registry_name
        d = int(cfg["d"])
        if d < DAG16_A:
            raise ValueError(f"d={d} < A={DAG16_A}: target cannot fit in ambient space")
        law = functional_anova_law_from_cell(cfg)

        structural_rng = np.random.default_rng(derive(family, int(function_seed), "structure"))
        embedding_rng = np.random.default_rng(derive(family, int(function_seed), int(d), "embedding"))
        active_raw = embedding_rng.choice(d, size=DAG16_A, replace=False)

        return draw_graph_target(
            family=family,
            cfg=cfg,
            function_seed=function_seed,
            law=law,
            graph=dag16_graph(),
            structural_rng=structural_rng,
            active_raw=active_raw,
            parents_attr=tuple(np.asarray(p, dtype=np.int64) for p in DAG16_NODE_PARENTS),
            layer_profile=DAG16_PROFILE,
            k=2,
            task_options=law.describe(),
            wiring={"mode": "reuse_dag16"},
            allowed_inputs=self.allowed_inputs,
        )


# ------------------------------------------------------ random graphs, 16 leaves

RAND16_A = 16
RAND16_K = 2
RAND16_DEPTH = 4                  # longest leaf-to-readout path, as tree16
RAND16_NODES = 15                 # local functions, as tree16 and dag16
RAND16_STRANDS = (2, 3, 4)        # base-forest strands; reuse taps = strands - 1
_RAND16_MAX_DRAW_TRIES = 64
_RAND16_MAX_GROUP_TRIES = 16


def _partitions(total: int, parts: int, least: int = 2) -> tuple[tuple[int, ...], ...]:
    """Non-decreasing partitions of `total` into `parts` parts of at least `least`.

    For 8 leaves this reproduces the benchmark's own lists, (2, 6) (3, 5) (4, 4)
    and (2, 2, 4) (2, 3, 3): a strand needs two leaves to carry a node.
    """
    out: list[tuple[int, ...]] = []

    def rec(remaining, k, floor, acc):
        if k == 1:
            if remaining >= floor:
                out.append(tuple(acc + [remaining]))
            return
        x = floor
        while x * k <= remaining:
            rec(remaining - x, k - 1, x, acc + [x])
            x += 1

    rec(total, parts, least, [])
    return tuple(out)


_PARTITIONS16 = {s: _partitions(RAND16_A, s) for s in RAND16_STRANDS}


class _Unit:
    """One wiring unit during sampling: a ref with its depth and leaf ancestry."""

    __slots__ = ("ref", "depth", "leaves")

    def __init__(self, ref: int, depth: int, leaves: frozenset):
        self.ref, self.depth, self.leaves = ref, depth, leaves


def _depth16(ref: int, node_parents) -> int:
    if ref < RAND16_A:
        return 0
    a, b = node_parents[ref - RAND16_A]
    return 1 + max(_depth16(a, node_parents), _depth16(b, node_parents))


def _leaves16(ref: int, node_parents) -> frozenset:
    if ref < RAND16_A:
        return frozenset((ref,))
    a, b = node_parents[ref - RAND16_A]
    return _leaves16(a, node_parents) | _leaves16(b, node_parents)


def _grow_strand16(leaf_refs, rng, node_parents):
    """Random binary tree over one leaf group under the depth cap; root unit or None."""
    for _ in range(_RAND16_MAX_GROUP_TRIES):
        units = [_Unit(r, 0, frozenset((r,))) for r in leaf_refs]
        parents_added: list[tuple[int, int]] = []
        ok = True
        while len(units) > 1:
            pairs = [(i, j) for i in range(len(units)) for j in range(i + 1, len(units))
                     if max(units[i].depth, units[j].depth) + 1 <= RAND16_DEPTH]
            if not pairs:
                ok = False
                break
            i, j = pairs[rng.integers(len(pairs))]
            a, b = units[i], units[j]
            ref = RAND16_A + len(node_parents) + len(parents_added)
            parents_added.append((a.ref, b.ref))
            merged = _Unit(ref, max(a.depth, b.depth) + 1, a.leaves | b.leaves)
            units = [u for idx, u in enumerate(units) if idx not in (i, j)]
            units.append(merged)
        if ok:
            node_parents.extend(parents_added)
            return units[0]
    return None


def sample_random_graph16(rng: np.random.Generator) -> GraphSpec:
    """One random graph with 16 leaves, depth exactly 4 and exactly 15 nodes.

    The benchmark's `sample_random_graph7` with the envelope of the module
    docstring; deterministic in the rng stream. Draws that cannot attach the
    required taps, or whose deepest strand root is not at depth 4, are
    rejected and resampled.
    """
    for _ in range(_RAND16_MAX_DRAW_TRIES):
        n_strands = int(rng.choice(RAND16_STRANDS))
        options = _PARTITIONS16[n_strands]
        sizes = options[rng.integers(len(options))]
        leaf_order = rng.permutation(RAND16_A)

        node_parents: list[tuple[int, int]] = []
        roots: list[_Unit] = []
        pos = 0
        failed = False
        for size in sizes:
            group = [int(r) for r in leaf_order[pos:pos + size]]
            pos += size
            root = _grow_strand16(group, rng, node_parents)
            if root is None:
                failed = True
                break
            roots.append(root)
        if failed or max(u.depth for u in roots) != RAND16_DEPTH:
            continue

        n_taps = RAND16_NODES - len(node_parents)          # strands - 1
        taps: list[_Unit] = []
        for _tap in range(n_taps):
            refs = list(range(RAND16_A + len(node_parents)))
            sources = [r for r in refs if 1 <= _depth16(r, node_parents) <= RAND16_DEPTH - 1]
            if rng.random() < 0.25:                        # scope overlap through a leaf
                sources += [r for r in refs if r < RAND16_A]
            x = int(sources[rng.integers(len(sources))])
            x_leaves = _leaves16(x, node_parents)
            partners = [r for r in refs
                        if not (_leaves16(r, node_parents) & x_leaves)
                        and max(_depth16(x, node_parents), _depth16(r, node_parents)) + 1
                        <= RAND16_DEPTH]
            existing = {frozenset(p) for p in node_parents}
            partners = [r for r in partners if frozenset((x, r)) not in existing]
            if not partners:
                continue
            y = int(partners[rng.integers(len(partners))])
            ref = RAND16_A + len(node_parents)
            node_parents.append((x, y) if rng.random() < 0.5 else (y, x))
            taps.append(_Unit(ref, _depth16(ref, node_parents), frozenset()))

        readout = tuple(u.ref - RAND16_A for u in roots + taps)
        if len(node_parents) != RAND16_NODES or len(readout) < 2:
            continue
        try:
            return GraphSpec(RAND16_A, node_parents, readout, (1.0,) * len(readout))
        except ValueError:                                 # tripwire: reject and resample
            continue
    raise RuntimeError("16-leaf random-graph sampler exhausted its retry budget")


def rand16_graph(function_seed: int) -> GraphSpec:
    """The graph this family draws for a function seed (the structural stream only)."""
    return sample_random_graph16(
        np.random.default_rng(derive("rand16_functional_anova", int(function_seed), "structure")))


def rand16_profile(graph: GraphSpec) -> tuple[int, ...]:
    """Nodes per depth level plus the readout row (identity metadata)."""
    widths = [0] * RAND16_DEPTH
    for v in range(graph.S):
        widths[_depth16(RAND16_A + v, graph.node_parents) - 1] += 1
    return (*widths, 1)


@task("rand16_functional_anova")
class Rand16FunctionalAnova(TaskFamily):
    """Held-out random graphs on 16 leaves, exactly 15 local functions at depth 4."""

    cell_keys = ("k", "rho", "f_range")
    cell_defaults = {"rho": 0.5}
    allowed_inputs = FANOVA_INPUTS

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        return RAND16_A

    def draw(self, cfg: Mapping, function_seed: int):
        cfg = self.resolve_cell(cfg)
        family = self.registry_name
        d = int(cfg["d"])
        if d < RAND16_A:
            raise ValueError(f"d={d} < A={RAND16_A}: target cannot fit in ambient space")
        if int(cfg.get("k", 2)) != 2:
            raise ValueError(f"{family} is frozen at k = 2")
        law = functional_anova_law_from_cell(cfg)

        structural_rng = np.random.default_rng(derive(family, int(function_seed), "structure"))
        embedding_rng = np.random.default_rng(derive(family, int(function_seed), int(d), "embedding"))
        graph = sample_random_graph16(structural_rng)
        active_raw = embedding_rng.choice(d, size=RAND16_A, replace=False)

        target = draw_graph_target(
            family=family,
            cfg=cfg,
            function_seed=function_seed,
            law=law,
            graph=graph,
            structural_rng=structural_rng,
            active_raw=active_raw,
            parents_attr=tuple(np.asarray(p, dtype=np.int64) for p in graph.node_parents),
            layer_profile=rand16_profile(graph),
            k=2,
            task_options=law.describe(),
            wiring={"mode": "random_layered_16"},
            allowed_inputs=self.allowed_inputs,
        )
        target.topology_hash = canonical_topology_hash(graph)
        return target


F_RANGE = [0.25, 0.4]            # the frozen v0.5 frequency interval

# New presets, added to a resolved campaign catalog. `dense_fourier:a16` carries
# the placeholder shape; the calibration of DESIGN.md section 7 rewrites Q and
# sigma_omega here before the campaign builds.
PRESETS: dict[str, dict[str, dict]] = {
    "dag16_functional_anova": {
        "default": {"rho": 0.5, "sigma_sq": 0.01, "input_distribution": "uniform_pm1", "f_range": F_RANGE},
    },
    "rand16_functional_anova": {
        "default": {"rho": 0.5, "sigma_sq": 0.01, "input_distribution": "uniform_pm1", "f_range": F_RANGE},
    },
    "flat_functional_anova": {
        "w8": {"w": 8, "rho": 0.5, "sigma_sq": 0.01, "input_distribution": "uniform_pm1", "f_range": F_RANGE},
    },
    # The 32-leaf balanced binary tree (decision 2026-09-16, DESIGN amendment 5): the
    # depth-4 preset one level up, 31 local functions; the paper's convention calls it depth 6.
    "tree_functional_anova": {
        "depth5": {"k": 2, "layer_profile": [16, 8, 4, 2, 1], "rho": 0.5, "sigma_sq": 0.01,
                   "input_distribution": "uniform_pm1", "f_range": F_RANGE},
    },
    "tree_fourier": {
        "binary_depth4": {"k": 2, "layer_profile": [8, 4, 2, 1], "sigma_omega": 0.85, "sigma_sq": 0.01,
                          "acceptance": "catastrophic_only", "input_distribution": "uniform_pm1"},
    },
    # The arity-16 Fourier node's shape is chosen by the calibration of DESIGN
    # section 7 from these four candidates, on development seeds 6090-6093.
    # `a16` carries the arity-8 frozen value; the campaign preset is whichever
    # candidate the calibration selects, recorded in manifests/calibration.json.
    "dense_fourier": {
        "a16": {"A": 16, "Q": 24, "sigma_omega": 1.5, "sigma_sq": 0.01,
                "acceptance": "catastrophic_only", "input_distribution": "uniform_pm1"},
        "a16_q24s10": {"A": 16, "Q": 24, "sigma_omega": 1.0, "sigma_sq": 0.01,
                       "acceptance": "catastrophic_only", "input_distribution": "uniform_pm1"},
        "a16_q48s10": {"A": 16, "Q": 48, "sigma_omega": 1.0, "sigma_sq": 0.01,
                       "acceptance": "catastrophic_only", "input_distribution": "uniform_pm1"},
        "a16_q48s075": {"A": 16, "Q": 48, "sigma_omega": 0.75, "sigma_sq": 0.01,
                        "acceptance": "catastrophic_only", "input_distribution": "uniform_pm1"},
    },
}


def dense16_shape(Q: int, sigma_omega: float) -> None:
    """Set the arity-16 Fourier node's shape (DESIGN.md section 7)."""
    PRESETS["dense_fourier"]["a16"]["Q"] = int(Q)
    PRESETS["dense_fourier"]["a16"]["sigma_omega"] = float(sigma_omega)


def register(campaign: dict) -> dict:
    """Add the suite's presets to a resolved campaign and enable its families."""
    from csfrbench.configs import _deep_merge, _validate_task_cell

    catalog = campaign["task_catalog_resolved"]
    defaults = {}
    for family, presets in PRESETS.items():
        entry = catalog["tasks"].setdefault(family, {"presets": {}})
        for name, cell in presets.items():
            resolved = _deep_merge(defaults, cell)
            _validate_task_cell(family, resolved, f"m17 tasks.{family}.presets.{name}", require_d=False)
            entry["presets"][name] = resolved
        campaign["tasks"].setdefault(family, {"preset": next(iter(presets))})
    return campaign
