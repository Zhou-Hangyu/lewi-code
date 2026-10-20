"""CSFRBench v0.5 graph topologies: the 8-input reuse DAG, the exactly-7-node
held-out graph sampler, and a canonical topology hash.

The fixed reuse DAG (v0.5 §8) — 8 active inputs, 7 local functions, arity 2,
two reused level-1 intermediates, ancestry-disjoint parents at every local
function, shared ancestry merged only at the additive readout:

    level 1:  a1=g1(x1,x2)  a2=g2(x3,x4)  a3=g3(x5,x6)  a4=g4(x7,x8)
    level 2:  b1=g5(a1,a2)  b2=g6(a2,a3)  b3=g7(a3,a4)
    readout:  y = Std(b1 + b2 + b3)

The held-out graph distribution (v0.5 §9) fixes exactly 7 local-function
nodes so random graphs match the balanced tree and the reuse DAG in
nonlinear-node count while varying topology, reuse, and readout organization.
The sampler keeps the v1 two-phase construction (leaf strands + reuse taps)
with the tap count pinned to 7 - (base-forest nodes).

The canonical topology hash identifies a graph up to leaf relabeling and node
reindexing, distinguishing genuine fan-out from duplicated shapes (a recursive
shape signature cannot). It canonically labels the DAG of leaves, local
functions, and the readout by color refinement with full individualization.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .graph import GraphSpec
from .rand import (
    RAND_A,
    RAND_DEPTH,
    _MAX_DRAW_TRIES,
    _PARTITIONS,
    _Unit,
    _depth_of,
    _grow_strand,
    _leaves_of,
)

# ------------------------------------------------------ fixed reuse DAG --

DAG8_A = 8
DAG8_NODE_PARENTS = (
    (0, 1),      # a1 = g1(x1, x2)
    (2, 3),      # a2 = g2(x3, x4)
    (4, 5),      # a3 = g3(x5, x6)
    (6, 7),      # a4 = g4(x7, x8)
    (8, 9),      # b1 = g5(a1, a2)
    (9, 10),     # b2 = g6(a2, a3)   -- a2 reused
    (10, 11),    # b3 = g7(a3, a4)   -- a3 reused
)
DAG8_READOUT = (4, 5, 6)             # b1, b2, b3
DAG8_PROFILE = (4, 3, 1)             # nodes per level plus the readout row


def dag8_graph() -> GraphSpec:
    """The v0.5 fixed reuse DAG as a validated GraphSpec."""
    return GraphSpec(DAG8_A, DAG8_NODE_PARENTS, DAG8_READOUT,
                     (1.0,) * len(DAG8_READOUT))


# ------------------------------------------- exactly-7-node graph sampler --

RAND7_NODES = 7


def sample_random_graph7(rng: np.random.Generator) -> GraphSpec:
    """One held-out random graph with exactly 7 local-function nodes.

    The v1 constraint envelope is unchanged (8 leaves, arity 2, depth exactly
    3, connected, ancestry-disjoint parents, additive readout, no duplicated
    unordered parent pair); only the node count is pinned: 2 strands take 1
    reuse tap, 3 strands take 2, so base nodes (8 - strands) + taps = 7.
    Draws that cannot attach the required taps are rejected and resampled.
    """
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

        n_taps = RAND7_NODES - len(node_parents)        # 1 or 2
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
        if len(node_parents) != RAND7_NODES or len(readout) < 2:
            continue
        try:
            return GraphSpec(RAND_A, node_parents, readout, (1.0,) * len(readout))
        except ValueError:                      # tripwire: reject and resample
            continue
    raise RuntimeError("7-node random-graph sampler exhausted its retry budget")


def graph_profile7(graph: GraphSpec) -> tuple[int, ...]:
    """Nodes per depth level plus the readout row (identity metadata)."""
    widths = [0] * RAND_DEPTH
    for v in range(graph.S):
        widths[_depth_of(RAND_A + v, graph.node_parents) - 1] += 1
    return (*widths, 1)


# ------------------------------------------------ canonical topology hash --

def canonical_topology_hash(graph: GraphSpec) -> str:
    """Canonical hash of the computation-graph topology.

    Invariant under leaf relabeling and node reindexing; sensitive to sharing
    (a node read twice differs from two same-shape nodes). Vertices are the
    leaves, the local-function nodes, and one readout vertex; the certificate
    is the lexicographically smallest edge list over canonical labelings,
    found by color refinement with full individualization on ties.
    """
    A, S = graph.A, graph.S
    n = A + S + 1
    readout_vertex = A + S
    out_nb: list[list[int]] = [[] for _ in range(n)]
    in_nb: list[list[int]] = [[] for _ in range(n)]
    for v, refs in enumerate(graph.node_parents):
        for r in refs:
            out_nb[r].append(A + v)
            in_nb[A + v].append(r)
    for i in graph.readout_indices:
        out_nb[A + i].append(readout_vertex)
        in_nb[readout_vertex].append(A + i)

    def refine(colors: list[int]) -> list[int]:
        while True:
            signature = [
                (colors[v],
                 tuple(sorted(colors[u] for u in in_nb[v])),
                 tuple(sorted(colors[u] for u in out_nb[v])))
                for v in range(n)
            ]
            order = {s: i for i, s in enumerate(sorted(set(signature)))}
            refined = [order[s] for s in signature]
            if refined == colors:
                return colors
            colors = refined

    def certificate(colors: list[int]):
        colors = refine(colors)
        classes: dict[int, list[int]] = {}
        for v, c in enumerate(colors):
            classes.setdefault(c, []).append(v)
        split = next((vs for _, vs in sorted(classes.items()) if len(vs) > 1),
                     None)
        if split is None:
            label = {v: i for i, v in
                     enumerate(sorted(range(n), key=lambda v: colors[v]))}
            return tuple(sorted((label[u], label[v])
                                for u in range(n) for v in out_nb[u]))
        best = None
        for v in split:
            branched = list(colors)
            branched[v] = n + 1                 # individualize
            cert = certificate(branched)
            if best is None or cert < best:
                best = cert
        return best

    base = [0] * A + [1] * S + [2]
    payload = repr((A, S, certificate(base))).encode()
    return hashlib.blake2b(payload, digest_size=10, person=b"aritygraph").hexdigest()
