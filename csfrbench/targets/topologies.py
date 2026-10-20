"""Graph templates and the sampler of the depth-4 DAGs.

Depth in a public name counts the readout node (paper Figure 3): the depth-4
tree has three nonlinear levels, the reuse DAG has two, the depth-5 tree has four.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from .graph import GraphSpec

# ------------------------------------------------------------ balanced trees


def balanced_wiring(d: int, k: int, profile: Sequence[int], structure_rng: np.random.Generator,
                    embedding_rng: np.random.Generator) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Parents of a balanced ``k``-ary tree and the active coordinates of its leaves.

    ``profile`` gives the number of local functions per level and ends in 1.
    The structure stream does not depend on ``d``, so an instance has the same
    graph and the same local functions at every ambient dimension. Only the
    assignment of the leaves to coordinates depends on ``d``.
    """
    d, k = int(d), int(k)
    profile = tuple(int(w) for w in profile)
    if d <= 0 or k <= 0:
        raise ValueError("d and k must be positive")
    if not profile or any(w <= 0 for w in profile) or profile[-1] != 1:
        raise ValueError("the level profile must hold positive widths and end in 1")
    active_count = profile[0] * k
    if d < active_count:
        raise ValueError(f"d = {d} is smaller than the number of active inputs {active_count}")
    parents: list[np.ndarray] = []
    for level in range(1, len(profile)):
        previous, width = profile[level - 1], profile[level]
        if previous != k * width:
            raise ValueError(f"level {level}: width {previous} must equal k * {width}")
        parents.append(structure_rng.permutation(previous).reshape(width, k))
    active = embedding_rng.choice(d, size=active_count, replace=False)
    return active.reshape(profile[0], k), tuple(parents)


def balanced_graph(profile: Sequence[int], k: int, parents: Sequence[np.ndarray]) -> GraphSpec:
    """The graph of a balanced tree with the given parents; the readout is the root."""
    profile = tuple(int(w) for w in profile)
    A = profile[0] * k
    offsets = [0]
    for width in profile[:-1]:
        offsets.append(offsets[-1] + width)
    node_parents: list[tuple[int, ...]] = [tuple(j * k + i for i in range(k)) for j in range(profile[0])]
    for level in range(1, len(profile)):
        for j in range(profile[level]):
            node_parents.append(tuple(A + offsets[level - 1] + p for p in parents[level - 1][j]))
    return GraphSpec(A, node_parents, (len(node_parents) - 1,), (1.0,))


# ------------------------------------------------------------ the reuse DAG

DAG8_A = 8
DAG8_NODE_PARENTS = (
    (0, 1),      # z1 = g1(x1, x2)
    (2, 3),      # z2 = g2(x3, x4)
    (4, 5),      # z3 = g3(x5, x6)
    (6, 7),      # z4 = g4(x7, x8)
    (8, 9),      # z5 = g5(z1, z2)
    (9, 10),     # z6 = g6(z2, z3)   z2 is used a second time
    (10, 11),    # z7 = g7(z3, z4)   z3 is used a second time
)
DAG8_READOUT = (4, 5, 6)             # z5, z6, z7
DAG8_PROFILE = (4, 3, 1)             # local functions per level, then the readout


def dag8_graph() -> GraphSpec:
    return GraphSpec(DAG8_A, DAG8_NODE_PARENTS, DAG8_READOUT, (1.0,) * len(DAG8_READOUT))


# ------------------------------------------------------------ sampled depth-4 DAGs

RAND_A = 8
RAND_DEPTH = 3                       # nonlinear levels
RAND_NODES = 7                       # local functions
_MAX_DRAW_TRIES = 64
_MAX_GROUP_TRIES = 16
#: Partitions of the 8 leaves into 2 or 3 strands; each part admits a binary tree of depth <= 3.
_PARTITIONS = {2: ((2, 6), (3, 5), (4, 4)), 3: ((2, 2, 4), (2, 3, 3))}


class _Unit:
    __slots__ = ("ref", "depth", "leaves")

    def __init__(self, ref: int, depth: int, leaves: frozenset):
        self.ref, self.depth, self.leaves = ref, depth, leaves


def _grow_strand(leaf_refs, rng, node_parents):
    """A random binary tree over one group of leaves; returns its root or None."""
    for _ in range(_MAX_GROUP_TRIES):
        units = [_Unit(r, 0, frozenset((r,))) for r in leaf_refs]
        added: list[tuple[int, int]] = []
        ok = True
        while len(units) > 1:
            pairs = [(i, j) for i in range(len(units)) for j in range(i + 1, len(units))
                     if max(units[i].depth, units[j].depth) + 1 <= RAND_DEPTH]
            if not pairs:
                ok = False
                break
            i, j = pairs[rng.integers(len(pairs))]
            a, b = units[i], units[j]
            ref = RAND_A + len(node_parents) + len(added)
            added.append((a.ref, b.ref))
            merged = _Unit(ref, max(a.depth, b.depth) + 1, a.leaves | b.leaves)
            units = [u for index, u in enumerate(units) if index not in (i, j)]
            units.append(merged)
        if ok:
            node_parents.extend(added)
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


def sample_graph(rng: np.random.Generator) -> GraphSpec:
    """One sampled depth-4 DAG: 8 leaves, 7 local functions of arity 2, 3 nonlinear levels.

    The sampler makes 2 or 3 strands (a random binary tree over each part of a
    random partition of the leaves) and then adds reuse connections: 1 for 2
    strands, 2 for 3 strands, so the graph has 7 local functions. A reuse node
    reads one value that is already used (an intermediate, or a leaf with
    probability 1/4) and one partner with disjoint input ancestry. The readout
    adds the strand roots and the reuse nodes. A draw is rejected when its
    deepest strand is not at level 3, when a reuse node cannot be attached, or
    when two nodes have the same unordered pair of parents.
    """
    for _ in range(_MAX_DRAW_TRIES):
        n_strands = int(rng.choice((2, 3)))
        sizes = _PARTITIONS[n_strands][rng.integers(len(_PARTITIONS[n_strands]))]
        leaf_order = rng.permutation(RAND_A)

        node_parents: list[tuple[int, int]] = []
        roots: list[_Unit] = []
        position = 0
        failed = False
        for size in sizes:
            group = [int(r) for r in leaf_order[position:position + size]]
            position += size
            root = _grow_strand(group, rng, node_parents)
            if root is None:
                failed = True
                break
            roots.append(root)
        if failed or max(u.depth for u in roots) != RAND_DEPTH:
            continue

        n_taps = RAND_NODES - len(node_parents)
        taps: list[_Unit] = []
        for _tap in range(n_taps):
            refs = list(range(RAND_A + len(node_parents)))
            sources = [r for r in refs if 1 <= _depth_of(r, node_parents) <= RAND_DEPTH - 1]
            if rng.random() < 0.25:
                sources += [r for r in refs if r < RAND_A]
            x = int(sources[rng.integers(len(sources))])
            x_leaves = _leaves_of(x, node_parents)
            partners = [r for r in refs
                        if not (_leaves_of(r, node_parents) & x_leaves)
                        and max(_depth_of(x, node_parents), _depth_of(r, node_parents)) + 1 <= RAND_DEPTH]
            existing = {frozenset(p) for p in node_parents}
            partners = [r for r in partners if frozenset((x, r)) not in existing]
            if not partners:
                continue
            y = int(partners[rng.integers(len(partners))])
            ref = RAND_A + len(node_parents)
            node_parents.append((x, y) if rng.random() < 0.5 else (y, x))
            taps.append(_Unit(ref, _depth_of(ref, node_parents), frozenset()))

        readout = tuple(u.ref - RAND_A for u in roots + taps)
        if len(node_parents) != RAND_NODES or len(readout) < 2:
            continue
        try:
            return GraphSpec(RAND_A, node_parents, readout, (1.0,) * len(readout))
        except ValueError:
            continue
    raise RuntimeError("the graph sampler used all its attempts")


def level_profile(graph: GraphSpec) -> tuple[int, ...]:
    """Local functions per nonlinear level, then the readout."""
    widths = [0] * RAND_DEPTH
    for v in range(graph.S):
        widths[_depth_of(RAND_A + v, graph.node_parents) - 1] += 1
    return (*widths, 1)


# ------------------------------------------------------------ canonical topology hash


def canonical_topology_hash(graph: GraphSpec) -> str:
    """A hash of the topology that does not depend on the labels of leaves and nodes.

    The vertices are the leaves, the local functions and one readout vertex.
    The certificate is the smallest edge list over canonical labelings, found by
    colour refinement with individualization of tied vertices. Isomorphic graphs
    have the same certificate, and graphs that are not isomorphic have different
    certificates.
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
            signature = [(colors[v], tuple(sorted(colors[u] for u in in_nb[v])),
                          tuple(sorted(colors[u] for u in out_nb[v]))) for v in range(n)]
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
        split = next((vs for _, vs in sorted(classes.items()) if len(vs) > 1), None)
        if split is None:
            label = {v: i for i, v in enumerate(sorted(range(n), key=lambda v: colors[v]))}
            return tuple(sorted((label[u], label[v]) for u in range(n) for v in out_nb[u]))
        best = None
        for v in split:
            branched = list(colors)
            branched[v] = n + 1
            cert = certificate(branched)
            if best is None or cert < best:
                best = cert
        return best

    base = [0] * A + [1] * S + [2]
    payload = repr((A, S, certificate(base))).encode()
    return hashlib.blake2b(payload, digest_size=10, person=b"aritygraph").hexdigest()
