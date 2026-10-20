"""The frozen structure-supply constructor.

One deterministic lift takes any admitted computation graph into the profile
backbone, and every same-backbone reference is a view of it.  Together the
views form the GraphOracle ladder GO(0) < GO(1) < ... < GraphOracle(all),
indexed by how many nonlinear teacher levels the masks enforce:

- ``go0``               -- GO(0), the ladder's bottom rung: first layer
                           restricted to the true active support, dense
                           above.  ``support_oracle`` is the legacy alias.
- ``graph_oracle``      -- GO(k): the lift's masks enforced through the
                           first ``levels`` nonlinear teacher levels;
                           connectivity after the revealed prefix is
                           unrestricted.  ``levels`` counts nonlinear
                           levels -- the readout is not a level.  Public
                           workload names count depth graph-theoretically
                           (readout node included, amendment 13), one more
                           than the nonlinear depth.  Nonlinear depths of
                           the frozen workloads: the depth-4 tree 3, the
                           depth-3 reuse DAGs 2, the depth-5 tree 4,
                           held-out graphs 3, flat 1, dense Fourier 1.
                           ``levels="all"`` (the default) enforces every
                           stage.  Two facts bound the ladder from above:
                           the readout mask is all-ones for every admitted
                           graph (the blocks live at the last stage are
                           exactly the readout-read blocks, and live blocks
                           partition the width), and on any graph whose top
                           level holds only readout-read nodes the top
                           stage is vacuous by liveness, so
                           ``masks(L-1) == masks(all)``.  Every frozen
                           workload is of that kind, so the only
                           informative explicit rungs are those with
                           ``masks(k) != masks(all)``; an explicit
                           ``levels`` that duplicates the full oracle is
                           refused at materialization (``duplicates_all``).
- ``permutation_control`` -- ``graph_oracle`` at ``levels=1`` with the
                           leaf-to-block assignment permuted; fan-in
                           profile, block sizes and retained weights match
                           the ``levels=1`` oracle exactly.  Defined at
                           level 1 only: level-1 read-sets are disjoint by
                           the target grammar, so the pooled re-deal is
                           exact; a deeper level can read reused blocks,
                           where no count-preserving permutation exists.

The lift itself is fixed by four rules.  Stages: teacher level ``l`` owns
``s_l = floor(l*H/L) - floor((l-1)*H/L)`` contiguous learner layers, so stage
lengths differ by at most one.  Liveness: a value is carried, block-locally,
from the stage that computes it until its last consumer (the readout counts
as level ``L+1``); a raw leaf read above level 1 gets a carrier block.  Width:
every live block first receives its minimum -- two units for any block
carried across a ReLU layer (a raw-leaf carrier, or a computed node held
past its computing stage), reserving a sign-preserving representation
without a bias-shift construction; one unit for a block in the stage
that computes it -- and the
remaining units are dealt as evenly as possible in canonical order --
computed nodes by index, then leaf carriers.
Within a stage, the first layer introduces that level's teacher edges and
every further layer is block-local refinement; no refinement layer adds a
cross-block edge.  The schedule and allocation never depend on ``levels``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..registry import method
from ..utils.seeds import derive
from .base import Method

from .maskladder import MaskedMLP, support_masks


def node_levels(target) -> tuple[list[int], list[int]]:
    """Each node slot's nonlinear level (leaves are 0) and canonical alias.

    Two slots holding the same parameter object with the same parents are one
    computational node read twice; they share a block, so reuse is fan-out.
    """
    parents, A = target.graph.node_parents, target.graph.A
    nodes = target.params["nodes"]
    level, canon = [], []
    for v, refs in enumerate(parents):
        level.append(1 + max((level[r - A] for r in refs if r >= A), default=0))
        alias = next((u for u in range(v) if nodes[u] is nodes[v]
                      and parents[u] == parents[v]), v)
        canon.append(alias)
    return level, canon


def stage_lengths(hidden: int, depth: int) -> list[int]:
    """Balanced contiguous stage lengths: differ by at most one, sum to H."""
    if hidden < depth:
        raise ValueError(f"backbone has {hidden} hidden layers for {depth} "
                         "teacher levels; the wiring cannot be embedded")
    return [(l * hidden) // depth - ((l - 1) * hidden) // depth
            for l in range(1, depth + 1)]


@dataclass
class Lift:
    """The deterministic plan: stages, live blocks, and unit allocation."""
    depth: int                      # L, teacher nonlinear depth
    stages: list[int]               # learner layers per stage, len L
    level: list[int]                # per slot
    canon: list[int]                # per slot
    # per stage 1..L: ordered dict of block key -> unit range at stage output.
    # keys: ("node", canonical slot) or ("leaf", leaf index)
    alloc: list[dict]
    horizon: dict                   # block key -> last consumer level (readout = L+1)


def plan(target, width: int, hidden: int) -> Lift:
    graph = target.graph
    level, canon = node_levels(target)
    depth = max(level)
    stages = stage_lengths(hidden, depth)

    horizon: dict = {}
    for v, refs in enumerate(graph.node_parents):
        for r in refs:
            key = ("leaf", int(r)) if r < graph.A else ("node", canon[r - graph.A])
            horizon[key] = max(horizon.get(key, 0), level[v])
    for i in graph.readout_indices:
        horizon[("node", canon[i])] = depth + 1

    alloc: list[dict] = []
    for l in range(1, depth + 1):
        nodes_here = sorted(v for v in range(len(level))
                            if canon[v] == v and level[v] <= l
                            and horizon.get(("node", v), 0) > l)
        leaves_here = sorted(a for a in range(graph.A)
                             if horizon.get(("leaf", a), 0) > l)
        blocks = [("node", v) for v in nodes_here] + \
                 [("leaf", a) for a in leaves_here]
        minimum = {b: (1 if b[0] == "node" and level[b[1]] == l else 2)
                   for b in blocks}
        need = sum(minimum.values())
        if need > width:
            raise ValueError(
                f"stage {l} needs {need} units for {len(blocks)} live blocks "
                f"but the backbone is {width} wide; cannot embed")
        extra, rem = divmod(width - need, len(blocks))
        sizes = [minimum[b] + extra + (1 if i < rem else 0)
                 for i, b in enumerate(blocks)]
        out, start = {}, 0
        for b, size in zip(blocks, sizes):
            out[b] = range(start, start + size)
            start += size
        alloc.append(out)
    return Lift(depth, stages, level, canon, alloc, horizon)


def lift_masks(target, width: int, hidden: int, *, levels="all",
               permute_level: int | None = None, seed: int = 0) -> list[torch.Tensor]:
    """Masks for every affine layer (H hidden + readout) of the lifted arm.

    ``levels`` counts nonlinear teacher levels (the readout is not a
    level); an explicit value above the target's nonlinear depth does not
    parse.  ``levels="all"`` enforces every stage.
    """
    graph = target.graph
    lf = plan(target, width, hidden)
    k = lf.depth if levels in (None, "all") else int(levels)
    if not 1 <= k <= lf.depth:
        raise ValueError(
            f"levels={levels} does not parse on this target: its nonlinear "
            f"depth is {lf.depth} (the readout is not a level)")
    if permute_level is not None and int(permute_level) != 1:
        raise ValueError(
            "the permutation control is defined at level 1 only: deeper "
            "levels can read reused blocks, so a pooled re-deal cannot "
            "preserve retained counts exactly")
    if permute_level is not None and k < 1:
        raise ValueError("permute_level must lie within the enforced levels")

    raw = np.asarray(target.active_raw, dtype=int).ravel()
    d = int(target.d)
    masks: list[torch.Tensor] = []
    prev_alloc: dict = {}
    prev_width = d
    for l in range(1, lf.depth + 1):
        cur = lf.alloc[l - 1]
        new_nodes = [b for b in cur if b[0] == "node" and lf.level[b[1]] == l]
        parent_cols = {}
        for b in new_nodes:
            cols: list[int] = []
            for r in graph.node_parents[b[1]]:
                if r < graph.A:
                    if l == 1:
                        cols.append(int(raw[r]))
                    else:
                        cols.extend(prev_alloc[("leaf", int(r))])
                else:
                    cols.extend(prev_alloc[("node", lf.canon[r - graph.A])])
            parent_cols[b] = cols
        if permute_level == l and len(new_nodes) > 1:
            pool = np.concatenate([parent_cols[b] for b in new_nodes])
            if len(pool) != len(set(pool.tolist())):
                raise ValueError(
                    "level-1 read-sets overlap on this graph (a leaf is "
                    "shared); the permutation control is undefined here")
            rng = np.random.default_rng(
                derive("graph_oracle", "perm", int(l), int(seed)))
            pool = rng.permutation(pool)
            cut = np.cumsum([0] + [len(parent_cols[b]) for b in new_nodes])
            parent_cols = {b: pool[cut[i]:cut[i + 1]].tolist()
                           for i, b in enumerate(new_nodes)}

        for j in range(lf.stages[l - 1]):
            m = torch.zeros(width, prev_width if j == 0 else width)
            for b, rows in cur.items():
                if j == 0 and b in parent_cols:
                    cols = parent_cols[b]
                elif j == 0 and l == 1:               # leaf carrier is born
                    cols = [int(raw[b[1]])]
                elif j == 0:                          # carried from last stage
                    cols = list(prev_alloc[b])
                else:                                 # block-local refinement
                    cols = list(cur[b])
                m[np.ix_(list(rows), cols)] = 1.0
            masks.append(m)
        prev_alloc = dict(cur)
        prev_width = width
    out = torch.zeros(1, width)
    for key in {("node", lf.canon[i]) for i in graph.readout_indices}:
        out[0, list(lf.alloc[-1][key])] = 1.0
    masks.append(out)

    if k < lf.depth:                # unrestricted above the enforced levels
        cut_layer = sum(lf.stages[:k])
        for i in range(cut_layer, hidden):
            masks[i] = torch.ones(width, width if i else d)
        masks[-1] = torch.ones(1, width)
    return masks


class _LiftArm(Method):
    def _masks(self, cfg, target):
        raise NotImplementedError

    def build(self, cfg, target):
        width, hidden = int(cfg["width"]), int(cfg["hidden"])
        return MaskedMLP(int(target.d), width, hidden,
                         self._masks(cfg, target, width, hidden))

    def effective_weights(self, model):
        """Masked matrices, read explicitly rather than through the raw
        default. The raw path is numerically identical today only by a
        three-legged invariant (masked entries start at zero, receive
        exactly zero gradients, and optimizer weight decay is pinned off);
        reading the model's masked matrices removes the dependence (audit
        round 2, finding 3.1)."""
        return tuple(model.effective_weights())


def duplicates_all(target, width: int, hidden: int, levels: int) -> bool:
    """True when the explicit ``levels`` prefix already equals the full
    oracle on this target -- the arm would silently duplicate
    ``levels="all"``, so materialization refuses it (amendment 12)."""
    full = lift_masks(target, width, hidden, levels="all")
    part = lift_masks(target, width, hidden, levels=int(levels))
    return all(torch.equal(a, b) for a, b in zip(part, full))


@method("go0")
class GO0(_LiftArm):
    """GO(0), the ladder's bottom rung: the true active-variable support
    only, dense above.  Its masks nest exactly under GO(1)."""

    hp_keys = frozenset()

    def _masks(self, cfg, target, width, hidden):
        return support_masks(target, width, hidden)


@method("support_oracle")
class SupportOracle(GO0):
    """Legacy alias for GO(0) (amendment 12): every record as run and the
    frozen v0.5 campaign configuration name the arm this way."""


@method("graph_oracle")
class GraphOracle(_LiftArm):
    """GO(k): true connectivity through the first ``levels`` nonlinear
    levels; ``levels="all"`` is the full oracle, the ladder's top."""

    hp_keys = frozenset({"levels"})

    def _masks(self, cfg, target, width, hidden):
        levels = cfg.get("hp", {}).get("levels", "all")
        return lift_masks(target, width, hidden, levels=levels)


@method("permutation_control")
class PermutationControl(_LiftArm):
    """graph_oracle(levels=k) with level ``k``'s parent assignment permuted."""

    def _masks(self, cfg, target, width, hidden):
        k = int(cfg.get("hp", {}).get("level", 1))
        return lift_masks(target, width, hidden, levels=k, permute_level=k,
                          seed=int(getattr(target, "function_seed", 0)))
