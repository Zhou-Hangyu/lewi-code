"""Structure-privileged reference arms that live inside the baseline's own model.

The frozen ``oracle`` arm answers "is the target learnable once the wiring is
given?" with a differently parameterised network (one width-256 MLP per graph
node), so the oracle-minus-dense gap mixes the value of the wiring with the
change of hypothesis class.  The arms here remove that confound: each one is
the *same* width-32 depth-4 MLP the generic learner trains, with a fixed binary
mask applied to its weights.  Every arm is therefore a subnetwork of the
baseline -- the baseline could have expressed it -- and the gap that remains is
a search gap, not a capacity gap.

Four rungs, in order of the information supplied:

``inputoracle``  the active coordinates, nothing else (Donoho & Johnstone 1994)
``randmask``     the fan-in profile at a random location -- the sparsity control
                 that separates "sparse helps" from "this sparsity helps"
                 (Frankle et al. 2021; Su et al. 2020)
``pairmask``     the active coordinates and the fan-in profile, grouped WRONGLY
``maskoracle``   the true graph

``randmask`` and ``pairmask`` are controls, not competitors: they are matched to
``maskoracle`` layer by layer on the number of retained weights.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..registry import method
from ..utils.seeds import derive
from .base import Method
from .nns import DenseLinear, ReLUMLP


class MaskedLinear(DenseLinear):
    """Affine layer with a fixed binary mask on the weight matrix."""

    def __init__(self, in_features: int, out_features: int, *, is_output: bool):
        super().__init__(in_features, out_features, is_output=is_output)
        self.register_buffer("mask", torch.ones_like(self.weight), persistent=True)

    def set_mask(self, mask: torch.Tensor) -> None:
        if mask.shape != self.weight.shape:
            raise ValueError(f"mask {tuple(mask.shape)} != weight {tuple(self.weight.shape)}")
        self.mask.copy_(mask.to(self.mask.dtype))
        self.reinit_for_mask()

    def reinit_for_mask(self) -> None:
        """Kaiming initialisation at the *retained* fan-in of each output unit.

        Initialising at the dense fan-in would shrink every masked unit's
        pre-activation variance in proportion to how much of its row was
        removed, so a sparser arm would start further from the edge of chaos
        than the dense baseline and lose for a reason that is not structural.
        """
        gain = 1.0 if self.is_output else 2.0 ** 0.5
        fan_in = self.mask.sum(dim=1).clamp(min=1.0)
        std = gain / torch.sqrt(fan_in)
        with torch.no_grad():
            self.weight.normal_(0.0, 1.0)
            self.weight.mul_(std.unsqueeze(1)).mul_(self.mask)
            self.bias.zero_()

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.mask


class MaskedMLP(ReLUMLP):
    """The benchmark backbone with a fixed mask on every affine layer."""

    def __init__(self, input_dim: int, width: int, num_hidden_layers: int,
                 masks: list[torch.Tensor]):
        super().__init__(input_dim=input_dim, width=width,
                         num_hidden_layers=num_hidden_layers, out_dim=1,
                         layer_factory=lambda i, o, is_out: MaskedLinear(
                             i, o, is_output=is_out))
        layers = self.linear_layers()
        if len(masks) != len(layers):
            raise ValueError(f"{len(masks)} masks for {len(layers)} layers")
        for layer, mask in zip(layers, masks):
            layer.set_mask(mask)

    def retained_weights(self) -> tuple[int, ...]:
        return tuple(int(layer.mask.sum().item()) for layer in self.linear_layers())


# --------------------------------------------------------------------------
# mask construction


def _levels(target) -> tuple[list[int], list[int]]:
    """Each node slot's level (leaves are 0), and the canonical slot it aliases.

    Two slots that hold the same parameter object are the same computational
    node read twice -- that is how the DAG template encodes reuse.  They must
    share one block, because reuse in a masked MLP is fan-out, not duplication.
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


def _layer_plan(depth: int, hidden: int) -> list[int]:
    """Hidden layers per graph level: one each, extras to the earliest levels."""
    if hidden < depth:
        raise ValueError(f"backbone has {hidden} hidden layers for {depth} graph "
                         "levels; the wiring cannot be embedded")
    plan = [1] * depth
    for i in range(hidden - depth):
        plan[i % depth] += 1
    return plan


def _blocks(width: int, count: int) -> list[range]:
    base, extra = divmod(width, count)
    out, start = [], 0
    for i in range(count):
        stop = start + base + (1 if i < extra else 0)
        out.append(range(start, stop))
        start = stop
    return out


def graph_masks(target, width: int, hidden: int, *, mode: str,
                seed: int) -> list[torch.Tensor]:
    """Binary masks realising ``mode`` on the frozen backbone.

    ``mode`` is one of ``graph`` (the true wiring), ``pairing`` (true active
    set, permuted grouping), ``random`` (fan-in profile at random coordinates)
    or ``support`` (active coordinates only, dense above).
    """
    d = int(target.d)
    level, canon = _levels(target)
    depth = max(level)
    plan = _layer_plan(depth, hidden)
    rng = np.random.default_rng(derive("maskladder", mode, int(seed)))

    leaves_of = {}                    # canonical slot -> its raw coordinates
    raw = np.asarray(target.active_raw, dtype=int).ravel()
    for v, refs in enumerate(target.graph.node_parents):
        if canon[v] == v and level[v] == 1:
            leaves_of[v] = [raw[r] for r in refs]
    order = sorted(leaves_of)
    if mode == "pairing":             # same active set, grouped wrongly
        pool = rng.permutation(np.concatenate([leaves_of[v] for v in order]))
        cut = np.cumsum([0] + [len(leaves_of[v]) for v in order])
        leaves_of = {v: pool[cut[i]:cut[i + 1]].tolist() for i, v in enumerate(order)}
    elif mode == "random":            # same fan-in, anywhere in the ambient space
        leaves_of = {v: rng.choice(d, size=len(leaves_of[v]), replace=False).tolist()
                     for v in order}

    per_level = [[v for v in range(len(level)) if level[v] == L and canon[v] == v]
                 for L in range(1, depth + 1)]
    masks, prev_blocks, prev_width = [], None, d
    for L, nodes_here in enumerate(per_level, start=1):
        blocks = dict(zip(nodes_here, _blocks(width, len(nodes_here))))
        for j in range(plan[L - 1]):
            m = torch.zeros(width, prev_width)
            for v, rows in blocks.items():
                if j > 0:                                  # inside one node
                    cols = list(blocks[v])
                elif L == 1:
                    cols = leaves_of[v]
                else:
                    cols = [c for r in target.graph.node_parents[v]
                            for c in prev_blocks[canon[r - target.graph.A]]]
                m[np.ix_(list(rows), cols)] = 1.0
            masks.append(m)
            prev_width = width
        prev_blocks = blocks

    out = torch.zeros(1, width)
    for v in {canon[i] for i in target.graph.readout_indices}:
        out[0, list(prev_blocks[v])] = 1.0
    masks.append(out)
    return masks


def support_masks(target, width: int, hidden: int) -> list[torch.Tensor]:
    """Dense backbone reading only the active coordinates."""
    d = int(target.d)
    raw = np.unique(np.asarray(target.active_raw, dtype=int).ravel())
    first = torch.zeros(width, d)
    first[:, raw] = 1.0
    return ([first] + [torch.ones(width, width) for _ in range(hidden - 1)]
            + [torch.ones(1, width)])


class _MaskArm(Method):
    mode = "graph"

    def build(self, cfg, target):
        width, hidden = int(cfg["width"]), int(cfg["hidden"])
        if self.mode == "support":
            masks = support_masks(target, width, hidden)
        else:
            masks = graph_masks(target, width, hidden, mode=self.mode,
                                seed=int(getattr(target, "function_seed", 0)))
        return MaskedMLP(int(target.d), width, hidden, masks)


@method("maskoracle")
class MaskOracle(_MaskArm):
    """The true computation graph, as a subnetwork of the generic learner."""
    mode = "graph"


@method("pairmask")
class PairMask(_MaskArm):
    """Right active set and right fan-in, wrong grouping -- the grouping control."""
    mode = "pairing"


@method("randmask")
class RandMask(_MaskArm):
    """Right fan-in profile at random coordinates -- the sparsity control."""
    mode = "random"


@method("inputoracle")
class InputOracle(_MaskArm):
    """The active coordinates only -- the classical support oracle."""
    mode = "support"
