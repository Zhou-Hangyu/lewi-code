"""GraphOracle: reference networks with masks from the target graph.

``GraphOracle(l)`` is the common backbone with fixed binary masks on its weight
matrices. The masks give the network the connectivity of the first ``l``
nonlinear levels of the target graph; the connections after these levels are
not restricted. ``GraphOracle(0)`` gives only the active inputs.
``GraphOracle(all)`` gives the full target connectivity.

Mask construction for a target with ``L`` nonlinear levels and a backbone with
``H >= L`` hidden layers:

* level ``r`` has ``s_r = floor(r H / L) - floor((r - 1) H / L)`` consecutive hidden layers;
* the hidden units of a level are divided into blocks, one per value that is
  alive at that level: a value is alive from the level that computes it until
  its last use (the readout is the last use of the values that it adds);
* a block gets 1 unit at the level that computes its value and 2 units when it
  carries a value through a further ReLU layer, so a signed value can be
  represented; the other units are divided as evenly as possible;
* the first layer of a level connects a block to the blocks of its parents;
  the further layers of the level connect each block only to itself.

A masked layer is initialized with Kaiming normal weights at the fan-in that
the mask keeps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..models import DenseLinear, ReLUMLP
from .base import Method, register


class MaskedLinear(DenseLinear):
    def __init__(self, in_features: int, out_features: int, *, is_output: bool):
        super().__init__(in_features, out_features, is_output=is_output)
        self.register_buffer("mask", torch.ones_like(self.weight), persistent=True)

    def set_mask(self, mask: torch.Tensor) -> None:
        if mask.shape != self.weight.shape:
            raise ValueError(f"mask {tuple(mask.shape)} != weight {tuple(self.weight.shape)}")
        self.mask.copy_(mask.to(self.mask.dtype))
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
    def __init__(self, input_dim: int, width: int, hidden: int, masks: list[torch.Tensor]):
        super().__init__(input_dim, width, hidden,
                         layer_factory=lambda i, o, is_output: MaskedLinear(i, o, is_output=is_output))
        layers = self.linear_layers()
        if len(masks) != len(layers):
            raise ValueError(f"{len(masks)} masks for {len(layers)} layers")
        for layer, mask in zip(layers, masks):
            layer.set_mask(mask)


def level_lengths(hidden: int, depth: int) -> list[int]:
    """Hidden layers per nonlinear level: the lengths differ by at most one and add up to ``hidden``."""
    if hidden < depth:
        raise ValueError(f"the backbone has {hidden} hidden layers for {depth} nonlinear levels")
    return [(lv * hidden) // depth - ((lv - 1) * hidden) // depth for lv in range(1, depth + 1)]


@dataclass
class Plan:
    depth: int                  # nonlinear levels of the target
    lengths: list[int]          # hidden layers per level
    level: list[int]            # nonlinear level of every node
    blocks: list[dict]          # per level: block key -> range of units; key = ("node", j) or ("leaf", i)


def plan(target, width: int, hidden: int) -> Plan:
    graph = target.graph
    level = graph.levels()
    depth = max(level)
    last_use: dict = {}
    for v, refs in enumerate(graph.node_parents):
        for r in refs:
            key = ("leaf", int(r)) if r < graph.A else ("node", r - graph.A)
            last_use[key] = max(last_use.get(key, 0), level[v])
    for i in graph.readout_indices:
        last_use[("node", i)] = depth + 1

    blocks: list[dict] = []
    for lv in range(1, depth + 1):
        nodes = sorted(v for v in range(graph.S) if level[v] <= lv and last_use.get(("node", v), 0) > lv)
        leaves = sorted(a for a in range(graph.A) if last_use.get(("leaf", a), 0) > lv)
        keys = [("node", v) for v in nodes] + [("leaf", a) for a in leaves]
        minimum = {b: (1 if b[0] == "node" and level[b[1]] == lv else 2) for b in keys}
        need = sum(minimum.values())
        if need > width:
            raise ValueError(f"level {lv} needs {need} units for {len(keys)} blocks; the backbone has {width}")
        extra, remainder = divmod(width - need, len(keys))
        out, start = {}, 0
        for index, b in enumerate(keys):
            size = minimum[b] + extra + (1 if index < remainder else 0)
            out[b] = range(start, start + size)
            start += size
        blocks.append(out)
    return Plan(depth, level_lengths(hidden, depth), level, blocks)


def support_masks(target, width: int, hidden: int) -> list[torch.Tensor]:
    """GraphOracle(0): the first layer reads only the active inputs."""
    first = torch.zeros(width, int(target.d))
    first[:, np.unique(np.asarray(target.active_raw, dtype=int).ravel())] = 1.0
    return [first] + [torch.ones(width, width) for _ in range(hidden - 1)] + [torch.ones(1, width)]


def graph_masks(target, width: int, hidden: int, levels="all") -> list[torch.Tensor]:
    """The masks of every affine layer (``hidden`` hidden layers, then the readout)."""
    if levels == 0:
        return support_masks(target, width, hidden)
    graph = target.graph
    p = plan(target, width, hidden)
    k = p.depth if levels in (None, "all") else int(levels)
    if not 1 <= k <= p.depth:
        raise ValueError(f"levels = {levels} is not valid: the target has {p.depth} nonlinear levels")

    raw = np.asarray(target.active_raw, dtype=int).ravel()
    d = int(target.d)
    masks: list[torch.Tensor] = []
    previous: dict = {}
    previous_width = d
    for lv in range(1, p.depth + 1):
        current = p.blocks[lv - 1]
        parent_columns = {}
        for b in current:
            if not (b[0] == "node" and p.level[b[1]] == lv):
                continue
            columns: list[int] = []
            for r in graph.node_parents[b[1]]:
                if r < graph.A:
                    columns.extend([int(raw[r])] if lv == 1 else previous[("leaf", int(r))])
                else:
                    columns.extend(previous[("node", r - graph.A)])
            parent_columns[b] = columns
        for j in range(p.lengths[lv - 1]):
            m = torch.zeros(width, previous_width if j == 0 else width)
            for b, rows in current.items():
                if j == 0 and b in parent_columns:
                    columns = parent_columns[b]
                elif j == 0 and lv == 1:            # a leaf that a later level reads
                    columns = [int(raw[b[1]])]
                elif j == 0:                        # a value carried from the level before
                    columns = list(previous[b])
                else:                               # inside the level: each block reads itself
                    columns = list(current[b])
                m[np.ix_(list(rows), columns)] = 1.0
            masks.append(m)
        previous = dict(current)
        previous_width = width
    out = torch.zeros(1, width)
    for key in {("node", i) for i in graph.readout_indices}:
        out[0, list(p.blocks[-1][key])] = 1.0
    masks.append(out)

    if k < p.depth:                                 # no restriction after the supplied levels
        for i in range(sum(p.lengths[:k]), hidden):
            masks[i] = torch.ones(width, width if i else d)
        masks[-1] = torch.ones(1, width)
    return masks


def is_full(target, width: int, hidden: int, levels: int) -> bool:
    """True when ``GraphOracle(levels)`` has the same masks as ``GraphOracle(all)`` on this target."""
    full = graph_masks(target, width, hidden, "all")
    part = graph_masks(target, width, hidden, int(levels))
    return all(torch.equal(a, b) for a, b in zip(part, full))


@register("graph_oracle")
class GraphOracle(Method):
    hp_keys = frozenset({"levels"})
    graph_informed = True

    def build(self, cfg, target):
        width, hidden = int(cfg["width"]), int(cfg["hidden"])
        levels = cfg.get("hp", {}).get("levels", "all")
        return MaskedMLP(int(target.d), width, hidden, graph_masks(target, width, hidden, levels))
