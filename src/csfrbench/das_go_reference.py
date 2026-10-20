"""The known-block intervention for GraphOracle learners (fb2 section 3.2).

GraphOracle is given the target computation, and its lift plan states which
neuron block carries each teacher node at each stage. Swapping that block is
the causal edit the architecture was built to make, so its CF-R^2 is the
attainable reading for a learned probe at the same layer -- bounded only by
how well the learner fits.

The plan is not a unique representational ground truth: within a block the
learner may refine or duplicate content, and blocks may be wider than the
node needs. The reading is therefore an attainable reference, not a ceiling
proof.
"""
from __future__ import annotations

import torch

from csfrbench.methods.graph_oracle import plan, stage_lengths


def layer_stage_map(model, target, dim: int) -> list[int]:
    lift = plan(target, dim, len(model.hidden_layers))
    lengths = stage_lengths(len(model.hidden_layers), max(lift.level))
    mapping = []
    for stage, count in enumerate(lengths, start=1):
        mapping += [stage] * count
    return mapping, lift


def known_block_masks(model, target, roster: list[dict], layer: int,
                      dim: int) -> tuple[torch.Tensor, list[bool], dict]:
    """(m, d) masks selecting each variable's planned block at ``layer``."""
    mapping, lift = layer_stage_map(model, target, dim)
    stage = mapping[layer - 1]
    alloc = lift.alloc[stage - 1]
    masks = torch.zeros(len(roster), dim)
    present, units = [], {}
    for vi, entry in enumerate(roster):
        key = ("node", int(entry["canon_slot"]))
        cols = list(alloc.get(key, []))
        units[entry["var"]] = len(cols)
        if cols:
            masks[vi, cols] = 1.0
        present.append(bool(cols))
    return masks, present, {"stage": stage, "block_units": units}
