"""Discovering Neural Wirings (Wortsman et al., 2019) on the shared MLP
backbone (per-workload profile; see configs/profile.yaml).

Sources: arXiv:1906.00586 and the official implementation allenai/dnw
(models/graphs/util.py::ChooseTopEdges, ::DNW). Method card:
studies/csfrbench/final-design/versions/v1.0/method_cards/dnw.md

Faithful mechanics, per source:
  * every affine weight tensor is a wiring graph: the forward pass uses
    only the largest-magnitude edges - the lowest ``prune_rate`` fraction
    of |w| within the tensor is zeroed (util.py zeroes
    int(prune_rate * numel) entries after a whole-tensor sort);
  * the backward pass is the identity to the full tensor ("we allow the
    gradient to flow to edges which did not exist during the forwards
    pass ... not through them"): every edge, active or not, receives the
    gradient it would have received at the masked-product node, so
    inactive edges keep training and can re-enter the active set;
  * no mask state is stored - the active set is recomputed from |w| at
    every forward, and the evaluation-time effective weights are the same
    deterministic top-k selection;
  * biases stay dense (the source's graphs are bias-free; the benchmark
    envelope keeps its dense biases).

Adaptation, disclosed in the method card: the source initializes its 1x1
convolutions with the framework default; here the weights use the common
benchmark envelope initialization. The edge budget is parameterized as
``density`` = 1 - prune_rate, one shared value for every layer, searched
by HPO alongside the learning rate.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import autograd

from ..registry import method
from .base import Method, readout_structural
from .nns import DenseLinear, ReLUMLP


class ChooseTopEdges(autograd.Function):
    """Top-|w| edges forward; identity gradient to every edge backward."""

    @staticmethod
    def forward(ctx, weight, prune_count):
        output = weight.clone()
        if prune_count > 0:
            idx = weight.flatten().abs().argsort()
            output.flatten()[idx[:prune_count]] = 0.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class DNWLinear(DenseLinear):
    """Affine layer whose active edge set is the running top-|w| subset."""

    def __init__(self, in_features, out_features, *, is_output):
        super().__init__(in_features, out_features, is_output=is_output)
        self.register_buffer("prune_count", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        w_eff = ChooseTopEdges.apply(self.weight, int(self.prune_count))
        return F.linear(x, w_eff, self.bias)

    def effective_weight(self):
        with torch.no_grad():
            return ChooseTopEdges.apply(self.weight, int(self.prune_count))


@method("dnw")
class DNW(Method):
    """DNW: magnitude-chosen wiring with straight-through edge updates."""

    hp_keys = frozenset({"density", "readout"})

    def build(self, cfg, target):
        density = float(cfg["hp"]["density"])
        if not 0.0 < density <= 1.0:
            raise ValueError(f"density must lie in (0, 1]; received {density}")
        prune_rate = 1.0 - density
        structural = readout_structural(cfg.get("hp"))
        model = ReLUMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
            # v1.0 rule: the readout is never pruned; v1.1 ``readout:
            # structural``: the readout is a wiring graph like every layer
            layer_factory=lambda i, o, is_output: (
                DenseLinear(i, o, is_output=True) if (is_output and not structural)
                else DNWLinear(i, o, is_output=is_output)),
            readout_structural=structural,
        )
        for layer in model.structural_layers():
            # The source zeroes int(prune_rate * numel) entries.
            layer.prune_count.fill_(int(prune_rate * layer.weight.numel()))
        model.dnw_density = density
        return model

    def effective_weights(self, model):
        # The top-|w| selection; the raw tensors are fully dense (inactive
        # edges keep training) and would report a dense network into the
        # structure diagnostics (audit finding D2, repaired 2026-08-28).
        return tuple(model.effective_weights())

    def extra_diagnostics(self, model):
        active = total = 0
        for layer in model.structural_layers():
            n = layer.weight.numel()
            total += n
            active += n - int(layer.prune_count)
        return {"active_connections": float(active),
                "density_realized": active / total,
                "density_target": float(model.dnw_density)}
