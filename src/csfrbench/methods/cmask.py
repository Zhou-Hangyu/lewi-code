"""Code-factorized connectivity masks: ``cmaskhir3`` and ``cmaskcg6``.

Both arms replace the free per-edge mask with a rank-``code_dim`` bilinear
form on neuron identities. Each neuron carries one code vector, and the layer
weight is multiplied by a deterministic gate built from the inner products of
the codes at the two ends of the edge,

    W_eff^l = W^l * sigmoid( a_l <phi^{l+1}_i, phi^l_j> + b_l ),

so connectivity is expressed through the code geometry rather than through one
free parameter per edge. On top of that soft gate a boundary may carry a
persistent binary *event mask*: every ``HARDEN_EVERY`` steps until
``HARDEN_UNTIL``, that boundary re-commits to a subset of its edges. Commitment
is reversible -- each event recomputes the mask from the current gate, so an
edge dropped at one event can return at a later one -- and it stops at
``HARDEN_UNTIL``, after which the last mask stands for the rest of training.

The two arms differ only in which boundaries commit and how:

``cmaskhir3``  the input boundary alone commits, by plain top-``fanin0`` over
               its own gate row. Every deeper boundary keeps its soft gate for
               the whole run. Searched: ``fanin0``, ``lr``.

``cmaskcg6``   the input boundary commits exactly as above, and boundary 1
               then commits too -- but by whole *modules*, where a module is a
               cluster of layer-1 units that boundary 0 just committed to read
               the same inputs. Searched: ``fanin0``, ``fanin1``, ``lr``.

The code dimension, the code initialization scale, the hardening schedule and
the module count are fixed constants of the methods, carried over unchanged
from the study that developed them (an earlier internal study, cards
``cmaskhir3-v1`` and ``modroute-v1``): the schedule is defined in absolute
optimizer steps, not as a fraction of the budget.

Auxiliary parameters (codes, per-layer gate scale and offset) are part of the
method and are disclosed with their count; the benchmark equalizes no
parameter count across arms (specification section 16.1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import method
from .base import Method
from .nns import DenseLinear, ReLUMLP, boundary_widths

CODE_DIM = 3
CODE_INIT = 0.5
HARDEN_EVERY = 2000
HARDEN_UNTIL = 16000


class SmoothCodeMaskedLinear(DenseLinear):
    """Affine layer with a deterministic sigmoid code-factorized mask."""

    def __init__(self, in_features, out_features, *, is_output):
        super().__init__(in_features, out_features, is_output=is_output)
        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.gate_offset = nn.Parameter(torch.tensor(0.0))    # gate 0.5 at init
        self._codes = None

    def bind_codes(self, recv: nn.Parameter, send: nn.Parameter) -> None:
        self._codes = (recv, send)

    def gate(self) -> torch.Tensor:
        recv, send = self._codes
        return torch.sigmoid(self.gate_scale * (recv @ send.t()) + self.gate_offset)

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.gate()

    def forward(self, x):
        return F.linear(x, self.weight * self.gate(), self.bias)


class HardenedCodeMaskedLinear(SmoothCodeMaskedLinear):
    """SmoothCodeMaskedLinear with a persistent binary event mask."""

    def __init__(self, in_features, out_features, *, is_output):
        super().__init__(in_features, out_features, is_output=is_output)
        self.register_buffer("event_mask", torch.ones(out_features, in_features))

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.gate() * self.event_mask

    def forward(self, x):
        return F.linear(x, self.weight * self.gate() * self.event_mask, self.bias)

    @torch.no_grad()
    def harden_(self, fanin: int) -> None:
        k = min(fanin, self.in_features)
        gate = self.gate()
        mask = torch.zeros_like(self.event_mask)
        mask.scatter_(1, gate.topk(k, dim=1).indices, 1.0)
        self.event_mask.copy_(mask)


@method("cmaskhir3")
class InputOnlyHardenedCodeMask(Method):
    """Smooth code mask everywhere, event hardening at the input boundary."""

    code_dim = CODE_DIM

    def build(self, cfg, target):
        fanin0 = max(2, int(round(float(cfg["hp"]["fanin0"]))))
        model = ReLUMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
            layer_factory=lambda i, o, is_output: HardenedCodeMaskedLinear(
                i, o, is_output=is_output),
        )
        widths = boundary_widths(model)
        generator = torch.Generator().manual_seed(int(cfg["auxiliary_init_seed"]))
        model.gate_codes = nn.ParameterList([
            nn.Parameter(torch.randn(int(w), self.code_dim, generator=generator)
                         * CODE_INIT)
            for w in widths
        ])
        for boundary, layer in enumerate(model.linear_layers()):
            layer.bind_codes(model.gate_codes[boundary + 1],
                             model.gate_codes[boundary])
        model.hard_fanin0 = fanin0
        return model

    def after_optimizer_step(self, model, optimizer, *, step, total_steps):
        if step % HARDEN_EVERY or step > HARDEN_UNTIL or step >= total_steps:
            return
        first = next(iter(model.linear_layers()))
        first.harden_(model.hard_fanin0)

    def effective_weights(self, model):
        return tuple(model.effective_weights())

    def extra_diagnostics(self, model):
        with torch.no_grad():
            gates = torch.cat([layer.gate().reshape(-1)
                               for layer in model.linear_layers()])
            masks = torch.cat([layer.event_mask.reshape(-1)
                               for layer in model.linear_layers()])
        return {"gate_open_mean": float(gates.mean()),
                "gate_frac_decided": float(((gates < 0.05) | (gates > 0.95))
                                           .to(torch.float32).mean()),
                "event_mask_density": float(masks.mean())}


GROUPS = 6


@torch.no_grad()
def cluster_rows(mask: torch.Tensor, groups: int, iters: int = 12) -> torch.Tensor:
    """k-means over the binary read-set rows; deterministic furthest-point seeding."""
    n = mask.shape[0]
    g = max(1, min(int(groups), n))
    centres = mask[:1].clone()
    for _ in range(g - 1):
        d = torch.cdist(mask, centres).min(dim=1).values
        centres = torch.cat([centres, mask[int(d.argmax())].unsqueeze(0)])
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        assign = torch.cdist(mask, centres).argmin(dim=1)
        for c in range(g):
            members = mask[assign == c]
            if len(members):
                centres[c] = members.mean(0)
    return assign


@torch.no_grad()
def harden_by_module_(layer, fanin: int, assign: torch.Tensor) -> None:
    """Commit whole modules: each receiver takes classes whole, best mean first."""
    recv, send = layer._codes
    score = layer.gate_scale * (recv @ send.t()) + layer.gate_offset
    z = ((score - score.mean(1, keepdim=True))
         / score.std(1, keepdim=True).clamp_min(1e-12))
    groups = int(assign.max()) + 1
    members = [(assign == c).nonzero(as_tuple=True)[0] for c in range(groups)]
    mask = torch.zeros_like(layer.event_mask)
    for i in range(mask.shape[0]):
        means = torch.tensor([float(z[i, m].mean()) if len(m) else -1e30
                              for m in members])
        used = 0
        for c in torch.argsort(means, descending=True).tolist():
            if not len(members[c]):
                continue
            if used and used + len(members[c]) > int(fanin):
                break
            mask[i, members[c]] = 1.0
            used += len(members[c])
            if used >= int(fanin):
                break
    layer.event_mask.copy_(mask)


@method("cmaskcg6")
class ModuleRoutedCodeMask(InputOnlyHardenedCodeMask):
    """Two committed boundaries, where the second selects whole modules.

    Boundary 0 commits first and exactly as in ``cmaskhir3``. Its freshly
    committed mask then defines the modules: the units of layer 1 are
    clustered into ``groups`` classes by their own committed read-sets, and
    boundary 1 ranks those classes by their mean row-standardized gate logit
    and takes them **whole** while the accumulated column count stays within
    ``fanin1``. Boundary 1's structure is therefore a function of boundary 0's
    structure, rather than an independently chosen top-k.

    ``fanin1`` is a soft cap in both directions, because the executed fan-in is
    always a sum of whole class sizes: a receiver whose best class alone exceeds
    ``fanin1`` takes that class whole and overshoots, and a receiver whose best
    class is a singleton undershoots, because adding the next class would break
    the cap. A module is only useful if it is read whole, so the class boundary
    wins over the budget. The executed count is reported per run as
    ``executed_fanin1_mean``.
    """

    groups = GROUPS

    def build(self, cfg, target):
        model = super().build(cfg, target)
        model.hard_fanin1 = max(2, int(round(float(cfg["hp"]["fanin1"]))))
        return model

    def after_optimizer_step(self, model, optimizer, *, step, total_steps):
        if step % HARDEN_EVERY or step > HARDEN_UNTIL or step >= total_steps:
            return
        layers = list(model.linear_layers())
        layers[0].harden_(model.hard_fanin0)
        assign = cluster_rows(layers[0].event_mask.detach(), self.groups)
        harden_by_module_(layers[1], model.hard_fanin1, assign)

    def extra_diagnostics(self, model):
        out = dict(super().extra_diagnostics(model))
        with torch.no_grad():
            layers = list(model.linear_layers())
            out["groups"] = float(self.groups)
            out["executed_fanin1_mean"] = float(layers[1].event_mask.sum(1).mean())
        return out
