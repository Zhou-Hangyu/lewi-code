"""Wave-5 arms (intake/wave-5.md): LeWi-v1 with its readout transition gated.

The package's ``lewi-v1`` (src/csfrbench/methods/lewi_v1.py) is the floorless
ball: one position per input and hidden unit, the score
s_ij = R^2 - ||e_i - e_j||^2 with R^2 = 2 d s^2 = 1.5 frozen, the soft gate
g_ij = sigmoid(s_ij) and the binary mask M_ij = 1[g_ij > 1/2] on every
transition but the readout, which is a plain linear map (v1.0 plain-readout
rule). The two arms below change the readout only; everything else is the
package class unchanged (the same code, the same seeds, the same knobs).

``lewi-v1-rg``  - the readout carries the same soft gate as every other
                  transition: the output unit gets one position e_out, drawn
                  from the auxiliary seed stream right after the hidden
                  positions (so every input and hidden position equals
                  LeWi-v1's at matched seeds); the readout weight is
                  W * g with g_j = sigmoid(R^2 - ||e_out - e_j||^2), scale and
                  offset frozen; NO mask on the readout (M = 1 throughout).
``lewi-v1-rgm`` - the readout is gated AND masked like every other
                  transition: M_j = 1[g_j > 1/2], derived at build and
                  re-derived after every optimizer step. With ``floor=1`` the
                  fan-out floor also covers the readout (its senders are
                  hidden units); with one output unit that floor reopens every
                  closed readout edge, i.e. it is vacuous there.

Dead readout (found 2026-09-12): an output position drawn outside every
last-layer unit's ball has no open readout edge; the hard mask then blocks
every gradient to that position and the network is a constant for the whole
run, for every learning rate of the instance (wave 5: the Fourier reuse DAG,
instance 7013, n = 8192, all 32 trials). The fixed argument ``readout_fix``
selects the remedy under study (studies/main-experiments/12-extra/report/
readout-mask.html):
  "none"    the arm as run in wave 5 before the repair;
  "redraw"  the default since the authors adopted it for the arm (2026-09-12,
            the wave-5 records repaired by w5_repair_merge.py): rejection
            sampling at build under the generic initialization condition:
            every transition must hold at least one open edge, i.e. a
            disconnected initialization is rejected.
            Implementation (design decision): the receivers of a transition are drawn
            by indexed attempts, e^(a) ~ N(0, s^2 I) with the generator
            seeded by hash(run seed, key, a) - key "output" for the readout,
            ("hidden", boundary) otherwise - and the first attempt with an
            open edge is kept. Attempt 0 is the auxiliary stream's own draw,
            so every connected initialization equals the record bit for bit;
            attempts a >= 1 use their own generators, so the auxiliary stream
            is never advanced and no other quantity (hidden positions,
            weights, data order) depends on whether a redraw happened. A
            hidden transition (32 x 32 edges, 61 % open) is never empty in
            practice, so the rule acts on the readout; a single dead unit
            among several receivers is the method's ordinary sparsity and is
            not rejected.
  "floor"   the readout's fan-in floor at every re-derivation: an output unit
            whose readout edges are all closed keeps its largest-score edge
            (the mirror of the record's fan-out floor, rule 52.11); acts only
            while the readout would otherwise be empty;
  "origin"  the output unit's position is the origin (the centre of the
            position law) instead of a draw; every run changes.
  "redraw_all"  rejection sampling of the whole position set under the same
            condition: while any transition has no open edge, every position
            (inputs, hidden units, output) is drawn again from the continuing
            stream, so the accepted set keeps the joint law of a draw
            conditioned on a connected network; nothing else changes.
  "radius2", "radius4"  one global squared radius as before, but 2 x or 4 x
            the frozen R^2 on every transition (an edge of a random pair is
            open with probability 0.89 or 0.99 instead of 0.61, the readout
            included); every run changes.
  "st"      straight-through masks on every transition (decision: "st on all
            layers"): the forward pass keeps the hard masks, the backward
            pass treats each mask as its soft gate (M + g - stopgrad(g)), so
            a closed edge still moves its two positions and any transition,
            the readout included, can reopen from any state; every run
            changes.

Weight initialization is unchanged: the readout layer draws its weights from
the global RNG in the same order as the plain readout (DenseLinear.
reset_parameters with the "linear" gain), so the initial weights of every
layer equal LeWi-v1's at matched seeds; the arms differ from LeWi-v1 only by
the readout gate (and mask) from step 0 on.
"""
from __future__ import annotations

import torch
from torch import nn

from csfrbench.methods.cmask import CODE_INIT
from csfrbench.methods.lewi_v05 import _rederive_mask_
from csfrbench.methods.lewi_v1 import (BallMaskedLinear, LeWiV1,
                                       _rederive_mask_floor_)
from csfrbench.methods.nns import ReLUMLP, boundary_widths
from csfrbench.registry import method
from csfrbench.utils.seeds import derive

ARMS = ("lewi-v1", "lewi-v1-rg", "lewi-v1-rgm")
PAPER = {"lewi-v1": "LeWi-v1",
         "lewi-v1-rg": "LeWi-v1 + readout gate",
         "lewi-v1-rgm": "LeWi-v1 + readout gate and mask"}


@torch.no_grad()
def _rederive_readout_mask_(layer) -> None:
    """M <- 1[s > 0], then every receiver (output unit) with no open incoming
    edge keeps its largest-score edge: the readout's fan-in floor."""
    score = layer.score()
    keep = score > 0
    dead = ~keep.any(dim=1)                       # receivers are the rows
    if bool(dead.any()):
        cols = score.argmax(dim=1)[dead]
        keep[torch.nonzero(dead, as_tuple=True)[0], cols] = True
    layer.event_mask.copy_(keep.to(layer.event_mask.dtype))


READOUT_FIXES = ("none", "redraw", "redraw_all", "floor", "origin", "radius2", "radius4", "st")
READOUT_RADIUS_SCALE = {"radius2": 2.0, "radius4": 4.0}      # a global radius: every transition, the readout included


class STBallMaskedLinear(BallMaskedLinear):
    """The ball-masked layer with a straight-through mask: hard in the forward
    pass, the soft gate in the backward pass (closed edges keep a gradient)."""

    def effective_weight(self) -> torch.Tensor:
        gate = self.gate()
        return self.weight * gate * (self.event_mask + gate - gate.detach())

    def forward(self, x):
        return torch.nn.functional.linear(x, self.effective_weight(), self.bias)


class _ReadoutGatedLeWiV1(LeWiV1):
    """LeWi-v1 whose readout is a BallMaskedLinear bound to an output position."""

    readout_masked = False
    hp_keys = frozenset({"floor", "code_dim", "radius", "readout_fix"})

    def build(self, cfg, target):
        hp = cfg.get("hp") or {}
        code_dim = int(hp.get("code_dim", self.code_dim))
        radius = float(hp.get("radius", 1.0))
        floor = int(hp.get("floor", 0))
        fix = str(hp.get("readout_fix", "redraw"))
        if fix not in READOUT_FIXES:
            raise ValueError(f"readout_fix must be one of {READOUT_FIXES}; received {fix!r}")
        if code_dim < 1:
            raise ValueError("code_dim must be a positive integer")
        if radius <= 0:
            raise ValueError("radius must be positive")
        if floor not in (0, 1):
            raise ValueError("floor must be 0 or 1")
        model = ReLUMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
            layer_factory=lambda i, o, is_output: (STBallMaskedLinear if fix == "st" else BallMaskedLinear)(
                i, o, is_output=is_output),
        )
        widths = boundary_widths(model)
        generator = torch.Generator().manual_seed(int(cfg["auxiliary_init_seed"]))
        # The record's order (boundary by boundary): the first len(widths) - 1
        # draws are LeWi-v1's positions; the output unit's position is the
        # one extra draw at the end of the stream.
        model.gate_codes = nn.ParameterList([
            nn.Parameter(torch.randn(int(w), code_dim, generator=generator)
                         * CODE_INIT)
            for w in widths
        ])
        squared_radius = 2.0 * code_dim * CODE_INIT ** 2 * radius
        for boundary, layer in enumerate(model.linear_layers()):
            layer.bind_codes(model.gate_codes[boundary + 1],
                             model.gate_codes[boundary])
            with torch.no_grad():
                layer.gate_offset.fill_(squared_radius * READOUT_RADIUS_SCALE.get(fix, 1.0))
            layer.gate_scale.requires_grad_(False)
            layer.gate_offset.requires_grad_(False)
        model.lewi_floor = floor
        model.lewi_readout_masked = bool(self.readout_masked)
        model.lewi_readout_fix = fix
        readout = model.output_layer
        with torch.no_grad():
            if fix == "origin":
                model.gate_codes[-1].zero_()
            elif fix == "redraw":
                model.lewi_readout_redraws = 0
                last = len(widths) - 2
                for boundary, layer in enumerate(model.linear_layers()):
                    attempt = 0
                    while not bool((layer.score() > 0).any()):        # the transition has no open edge: next attempt for its receivers
                        attempt += 1
                        key = ("output",) if boundary == last else ("hidden", boundary)
                        g = torch.Generator().manual_seed(int(derive(int(cfg["auxiliary_init_seed"]), *key, attempt)))
                        codes = model.gate_codes[boundary + 1]
                        codes.copy_(torch.randn(codes.shape[0], code_dim, generator=g) * CODE_INIT)
                        model.lewi_readout_redraws += 1
            elif fix == "redraw_all":
                model.lewi_readout_redraws = 0
                while not all(bool((layer.score() > 0).any()) for layer in model.linear_layers()):
                    for codes in model.gate_codes:
                        codes.copy_(torch.randn(codes.shape[0], code_dim, generator=generator) * CODE_INIT)
                    model.lewi_readout_redraws += 1
        self._rederive_masks(model)               # the masks are live from step 0
        return model

    @staticmethod
    def _rederive_masks(model) -> None:
        layers = list(model.structural_layers())
        floor = getattr(model, "lewi_floor", 0)
        if floor:
            _rederive_mask_(layers[0])            # inputs: the plain rule
            for layer in layers[1:]:
                _rederive_mask_floor_(layer)
        else:
            for layer in layers:
                _rederive_mask_(layer)
        if getattr(model, "lewi_readout_masked", False):
            if floor:
                _rederive_mask_floor_(model.output_layer)     # the fan-out floor, vacuous with one output unit
            elif getattr(model, "lewi_readout_fix", "none") == "floor":
                _rederive_readout_mask_(model.output_layer)   # the fan-in floor
            else:
                _rederive_mask_(model.output_layer)

    def extra_diagnostics(self, model):
        out = super().extra_diagnostics(model)    # the hidden transitions, as LeWi-v1
        readout = model.output_layer
        with torch.no_grad():
            gate = readout.gate().reshape(-1)
            out["readout_gate_open_mean"] = float(gate.mean())
            out["readout_open_frac"] = float((gate > 0.5).to(torch.float32).mean())
            out["readout_mask_density"] = float(readout.event_mask.mean())
            score = readout.score()
            out["readout_floored_frac"] = float((score.max(dim=1).values <= 0).to(torch.float32).mean())
            out["readout_redraws"] = int(getattr(model, "lewi_readout_redraws", 0))
        return out


@method("lewi-v1-rg")
class LeWiV1ReadoutGate(_ReadoutGatedLeWiV1):
    """LeWi-v1 + the soft gate on the readout, no mask there."""

    readout_masked = False


@method("lewi-v1-rgm")
class LeWiV1ReadoutGateMask(_ReadoutGatedLeWiV1):
    """LeWi-v1 + gate and mask on the readout, as on every other transition."""

    readout_masked = True
