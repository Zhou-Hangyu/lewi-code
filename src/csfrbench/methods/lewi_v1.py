"""LeWi-v1 - the ball with one global fixed radius, floorless (CSFRBench v1.0; v1.1 adds the
structural readout, `readout: structural`).

Method card:
studies/csfrbench/final-design/versions/v1.0/method_cards/lewi-v1.md

Ported from the winner of an earlier internal LeWi study (its arm
``ng_n_eu_hid_fx_ro0``, a subclass of the LeWi-v05 record whose flags-off
form reproduces ``lewi-v05`` bit for bit).
Design decision 2026-09-09: the floorless ball is the main method; the
floored ball is an ablation (``floor=1``).

The method:

  * one position e_i in R^3 per unit (inputs and hidden units), initialized
    N(0, 0.5^2 I) from the auxiliary seed stream exactly as LeWi-v05 draws
    its codes; the same position serves the unit's sender and receiver roles;
  * on every transition but the readout, the score s_ij = R^2 - ||e_i - e_j||^2
    with R^2 = 2 d s^2 = 1.5 (61% of the edges open at initialization), the
    gate g_ij = sigmoid(s_ij) with the gate scale 1 and offset R^2 frozen
    (never learned), and the binary mask M_ij = 1[g_ij > 1/2], derived at
    build (the masks are live from step 0) and re-derived after every
    optimizer step; the forward pass is y = (W * G * M) x + b (soft gate
    times hard mask, no straight-through);
  * the readout: plain (v1.0 rule, the default) or, under the CSFRBench
    v1.1 fixed argument ``readout: structural``, gated and masked exactly
    like every other transition - the output unit gets one position drawn
    from the auxiliary stream right after the hidden positions (so every
    input and hidden position equals the plain form's at matched seeds),
    the readout weight is W * G * M with the same frozen radius, and its
    mask is re-derived after every step (the 12-extra wave-5 arm
    ``lewi-v1-rgm``; weight initialization unchanged, the readout draws
    its weights in the same order as the plain readout);
  * the connected-initialization rule (CSFRBench v1.1 item 7, decision
    2026-09-12): a transition whose draw leaves no edge open (score <= 0 on
    every pair; the readout under ``readout: structural`` has one receiver,
    so its draw lands outside every last-layer ball about once in a
    thousand cells, and the network is then a constant for the whole run,
    for every trial of that cell) has its receivers' positions redrawn by
    indexed attempts: attempt a >= 1 draws N(0, 0.5^2 I) from its own
    generator seeded by derive(auxiliary_init_seed, key, a) with key
    ("output",) for the readout and ("hidden", boundary) otherwise, and
    the first attempt with an open edge stands. Attempt 0 is the auxiliary
    stream's own draw, so every connected initialization is bit for bit
    the record and the auxiliary stream is never advanced (the hidden
    positions, the weights and the data order never depend on whether a
    redraw happened); ``readout_redraws`` in the diagnostics counts the
    attempts taken (the 12-extra readout study, report readout-mask.html);
  * no penalty, no warm-up; the learning rate is the only searched number.

Knobs (``hp_keys``; the defaults reproduce the method above):

  * ``floor``     0 (default) or 1: the fan-out floor on the transitions
                  whose senders are hidden units - a sender whose outgoing
                  edges would all close keeps its largest-score edge open;
                  inputs are exempt (an input with no reader is a discarded
                  input, not a dead unit); the next-gen arm is
                  ``ng_n_eu_hid_fo_fx_ro0``;
  * ``code_dim``  the position dimension d (default 3); R^2 scales with d;
  * ``radius``    multiplier of R^2 (default 1.0; next-gen 6.36-6.37 read
                  0.5 and 1.5);
  * ``readout``   "plain" (default) or "structural" (v1.1): see above.
"""
from __future__ import annotations

import torch
from torch import nn

from ..registry import method
from .base import Method, readout_structural
from .cmask import CODE_INIT
from .lewi_v05 import CODE_RANK, EuclideanCodeMaskedLinear, _rederive_mask_
from .nns import DenseLinear, ReLUMLP, boundary_widths
from ..utils.seeds import derive


class BallMaskedLinear(EuclideanCodeMaskedLinear):
    """Score R^2 - ||e_i - e_j||^2 (the offset is R^2, the scale 1)."""

    def score(self) -> torch.Tensor:
        recv, send = self._codes
        return self.gate_scale * (-torch.cdist(recv, send) ** 2) + self.gate_offset

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.score())


@torch.no_grad()
def _rederive_mask_floor_(layer) -> None:
    """M <- 1[s > 0], then every sender with no open outgoing edge keeps its
    largest-score edge (the record's fan-out floor, rule 52.11)."""
    score = layer.score()
    keep = score > 0
    dead = ~keep.any(dim=0)                       # senders are the columns
    if bool(dead.any()):
        rows = score.argmax(dim=0)[dead]
        keep[rows, torch.nonzero(dead, as_tuple=True)[0]] = True
    layer.event_mask.copy_(keep.to(layer.event_mask.dtype))


@method("lewi-v1")
class LeWiV1(Method):
    """LeWi-v1: positions in a fixed ball, masks on every transition but the
    readout (plain readout) or on every transition (structural readout)."""

    code_dim = CODE_RANK
    hp_keys = frozenset({"floor", "code_dim", "radius", "readout"})

    def build(self, cfg, target):
        hp = cfg.get("hp") or {}
        code_dim = int(hp.get("code_dim", self.code_dim))
        radius = float(hp.get("radius", 1.0))
        floor = int(hp.get("floor", 0))
        if code_dim < 1:
            raise ValueError("code_dim must be a positive integer")
        if radius <= 0:
            raise ValueError("radius must be positive")
        if floor not in (0, 1):
            raise ValueError("floor must be 0 or 1")
        structural = readout_structural(hp)
        model = ReLUMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
            layer_factory=lambda i, o, is_output: (
                DenseLinear(i, o, is_output=True) if (is_output and not structural)
                else BallMaskedLinear(i, o, is_output=is_output)),
            readout_structural=structural,
        )
        widths = boundary_widths(model)
        generator = torch.Generator().manual_seed(int(cfg["auxiliary_init_seed"]))
        # One position per input and hidden unit, drawn in the record's order
        # (boundary by boundary), so the positions equal LeWi-v05's codes;
        # under the structural readout the output unit's position is the one
        # extra draw at the end of the stream (12-extra wave 5).
        positioned = widths if structural else widths[:-1]
        model.gate_codes = nn.ParameterList([
            nn.Parameter(torch.randn(int(w), code_dim, generator=generator)
                         * CODE_INIT)
            for w in positioned
        ])
        squared_radius = 2.0 * code_dim * CODE_INIT ** 2 * radius
        for boundary, layer in enumerate(model.structural_layers()):
            layer.bind_codes(model.gate_codes[boundary + 1],
                             model.gate_codes[boundary])
            with torch.no_grad():
                layer.gate_offset.fill_(squared_radius)
            layer.gate_scale.requires_grad_(False)
            layer.gate_offset.requires_grad_(False)
        model.lewi_floor = floor
        # The connected-initialization rule (v1.1 item 7): a transition with
        # no open edge redraws its receivers' positions by indexed attempts
        # from their own generators; attempt 0 is the stream's draw above.
        model.lewi_readout_redraws = 0
        aux_seed = int(cfg["auxiliary_init_seed"])
        last = len(positioned) - 2               # the readout boundary (structural only)
        with torch.no_grad():
            for boundary, layer in enumerate(model.structural_layers()):
                attempt = 0
                while not bool((layer.score() > 0).any()):
                    attempt += 1
                    key = ("output",) if (structural and boundary == last) else ("hidden", boundary)
                    g = torch.Generator().manual_seed(int(derive(aux_seed, *key, attempt)))
                    codes = model.gate_codes[boundary + 1]
                    codes.copy_(torch.randn(codes.shape[0], code_dim, generator=g) * CODE_INIT)
                    model.lewi_readout_redraws += 1
        self._rederive_masks(model)               # the masks are live from step 0
        return model

    @staticmethod
    def _rederive_masks(model) -> None:
        layers = model.structural_layers()
        if getattr(model, "lewi_floor", 0):
            _rederive_mask_(layers[0])            # inputs: the plain rule
            for layer in layers[1:]:
                _rederive_mask_floor_(layer)
            return
        for layer in layers:
            _rederive_mask_(layer)

    def after_optimizer_step(self, model, optimizer, *, step, total_steps):
        self._rederive_masks(model)

    def effective_weights(self, model):
        return tuple(layer.effective_weight()
                     for layer in model.linear_layers())

    def extra_diagnostics(self, model):
        # The hidden-transition statistics are the same quantities under both
        # readout forms (comparable across v1.0 and v1.1 records); the
        # structural readout adds its own three numbers.
        layers = tuple(model.hidden_layers)
        with torch.no_grad():
            gates = torch.cat([layer.gate().reshape(-1) for layer in layers])
            masks = [layer.event_mask for layer in layers]
            active = sum(int(m.sum()) for m in masks)
            total = sum(m.numel() for m in masks)
            fan_in_0 = float(masks[0].sum(dim=1).mean())
            out = {"gate_open_mean": float(gates.mean()),
                   "gate_frac_decided": float(((gates < 0.05) | (gates > 0.95))
                                              .to(torch.float32).mean()),
                   "masked_boundary_density": (active / total) if total else 1.0,
                   "input_fan_in_mean": fan_in_0}
            if model.readout_structural:
                readout = model.output_layer
                gate = readout.gate().reshape(-1)
                out["readout_gate_open_mean"] = float(gate.mean())
                out["readout_open_frac"] = float((gate > 0.5).to(torch.float32).mean())
                out["readout_mask_density"] = float(readout.event_mask.mean())
            out["readout_redraws"] = int(getattr(model, "lewi_readout_redraws", 0))
        return out
