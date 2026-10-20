"""The record arm with an input-side initialization condition (study-local; the record arm `lewi-v1-rgm` is untouched).

``lewi-v1-rgm-in<m>``: after the record's build, every input that holds fewer than m open first-transition connections
has its position drawn again by indexed attempts, e^(a) ~ N(0, s^2 I) from the generator seeded by
derive(auxiliary seed, "input", i, a), a = 1, 2, ...; the first attempt with at least m open connections is kept.
Attempt 0 is the record's own draw, so a cell whose inputs all meet the condition is the record bit for bit, and the
attempts use their own generators, so no other quantity (weights, other positions, data order) depends on a redraw.
m = 1 is the input-side twin of the readout rule adopted on 2026-09-12 (every transition holds an open connection):
every input holds an open connection. The diagnostic `input_redraws` counts the attempts.

``lewi-v1-rgm-infloor``: the record's fan-out floor (rule 52.11, csfrbench.methods.lewi_v1._rederive_mask_floor_) on the
input boundary, which the record's floor option leaves out: at every re-derivation an input whose connections are all
closed keeps its largest-score one, so its position keeps a gradient. A run in which no input ever loses every
connection is the record bit for bit. The diagnostic `input_floor_steps` counts the re-derivations where the floor acted.
"""
from __future__ import annotations

import torch

import w5_methods
from csfrbench.methods.cmask import CODE_INIT
from csfrbench.methods.lewi_v1 import _rederive_mask_floor_
from csfrbench.registry import method
from csfrbench.utils.seeds import derive

MIN_OPEN = (1, 5)


class _InputCondition(w5_methods.LeWiV1ReadoutGateMask):
    min_open = 1

    def build(self, cfg, target):
        model = super().build(cfg, target)
        layer, codes = model.hidden_layers[0], model.gate_codes[0]
        model.lewi_input_redraws = 0
        with torch.no_grad():
            for i in range(codes.shape[0]):
                attempt = 0
                while int((layer.score()[:, i] > 0).sum()) < self.min_open:
                    attempt += 1
                    g = torch.Generator().manual_seed(int(derive(int(cfg["auxiliary_init_seed"]), "input", i, attempt)))
                    codes[i].copy_(torch.randn(codes.shape[1], generator=g) * CODE_INIT)
                    model.lewi_input_redraws += 1
        if model.lewi_input_redraws:
            self._rederive_masks(model)
        return model

    def extra_diagnostics(self, model):
        out = dict(super().extra_diagnostics(model))
        out["input_redraws"] = int(getattr(model, "lewi_input_redraws", 0))
        return out


for _m in MIN_OPEN:
    method(f"lewi-v1-rgm-in{_m}")(type(f"LeWiInputCondition{_m}", (_InputCondition,), {"min_open": _m}))


@method("lewi-v1-rgm-infloor")
class LeWiInputFloor(w5_methods.LeWiV1ReadoutGateMask):
    @staticmethod
    def _rederive_masks(model) -> None:
        w5_methods.LeWiV1ReadoutGateMask._rederive_masks(model)
        layer = model.hidden_layers[0]
        if not bool(layer.event_mask.any(dim=0).all()):
            _rederive_mask_floor_(layer)
            model.lewi_input_floor_steps = getattr(model, "lewi_input_floor_steps", 0) + 1

    def extra_diagnostics(self, model):
        out = dict(super().extra_diagnostics(model))
        out["input_floor_steps"] = int(getattr(model, "lewi_input_floor_steps", 0))
        return out
