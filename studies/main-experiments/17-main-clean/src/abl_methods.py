"""Mechanism ablations of the final LeWi (18-main-d8 intake/analysis.md section 3, decision 2026-09-18).

Every arm is ``lewi-v1-rgm`` built exactly as the record builds it (the same seeds, the same positions, the same
connected-initialization rule, the same initial weights), then changed in one mechanism:

  ``lewi-abl-soft``    soft-only: the trainable positions and the distance gates stay, the binary mask is removed
                       on every transition, the readout included (W * G in the forward pass; the would-be masks
                       are still re-derived after every step for the density diagnostics);
  ``lewi-abl-fixed``   fixed embeddings: the positions are drawn as usual and frozen for the whole run, so the
                       gates and masks are those of the initialization; the connection weights train;
  ``lewi-abl-random``  random sparse connectivity: a fixed random binary mask per transition with exactly as many
                       open edges as LeWi's initial mask at the matched seed (the readout included), no gate
                       (G = 1), positions frozen; the weights train under that mask;
  ``lewi-abl-ste``     straight-through: the forward pass keeps the hard masks, the backward pass treats each mask as
                       its soft gate (M + g - stopgrad(g); w5_methods.STBallMaskedLinear), so a closed edge still
                       moves its two positions and any transition can reopen.

The sensitivity arms of the same intake (the position dimension r and the radius multiplier) are the record's method
with a fixed hyperparameter (m17.ARM_HP), not classes here.
"""
from __future__ import annotations

import torch

from csfrbench.registry import method
from csfrbench.methods.lewi_v1 import BallMaskedLinear
from csfrbench.utils.seeds import derive
from w5_methods import LeWiV1ReadoutGateMask, STBallMaskedLinear

ARMS = ("lewi-abl-soft", "lewi-abl-fixed", "lewi-abl-random", "lewi-abl-ste")
# Initialization-scale sensitivity (the same intake): the positions AND the radius are scaled together by c (R^2 by c^2), so
# the initial masks are exactly the record's (score = R^2 - dist^2 scales by c^2, its sign is unchanged) and only the scale of
# the geometry changes: the soft gate sigmoid(score) is sharper (c > 1) or flatter (c < 1) and a position step of a given size
# is smaller or larger relative to the radius.
INIT_ARMS = {"lewi-init05": 0.5, "lewi-init2": 2.0}
PAPER = {"lewi-abl-soft": "LeWi, soft-only (no mask)", "lewi-abl-fixed": "LeWi, fixed embeddings",
         "lewi-abl-random": "random sparse (LeWi's initial density)", "lewi-abl-ste": "LeWi, straight-through mask"}


class SoftBallLinear(BallMaskedLinear):
    """The ball layer without its binary mask: W * G in the forward pass."""

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.gate()

    def forward(self, x):
        return torch.nn.functional.linear(x, self.effective_weight(), self.bias)


class GateFreeMaskedLinear(BallMaskedLinear):
    """A fixed binary mask and no gate: W * M in the forward pass (the mask is set once at build)."""

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.event_mask

    def forward(self, x):
        return torch.nn.functional.linear(x, self.effective_weight(), self.bias)


def _swap(model, cls) -> None:
    """Give every structural layer (the readout included) the forward pass of `cls`; parameters and buffers stay."""
    for layer in model.linear_layers():
        layer.__class__ = cls


@method("lewi-abl-soft")
class LeWiSoftOnly(LeWiV1ReadoutGateMask):
    def build(self, cfg, target):
        model = super().build(cfg, target)
        _swap(model, SoftBallLinear)
        return model


@method("lewi-abl-fixed")
class LeWiFixedEmbeddings(LeWiV1ReadoutGateMask):
    def build(self, cfg, target):
        model = super().build(cfg, target)
        for codes in model.gate_codes:
            codes.requires_grad_(False)
        return model


@method("lewi-abl-random")
class LeWiRandomSparse(LeWiV1ReadoutGateMask):
    def build(self, cfg, target):
        model = super().build(cfg, target)
        aux = int(cfg["auxiliary_init_seed"])
        with torch.no_grad():
            for boundary, layer in enumerate(model.linear_layers()):
                k = int(layer.event_mask.sum())                      # LeWi's open edges on this transition at this seed
                g = torch.Generator().manual_seed(int(derive(aux, "random_mask", boundary)))
                perm = torch.randperm(layer.event_mask.numel(), generator=g)
                mask = torch.zeros(layer.event_mask.numel(), dtype=layer.event_mask.dtype)
                mask[perm[:k]] = 1
                layer.event_mask.copy_(mask.reshape(layer.event_mask.shape))
                layer.gate_scale.zero_()
                layer.gate_offset.fill_(30.0)                        # sigmoid(30) = 1 in float32: no gate
        for codes in model.gate_codes:
            codes.requires_grad_(False)
        _swap(model, GateFreeMaskedLinear)
        model.lewi_random_mask = True
        return model

    @staticmethod
    def _rederive_masks(model) -> None:
        if getattr(model, "lewi_random_mask", False):
            return                                                   # the mask is fixed for the whole run
        LeWiV1ReadoutGateMask._rederive_masks(model)


@method("lewi-abl-ste")
class LeWiStraightThrough(LeWiV1ReadoutGateMask):
    def build(self, cfg, target):
        model = super().build(cfg, target)
        _swap(model, STBallMaskedLinear)
        return model


class _LeWiInitScale(LeWiV1ReadoutGateMask):
    scale = 1.0

    def build(self, cfg, target):
        model = super().build(cfg, target)
        c = float(self.scale)
        with torch.no_grad():
            for codes in model.gate_codes:
                codes.mul_(c)
            for layer in model.linear_layers():
                layer.gate_offset.mul_(c * c)
        self._rederive_masks(model)          # unchanged by construction; re-derived so the buffers are the scaled geometry's own
        return model


@method("lewi-init05")
class LeWiInitHalf(_LeWiInitScale):
    scale = 0.5


@method("lewi-init2")
class LeWiInitDouble(_LeWiInitScale):
    scale = 2.0
