"""Two prespecified LeWi variants on the adopted arm (18-main-d8 intake/lewi_mask_only_r_sensitivity_goal.md, decision 2026-09-22).

  ``lewi-mask-in5``   the mask-only forward: the multiplier of every transition (the readout included) is the binary mask m
                      alone - exactly m in the forward pass; the embeddings train through the active-edge surrogate
                      d(multiplier)/ds = m * sigmoid'(s), implemented as m + m * (g - g.detach()), so a closed edge gets no
                      direct score gradient and no weight is rescaled. Everything else is the adopted arm's (lewi-v1-rgm-in5:
                      the Euclidean scores and radius, the embeddings shared across incoming and outgoing edges, the readout's
                      gate and mask, the biases, the initialization and both rejection rules, the mask refresh after every step).

The embedding-dimension arms need no class: they are the adopted arm with code_dim r (its own rule R^2 = 2 r sigma^2 = r/2,
sigma = 0.5; methods/lewi_v1.py), carried by m17.ARM_HP under the keys lewi-dim1 ... lewi-dim8.
"""
from __future__ import annotations

import torch
from csfrbench.registry import method
from csfrbench.methods.lewi_v1 import BallMaskedLinear

import abl_methods
import abl_in5_methods


class MaskOnlyBallLinear(BallMaskedLinear):
    """The ball layer with the mask alone in the forward pass and the active-edge surrogate in the backward pass."""

    def effective_weight(self) -> torch.Tensor:
        m = self.event_mask.to(self.weight.dtype)
        g = self.gate()
        return self.weight * (m + m * (g - g.detach()))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.effective_weight(), self.bias)


@method("lewi-mask-in5")
class LeWiMaskOnlyIn5(abl_in5_methods._In5):
    def build(self, cfg, target):
        model = super().build(cfg, target)
        abl_methods._swap(model, MaskOnlyBallLinear)
        return model


ARMS = ("lewi-mask-in5",)
