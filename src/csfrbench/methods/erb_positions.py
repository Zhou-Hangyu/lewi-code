"""Spatially embedded MLP - Erb et al., Training Neural Networks by
Optimizing Neuron Positions (arXiv:2506.13410, Living Machines 2025).

Source: the paper only - no official implementation is public (verified
2026-08-28), so every constant the paper leaves unspecified is an
explicit documented choice in the method card:
studies/csfrbench/final-design/versions/v1.0/method_cards/erb_positions.md

Paper mechanics:
  * neurons live in 3-D Euclidean space; the z-coordinate is fixed to the
    neuron's layer index (standard, non-relaxed setup) and the two
    in-layer coordinates are free, learned parameters;
  * synaptic weights are not parameters: w_ij = 1 / ||p_i - p_j||_2
    between consecutive layers (inverse Euclidean distance, the paper's
    conductance analogy, no decay constant, no fan-in normalization);
  * signs come from the excitatory/inhibitory parameterization: each
    neuron that sends output onward - input and hidden neurons, per the
    paper's Eq. 4 parameter count sum_{i=1}^{N_L - 1} n_i, which uniquely
    reproduces the paper's reported totals (10,574 / 10,576 / 13,416) -
    carries a continuous inhibition value v, mapped by a scaled steep
    sigmoid to a mask close to +-1 that multiplies the neuron's output.
    The output layer carries no mask (2026-08-28 repair; the earlier
    revision masked hidden + output neurons instead, audited as finding
    E1 in studies/csfrbench/baseline-audit/v05-suite);
  * learned parameters: positions (2 per neuron), biases (non-input
    neurons) and inhibition values (non-output neurons) - the count is
    linear in the neuron count (for example 4,195 against the dense
    backbone's 119,937 at the d=32 128x8 pilot shape; the per-workload
    profile sets the operative shape), which is disclosed, never hidden.

Readout rule. CSFRBench v1.0 left the readout plain: a linear layer with
free weights and bias, positions, biases and inhibition values covering the
input and hidden neurons only, the readout unit without a position (disclosed
in the method card as a deviation from the paper). CSFRBench v1.1 restores
the paper's all-layer parameterization under the fixed argument
``readout: structural``: the readout unit is positioned, the readout weights
are inverse distances and the readout carries its bias (the v0.5 form, bit
for bit). The auxiliary stream is consumed in the same order in both forms,
so the hidden geometry and the inhibition values are identical.

Documented choices where the paper is silent (method card, section
"unspecified constants"): mask = 2*sigmoid(10 v) - 1; distances clamped
below at 1e-4 (vacuous under the standard unit layer spacing, kept for
the relaxed-z variant's sake); free coordinates initialized uniformly in
[0, 8)^2 from the benchmark auxiliary seed stream; inhibition values
initialized standard normal (so both signs are present at start); biases
start at zero; the activation is the envelope's ReLU. Depth-8
composition of unnormalized inverse-distance weights is outside the
source's tested regime (single-hidden-layer networks); no normalization
is added because the source defines none.
"""
from __future__ import annotations

import torch
from torch import nn

from ..registry import method
from .base import Method, readout_structural
from .nns import DenseLinear

MASK_STEEPNESS = 10.0
DIST_EPS = 1e-4
# Free-coordinate initialization scale (documented choice; the paper gives
# none). At small scales every inverse-distance matrix on a 128-wide layer
# is nearly rank-1 (all distances ~ the unit layer spacing) and the depth-8
# forward collapses to a constant with dead ReLUs everywhere; 8.0 was the
# smallest screened scale at which gradients reach the positions at
# initialization. Recorded in the method card; no further tuning.
POS_SPREAD = 8.0


class SpatialMLP(nn.Module):
    """MLP whose weights are inverse distances between positioned neurons.

    ``readout_structural`` (CSFRBench v1.1): the readout unit is positioned
    and the readout weights are inverse distances with their own bias (the
    paper's all-layer form, the v0.5 record). Without it (v1.0): the readout
    is a plain linear layer and the readout unit carries no position.

    Inhibition masks follow the paper's Eq. 4 placement: input and hidden
    neurons carry one +-1-ish factor each, multiplying the neuron's
    output; the output layer carries none.
    """

    def __init__(self, layer_sizes: list[int], generator: torch.Generator,
                 readout_structural: bool = False):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        self.out_dim = layer_sizes[-1]
        self.readout_structural = bool(readout_structural)
        positioned = layer_sizes if self.readout_structural else layer_sizes[:-1]
        self.positions = nn.ParameterList(
            nn.Parameter(torch.rand(n, 2, generator=generator) * POS_SPREAD)
            for n in positioned
        )
        if not self.readout_structural:
            # The source drew a position for the readout unit too; the draw
            # is consumed so the auxiliary stream, and with it the hidden
            # geometry and the inhibition values, equal the structural form
            # and the sealed 08-main initialization.
            torch.rand(layer_sizes[-1], 2, generator=generator)
        # Every non-input layer carries a bias per neuron; under the plain
        # readout the readout's bias belongs to its DenseLinear.
        biased = layer_sizes[1:] if self.readout_structural else layer_sizes[1:-1]
        self.biases = nn.ParameterList(
            nn.Parameter(torch.zeros(n)) for n in biased
        )
        # Non-output layers carry an inhibition value per neuron
        # (paper Eq. 4: sum_{i=1}^{N_L - 1} n_i inhibition parameters).
        self.inhibitions = nn.ParameterList(
            nn.Parameter(torch.randn(n, generator=generator))
            for n in layer_sizes[:-1]
        )
        if not self.readout_structural:
            self.output_layer = DenseLinear(layer_sizes[-2], layer_sizes[-1],
                                            is_output=True)

    def _weight(self, layer: int) -> torch.Tensor:
        """w_ij = 1 / ||p_i - p_j||_2 with unit z-spacing between layers."""
        p_in = self.positions[layer]
        p_out = self.positions[layer + 1]
        diff = p_out.unsqueeze(1) - p_in.unsqueeze(0)      # (out, in, 2)
        dist = (diff.pow(2).sum(-1) + 1.0).sqrt()          # dz = 1
        return 1.0 / dist.clamp_min(DIST_EPS)

    def _mask(self, layer: int) -> torch.Tensor:
        """Mask of neuron layer ``layer`` (0 = input, 1.. = hidden):
        scaled steep sigmoid, close to +1 or -1 by the sign of v."""
        v = self.inhibitions[layer]
        return 2.0 * torch.sigmoid(MASK_STEEPNESS * v) - 1.0

    def forward(self, x):
        h = x * self._mask(0)
        last = len(self.layer_sizes) - 2
        for layer in range(last):
            pre = torch.nn.functional.linear(
                h, self._weight(layer), self.biases[layer])
            h = torch.relu(pre) * self._mask(layer + 1)
        if self.readout_structural:
            out = torch.nn.functional.linear(
                h, self._weight(last), self.biases[last])
        else:
            out = self.output_layer(h)
        return out.squeeze(-1) if self.out_dim == 1 else out

    def n_positioned_transitions(self) -> int:
        """Transitions whose weights are inverse distances."""
        return len(self.layer_sizes) - (1 if self.readout_structural else 2)

    def regularized_weights(self):
        return tuple(self._weight(i)
                     for i in range(self.n_positioned_transitions()))


@method("erb_positions")
class ErbPositions(Method):
    """Erb et al. 2025: weights from learned neuron positions."""

    hp_keys = frozenset({"readout"})

    def build(self, cfg, target):
        sizes = [target.d] + [int(cfg["width"])] * int(cfg["hidden"]) + [1]
        generator = torch.Generator().manual_seed(int(cfg["auxiliary_init_seed"]))
        return SpatialMLP(sizes, generator,
                          readout_structural=readout_structural(cfg.get("hp")))

    def effective_weights(self, model):
        """Signed effective matrices: inverse distances times the source
        layer's inhibition mask (columns), input layer included; under the
        plain readout (v1.0) the readout weight times the last hidden
        layer's mask, under the structural readout (v1.1) the readout's
        inverse distances times that mask."""
        mats = []
        for i in range(model.n_positioned_transitions()):
            w = model._weight(i) * model._mask(i).unsqueeze(0)
            mats.append(w)
        if not model.readout_structural:
            last = len(model.layer_sizes) - 2
            mats.append(model.output_layer.weight * model._mask(last).unsqueeze(0))
        return tuple(m.detach() for m in mats)

    def extra_diagnostics(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        masks = torch.cat([model._mask(i).detach()
                           for i in range(len(model.layer_sizes) - 1)])
        return {"n_parameters": float(n_params),
                "inhibitory_fraction": float((masks < 0).float().mean())}
