"""L0 regularization with hard-concrete gates (Louizos, Welling and Kingma, 2017).

Every weight of a hidden transition has one hard-concrete gate. In training the
gate is sampled once per minibatch; at evaluation the deterministic gate acts.
The penalty is ``lam`` times the expected fraction of open gates. The gates use
the parameter-wise form of the paper (the authors' released code gates input
units). Biases and the readout have no gate and no penalty.

Fixed: ``p_open_init = 0.8318``, the initial probability that a gate is open.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import DenseLinear, ReLUMLP
from .base import Method, register

P_OPEN_INIT = 0.8318


class HardConcreteLinear(DenseLinear):
    BETA = 2.0 / 3.0
    GAMMA = -0.1
    ZETA = 1.1
    SAMPLE_EPS = 1e-6
    LOG_ALPHA_MIN = math.log(1e-2)
    LOG_ALPHA_MAX = math.log(1e2)
    GATE_INIT_STD = 1e-2

    def __init__(self, in_features: int, out_features: int, *, is_output: bool, p_open_init: float,
                 gate_generator: torch.Generator):
        if not 0.0 < float(p_open_init) < 1.0:
            raise ValueError("p_open_init must lie strictly between 0 and 1")
        super().__init__(in_features, out_features, is_output=is_output)      # the weight: global random state
        self._open_logit_shift = self.BETA * math.log(-self.GAMMA / self.ZETA)
        p = float(p_open_init)
        initial = math.log(p) - math.log1p(-p) + self._open_logit_shift
        noise = torch.randn(self.weight.shape, generator=gate_generator, dtype=self.weight.dtype, device="cpu")
        self.log_alpha = nn.Parameter(torch.full_like(self.weight, initial) + self.GATE_INIT_STD * noise)

    def sampled_gate(self) -> torch.Tensor:
        uniform = torch.empty_like(self.log_alpha).uniform_(self.SAMPLE_EPS, 1.0 - self.SAMPLE_EPS)
        logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
        concrete = torch.sigmoid((logistic_noise + self.log_alpha) / self.BETA)
        return (concrete * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)

    def deterministic_gate(self) -> torch.Tensor:
        stretched = torch.sigmoid(self.log_alpha) * (self.ZETA - self.GAMMA) + self.GAMMA
        return stretched.clamp(0.0, 1.0)

    def open_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha - self._open_logit_shift)

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.deterministic_gate()

    def forward(self, x):
        gate = self.sampled_gate() if self.training else self.deterministic_gate()
        return F.linear(x, self.weight * gate, self.bias)


@register("l0")
class L0(Method):
    hp_keys = frozenset({"lam", "p_open_init"})

    def build(self, cfg, target):
        p_open = float(cfg["hp"].get("p_open_init", P_OPEN_INIT))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(cfg["auxiliary_seed"]))

        def layer(i, o, is_output):
            if is_output:
                return DenseLinear(i, o, is_output=True)
            return HardConcreteLinear(i, o, is_output=False, p_open_init=p_open, gate_generator=generator)

        return ReLUMLP(target.d, cfg["width"], cfg["hidden"], layer_factory=layer)

    def penalty(self, model, hp):
        lam = float(hp["lam"])
        if not math.isfinite(lam) or lam < 0:
            raise ValueError(f"lam must be finite and non-negative; received {lam}")
        count = sum(layer.weight.numel() for layer in model.hidden_layers)
        expected_open = torch.stack([layer.open_probability().sum() for layer in model.hidden_layers]).sum()
        return lam * expected_open / count

    def after_step(self, model, optimizer, *, step, total_steps):
        with torch.no_grad():
            for layer in model.hidden_layers:
                layer.log_alpha.clamp_(min=layer.LOG_ALPHA_MIN, max=layer.LOG_ALPHA_MAX)

    def diagnostics(self, model):
        with torch.no_grad():
            layers = list(model.hidden_layers)
            count = sum(layer.weight.numel() for layer in layers)
            gates = [layer.deterministic_gate() for layer in layers]
            return {
                "open_fraction": float(sum((g > 0).sum() for g in gates) / count),
                "saturated_fraction": float(sum((g >= 1).sum() for g in gates) / count),
                "expected_open_fraction": float(sum(layer.open_probability().sum() for layer in layers) / count),
            }
