"""Input L0: one hard-concrete gate per input coordinate before a Dense MLP.

The input-selection control of the final rerun (CSFRBench v1.0; paper
Figure 5a): a length-``d`` vector of hard-concrete gates multiplies the raw
input coordinates before an otherwise ordinary ``DenseMLP``, fully dense
above the input and with the plain readout every arm shares. The gate math
is the package's edge-wise L0 (Louizos, Welling & Kingma 2018) verbatim:
beta 2/3, stretch (-0.1, 1.1), a stochastic gate per minibatch in training,
the deterministic estimator at evaluation, ``p_open_init`` 0.8318 with the
1e-2 initialization noise from the auxiliary seed, the log-alpha clamp after
every step. Penalty: ``lam`` times the expected fraction of open input
gates (mean-normalized, the package's L0 convention). Searched: ``lr`` and
``lam``. The backbone is built first, so it consumes the global
initialization stream exactly as the Dense arm does. Ported unchanged from
the 12-extra harness (``12-extra/src/x12_methods.py``, wave 1). Method card:
``versions/v1.0/method_cards/input_l0.md``.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from csfrbench.methods.base import Method
from csfrbench.methods.nns import DenseMLP, HardConcreteLinear
from csfrbench.registry import method


class InputHardConcreteGate(nn.Module):
    """One hard-concrete gate per input coordinate (the package's edge-wise
    gate math, HardConcreteLinear, applied to a vector of d gates)."""

    def __init__(self, d: int, *, p_open_init: float, generator: torch.Generator,
                 gate_init_std: float = 1e-2):
        super().__init__()
        self.beta = HardConcreteLinear.DEFAULT_BETA
        self.gamma = HardConcreteLinear.DEFAULT_GAMMA
        self.zeta = HardConcreteLinear.DEFAULT_ZETA
        if not 0.0 < float(p_open_init) < 1.0:
            raise ValueError("p_open_init must lie strictly between 0 and 1.")
        self._open_logit_shift = self.beta * math.log(-self.gamma / self.zeta)
        p = float(p_open_init)
        initial = math.log(p) - math.log1p(-p) + self._open_logit_shift
        noise = torch.randn(int(d), generator=generator, dtype=torch.float32, device="cpu")
        self.log_alpha = nn.Parameter(torch.full((int(d),), initial) + float(gate_init_std) * noise)

    def sampled_gate(self) -> torch.Tensor:
        uniform = torch.empty_like(self.log_alpha).uniform_(
            HardConcreteLinear.SAMPLE_EPS, 1.0 - HardConcreteLinear.SAMPLE_EPS)
        logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
        concrete = torch.sigmoid((logistic_noise + self.log_alpha) / self.beta)
        return (concrete * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    def deterministic_gate(self) -> torch.Tensor:
        return (torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    def expected_open_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha - self._open_logit_shift)

    def expected_l0(self) -> torch.Tensor:
        return self.expected_open_probability().sum()

    @torch.no_grad()
    def clamp_log_alpha_(self) -> None:
        self.log_alpha.clamp_(min=HardConcreteLinear.LOG_ALPHA_MIN, max=HardConcreteLinear.LOG_ALPHA_MAX)

    def forward(self, x):
        gate = self.sampled_gate() if self.training else self.deterministic_gate()
        return x * gate


class InputGatedDenseMLP(DenseMLP):
    """The benchmark's DenseMLP with a hard-concrete gate on every input
    coordinate. The backbone is built first, so it consumes the global
    initialization stream exactly as the Dense arm does; the gate draws its
    initialization noise from the auxiliary generator."""

    def __init__(self, input_dim, width, num_hidden_layers, *, p_open_init, gate_init_seed):
        super().__init__(input_dim=input_dim, width=width, num_hidden_layers=num_hidden_layers)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(gate_init_seed))
        self.input_gate = InputHardConcreteGate(input_dim, p_open_init=p_open_init, generator=generator)

    def forward(self, x):
        return super().forward(self.input_gate(x))

    def effective_weights(self) -> tuple[torch.Tensor, ...]:
        layers = self.linear_layers()
        first = layers[0].weight * self.input_gate.deterministic_gate().unsqueeze(0)
        return (first, *(layer.effective_weight() for layer in layers[1:]))


@method("input_l0")
class InputL0(Method):
    """Input-wise hard-concrete gate before an ordinary Dense MLP."""

    hp_keys = frozenset({"lam", "p_open_init"})

    def build(self, cfg, target):
        hp = cfg.get("hp") or {}
        return InputGatedDenseMLP(
            input_dim=int(target.d),
            width=int(cfg["width"]),
            num_hidden_layers=int(cfg["hidden"]),
            p_open_init=float(hp.get("p_open_init", 0.8318)),
            gate_init_seed=int(cfg["auxiliary_init_seed"]),
        )

    def penalty(self, model, hp):
        lam = float(hp["lam"])
        if not math.isfinite(lam) or lam < 0:
            raise ValueError(f"lam must be finite and nonnegative; received {lam}.")
        return lam * model.input_gate.expected_l0() / model.input_gate.log_alpha.numel()

    def after_optimizer_step(self, model, optimizer, *, step, total_steps):
        model.input_gate.clamp_log_alpha_()

    def effective_weights(self, model):
        return tuple(model.effective_weights())

    def extra_diagnostics(self, model):
        with torch.no_grad():
            gate = model.input_gate.deterministic_gate()
            expected = model.input_gate.expected_open_probability()
            open_bits = sum(2 ** k for k, g in enumerate(gate.tolist()) if g > 0)
        return {"input_gate_open_count": float((gate > 0).sum()),
                "input_gate_full_count": float((gate >= 1).sum()),
                "input_gate_mean": float(gate.mean()),
                "input_gate_expected_open_frac": float(expected.mean()),
                "input_gate_open_bits": float(open_bits)}   # bit k set = coordinate k open (exact below 2^53)
