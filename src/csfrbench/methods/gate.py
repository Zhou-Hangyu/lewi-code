import math

import torch

from ..registry import method
from .base import Method
from .nns import HardConcreteMLP


@method("gate")
class GatedControl(Method):
    """Hard-concrete gates as a learning algorithm, with the L0 penalty off by default.

    Setting an L0 method's lambda to zero does not recover the dense baseline: the gates
    remain, the logits are still trained by the task loss, and fresh gates are still
    sampled each step. This arm exposes the two mechanisms as independent switches so
    the effect can be attributed:

        learn_gates   the logits receive task-loss gradient (reparameterization plus
                      task-driven edge selection)
        stochastic    a fresh gate is sampled per minibatch (multiplicative noise)

    The four corners are a factorial control for `l0hc`; ``learn_gates`` and
    ``stochastic`` both true with ``lam`` zero reproduces `l0hc` at lambda = 0 exactly.
    """

    hp_keys = frozenset(
        {"lam", "p_open_init", "stochastic", "learn_gates"})

    def build(self, cfg, target):
        hp = cfg.get("hp") or {}
        model = HardConcreteMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
            p_open_init=float(hp.get("p_open_init", 0.8318)),
            gate_init_seed=int(cfg["auxiliary_init_seed"]),
            stochastic=bool(hp.get("stochastic", True)),
        )
        if not bool(hp.get("learn_gates", True)):
            for layer in model.structural_layers():
                layer.log_alpha.requires_grad_(False)
        return model

    def penalty(self, model, hp):
        lam = float(hp.get("lam", 0.0))
        if not math.isfinite(lam) or lam < 0:
            raise ValueError(f"lam must be finite and nonnegative; received {lam}.")
        if lam == 0.0:
            return None

        num_weights = sum(weight.numel() for weight in model.regularized_weights())
        return lam * model.expected_l0() / num_weights

    def after_optimizer_step(self, model, optimizer, *, step, total_steps):
        model.clamp_log_alpha_()

    def effective_weights(self, model):
        """Gate-multiplied weights; the raw matrices never act alone at evaluation."""
        return tuple(model.effective_weights())

    def extra_diagnostics(self, model):
        return gate_diagnostics(model)


def gate_diagnostics(model) -> dict[str, float]:
    """Gate state, and the effective weight scale the gate actually produces.

    The mean gate alone cannot distinguish shrinkage from reparameterization: a network
    is free to grow its raw weights to offset a small gate. ``eff_weight_absmean`` is
    the quantity that settles it.
    """
    with torch.no_grad():
        layers = model.structural_layers()
        num_weights = sum(layer.weight.numel() for layer in layers)
        gates = [layer.deterministic_gate() for layer in layers]

        raw_abs = sum(layer.weight.abs().sum() for layer in layers) / num_weights
        eff_abs = sum(
            (layer.weight * gate).abs().sum() for layer, gate in zip(layers, gates)
        ) / num_weights
        gate_sum = sum(gate.sum() for gate in gates) / num_weights
        nonzero = sum((gate > 0).sum() for gate in gates) / num_weights
        expected_open = sum(
            layer.expected_open_probability().sum() for layer in layers
        ) / num_weights

    return {
        "raw_weight_absmean": float(raw_abs),
        "eff_weight_absmean": float(eff_abs),
        "mean_deterministic_gate": float(gate_sum),
        "deterministic_nonzero_frac": float(nonzero),
        "expected_open_frac": float(expected_open),
        "gate_compensation": float(raw_abs / max(float(eff_abs), 1e-12)),
    }
