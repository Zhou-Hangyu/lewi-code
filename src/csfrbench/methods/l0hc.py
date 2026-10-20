"""Hard-concrete L0 (Louizos et al.) on the structural layers: the hidden
transitions, plus the readout under the v1.1 fixed argument
``readout: structural`` (a plain dense readout under the v1.0 rule).

Method card:
studies/csfrbench/final-design/versions/v1.0/method_cards/l0hc.md
"""
import math
import torch

from ..registry import method
from .base import Method, readout_structural
from .nns import HardConcreteMLP


@method("l0hc")
class HardConcreteL0(Method):
    """Mean-normalized edge-wise hard-concrete L0 regularization.

    Every scalar affine weight has an independent hard-concrete gate. The
    additive penalty is lambda times the expected fraction of active weight
    gates.

    This follows the paper's general parameter-wise hard-concrete formulation.
    It differs from the authors' released MLP experiments, which use one gate
    per input unit and therefore induce group/neuron sparsity.

    Biases are not gated or penalized, matching the benchmark's L1 and SoFi
    weight-only regularizers.
    """

    hp_keys = frozenset({"lam", "p_open_init", "readout"})

    def build(self, cfg, target):
        return HardConcreteMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
            p_open_init=float(cfg["hp"].get("p_open_init", 0.8318)),
            gate_init_seed=int(cfg["auxiliary_init_seed"]),
            readout_structural=readout_structural(cfg.get("hp")),
        )

    def penalty(self, model, hp):
        lam = float(hp["lam"])
        if not math.isfinite(lam) or lam < 0:
            raise ValueError(
                f"lam must be finite and nonnegative; received {lam}."
            )

        num_weights = sum(
            weight.numel()
            for weight in model.regularized_weights()
        )
        if num_weights == 0:
            raise ValueError("The model contains no regularized weights.")

        # Mean-normalized expected active-weight fraction. This is a benchmark
        # scaling convention; lambda must still be tuned independently.
        return lam * model.expected_l0() / num_weights

    def after_optimizer_step(
        self,
        model,
        optimizer,
        *,
        step,
        total_steps,
    ):
        # Match the released implementation's post-update log-alpha bounds.
        model.clamp_log_alpha_()

    def effective_weights(self, model):
        """Return gate-multiplied weights.

        The raw matrices never act alone at evaluation, so the base
        implementation's regularized_weights() would report a fully gated-off
        network as dense.
        """
        return tuple(model.effective_weights())

    def extra_diagnostics(self, model):
        from .gate import gate_diagnostics

        with torch.no_grad():
            saturated = sum(
                (layer.deterministic_gate() >= 1).sum() for layer in model.structural_layers()
            ) / sum(layer.weight.numel() for layer in model.structural_layers())
        return {**gate_diagnostics(model), "deterministic_one_frac": float(saturated)}