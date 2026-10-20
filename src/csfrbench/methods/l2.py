import torch

from ..registry import method
from .base import Method, readout_structural
from .nns import DenseMLP


@method("l2")
class GlobalL2(Method):
    """Mean-normalized global L2 weight regularization."""

    hp_keys = frozenset({"lam", "readout"})

    def build(self, cfg, target):
        return DenseMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
            readout_structural=readout_structural(cfg.get("hp")),
        )

    def penalty(self, model, hp):
        lam = float(hp["lam"])
        if lam < 0:
            raise ValueError(f"lam must be nonnegative, received {lam}.")

        weights = tuple(model.regularized_weights())
        if not weights:
            raise ValueError("The model exposes no regularized weights.")

        num_weights = sum(weight.numel() for weight in weights)
        l2_sum = torch.stack(
            [weight.square().sum() for weight in weights]
        ).sum()

        return lam * l2_sum / num_weights
