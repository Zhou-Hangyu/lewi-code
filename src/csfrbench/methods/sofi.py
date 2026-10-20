import math

import torch

from ..registry import method
from .base import Method
from .nns import DenseMLP



@method("sofi")
class SoFi(Method):
    """Sorted Fan-In: a nonconvex per-neuron sorted-tail penalty.

    For each neuron, the absolute incoming weights are sorted in descending
    order,

        |w|_(1) >= |w|_(2) >= ... >= |w|_(F),

    and rank r receives coefficient

        alpha_r = r**c / mean_{s=1,...,F}(s**c).

    For c > 0, small tail weights receive progressively larger coefficients,
    encouraging each neuron to retain only a few substantial incoming edges.
    This is deliberately a local fan-in prior rather than a global sparsity
    count.

    The coefficients have unit mean within every layer, so their sum equals
    the layer fan-in F for every c. This controls the gross penalty scale across
    c, but lambda must still be tuned separately.

    At c = 0, every alpha_r equals 1, so this penalty is exactly the
    mean-normalized global L1 penalty, provided both methods regularize the
    same weight tensors.
    """

    hp_keys = frozenset({"lam", "c"})

    def build(self, cfg, target):
        return DenseMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
        )

    def penalty(self, model, hp):
        lam = float(hp["lam"])
        c = float(hp["c"])

        if not math.isfinite(lam) or lam < 0:
            raise ValueError(f"lam must be finite and nonnegative; received {lam}.")
        if not math.isfinite(c) or c < 0:
            raise ValueError(f"c must be finite and nonnegative; received {c}.")

        weights = tuple(model.regularized_weights())
        if not weights:
            raise ValueError("The model exposes no regularized weights.")

        terms = []
        num_weights = 0

        for weight in weights:
            if weight.ndim != 2:
                raise ValueError(
                    "SoFi expects two-dimensional affine weight matrices; "
                    f"received shape {tuple(weight.shape)}."
                )

            # Rows correspond to receiving neurons; sort each neuron's incoming
            # edge magnitudes independently.
            sorted_magnitudes = weight.abs().sort(
                dim=1,
                descending=True,
            ).values

            fan_in = weight.shape[1]
            ranks = torch.arange(
                1,
                fan_in + 1,
                device=weight.device,
                dtype=weight.dtype,
            )

            rank_coefficients = ranks.pow(c)
            rank_coefficients = (
                rank_coefficients / rank_coefficients.mean()
            )

            terms.append(
                (sorted_magnitudes * rank_coefficients.unsqueeze(0)).sum()
            )
            num_weights += weight.numel()

        return lam * torch.stack(terms).sum() / num_weights