"""`dense_fourier`: one fourier node over all A signal leaves.

The high-arity fourier control -- the independent-family counterpart of
`dense_anova`: remove low-arity decomposability under the fourier law and ask
whether the target stays learnable at benchmark sample sizes. A new k = 8
target-design problem: `Q` and `sigma_omega` are candidate design parameters
calibrated by target-only health criteria in the v1 finalization (Stage A);
the defaults below are starting proposals, not inherited frozen values.
"""
from __future__ import annotations

from collections.abc import Mapping

from ..registry import task
from .graph import BalancedGraphFamily
from .laws import FourierLaw, _acceptance_is_catastrophic


@task("dense_fourier")
class DenseFourier(BalancedGraphFamily):
    """y = fourier node on all A leaves; interaction at order A."""

    cell_keys = ("A", "Q", "sigma_omega", "acceptance")
    cell_defaults = {"A": 8, "Q": 8, "sigma_omega": 0.85}

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        A = int(cell["A"])
        if A < 2:
            raise ValueError(f"dense_fourier requires A >= 2; received {A}")
        return A

    def make_law(self, cfg: Mapping):
        return FourierLaw(float(cfg["sigma_omega"]), Q=int(cfg["Q"]),
                          catastrophic_only=_acceptance_is_catastrophic(cfg))

    def family_options(self, cfg: Mapping, law) -> dict:
        return {**law.describe(), "A": int(cfg["A"])}

    def draw(self, cfg: Mapping, function_seed: int):
        cfg = self.resolve_cell(cfg)
        A = int(cfg["A"])
        return super().draw({**cfg, "k": A, "layer_profile": (1,)}, function_seed)
