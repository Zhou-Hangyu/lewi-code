"""Slot F -- `tree_tri`: the exact zero-guidance stress pole (§4.4).
Triangle-wave tree on U[-1, 1] inputs: every node value is exactly U[-1, 1] and
Y is fully independent of every strict subset of the active coordinates.
Frozen canonical form: f = 1, with the depth-2 <-> depth-3 pair as the stress
experiment (depth 2 = structure-search failure; depth 3 = SGD composition
wall). Frequencies f >= 2 add local difficulty as a second bottleneck and are
appendix stress only (§10)."""
from __future__ import annotations

from collections.abc import Mapping

from ..registry import task
from .graph import BalancedGraphFamily
from .laws import DEFAULT_FREQUENCIES, TriLaw, validate_frequencies


@task("tree_tri")
class TreeTri(BalancedGraphFamily):
    """Zero-guidance pole, quarantined as stress: expected failure of I/O learners."""

    cell_keys = ("k", "layer_profile", "frequencies", "random_signs", "random_phases")
    cell_defaults = {
        "k": 2, "layer_profile": (4, 2, 1), "frequencies": DEFAULT_FREQUENCIES,
        "random_signs": True, "random_phases": True,
    }
    allowed_inputs = ("uniform_pm1",)

    @classmethod
    def check_cell(cls, cell: Mapping) -> int:
        validate_frequencies(cell.get("frequencies", DEFAULT_FREQUENCIES))
        return super().check_cell(cell)

    def make_law(self, cfg: Mapping):
        return TriLaw(
            frequencies=cfg.get("frequencies", DEFAULT_FREQUENCIES),
            random_signs=cfg.get("random_signs", True),
            random_phases=cfg.get("random_phases", True),
        )

    def family_options(self, cfg: Mapping, law) -> dict:
        return {
            "frequencies": list(law.frequencies),
            "random_signs": law.random_signs,
            "random_phases": law.random_phases,
        }
