"""Slot D -- `tree_relu`: balanced tree of signed ReLU ridges (§4.3, diagnostic).
Exactly realizable by a sparse ReLU network, so the oracle-graph learner has zero
approximation error and the family isolates pure discovery cost."""
from __future__ import annotations

from collections.abc import Mapping

from ..registry import task
from .graph import BalancedGraphFamily
from .laws import ReluLaw


@task("tree_relu")
class TreeRelu(BalancedGraphFamily):
    """Can dense ReLU discover a sparse ReLU teacher realizable in its own class?"""

    cell_keys = ("k", "layer_profile")
    cell_defaults = {"k": 2, "layer_profile": (4, 2, 1)}

    def make_law(self, cfg: Mapping):
        return ReluLaw()
