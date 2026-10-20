"""Dense: the backbone without regularization."""
from __future__ import annotations

from ..models import DenseMLP
from .base import Method, register


@register("dense")
class Dense(Method):
    def build(self, cfg, target):
        return DenseMLP(target.d, cfg["width"], cfg["hidden"])
