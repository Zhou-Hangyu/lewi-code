from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from torch import Tensor, nn
from torch.optim import Optimizer


READOUT_FORMS = ("plain", "structural")


def readout_structural(hp: Mapping | None) -> bool:
    """CSFRBench v1.1: the fixed method argument ``readout`` selects whether
    the readout transition is *structural* - penalized, gated, masked or
    pruned like every hidden transition - or *plain* (the v1.0 rule: an
    ordinary linear map no mechanism touches). Absent means plain, so every
    specification written before v1.1 rebuilds unchanged."""
    form = str((hp or {}).get("readout", "plain"))
    if form not in READOUT_FORMS:
        raise ValueError(
            f"readout must be one of {READOUT_FORMS}; received {form!r}")
    return form == "structural"


class Method(ABC):
    """Stateless definition of one benchmark method.

    Method instances must not hold mutable run-specific state. Any state needed
    to reproduce model predictions, effective weights, or diagnostics after
    checkpoint restoration must be stored in the model as an ``nn.Parameter``
    or a persistent registered buffer. Training-only optimizer state may be
    stored in the optimizer.

    Hyperparameter search spaces belong to campaign configuration, not to this
    interface.
    """

    weight_graph: bool = True

    # Hyperparameter keys this method consumes (search dimensions and fixed
    # arguments), excluding the envelope's ``lr``. When declared (not None),
    # spec validation rejects any unknown hp key, so a typo'd pin fails
    # loudly instead of silently running at a code default. ``None`` means
    # undeclared: legacy methods keep the permissive behavior.
    hp_keys: frozenset[str] | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "Method instances must remain stateless. "
            f"Attempted to assign {name!r}. Store persistent state in the "
            "model as an nn.Parameter or persistent registered buffer; store "
            "training-only optimizer state in optimizer.state."
        )

    @abstractmethod
    def build(self, cfg: dict, target) -> nn.Module:
        """Construct and return the model used by this method."""
        raise NotImplementedError

    def penalty(
        self,
        model: nn.Module,
        hp: dict,
    ) -> Tensor | None:
        """Return the scalar additive penalty, or ``None`` for no penalty."""
        return None

    def after_optimizer_step(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        *,
        step: int,
        total_steps: int,
    ) -> None:
        """Apply method-specific post-update operations.

        This hook may mutate model parameters, model buffers, or optimizer
        state. It must not place checkpoint-relevant state on the ``Method``
        instance or in unregistered model attributes.

        State required for later evaluation or diagnostics must appear in
        ``model.state_dict()``. For example, a temperature, threshold, running
        statistic, or deterministic gate value should be registered on the
        model using ``register_buffer(..., persistent=True)``.

        The default implementation performs no operation.
        """
        return None

    def effective_weights(self, model):
        """Return deterministic effective weight matrices in forward order.

        Override this for gated or reparameterized methods. The default reads
        the model's own effective_weights() (every affine layer, the plain
        readout included) and falls back to regularized_weights().
        """
        if hasattr(model, "effective_weights"):
            return tuple(model.effective_weights())
        if not hasattr(model, "regularized_weights"):
            raise NotImplementedError(
                f"{type(self).__name__} must override effective_weights() or its "
                "model must implement effective_weights() or regularized_weights()."
            )
        return tuple(model.regularized_weights())

    def extra_diagnostics(self, model: nn.Module) -> dict[str, float]:
        """Return optional scalar diagnostics for the selected checkpoint."""
        return {}