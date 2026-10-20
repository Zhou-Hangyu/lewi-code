"""The interface of a method.

A method builds its model on the common backbone and may add a penalty to the
loss and an operation after each optimizer step. A method object holds no state
of a run: every quantity that a later evaluation needs is a parameter or a
persistent buffer of the model.

To add a method, subclass :class:`Method` and register it::

    from csfrbench.methods import Method, register
    from csfrbench.models import DenseMLP

    @register("my_method")
    class MyMethod(Method):
        hp_keys = frozenset({"strength"})

        def build(self, cfg, target):
            return DenseMLP(target.d, cfg["width"], cfg["hidden"])

        def penalty(self, model, hp):
            return hp["strength"] * sum(w.abs().mean() for w in model.effective_weights()[:-1])

``cfg`` holds ``width``, ``hidden``, ``hp`` (the hyperparameters of the run, with
the learning rate ``lr``) and ``auxiliary_seed`` (the seed of every random
quantity of the method that is not a backbone weight). A method must not read
the target's graph or its active inputs. Only the graph-informed references do.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn
from torch.optim import Optimizer

METHODS: dict[str, type] = {}


def register(name: str):
    """Class decorator: register a method under ``name``."""
    def decorator(cls):
        existing = METHODS.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"method {name!r} is already registered by {existing.__qualname__}")
        METHODS[name] = cls
        cls.name = name
        return cls
    return decorator


def get_method(name: str) -> Method:
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; registered: {sorted(METHODS)}")
    return METHODS[name]()


class Method(ABC):
    #: hyperparameter keys that the method reads, without the learning rate ``lr``
    hp_keys: frozenset[str] = frozenset()
    #: True when the method reads the target's graph or active inputs (a reference, not a learner)
    graph_informed: bool = False

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a method object holds no state (attempted to set {name!r}); "
                             "store run state in the model")

    @classmethod
    def check_hp(cls, hp: dict) -> None:
        unknown = set(hp) - set(cls.hp_keys) - {"lr"}
        if unknown:
            raise ValueError(f"method {getattr(cls, 'name', cls.__name__)!r} does not read the hyperparameters "
                             f"{sorted(unknown)}; it reads {sorted(set(cls.hp_keys) | {'lr'})}")

    @abstractmethod
    def build(self, cfg: dict, target) -> nn.Module:
        """Build the model. The global random state is seeded before the call."""

    def penalty(self, model: nn.Module, hp: dict) -> Tensor | None:
        """The term that is added to the mean squared error, or None."""
        return None

    def after_step(self, model: nn.Module, optimizer: Optimizer, *, step: int, total_steps: int) -> None:
        """An operation after the optimizer step ``step`` (1-based)."""
        return None

    def effective_weights(self, model: nn.Module) -> tuple[Tensor, ...]:
        """The weight matrices that act at evaluation, in forward order."""
        return tuple(model.effective_weights())

    def diagnostics(self, model: nn.Module) -> dict[str, float]:
        """Scalar diagnostics of the selected checkpoint."""
        return {}
