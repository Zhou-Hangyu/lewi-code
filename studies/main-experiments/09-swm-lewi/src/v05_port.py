"""Canonical CSFRBench v0.5 builders, ported to vector-output world models.

The port principle (goal doc, Phase 4): *import the canonical builder, do not
reimplement it*. Every arm here calls the frozen ``Method.build(cfg, target)``
of ``csfrbench.methods`` unchanged. The only adaptation is the output width:
the benchmark's targets are scalar, a world model predicts a state delta.

That adaptation is made by temporarily defaulting ``out_dim`` on the three
backbone classes the builders construct (``ReLUMLP``/``DenseMLP``/
``HardConcreteMLP``) and on ERB's ``SpatialMLP`` layer-size list, then calling
the untouched ``build``. At ``out_dim = 1`` the patch is a no-op, so parity
against the frozen builders is bit-exact by construction and is asserted in
``test_parity.py``.

The registry is shared process-wide; the provenance of every name this study
resolves is asserted below.
"""

from __future__ import annotations

import hashlib
import sys
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

import csfrbench.methods  # noqa: E402,F401  (registers the canonical roster)
import csfrbench.registry as _registry  # noqa: E402

from csfrbench.registry import METHODS as _METHODS  # noqa: E402
from csfrbench.registry import get_method  # noqa: E402
from csfrbench.methods import erb_positions as _erb  # noqa: E402
from csfrbench.methods import nns as _nns  # noqa: E402

# ---------------------------------------------------------------- roster

#: study arm name -> canonical registry key
REGISTRY_KEY = {
    "dense": "noreg",
    "l1": "l1",
    "l2": "l2",
    "l0hc": "l0hc",
    "dnw": "dnw",
    "rigl": "rigl",
    "fiedler": "fiedler",
    "bimt": "bimt",
    "erb": "erb_positions",
    "lewi": "lewi-v05",
    "lewi_reg": "lewi_reg-v05",
}

#: the competitive roster of the goal document (references excluded)
COMPETITIVE = ("dense", "l1", "l2", "l0hc", "dnw", "rigl", "fiedler", "bimt", "erb")
LEWI_ARMS = ("lewi", "lewi_reg")

#: fixed (non-searched) hyperparameters carried from the v0.5 campaign config
FIXED_HP = {"l0hc": {"p_open_init": 0.8318}}

for _arm, _key in REGISTRY_KEY.items():
    assert _METHODS[_key].__module__.startswith("csfrbench."), (_arm, _key)


def method_of(arm: str):
    return get_method(REGISTRY_KEY[arm])


def source_hashes() -> dict:
    """sha256 of every canonical method source file this study imports."""
    out = {}
    for arm, key in REGISTRY_KEY.items():
        module = sys.modules[_METHODS[key].__module__]
        path = Path(module.__file__)
        out[arm] = {"module": module.__name__, "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for extra in (_nns, _registry):
        path = Path(extra.__file__)
        out[extra.__name__] = {"module": extra.__name__, "path": str(path),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return out


# ---------------------------------------------------------------- out_dim shim

_BACKBONES = ("ReLUMLP", "DenseMLP", "HardConcreteMLP")


@contextmanager
def _vector_output(out_dim: int):
    """Default ``out_dim`` on the canonical backbones for the duration."""
    saved = {}
    for name in _BACKBONES:
        cls = getattr(_nns, name)
        saved[name] = cls.__init__

        def patched(self, *args, __orig=cls.__init__, **kwargs):
            kwargs.setdefault("out_dim", out_dim)
            return __orig(self, *args, **kwargs)

        cls.__init__ = patched
    erb_saved = _erb.SpatialMLP.__init__

    def erb_patched(self, layer_sizes, generator, __orig=erb_saved):
        sizes = list(layer_sizes)
        sizes[-1] = out_dim
        return __orig(self, sizes, generator)

    _erb.SpatialMLP.__init__ = erb_patched
    try:
        yield
    finally:
        for name in _BACKBONES:
            getattr(_nns, name).__init__ = saved[name]
        _erb.SpatialMLP.__init__ = erb_saved


class _Target:
    """The one attribute every competitive builder reads off ``target``."""

    def __init__(self, d: int):
        self.d = int(d)


def build_model(arm: str, in_dim: int, out_dim: int, hp: dict, aux_seed: int,
                width: int, hidden: int) -> nn.Module:
    """Canonical ``build`` for ``arm`` with a vector output head."""
    cfg = {"width": int(width), "hidden": int(hidden),
           "auxiliary_init_seed": int(aux_seed), "hp": dict(hp)}
    with _vector_output(int(out_dim)):
        return method_of(arm).build(cfg, _Target(in_dim))


def structural_params(arm: str, model: nn.Module) -> int:
    """Parameters that exist only because of the method (codes, gates,
    positions, inhibitions) - everything beyond the plain MLP envelope."""
    plain = set()
    for layer in getattr(model, "linear_layers", lambda: ())():
        plain.add(id(layer.weight))
        if getattr(layer, "bias", None) is not None:
            plain.add(id(layer.bias))
    return int(sum(p.numel() for p in model.parameters() if id(p) not in plain))


def hp_defaults(arm: str) -> dict:
    return dict(FIXED_HP.get(arm, {}))
