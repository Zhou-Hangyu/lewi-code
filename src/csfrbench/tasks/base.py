"""Shared target machinery for configurable layered compositional tasks."""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..utils.seeds import derive

DEFAULT_CALIBRATION = {
    "mc_pool": 1 << 18,
    "activation_band": (0.35, 0.65),
    "max_redraws": 40,
    "min_node_sd": 1e-3,
}
DEFAULT_WIRING = {"mode": "balanced_partition"}

INPUT_DISTRIBUTIONS = ("normal", "uniform", "uniform_pm1")
DEFAULT_INPUT_DISTRIBUTION = "normal"
_UNIFORM_BOUND = float(np.sqrt(3.0))    # variance 1, matching N(0, 1)


def standardize_input_distribution(distribution: str | None) -> str:
    """Normalize and check an input-distribution name."""
    distribution = str(distribution or DEFAULT_INPUT_DISTRIBUTION)
    if distribution not in INPUT_DISTRIBUTIONS:
        raise ValueError(
            f"unknown input_distribution {distribution!r}; expected one of {list(INPUT_DISTRIBUTIONS)}"
        )
    return distribution


def sample_features(rng: np.random.Generator, shape, distribution: str) -> np.ndarray:
    """Draw float32 inputs from the target's input distribution."""
    distribution = standardize_input_distribution(distribution)
    if distribution == "normal":
        X = rng.standard_normal(shape)
    elif distribution == "uniform":
        X = rng.uniform(-_UNIFORM_BOUND, _UNIFORM_BOUND, size=shape)
    else:
        X = rng.uniform(-1.0, 1.0, size=shape)
    return X.astype(np.float32)


def _jsonable(value: Any) -> Any:
    """Convert numpy/nested values into JSON-serializable ones."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize value of type {type(value).__name__}")


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    """Stable short hash of a canonical JSON payload."""
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(encoded, digest_size=10, person=b"aritytask").hexdigest()


def _standardize_calibration(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fill in and validate the calibration options."""
    result = dict(DEFAULT_CALIBRATION)
    if cfg:
        unknown = set(cfg) - set(result)
        if unknown:
            raise ValueError(f"unknown calibration options: {sorted(unknown)}")
        result.update(cfg)

    result["mc_pool"] = int(result["mc_pool"])
    result["max_redraws"] = int(result["max_redraws"])
    result["min_node_sd"] = float(result["min_node_sd"])
    band = tuple(float(x) for x in result["activation_band"])

    if result["mc_pool"] <= 1:
        raise ValueError("calibration.mc_pool must exceed 1")
    if result["max_redraws"] <= 0:
        raise ValueError("calibration.max_redraws must be positive")
    if result["min_node_sd"] <= 0:
        raise ValueError("calibration.min_node_sd must be positive")
    if len(band) != 2 or not 0.0 < band[0] < band[1] < 1.0:
        raise ValueError("calibration.activation_band must satisfy 0 < low < high < 1")

    result["activation_band"] = band
    return result


def _standardize_wiring(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fill in and validate the wiring options."""
    result = dict(DEFAULT_WIRING)
    if cfg:
        unknown = set(cfg) - set(result)
        if unknown:
            raise ValueError(f"unknown wiring options: {sorted(unknown)}")
        result.update(cfg)
    if result["mode"] != "balanced_partition":
        raise ValueError("only wiring.mode='balanced_partition' is currently supported")
    return result


# Cell keys owned by the harness rather than any one family.
CELL_GLOBAL_KEYS = frozenset(
    {"d", "sigma_sq", "input_distribution", "target_variance", "calibration", "wiring"}
)


class TaskFamily(ABC):
    """Factory for one registered target family."""

    #: family-specific cell keys accepted beyond the global ones
    cell_keys: tuple[str, ...] = ("k", "layer_profile")
    #: defaults merged under the cell before validation and drawing
    cell_defaults: Mapping[str, Any] = {}
    #: input distributions the family accepts (None = all)
    allowed_inputs: tuple[str, ...] | None = None

    @abstractmethod
    def draw(self, cfg: Mapping[str, Any], function_seed: int) -> "Target":
        """Draw one deterministic target from a resolved task configuration."""
        raise NotImplementedError

    @classmethod
    def resolve_cell(cls, cell: Mapping[str, Any]) -> dict[str, Any]:
        """The cell with the family defaults filled in."""
        return {**cls.cell_defaults, **cell}

    @classmethod
    def check_cell(cls, cell: Mapping[str, Any]) -> int:
        """Family-specific structural checks; returns the active-coordinate count.

        The default is the balanced layered contract used by the tree workloads.
        """
        k = int(cell["k"])
        profile = tuple(int(w) for w in cell["layer_profile"])
        if k <= 0 or not profile or any(w <= 0 for w in profile) or profile[-1] != 1:
            raise ValueError("invalid k/layer_profile")
        for previous, current in zip(profile, profile[1:]):
            if previous != k * current:
                raise ValueError(
                    f"balanced_partition requires previous width {previous} to equal "
                    f"k * current width {k * current}"
                )
        return profile[0] * k

    @classmethod
    def validate_cell(cls, cell: Mapping[str, Any], *, require_d: bool) -> None:
        """Validate one resolved task cell: keys, shape, noise, input law, d bound."""
        unknown = set(cell) - CELL_GLOBAL_KEYS - set(cls.cell_keys)
        if unknown:
            raise ValueError(f"unknown cell keys: {sorted(unknown)}")
        resolved = cls.resolve_cell(cell)

        if require_d and "sigma_sq" not in resolved:
            raise ValueError("missing keys: ['sigma_sq']")
        if "sigma_sq" in resolved:
            sigma_sq = float(resolved["sigma_sq"])
            if not 0.0 <= sigma_sq < float("inf"):
                raise ValueError("sigma_sq must be finite and nonnegative")

        distribution = standardize_input_distribution(resolved.get("input_distribution"))
        if cls.allowed_inputs is not None and distribution not in cls.allowed_inputs:
            raise ValueError(
                f"input_distribution={distribution!r} is not supported; expected one "
                f"of {sorted(cls.allowed_inputs)}"
            )

        active = int(cls.check_cell(resolved))
        if require_d:
            if "d" not in resolved:
                raise ValueError("missing keys: ['d']")
            if int(resolved["d"]) < active:
                raise ValueError("d is too small for the active inputs")


class Target(ABC):
    """A realized, noise-free scalar regression target."""

    def __init__(
        self,
        *,
        d: int, # ambient dimension
        k: int, # local arity
        layer_profile: Sequence[int],   # number of nodes per layer
        family: str,
        sigma_sq: float,
        function_seed: int,
        active_raw: np.ndarray, # active raw coordinates
        parents: Sequence[np.ndarray],
        params: Mapping[str, Any],  # node-function parameters
        node_mu: Sequence[np.ndarray | float], # calibration means
        node_sd: Sequence[np.ndarray | float],  # calibration stddevs
        out_mu: float,
        out_sd: float,
        task_options: Mapping[str, Any] | None = None,
        calibration: Mapping[str, Any] | None = None,
        wiring: Mapping[str, Any] | None = None,
        input_distribution: str | None = None,
        target_variance: float = 1.0,
    ):
        """Store the realized target and derive its identity fingerprint."""
        self.d = int(d)
        self.k = int(k)
        self.layer_profile = tuple(int(w) for w in layer_profile)
        self.family = str(family)
        self.sigma_sq = float(sigma_sq)
        self.function_seed = int(function_seed)
        self.active_raw = np.asarray(active_raw, dtype=np.int64)
        self.parents = tuple(np.asarray(p, dtype=np.int64) for p in parents)
        self.params = dict(params)
        self.node_mu = list(node_mu)
        self.node_sd = list(node_sd)
        self.out_mu = float(out_mu)
        self.out_sd = float(out_sd)
        self.task_options = dict(task_options or {})
        self.calibration = dict(calibration or DEFAULT_CALIBRATION)
        self.wiring = dict(wiring or DEFAULT_WIRING)
        self.input_distribution = standardize_input_distribution(input_distribution)
        self.target_variance = float(target_variance)
        if not self.target_variance > 0.0 or not np.isfinite(self.target_variance):
            raise ValueError(f"target_variance must be finite and positive; received {target_variance}")
        self.S = int(sum(self.layer_profile))
        self.A = int(self.active_raw.size)

        identity = {
            "family": self.family,
            "d": self.d,
            "k": self.k,
            "layer_profile": self.layer_profile,
            "sigma_sq": self.sigma_sq,
            "function_seed": self.function_seed,
            "task_options": self.task_options,
            "calibration": self.calibration,
            "wiring": self.wiring,
            "input_distribution": self.input_distribution,
            "target_variance": self.target_variance,
        }
        self.fingerprint = f"{self.family}-{_canonical_hash(identity)}"

    @abstractmethod
    def _node_map(self, layer: int, Z: np.ndarray) -> np.ndarray:
        """Apply the family's node functions to one layer's parent values."""
        raise NotImplementedError

    def calibrate_layer(self, li: int, Z: np.ndarray) -> None:
        """Update family-specific calibration parameters before measuring a layer."""
        return None

    def redraw_node(self, li: int, j: int, rng: np.random.Generator) -> None:
        """Redraw a failed node during calibration, when supported by the family."""
        return None

    def activation_fraction(self, li: int, H: np.ndarray) -> np.ndarray | None:
        """Return per-node active fractions, or ``None`` for continuous families."""
        return None

    def _forward_nodes(self, Xact: np.ndarray, *, standardize: bool = True) -> list[np.ndarray]:
        """Evaluate every layer, returning per-layer node values."""
        Xact = np.asarray(Xact, dtype=np.float32)
        if Xact.ndim != 2 or Xact.shape[1] != self.A:
            raise ValueError(f"Xact must have shape (n, {self.A}); received {Xact.shape}")

        n = Xact.shape[0]
        activations: list[np.ndarray] = []
        previous = None

        for li, width in enumerate(self.layer_profile):
            if li == 0:
                Z = Xact.reshape(n, width, self.k)
            else:
                Z = previous[:, self.parents[li - 1]]

            H = np.asarray(self._node_map(li, Z), dtype=np.float32)
            if H.shape != (n, width):
                raise ValueError(
                    f"layer {li} node map must return {(n, width)}; received {H.shape}"
                )
            if not np.all(np.isfinite(H)):
                raise FloatingPointError(f"layer {li} produced nonfinite activations")

            if standardize:
                H = (H - self.node_mu[li]) / self.node_sd[li]
            activations.append(H)
            previous = H

        return activations

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate centered, approximately unit-variance ``f*(X)``."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(f"X must have shape (n, {self.d}); received {X.shape}")
        Xact = X[:, self.active_raw.reshape(-1)]
        out = self._forward_nodes(Xact)[-1][:, 0]
        result = (out - self.out_mu) / self.out_sd
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("target evaluation produced nonfinite values")
        return result.astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete realized target needed for auditing/reconstruction."""
        return _jsonable(
            {
                "family": self.family,
                "d": self.d,
                "k": self.k,
                "S": self.S,
                "A": self.A,
                "layer_profile": self.layer_profile,
                "function_seed": self.function_seed,
                "fingerprint": self.fingerprint,
                "sigma_sq": self.sigma_sq,
                "active_raw": self.active_raw,
                "parents": self.parents,
                "params": self.params,
                "node_mu": self.node_mu,
                "node_sd": self.node_sd,
                "out_mu": self.out_mu,
                "out_sd": self.out_sd,
                "task_options": self.task_options,
                "calibration": self.calibration,
                "wiring": self.wiring,
                "input_distribution": self.input_distribution,
                "target_variance": self.target_variance,
                "activation_audit": getattr(self, "activation_audit", None),
            }
        )


def wire(
    d: int,
    k: int,
    layer_profile: Sequence[int],
    structural_rng: np.random.Generator,
    embedding_rng: np.random.Generator,
    *,
    wiring: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Draw balanced-tree wiring and a separate ambient-coordinate embedding."""
    d = int(d)
    k = int(k)
    profile = tuple(int(w) for w in layer_profile)
    wiring_cfg = _standardize_wiring(wiring)

    if d <= 0 or k <= 0:
        raise ValueError("d and k must be positive")
    if not profile or any(w <= 0 for w in profile):
        raise ValueError("layer_profile must contain positive widths")
    if profile[-1] != 1:
        raise ValueError("layer_profile must end in one output node")

    n0 = profile[0]
    active_count = n0 * k
    if d < active_count:
        raise ValueError(
            f"d={d} < active_count={active_count}: target cannot fit in ambient space"
        )

    if wiring_cfg["mode"] != "balanced_partition":  # guarded by normalization
        raise AssertionError("unreachable wiring mode")

    parents: list[np.ndarray] = []
    for li in range(1, len(profile)):
        previous_width, width = profile[li - 1], profile[li]
        if previous_width != k * width:
            raise ValueError(
                f"layer {li}: previous width {previous_width} must equal "
                f"k * current width = {k * width} for balanced_partition wiring"
            )
        parents.append(structural_rng.permutation(previous_width).reshape(width, k))

    active = embedding_rng.choice(d, size=active_count, replace=False)
    return active.reshape(n0, k), tuple(parents)


def calibrate(
    target_cls: type[Target],
    *,
    d: int,
    k: int,
    layer_profile: Sequence[int],
    family: str,
    sigma_sq: float,
    function_seed: int,
    task_options: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    wiring: Mapping[str, Any] | None = None,
    input_distribution: str | None = None,
    target_variance: float = 1.0,
    **draw_options: Any,
) -> Target:
    """Draw and population-calibrate every node of a target."""
    function_seed = int(function_seed)
    profile = tuple(int(w) for w in layer_profile)
    calibration_cfg = _standardize_calibration(calibration)
    wiring_cfg = _standardize_wiring(wiring)
    input_distribution = standardize_input_distribution(input_distribution)
    options = dict(task_options or {})

    # Structural and node-parameter streams exclude d, preserving repeated
    # measures across ambient dimensions. Only the embedding stream includes d.
    structural_rng = np.random.default_rng(derive(family, function_seed, "structure"))
    embedding_rng = np.random.default_rng(derive(family, function_seed, int(d), "embedding"))
    pool_rng = np.random.default_rng(derive(family, function_seed, "calibration_pool"))

    active_raw, parents = wire(
        d,
        k,
        profile,
        structural_rng,
        embedding_rng,
        wiring=wiring_cfg,
    )
    params = target_cls.draw_params(profile, int(k), structural_rng, **draw_options)
    num_layers = len(profile)

    target = target_cls(
        d=d,
        k=k,
        layer_profile=profile,
        family=family,
        sigma_sq=sigma_sq,
        function_seed=function_seed,
        active_raw=active_raw,
        parents=parents,
        params=params,
        node_mu=[0.0] * num_layers,
        node_sd=[1.0] * num_layers,
        out_mu=0.0,
        out_sd=1.0,
        task_options=options,
        calibration=calibration_cfg,
        wiring=wiring_cfg,
        input_distribution=input_distribution,
        target_variance=target_variance,
    )

    mc_pool = calibration_cfg["mc_pool"]
    pool = sample_features(pool_rng, (mc_pool, active_raw.size), input_distribution)
    low, high = calibration_cfg["activation_band"]
    max_redraws = calibration_cfg["max_redraws"]
    min_node_sd = calibration_cfg["min_node_sd"]

    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    audit: list[list[float] | None] = []
    previous = None

    for li, width in enumerate(profile):
        Z = pool.reshape(mc_pool, width, k) if li == 0 else previous[:, parents[li - 1]]

        H = None
        frac = None
        bad: list[int] = []
        for _attempt in range(max_redraws):
            target.calibrate_layer(li, Z)
            H = np.asarray(target._node_map(li, Z), dtype=np.float32)
            if H.shape != (mc_pool, width):
                raise ValueError(
                    f"layer {li} node map must return {(mc_pool, width)}; received {H.shape}"
                )
            frac = target.activation_fraction(li, H)
            if frac is None:
                bad = []
                break
            frac = np.asarray(frac, dtype=np.float64)
            if frac.shape != (width,):
                raise ValueError(
                    f"activation_fraction for layer {li} must have shape {(width,)}; "
                    f"received {frac.shape}"
                )
            bad = [j for j, value in enumerate(frac) if not low <= value <= high]
            if not bad:
                break
            for j in bad:
                target.redraw_node(li, j, structural_rng)
        else:
            raise RuntimeError(
                f"layer {li}: nodes {bad} remain outside [{low}, {high}] after "
                f"{max_redraws} redraws"
            )

        assert H is not None
        audit.append(None if frac is None else [float(x) for x in frac])

        mean = H.mean(axis=0, dtype=np.float64)
        std = H.std(axis=0, dtype=np.float64)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise FloatingPointError(f"layer {li} calibration produced nonfinite moments")
        if float(std.min()) < min_node_sd:
            raise RuntimeError(
                f"layer {li}: minimum node sd {float(std.min()):.3e} < {min_node_sd:.3e}"
            )

        std = std + 1e-8
        means.append(mean.astype(np.float32))
        stds.append(std.astype(np.float32))
        target.node_mu = means + [0.0] * (num_layers - len(means))
        target.node_sd = stds + [1.0] * (num_layers - len(stds))
        previous = (H - means[-1]) / stds[-1]

    target.node_mu = means
    target.node_sd = stds
    target.activation_audit = audit

    # The final node has already been standardized above. Retain explicit
    # output fields for schema compatibility without applying a second noisy
    # Monte-Carlo normalization.
    target.out_mu = 0.0
    target.out_sd = 1.0 / float(np.sqrt(target_variance))
    return target
