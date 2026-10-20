"""Computation graphs and target instances.

A target is a directed acyclic graph of local functions over the active inputs,
followed by a readout:

* leaves ``0 .. A-1`` are the active inputs (an input ``x ~ U[-1, 1]`` enters a
  local function as ``sqrt(3) * x``, which has unit variance);
* node ``j`` (reference ``A + j``) applies one local function to its parents;
* the readout is one node, or the sum of several nodes.

Sequential normalization: the output of every local function is centred and
scaled to unit variance on a fixed pool of inputs, in topological order, and
the readout is normalized in the same way. The stored means and standard
deviations are part of the target instance.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..seeds import derive

SQRT3 = float(np.sqrt(3.0))
INPUT_DISTRIBUTION = "uniform_pm1"

#: Constants of the normalization pool. They enter the target fingerprint.
NORMALIZATION = {
    "mc_pool": 1 << 18,             # 262,144 inputs
    "activation_band": (0.35, 0.65),  # no local function of this version uses it; part of the fingerprint
    "max_redraws": 40,
    "min_node_sd": 1e-3,
}


def sample_inputs(rng: np.random.Generator, shape) -> np.ndarray:
    """Independent coordinates ``X_j ~ U[-1, 1]`` as float32."""
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def _jsonable(value: Any) -> Any:
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
    raise TypeError(f"cannot serialize a value of type {type(value).__name__}")


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(encoded, digest_size=10, person=b"aritytask").hexdigest()


class GraphSpec:
    """A validated topology: the parents of every node and a linear readout.

    The parents of a local function must have disjoint input ancestry, so they
    are independent random variables. Only the readout may add values that
    share ancestry.
    """

    def __init__(self, A, node_parents, readout_indices, readout_weights):
        self.A = int(A)
        self.node_parents = tuple(tuple(int(r) for r in refs) for refs in node_parents)
        self.readout_indices = tuple(int(i) for i in readout_indices)
        self.readout_weights = tuple(float(w) for w in readout_weights)
        self.S = len(self.node_parents)

        if self.A < 1:
            raise ValueError("a graph needs at least one leaf")
        if not self.node_parents:
            raise ValueError("a graph needs at least one node")
        if len(self.readout_indices) != len(self.readout_weights) or not self.readout_indices:
            raise ValueError("readout indices and weights must align and be non-empty")
        if any(not 0 <= i < self.S for i in self.readout_indices):
            raise ValueError("readout indices must refer to nodes")

        ancestors: list[frozenset[int]] = []
        for v, refs in enumerate(self.node_parents):
            if len(set(refs)) != len(refs):
                raise ValueError(f"node {v} repeats a parent")
            parent_sets = []
            for r in refs:
                if not 0 <= r < self.A + v:
                    raise ValueError(f"node {v}: parent {r} is out of range or not in topological order")
                parent_sets.append(frozenset((r,)) if r < self.A else ancestors[r - self.A])
            union: set[int] = set()
            for s in parent_sets:
                if union & s:
                    raise ValueError(f"node {v} has parents with shared input ancestry")
                union |= s
            ancestors.append(frozenset(union))
        self.leaf_ancestors = tuple(ancestors)

    @property
    def additive_readout(self) -> bool:
        return len(self.readout_indices) > 1

    def levels(self) -> list[int]:
        """The nonlinear level of every node (a node whose parents are all inputs is at level 1)."""
        level: list[int] = []
        for refs in self.node_parents:
            level.append(1 + max((level[r - self.A] for r in refs if r >= self.A), default=0))
        return level

    def to_dict(self) -> dict:
        return {
            "A": self.A,
            "node_parents": [list(refs) for refs in self.node_parents],
            "readout_indices": list(self.readout_indices),
            "readout_weights": list(self.readout_weights),
        }


class Target:
    """One target instance: a noise-free function ``f*`` with known intermediates."""

    def __init__(self, *, graph: GraphSpec, law, family: str, d: int, k: int, layer_profile: Sequence[int],
                 sigma_sq: float, function_seed: int, active_raw: np.ndarray, parents: Sequence[np.ndarray],
                 params: Mapping[str, Any], node_mu: Sequence[float], node_sd: Sequence[float],
                 out_mu: float, out_sd: float, task_options: Mapping[str, Any], wiring: Mapping[str, Any],
                 normalization: Mapping[str, Any], target_variance: float = 1.0,
                 normalization_report: Mapping[str, Any] | None = None):
        self.graph = graph
        self.law = law
        self.family = str(family)
        self.d = int(d)
        self.k = int(k)
        self.layer_profile = tuple(int(w) for w in layer_profile)
        self.sigma_sq = float(sigma_sq)
        self.function_seed = int(function_seed)
        self.active_raw = np.asarray(active_raw, dtype=np.int64)
        self.parents = tuple(np.asarray(p, dtype=np.int64) for p in parents)
        self.params = dict(params)
        self.node_mu = list(node_mu)
        self.node_sd = list(node_sd)
        self.out_mu = float(out_mu)
        self.out_sd = float(out_sd)
        self.task_options = dict(task_options)
        self.wiring = dict(wiring)
        self.calibration = dict(normalization)
        self.input_distribution = INPUT_DISTRIBUTION
        self.target_variance = float(target_variance)
        self.normalization_report = dict(normalization_report or {})
        self.topology_hash: str | None = None
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
        #: The identity of the instance. Seeds of the data and of the training are derived from it.
        self.fingerprint = f"{self.family}-{_canonical_hash(identity)}"

    @property
    def target_id(self) -> str:
        digest = self.fingerprint.split("-", 1)[-1][:8]
        return f"{self.family}_d{self.d}_fs{self.function_seed}_{digest}"

    @property
    def active_inputs(self) -> list[int]:
        """The coordinates of ``x`` that the target reads, sorted."""
        return sorted(int(i) for i in self.active_raw.reshape(-1))

    def _check_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(f"X must have shape (n, {self.d}); received {X.shape}")
        return X

    def forward(self, X: np.ndarray) -> dict:
        """Leaves, normalized node values and the raw readout, as float64 arrays."""
        X = self._check_input(X)
        leaves = SQRT3 * X[:, self.active_raw.reshape(-1)].astype(np.float64)
        values: list[np.ndarray] = []
        for v, refs in enumerate(self.graph.node_parents):
            Z = np.stack([leaves[:, r] if r < self.graph.A else values[r - self.graph.A] for r in refs], axis=1)
            h = self.law.value(self.params["nodes"][v], Z)
            values.append((h - self.node_mu[v]) / self.node_sd[v])
        raw = sum(w * values[i] for i, w in zip(self.graph.readout_indices, self.graph.readout_weights))
        return {"leaves": leaves, "values": values, "raw_readout": raw}

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """The noise-free target ``f*(X)`` as float32, with mean 0 and variance ``target_variance``."""
        raw = self.forward(X)["raw_readout"]
        result = (raw - self.out_mu) / self.out_sd
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("the target evaluation gave non-finite values")
        return result.astype(np.float32)

    def intermediates(self, X: np.ndarray) -> np.ndarray:
        """The normalized value of every local function, shape ``(n, S)``, float64."""
        return np.stack(self.forward(X)["values"], axis=1)

    def gradients(self, X: np.ndarray) -> np.ndarray:
        """The gradient of ``f*`` with respect to the active inputs, shape ``(n, A)``."""
        frame = self.forward(X)
        leaves, values = frame["leaves"], frame["values"]
        n, A = leaves.shape
        jacobians: list[np.ndarray] = []
        for v, refs in enumerate(self.graph.node_parents):
            Z = np.stack([leaves[:, r] if r < A else values[r - A] for r in refs], axis=1)
            local = self.law.grad(self.params["nodes"][v], Z) / self.node_sd[v]
            J = np.zeros((n, A))
            for j, r in enumerate(refs):
                if r < A:
                    J[:, r] += local[:, j] * SQRT3
                else:
                    J += local[:, j][:, None] * jacobians[r - A]
            jacobians.append(J)
        total = sum(w * jacobians[i] for i, w in zip(self.graph.readout_indices, self.graph.readout_weights))
        return total / self.out_sd

    def to_dict(self) -> dict:
        """The complete instance: graph, parameters and stored normalization."""
        return _jsonable({
            "family": self.family, "d": self.d, "k": self.k, "S": self.S, "A": self.A,
            "layer_profile": self.layer_profile, "function_seed": self.function_seed,
            "fingerprint": self.fingerprint, "sigma_sq": self.sigma_sq,
            "active_raw": self.active_raw, "parents": self.parents, "params": self.params,
            "node_mu": self.node_mu, "node_sd": self.node_sd, "out_mu": self.out_mu, "out_sd": self.out_sd,
            "task_options": self.task_options, "calibration": self.calibration, "wiring": self.wiring,
            "input_distribution": self.input_distribution, "target_variance": self.target_variance,
            "topology_hash": self.topology_hash, "normalization_report": self.normalization_report,
        })


def draw_target(*, family: str, d: int, sigma_sq: float, function_seed: int, law, graph: GraphSpec,
                structure_rng: np.random.Generator, active_raw: np.ndarray, parents: Sequence[np.ndarray],
                layer_profile: Sequence[int], k: int, task_options: Mapping, wiring: Mapping,
                target_variance: float = 1.0) -> Target:
    """Draw the local functions of a graph and normalize them in topological order.

    A local function is drawn again only when its raw standard deviation on the
    normalization pool is below ``min_node_sd``.
    """
    cfg = dict(NORMALIZATION)
    node_params = [law.draw(structure_rng, len(refs)) for refs in graph.node_parents]
    node_mu = [0.0] * graph.S
    node_sd = [1.0] * graph.S

    pool_rng = np.random.default_rng(derive(family, int(function_seed), "calibration_pool"))
    pool = sample_inputs(pool_rng, (cfg["mc_pool"], graph.A)).astype(np.float64)
    leaves = SQRT3 * pool

    values: list[np.ndarray] = []
    moments: list[tuple[float, float]] = []
    redraws = 0
    for v, refs in enumerate(graph.node_parents):
        Z = np.stack([leaves[:, r] if r < graph.A else values[r - graph.A] for r in refs], axis=1)
        for _attempt in range(cfg["max_redraws"]):
            law.calibrate(node_params[v], Z)
            h = law.value(node_params[v], Z)
            sd = float(h.std())
            if sd >= cfg["min_node_sd"]:
                break
            node_params[v] = law.draw(structure_rng, len(refs))
            redraws += 1
        else:
            raise RuntimeError(f"{family} node {v}: no valid draw after {cfg['max_redraws']} attempts")
        mu = float(h.mean())
        moments.append((mu, sd))
        node_mu[v], node_sd[v] = mu, sd
        values.append((h - mu) / sd)

    raw = sum(w * values[i] for i, w in zip(graph.readout_indices, graph.readout_weights))
    pool_readout = (float(raw.mean()), float(raw.std()))
    if graph.additive_readout:
        out_mu, out_sd = pool_readout[0], float(pool_readout[1] / np.sqrt(target_variance))
    else:                       # the readout node is already normalized
        out_mu, out_sd = 0.0, float(np.sqrt(1.0 / target_variance))
    if not out_sd > 0.0 or not np.isfinite(out_sd):
        raise RuntimeError(f"{family}: degenerate readout normalization {out_sd!r}")

    mu_list, sd_list = list(node_mu), list(node_sd)
    if graph.additive_readout:  # the readout normalization is the last entry
        mu_list.append(out_mu)
        sd_list.append(out_sd * float(np.sqrt(target_variance)))

    return Target(
        graph=graph, law=law, family=family, d=int(d), k=int(k), layer_profile=layer_profile,
        sigma_sq=float(sigma_sq), function_seed=int(function_seed), active_raw=active_raw, parents=parents,
        params={"nodes": node_params, "graph": graph.to_dict(), "law": law.describe()},
        node_mu=mu_list, node_sd=sd_list, out_mu=out_mu, out_sd=out_sd,
        task_options=task_options, wiring=wiring,
        normalization={"mode": "audited_mc", **cfg}, target_variance=target_variance,
        normalization_report={"mc_pool": cfg["mc_pool"], "redraws": redraws, "node_moments": moments,
                              "pool_readout": pool_readout},
    )
