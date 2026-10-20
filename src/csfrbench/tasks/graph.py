"""Composition-graph target machinery for the CSFRBench suite (target-design.md §5).

A target is a DAG of law nodes over standardized leaves plus a readout:

  * leaves 0..A-1 carry the active input coordinates (uniform_pm1 leaves enter as
    sqrt(3) * x for unit variance; tri consumes U[-1, 1] directly);
  * node j (ref A + j) applies one law to its parent refs;
  * the readout returns one node or a fixed-weight sum of nodes (flat root, dag).

Calibration follows the §7 policy: exact law (plain anova, tri; the additive
readout normalizer comes from the exact multilinear expansion), or audited
deterministic MC on the fixed calibration pool (shaped anova standardizes its
shapes; relu/fourier standardize node outputs and honor the activation band).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..utils.seeds import derive
from .base import (
    Target,
    TaskFamily,
    _standardize_calibration,
    sample_features,
    standardize_input_distribution,
    wire,
)
from .laws import AnovaLaw

_SQRT3 = float(np.sqrt(3.0))
_EXPANSION_CAP = 1 << 20
_REDRAW_MIN_SD = 0.1        # §7: node sd < 0.1 triggers a counted redraw


class GraphSpec:
    """Validated topology: parent refs per node and a linear readout."""

    def __init__(self, A, node_parents, readout_indices, readout_weights):
        self.A = int(A)
        self.node_parents = tuple(tuple(int(r) for r in refs) for refs in node_parents)
        self.readout_indices = tuple(int(i) for i in readout_indices)
        self.readout_weights = tuple(float(w) for w in readout_weights)
        self.S = len(self.node_parents)

        if self.A < 1:
            raise ValueError("GraphSpec requires at least one leaf")
        if not self.node_parents:
            raise ValueError("GraphSpec requires at least one node")
        if len(self.readout_indices) != len(self.readout_weights) or not self.readout_indices:
            raise ValueError("readout indices and weights must align and be nonempty")
        if any(not 0 <= i < self.S for i in self.readout_indices):
            raise ValueError("readout indices must reference nodes")

        # Leaf-ancestor sets, and the §8 merge tripwire: every *law* node must have
        # parents with pairwise-disjoint leaf ancestries (independence under
        # independent leaves); only the linear readout may merge shared ancestry.
        ancestors: list[frozenset[int]] = []
        for v, refs in enumerate(self.node_parents):
            if len(set(refs)) != len(refs):
                raise ValueError(f"node {v} repeats a parent ref")
            parent_sets = []
            for r in refs:
                if not 0 <= r < self.A + v:
                    raise ValueError(f"node {v} ref {r} is out of range or non-topological")
                parent_sets.append(frozenset((r,)) if r < self.A else ancestors[r - self.A])
            union: set[int] = set()
            for s in parent_sets:
                if union & s:
                    raise ValueError(
                        f"node {v} merges parents with shared leaf ancestry; only the "
                        "additive readout may do that (merge_calibration tripwire)"
                    )
                union |= s
            ancestors.append(frozenset(union))
        self.leaf_ancestors = tuple(ancestors)

    @property
    def additive_readout(self) -> bool:
        return len(self.readout_indices) > 1

    def to_dict(self) -> dict:
        return {
            "A": self.A,
            "node_parents": [list(refs) for refs in self.node_parents],
            "readout_indices": list(self.readout_indices),
            "readout_weights": list(self.readout_weights),
        }


# ------------------------------------------------------------- expansion --

def multilinear_expansion(graph: GraphSpec, law, node_params) -> dict[frozenset, float]:
    """Exact multilinear expansion of the raw readout in the orthonormal leaf basis.

    Valid for the plain anova law only: every node is multilinear and every
    product joins disjoint leaf supports, so the coefficients are exact and
    variance mass by interaction order is a finite sum (§4.1).
    """
    if not (isinstance(law, AnovaLaw) and law.plain):
        raise ValueError("multilinear_expansion applies to the plain anova law only")

    def scaled(poly, c):
        return {S: c * v for S, v in poly.items()}

    def added(polys):
        out: dict[frozenset, float] = {}
        for poly in polys:
            for S, v in poly.items():
                out[S] = out.get(S, 0.0) + v
        return out

    def multiplied(p, q):
        out: dict[frozenset, float] = {}
        for S, a in p.items():
            for T, b in q.items():
                if S & T:
                    raise ValueError("non-multilinear product: shared support")
                key = S | T
                out[key] = out.get(key, 0.0) + a * b
        if len(out) > _EXPANSION_CAP:
            raise RuntimeError("multilinear expansion exceeded the size cap")
        return out

    leaf_polys = [{frozenset((i,)): 1.0} for i in range(graph.A)]
    node_polys: list[dict[frozenset, float]] = []
    for refs, params in zip(graph.node_parents, node_params):
        parents = [leaf_polys[r] if r < graph.A else node_polys[r - graph.A] for r in refs]
        alpha, gamma = law._weights(len(refs))
        terms = [scaled(p, alpha * s) for p, s in zip(parents, params["sign_a"])]
        product = {frozenset(): gamma}
        for p, s in zip(parents, params["sign_b"]):
            product = multiplied(product, scaled(p, s))
        node_polys.append(added(terms + [product]))

    poly = added(
        scaled(node_polys[i], w)
        for i, w in zip(graph.readout_indices, graph.readout_weights)
    )
    poly.pop(frozenset(), None)     # exactly zero mean; drop the empty key if present
    return {S: c for S, c in poly.items() if c != 0.0}   # prune pole-artifact zeros


def expansion_variance(poly: Mapping[frozenset, float]) -> float:
    """Exact variance of an expanded target (orthonormal basis)."""
    return float(sum(c * c for S, c in poly.items() if S))


def expansion_mass_by_order(poly: Mapping[frozenset, float]) -> dict[int, float]:
    """Exact variance mass per interaction order."""
    mass: dict[int, float] = {}
    for S, c in poly.items():
        if S:
            mass[len(S)] = mass.get(len(S), 0.0) + c * c
    return dict(sorted(mass.items()))


def expansion_main_effects(poly: Mapping[frozenset, float], A: int) -> np.ndarray:
    """Exact per-leaf first-order variance mass."""
    out = np.zeros(A)
    for i in range(A):
        out[i] = poly.get(frozenset((i,)), 0.0) ** 2
    return out


# ---------------------------------------------------------------- target --

class CSFRGraphTarget(Target):
    """A realized composition-graph target; evaluates its own DAG."""

    def __init__(self, *, graph: GraphSpec, law, calibration_report=None, **base_kwargs):
        self.graph = graph
        self.law = law
        self.calibration_report = dict(calibration_report or {})
        super().__init__(**base_kwargs)

    def _node_map(self, layer, Z):
        raise NotImplementedError("CSFRGraphTarget evaluates its own graph; see forward()")

    @property
    def leaf_scale(self) -> float:
        if self.law.scale_leaves and self.input_distribution == "uniform_pm1":
            return _SQRT3
        return 1.0

    def _check_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(f"X must have shape (n, {self.d}); received {X.shape}")
        return X

    def forward(self, X: np.ndarray) -> dict:
        """Leaves, per-node values, and the raw readout, as float64 arrays."""
        X = self._check_input(X)
        Xact = X[:, self.active_raw.reshape(-1)].astype(np.float64)
        leaves = self.leaf_scale * Xact

        values: list[np.ndarray] = []
        for v, refs in enumerate(self.graph.node_parents):
            Z = np.stack(
                [leaves[:, r] if r < self.graph.A else values[r - self.graph.A]
                 for r in refs],
                axis=1,
            )
            h = self.law.value(self.params["nodes"][v], Z)
            values.append((h - self.node_mu[v]) / self.node_sd[v])

        raw = sum(
            w * values[i]
            for i, w in zip(self.graph.readout_indices, self.graph.readout_weights)
        )
        return {"leaves": leaves, "values": values, "raw_readout": raw}

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate centered ``f*(X)`` at variance ``target_variance``."""
        raw = self.forward(X)["raw_readout"]
        result = (raw - self.out_mu) / self.out_sd
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("target evaluation produced nonfinite values")
        return result.astype(np.float32)

    def gradients(self, X: np.ndarray) -> np.ndarray:
        """Exact gradient of ``f*`` w.r.t. the active raw coordinates, shape (n, A)."""
        frame = self.forward(self._check_input(X))
        leaves, values = frame["leaves"], frame["values"]
        n, A = leaves.shape

        jacobians: list[np.ndarray] = []
        for v, refs in enumerate(self.graph.node_parents):
            Z = np.stack(
                [leaves[:, r] if r < A else values[r - A] for r in refs],
                axis=1,
            )
            local = self.law.grad(self.params["nodes"][v], Z) / self.node_sd[v]
            J = np.zeros((n, A))
            for j, r in enumerate(refs):
                if r < A:
                    J[:, r] += local[:, j] * self.leaf_scale
                else:
                    J += local[:, j][:, None] * jacobians[r - A]
            jacobians.append(J)

        total = sum(
            w * jacobians[i]
            for i, w in zip(self.graph.readout_indices, self.graph.readout_weights)
        )
        return total / self.out_sd

    def expansion(self) -> dict[frozenset, float]:
        """Exact multilinear expansion of the *normalized* target (plain anova only)."""
        poly = multilinear_expansion(self.graph, self.law, self.params["nodes"])
        return {S: c / self.out_sd for S, c in poly.items()}

    def to_dict(self) -> dict:
        record = super().to_dict()
        record["calibration_report"] = self.calibration_report
        return record


# ---------------------------------------------------------------- driver --

def draw_graph_target(
    *,
    family: str,
    cfg: Mapping,
    function_seed: int,
    law,
    graph: GraphSpec,
    structural_rng: np.random.Generator,
    active_raw: np.ndarray,
    parents_attr: Sequence[np.ndarray],
    layer_profile: Sequence[int],
    k: int,
    task_options: Mapping,
    wiring: Mapping,
    ties: Sequence[tuple[int, int]] = (),
    allowed_inputs: Sequence[str] | None = None,
) -> CSFRGraphTarget:
    """Draw per-node parameters, calibrate per the §7 policy, and build the target."""
    distribution = standardize_input_distribution(cfg.get("input_distribution"))
    if allowed_inputs is not None and distribution not in allowed_inputs:
        raise ValueError(
            f"{family} requires input_distribution in {sorted(allowed_inputs)}; "
            f"received {distribution!r}"
        )
    target_variance = float(cfg.get("target_variance", 1.0))
    calibration_cfg = _standardize_calibration(cfg.get("calibration"))

    node_params = [
        law.draw(structural_rng, len(refs)) for refs in graph.node_parents
    ]
    tied_nodes: set[int] = set()
    for dst, src in ties:
        if law.activation_fraction(node_params[src], np.zeros(1)) is not None:
            raise ValueError("tied nodes are not supported for band-calibrated laws")
        node_params[dst] = node_params[src]     # shared object: one function, two slots
        tied_nodes |= {int(dst), int(src)}

    node_mu = [0.0] * graph.S
    node_sd = [1.0] * graph.S
    report: dict = {"mode": "exact" if law.exact else "audited_mc"}
    pool_out = None

    needs_pool = law.needs_pool or law.standardize_nodes
    if needs_pool:
        pool_rng = np.random.default_rng(derive(family, int(function_seed), "calibration_pool"))
        mc_pool = calibration_cfg["mc_pool"]
        low, high = calibration_cfg["activation_band"]
        max_redraws = calibration_cfg["max_redraws"]
        min_node_sd = calibration_cfg["min_node_sd"]

        pool = sample_features(pool_rng, (mc_pool, graph.A), distribution).astype(np.float64)
        scale = _SQRT3 if (law.scale_leaves and distribution == "uniform_pm1") else 1.0
        leaves = scale * pool

        # v0.5 acceptance is one uniform rule: node outputs are
        # pool-standardized, so ordinary scale variation is rescaled rather
        # than redrawn, and only the catastrophic floor (min_node_sd) forces
        # a counted redraw. The activation-fraction band belongs to the
        # dropped ReLU family and is not part of the v0.5 path; legacy
        # configurations keep both it and the sd < 0.1 rule.
        catastrophic_only = bool(getattr(law, "catastrophic_only", False))
        redraw_floor = min_node_sd if catastrophic_only else _REDRAW_MIN_SD

        values: list[np.ndarray] = []
        moments: list[tuple[float, float]] = []
        fractions: list[float | None] = []
        redraws = 0
        for v, refs in enumerate(graph.node_parents):
            Z = np.stack(
                [leaves[:, r] if r < graph.A else values[r - graph.A] for r in refs],
                axis=1,
            )
            for _attempt in range(max_redraws):
                law.calibrate(node_params[v], Z, structural_rng)
                h = law.value(node_params[v], Z)
                frac = law.activation_fraction(node_params[v], h)
                sd = float(h.std())
                band_ok = (True if catastrophic_only
                           else (frac is None or low <= frac <= high))
                if band_ok and sd >= redraw_floor:
                    break
                if v in tied_nodes:
                    raise RuntimeError(
                        f"{family} node {v}: a tied node failed calibration; redrawing "
                        "would break the tie"
                    )
                node_params[v] = law.draw(structural_rng, len(refs))
                redraws += 1
            else:
                raise RuntimeError(
                    f"{family} node {v}: calibration failed after {max_redraws} redraws"
                )
            if sd < min_node_sd:
                raise RuntimeError(
                    f"{family} node {v}: pool sd {sd:.3e} < {min_node_sd:.3e}"
                )
            mu = float(h.mean())
            moments.append((mu, sd))
            fractions.append(frac)
            if law.standardize_nodes:
                node_mu[v], node_sd[v] = mu, sd
                h = (h - mu) / sd
            values.append(h)

        raw = sum(
            w * values[i]
            for i, w in zip(graph.readout_indices, graph.readout_weights)
        )
        pool_out = (float(raw.mean()), float(raw.std()))
        report.update(
            mc_pool=mc_pool,
            redraws=redraws,
            node_moments=moments,
            activation_fractions=fractions,
            pool_readout=pool_out,
        )

    # §7 readout normalizer, split by calibration mode.
    if isinstance(law, AnovaLaw) and law.plain:
        variance = expansion_variance(multilinear_expansion(graph, law, node_params))
        out_mu, out_sd = 0.0, float(np.sqrt(variance / target_variance))
        report["readout_variance"] = variance
    elif law.exact:                                     # tri: every node is U[-1, 1]
        out_mu, out_sd = 0.0, float(np.sqrt((1.0 / 3.0) / target_variance))
    elif law.standardize_nodes and not graph.additive_readout:
        out_mu, out_sd = 0.0, float(np.sqrt(1.0 / target_variance))
    else:                                               # shaped anova, any readout
        mu, sd = pool_out
        out_mu, out_sd = mu, float(sd / np.sqrt(target_variance))
    if not out_sd > 0.0 or not np.isfinite(out_sd):
        raise RuntimeError(f"{family}: degenerate readout normalizer {out_sd!r}")

    mu_list = list(node_mu)
    sd_list = list(node_sd)
    if graph.additive_readout:      # record the readout normalizer as the final entry
        mu_list.append(out_mu)
        sd_list.append(out_sd * float(np.sqrt(target_variance)))

    calibration_identity = (
        {"mode": "exact"} if law.exact
        else {"mode": "audited_mc", **calibration_cfg}
    )
    return CSFRGraphTarget(
        graph=graph,
        law=law,
        calibration_report=report,
        d=int(cfg["d"]),
        k=int(k),
        layer_profile=layer_profile,
        family=family,
        sigma_sq=float(cfg["sigma_sq"]),
        function_seed=int(function_seed),
        active_raw=active_raw,
        parents=parents_attr,
        params={"nodes": node_params, "graph": graph.to_dict(), "law": law.describe()},
        node_mu=mu_list,
        node_sd=sd_list,
        out_mu=out_mu,
        out_sd=out_sd,
        task_options=task_options,
        calibration=calibration_identity,
        wiring=wiring,
        input_distribution=distribution,
        target_variance=target_variance,
    )


# ------------------------------------------------------- balanced helper --

def balanced_refs(profile: Sequence[int], k: int, parents: Sequence[np.ndarray]):
    """Graph refs for a balanced layered tree wired by ``base.wire``."""
    profile = tuple(int(w) for w in profile)
    A = profile[0] * k
    offsets = [0]
    for width in profile[:-1]:
        offsets.append(offsets[-1] + width)

    node_parents: list[tuple[int, ...]] = []
    for j in range(profile[0]):
        node_parents.append(tuple(j * k + i for i in range(k)))
    for li in range(1, len(profile)):
        for j in range(profile[li]):
            node_parents.append(
                tuple(A + offsets[li - 1] + p for p in parents[li - 1][j])
            )
    return GraphSpec(A, node_parents, (len(node_parents) - 1,), (1.0,))


class BalancedGraphFamily(TaskFamily):
    """Shared draw for balanced k-ary tree workloads (tree, node, dense shapes)."""

    allowed_inputs: tuple[str, ...] | None = None
    default_input = "uniform_pm1"

    def make_law(self, cfg: Mapping):
        raise NotImplementedError

    def family_options(self, cfg: Mapping, law) -> dict:
        return dict(law.describe())

    def draw(self, cfg: Mapping, function_seed: int) -> CSFRGraphTarget:
        cfg = self.resolve_cell(cfg)
        family = self.registry_name
        d, k = int(cfg["d"]), int(cfg["k"])
        profile = tuple(int(w) for w in cfg["layer_profile"])
        law = self.make_law(cfg)

        structural_rng = np.random.default_rng(derive(family, int(function_seed), "structure"))
        embedding_rng = np.random.default_rng(
            derive(family, int(function_seed), int(d), "embedding")
        )
        active_raw, parents = wire(
            d, k, profile, structural_rng, embedding_rng, wiring=cfg.get("wiring")
        )
        graph = balanced_refs(profile, k, parents)

        return draw_graph_target(
            family=family,
            cfg=cfg,
            function_seed=function_seed,
            law=law,
            graph=graph,
            structural_rng=structural_rng,
            active_raw=active_raw,
            parents_attr=parents,
            layer_profile=profile,
            k=k,
            task_options=self.family_options(cfg, law),
            wiring={"mode": "balanced_partition"},
            allowed_inputs=self.allowed_inputs,
        )
