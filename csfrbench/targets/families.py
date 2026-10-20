"""The target families of CSFRBench.

A family draws a target instance from an instance seed. The family identifier
and the instance seed fix the graph and the local functions; the ambient
dimension ``d`` only fixes which coordinates of the input are active. So the
instance with seed 6000 is the same function in the primary setting (every
input active) and in the nuisance-input setting (inert coordinates added).

The family identifiers are part of the seed derivation and of the target
fingerprint. Do not change them.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..seeds import derive
from .graph import GraphSpec, Target, draw_target
from .laws import AdditiveMultiplicative, RandomFourier
from .topologies import (DAG8_NODE_PARENTS, DAG8_PROFILE, RAND_A, balanced_graph, balanced_wiring,
                         canonical_topology_hash, dag8_graph, level_profile, sample_graph)

F_RANGE = (0.25, 0.4)       # frequency interval of the additive-multiplicative factors, in cycles
RHO = 0.5                   # share of the k-way interaction in the variance of a local function
SIGMA_SQ = 0.01             # variance of the observation noise
TARGET_VARIANCE = 1.0


def _streams(family: str, seed: int, d: int) -> tuple[np.random.Generator, np.random.Generator]:
    structure = np.random.default_rng(derive(family, int(seed), "structure"))
    embedding = np.random.default_rng(derive(family, int(seed), int(d), "embedding"))
    return structure, embedding


def _balanced(family: str, law, *, d: int, k: int, profile, seed: int, options: dict, sigma_sq: float) -> Target:
    structure, embedding = _streams(family, seed, d)
    active, parents = balanced_wiring(d, k, profile, structure, embedding)
    graph = balanced_graph(profile, k, parents)
    return draw_target(family=family, d=d, sigma_sq=sigma_sq, function_seed=seed, law=law, graph=graph,
                       structure_rng=structure, active_raw=active, parents=parents, layer_profile=profile, k=k,
                       task_options=options, wiring={"mode": "balanced_partition"},
                       target_variance=TARGET_VARIANCE)


def _fixed(family: str, law, graph: GraphSpec, *, d: int, seed: int, active_shape, parents, profile, options: dict,
           wiring: str, sigma_sq: float, sampled: bool = False) -> Target:
    if d < graph.A:
        raise ValueError(f"d = {d} is smaller than the number of active inputs {graph.A}")
    structure, embedding = _streams(family, seed, d)
    if sampled:
        graph = sample_graph(structure)
        parents = tuple(np.asarray(p, dtype=np.int64) for p in graph.node_parents)
        profile = level_profile(graph)
    active = embedding.choice(d, size=graph.A, replace=False)
    if active_shape is not None:
        active = active.reshape(active_shape)
    target = draw_target(family=family, d=d, sigma_sq=sigma_sq, function_seed=seed, law=law, graph=graph,
                         structure_rng=structure, active_raw=active, parents=parents, layer_profile=profile, k=2,
                         task_options=options, wiring={"mode": wiring}, target_variance=TARGET_VARIANCE)
    if sampled:
        target.topology_hash = canonical_topology_hash(graph)
    return target


# ---------------------------------------------------------------- the families


def tree(d: int, seed: int, *, profile=(4, 2, 1), sigma_sq: float = SIGMA_SQ) -> Target:
    """Balanced binary tree with additive-multiplicative local functions."""
    law = AdditiveMultiplicative(RHO, F_RANGE)
    return _balanced("tree_functional_anova", law, d=d, k=2, profile=tuple(profile), seed=seed,
                     options=law.describe(), sigma_sq=sigma_sq)


def fourier_tree(d: int, seed: int, *, profile=(4, 2, 1), sigma_sq: float = SIGMA_SQ) -> Target:
    """Balanced binary tree with random Fourier local functions (Q = 8, sigma_omega = 0.85)."""
    law = RandomFourier(0.85, Q=8)
    return _balanced("tree_fourier", law, d=d, k=2, profile=tuple(profile), seed=seed, options=law.describe(),
                     sigma_sq=sigma_sq)


def single(d: int, seed: int, *, k: int = 2, sigma_sq: float = SIGMA_SQ) -> Target:
    """One additive-multiplicative local function of arity ``k``."""
    law = AdditiveMultiplicative(RHO, F_RANGE)
    return _balanced("node_functional_anova", law, d=d, k=k, profile=(1,), seed=seed, options=law.describe(),
                     sigma_sq=sigma_sq)


def high_arity(d: int, seed: int, *, sigma_sq: float = SIGMA_SQ) -> Target:
    """One random Fourier function of all 8 active inputs (Q = 24, sigma_omega = 1.5)."""
    law = RandomFourier(1.5, Q=24)
    return _balanced("dense_fourier", law, d=d, k=8, profile=(1,), seed=seed,
                     options={**law.describe(), "A": 8}, sigma_sq=sigma_sq)


def high_arity_stress(d: int, seed: int, *, sigma_sq: float = SIGMA_SQ) -> Target:
    """One additive-multiplicative function of all 8 active inputs (stress setting)."""
    law = AdditiveMultiplicative(RHO, F_RANGE)
    return _balanced("dense_functional_anova", law, d=d, k=8, profile=(1,), seed=seed,
                     options={**law.describe(), "A": 8}, sigma_sq=sigma_sq)


def flat(d: int, seed: int, *, w: int = 4, sigma_sq: float = SIGMA_SQ) -> Target:
    """``w`` disjoint two-input local functions and an additive readout."""
    law = AdditiveMultiplicative(RHO, F_RANGE)
    graph = GraphSpec(2 * w, [(2 * j, 2 * j + 1) for j in range(w)], tuple(range(w)), (1.0,) * w)
    return _fixed("flat_functional_anova", law, graph, d=d, seed=seed, active_shape=(w, 2),
                  parents=(np.arange(w, dtype=np.int64).reshape(1, w),), profile=(w, 1),
                  options={**law.describe(), "w": w}, wiring="flat_additive", sigma_sq=sigma_sq)


def _dag_parents():
    return tuple(np.asarray(p, dtype=np.int64) for p in DAG8_NODE_PARENTS)


def reuse_dag(d: int, seed: int, *, sigma_sq: float = SIGMA_SQ) -> Target:
    """The fixed reuse DAG with additive-multiplicative local functions."""
    law = AdditiveMultiplicative(RHO, F_RANGE)
    return _fixed("dag_functional_anova", law, dag8_graph(), d=d, seed=seed, active_shape=None,
                  parents=_dag_parents(), profile=DAG8_PROFILE, options=law.describe(), wiring="reuse_dag8",
                  sigma_sq=sigma_sq)


def fourier_reuse_dag(d: int, seed: int, *, sigma_sq: float = SIGMA_SQ) -> Target:
    """The fixed reuse DAG with random Fourier local functions (Q = 8, sigma_omega = 0.85)."""
    law = RandomFourier(0.85, Q=8)
    return _fixed("dag_fourier_v2", law, dag8_graph(), d=d, seed=seed, active_shape=None,
                  parents=_dag_parents(), profile=DAG8_PROFILE, options=law.describe(), wiring="reuse_dag8",
                  sigma_sq=sigma_sq)


def sampled_dag(d: int, seed: int, *, sigma_sq: float = SIGMA_SQ) -> Target:
    """A sampled depth-4 DAG with additive-multiplicative local functions."""
    law = AdditiveMultiplicative(RHO, F_RANGE)
    placeholder = GraphSpec(RAND_A, [(0, 1)], (0,), (1.0,))      # replaced by the sampled graph
    return _fixed("rand_functional_anova", law, placeholder, d=d, seed=seed, active_shape=None, parents=(),
                  profile=(), options=law.describe(), wiring="random_layered_7", sigma_sq=sigma_sq, sampled=True)


#: family identifier -> (constructor, keyword arguments of the presets)
FAMILIES: dict[str, tuple[Callable[..., Target], dict[str, dict]]] = {
    "tree_functional_anova": (tree, {"depth3": {"profile": (4, 2, 1)}, "depth4": {"profile": (8, 4, 2, 1)}}),
    "tree_fourier": (fourier_tree, {"binary_depth3": {"profile": (4, 2, 1)}, "single": {"profile": (1,)}}),
    "node_functional_anova": (single, {"default": {}}),
    "dense_fourier": (high_arity, {"frozen_v1": {}}),
    "dense_functional_anova": (high_arity_stress, {"stress8": {}}),
    "flat_functional_anova": (flat, {"w4": {"w": 4}}),
    "dag_functional_anova": (reuse_dag, {"default": {}}),
    "dag_fourier_v2": (fourier_reuse_dag, {"default": {}}),
    "rand_functional_anova": (sampled_dag, {"default": {}}),
}


def draw(family: str, preset: str, d: int, seed: int, *, sigma_sq: float = SIGMA_SQ) -> Target:
    """Draw the target instance ``seed`` of a family preset at ambient dimension ``d``."""
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; available: {sorted(FAMILIES)}")
    constructor, presets = FAMILIES[family]
    if preset not in presets:
        raise KeyError(f"unknown preset {preset!r} of {family}; available: {sorted(presets)}")
    return constructor(int(d), int(seed), sigma_sq=float(sigma_sq), **presets[preset])
