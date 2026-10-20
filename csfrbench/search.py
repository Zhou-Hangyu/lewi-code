"""Hyperparameter search and model selection.

For each method, target instance and training-set size the benchmark trains
``T = 32`` configurations from a scrambled Sobol sequence over the frozen search
space of the method. Each run keeps its checkpoint with the lowest validation
NMSE. The configuration with the lowest validation NMSE is selected and
evaluated once on the test pool.

The seed of the sequence depends on the set of search spaces, the method, the
target family and the training-set size. It does not depend on the target
instance: every instance of a workload uses the same 32 configurations.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib import resources

import numpy as np
from scipy.stats import qmc

from . import suite
from .seeds import derive

PAD_LOG10 = 0.25 / 2        # 25% of the span in log10, half on each side
MIN_PAD_LOG10 = 0.05
MIN_CONDITIONS = 12         # a panel with fewer selected configurations cannot fix a span
MIN_LOG_SPAN = 1.0          # decades that the interval of such a panel covers at least


@lru_cache(maxsize=1)
def spaces() -> dict:
    """The frozen search spaces, by set and arm."""
    text = resources.files("csfrbench").joinpath("config/spaces.json").read_text()
    return json.loads(text)["sets"]


def space(space_set: str, arm: str) -> list[dict]:
    """The dimensions of the search space of an arm, in the order of the Sobol coordinates."""
    sets = spaces()
    if space_set not in sets or arm not in sets[space_set]["arms"]:
        raise KeyError(f"no frozen search space for arm {arm!r} in set {space_set!r}")
    return sets[space_set]["arms"][arm]["dimensions"]


def from_unit(u: float, low: float, high: float, log: bool) -> float:
    if log:
        return float(np.exp(np.log(low) + u * (np.log(high) - np.log(low))))
    return float(low + u * (high - low))


def sample(dimensions: list[dict], trials: int, seed: int) -> list[dict]:
    """``trials`` points of a scrambled Sobol sequence, mapped to the search space."""
    for dim in dimensions:
        if not (math.isfinite(dim["low"]) and math.isfinite(dim["high"])) or dim["low"] >= dim["high"]:
            raise ValueError(f"dimension {dim['name']}: the bounds must be finite with low < high")
        if dim["log"] and dim["low"] <= 0.0:
            raise ValueError(f"dimension {dim['name']}: a logarithmic scale needs low > 0")
    unit = qmc.Sobol(d=len(dimensions), scramble=True, seed=seed).random(trials)
    return [{dim["name"]: from_unit(float(u), dim["low"], dim["high"], dim["log"]) for dim, u in zip(dimensions, row)}
            for row in unit]


def trial_seed(workload: suite.Workload, arm: str, n: int) -> int:
    a = suite.ARMS[arm]
    return derive(suite.SEARCH_SEED, suite.SEARCH_KEY, workload.space_set_of(arm), a.trial_key, workload.family,
                  workload.preset, int(n))


def configurations(workload: suite.Workload, arm: str, n: int, trials: int = suite.TRIALS) -> list[dict]:
    """The hyperparameters of the ``trials`` runs of an arm at a training-set size.

    Each entry holds the searched values and the fixed arguments of the arm.
    """
    if arm not in workload.arms:
        raise KeyError(f"arm {arm!r} is not part of workload {workload.name!r} in the {workload.setting} setting")
    if int(n) not in workload.profile.sizes:
        raise ValueError(f"n = {n} is not on the grid {workload.profile.sizes}")
    fixed = dict(suite.ARMS[arm].fixed)
    drawn = sample(space(workload.space_set_of(arm), arm), trials, trial_seed(workload, arm, n))
    return [{**fixed, **hp} for hp in drawn]


def select(runs: list[dict], trials: int = suite.TRIALS) -> dict | None:
    """The selected run of one condition (method, target instance, training-set size).

    ``runs`` holds one record per trial with ``trial``, ``best_val_nmse`` and
    ``diverged``. A condition is selected only when all its ``trials`` runs are
    on record. A diverged run is never selected. Ties go to the lowest trial.
    """
    if len({int(r["trial"]) for r in runs}) < int(trials):
        return None
    valid = [r for r in runs if not int(r.get("diverged", 0)) and r.get("best_val_nmse") is not None
             and math.isfinite(float(r["best_val_nmse"]))]
    if not valid:
        return None
    return min(valid, key=lambda r: (float(r["best_val_nmse"]), int(r["trial"])))


def freeze(start: list[dict], selected: list[dict]) -> list[dict]:
    """The search space of a method from the configurations selected on a development panel.

    For each dimension the span of the selected values is expanded by 25% of
    its width in log10 (half on each side, at least 0.05 decades on each side).
    A hard boundary of the starting space is never crossed. A panel with fewer
    than 12 conditions gets an interval of at least one decade about its
    centre, clipped to the starting space.
    """
    out = []
    for dim in start:
        values = [float(hp[dim["name"]]) for hp in selected if hp.get(dim["name"]) is not None]
        if not values:
            out.append(dict(dim))
            continue
        lo, hi = min(values), max(values)
        small = len(selected) < MIN_CONDITIONS
        if dim["log"]:
            l10, h10 = math.log10(lo), math.log10(hi)
            pad = max((h10 - l10) * PAD_LOG10, MIN_PAD_LOG10)
            l10, h10 = l10 - pad, h10 + pad
            if small and (h10 - l10) < MIN_LOG_SPAN:
                mid = 0.5 * (l10 + h10)
                l10, h10 = mid - MIN_LOG_SPAN / 2, mid + MIN_LOG_SPAN / 2
                l10 = max(l10, math.log10(float(dim["low"])))
                h10 = min(h10, math.log10(float(dim["high"])))
            lo, hi = 10 ** l10, 10 ** h10
        else:
            pad = max((hi - lo) * PAD_LOG10, 1e-12)
            lo, hi = lo - pad, hi + pad
            if small:
                lo, hi = max(lo, float(dim["low"])), min(hi, float(dim["high"]))
        if "low" in dim["hard"]:
            lo = max(lo, float(dim["low"]))
        if "high" in dim["hard"]:
            hi = min(hi, float(dim["high"]))
        out.append({**dim, "low": float(lo), "high": float(hi)})
    return out


def boundary_occupancy(dimension: dict, values, margin: float = 0.10) -> dict:
    """The fraction of selected values within ``margin`` of a boundary, in the search coordinate."""
    lo, hi = float(dimension["low"]), float(dimension["high"])
    v = np.asarray(list(values), dtype=float)
    position = ((np.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))) if dimension["log"] else (v - lo) / (hi - lo)
    return {"at_low": float((position <= margin).mean()), "at_high": float((position >= 1.0 - margin).mean()),
            "median_position": float(np.median(position)), "n": int(v.size)}
