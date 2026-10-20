"""Campaign pipeline: calibrated spaces -> quasi-random HPO -> paired finals -> stats.

Implements the study protocol rules 3-8. This module is the whole control side:
campaign YAML -> validated campaign dict (Hydra compose), cell materialization
(one immutable ``spec.json`` per cell), and
the stage logic below. The worker side is ``csfrbench.train`` — one
``python -m csfrbench.train --run-dir <dir>`` call per manifest line, ``width``
of them at a time inside each job; the two sides share no state beyond
``spec.json`` down and the ``run.json`` shard back.

The search spaces are STATE, not just config: the campaign YAML's
``hpo.initial_spaces`` block holds only the broad, literature-informed STARTING
ranges, and ``pilot`` calibrates them in place into
``results/csfrbench/<campaign>/spaces.yaml`` (rule 3): quasi-random rounds on
representative development workloads (n_min / n_mid / n_max), widening any edge
where good configurations accumulate and contracting outer regions dominated by
poor or failed configurations, until competitive configurations sit comfortably
inside — then the spaces freeze. ``sample`` (rule 4) draws the main Sobol trials
from the frozen spaces; ``select`` (rules 5+8) freezes per-workload winners on
validation NMSE only and re-checks the boundary rule; ``final`` (rule 6) runs
paired held-out instances; ``analyze`` (rule 7) reports bootstrap-CI-primary
statistics with two-sided Wilcoxon + Holm against the campaign baseline.

Saturation is handled IN PLACE at every stage: a run measured under a wrong
space is invalid, so the space is widened where it stands (recorded in
``spaces.yaml`` history), the affected method's invalidated cells are deleted,
and the loop re-runs pilot -> sample -> select until nothing saturates. No new
campaign folders are created. Every re-sample draws fresh Sobol points (the
seed folds in the space generation), so a winner cannot sit at the same unit
coordinate by construction; a dimension that still saturates after
``SELECT_GATE_CAP`` in-place widenings stops the automation for human review.

Calibration thresholds (our operationalization of Choi et al. 1910.05446 /
Schmidt et al. 2007.01547, which prescribe the mechanism but no numbers):
top set = best max(3, 25%) completed trials per dev workload; widen an edge when
the best trial or >= half the top set sits within ``hpo.boundary_margin``
(default 0.1) of it in the sampling geometry; contract an outer-quartile band
only when every dev workload agrees it holds no top-set member and only
failed-or-below-median trials, with >= 3 completed band trials per dev workload
(or >= 6 pooled with every dev workload represented); contraction pulls the
edge to the top set padded by 0.15. Widening always wins over contraction on
the same edge.

CLI (runs root convention: ``results/csfrbench/<campaign>/runs``)::

    python -m csfrbench.pipeline pilot   <campaign> --runs R [--local | --dispatch --submit]
    python -m csfrbench.pipeline sample  <campaign> --runs R [--dispatch --submit]
    python -m csfrbench.pipeline run-pending --runs R [--limit N]   # local executor
    python -m csfrbench.pipeline select  <campaign> --runs R
    python -m csfrbench.pipeline final   <campaign> --runs R [--dispatch --submit]
    python -m csfrbench.pipeline analyze <campaign> --runs R

``pilot --local`` drives calibration rounds to convergence in-process; without
it, each invocation advances one step (materialize / analyze / adjust) so
cluster-dispatched rounds can land between calls.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shlex
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from hydra import compose, initialize_config_dir
from hydra.errors import MissingConfigException
from omegaconf import OmegaConf
from scipy.stats import qmc, wilcoxon

import runstore
from . import metrics
from .configs import (ConfigError, resolve_cell, resolve_run_spec,
                      resolve_task_catalog, validate_resolved_run_spec)
from .datasets import ensure_dataset
from .registry import available_methods, available_tasks, get_task
from .utils.seeds import derive

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
LIB_SRC = CONFIG_DIR.parents[1]              # src/ (`import csfrbench`, `import runstore`) or a bundle's implementation/


def _repo_root(start: Path) -> Path:
    """The first ancestor holding .git: the package may run from src/ or from a
    bundle snapshot (versions/v1.1/implementation), whose parent is not the repository."""
    for p in (start, *start.parents):
        if (p / ".git").exists():
            return p
    return start.parent


REPO = _repo_root(LIB_SRC)                   # repo root (cd target for dispatched jobs)

BASE_CONFIG = "base"
AXES = ("d", "function_seed", "dataset_seed")    # `n` lives in `data.n_values`

BUDGETS = (1, 4, 16, 32, 64)     # rule-8 robustness checkpoints
SELECT_GATE_CAP = 3              # in-place widenings per (method, hp, edge) before stopping
IMPUTED_VAL_NMSE = 1.0           # rule 5c: diverged/crashed trials keep their slot at the
                                 # mean-predictor reference value

TOP_FRACTION = 0.25              # top set = best max(3, ceil(25%)) completed trials
OUTER_BAND = 0.25                # contraction looks at the outer quarter per edge
CONTRACT_MIN_PER_WORKLOAD = 3    # completed band trials per dev workload to contract ...
CONTRACT_MIN_POOLED = 6          # ... or this many pooled, every workload represented
CONTRACT_PAD = 0.15              # unit-space padding kept beyond the top set


# ---------------------------------------------------------------- campaign --
# A campaign is one file: `configs/<name>.yaml` inheriting `configs/base.yaml`.
# A scalar hp (shared `hp:` block, or a method's own entry overriding it) is a
# fixed constant merged into every spec; starting search ranges live in the
# `hpo.initial_spaces` block and are calibrated in place by `pilot`.

def available_campaigns(config_dir: str | Path | None = None) -> tuple[str, ...]:
    """Every YAML in ``config_dir`` except the shared base."""
    config_dir = Path(config_dir or CONFIG_DIR)
    return tuple(sorted(p.stem for p in config_dir.glob("*.yaml") if p.stem != BASE_CONFIG))


def compose_campaign(
    name: str,
    overrides: list[str] | None = None,
    config_dir: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Compose one campaign file plus the ``base`` it inherits."""
    config_dir = Path(config_dir or CONFIG_DIR).resolve()
    name = str(name)
    if name == BASE_CONFIG:
        raise ConfigError(f"{name!r} is shared substrate, not a campaign; "
                          f"available: {list(available_campaigns(config_dir))}")
    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name=name, overrides=list(overrides or []))
    except MissingConfigException as exc:
        raise ConfigError(
            f"could not compose campaign {name!r} from {config_dir}; "
            f"campaigns available: {list(available_campaigns(config_dir))} ({exc})"
        ) from exc
    return OmegaConf.to_container(cfg, resolve=True), config_dir


def _require(composed: Mapping[str, Any], key: str, name: str) -> Any:
    """Fetch a required campaign key, or fail naming where to set it."""
    value = composed.get(key)
    if value is None:
        raise ConfigError(
            f"campaign {name!r} is missing {key!r}: set it in configs/{name}.yaml, "
            "or inherit it with `defaults: [base, _self_]`"
        )
    return value


def _axis(composed: Mapping[str, Any], key: str, name: str) -> list[Any]:
    """A bare scalar is accepted as a one-point axis."""
    value = _require(composed, key, name)
    values = list(value) if isinstance(value, list) else [value]
    if not values:
        raise ConfigError(f"campaign {name!r}: axis {key!r} is empty")
    return values


def _split_hp(block: Mapping[str, Any] | None, context: str) -> tuple[dict, dict]:
    """``{hp: value}`` -> (swept axes, fixed constants)."""
    axes: dict[str, list] = {}
    fixed: dict[str, Any] = {}
    for key, value in dict(block or {}).items():
        if isinstance(value, list):
            if not value:
                raise ConfigError(f"{context}: hp axis {key!r} is empty")
            axes[key] = list(value)
        else:
            fixed[key] = value
    return axes, fixed


def _pick(catalog: Mapping[str, Any] | None, key: str, what: str) -> dict[str, Any]:
    """Pick one entry from a base.yaml catalog by name."""
    entries = dict(catalog or {})
    if key not in entries:
        raise ConfigError(f"unknown {what} {key!r}; base.yaml defines {sorted(entries)}")
    return dict(entries[key])


def build_campaign_dict(
    composed: Mapping[str, Any],
    config_dir: str | Path,
    name: str,
) -> dict[str, Any]:
    """Assemble the campaign dict that ``configs.resolve_run_spec`` consumes."""
    config_dir = Path(config_dir)

    size = str(_require(composed, "model", name))
    arch = {"act": _require(composed, "model_act", name),
            **_pick(composed.get("model_sizes"), size, "model size")}

    profile = str(_require(composed, "dispatch", name))
    dispatch_cfg = _pick(composed.get("dispatch_profiles"), profile, "dispatch profile")

    tasks = dict(_require(composed, "tasks", name))
    unregistered = [t for t in tasks if t not in available_tasks()]
    if unregistered:
        raise ConfigError(f"campaign {name!r}: unregistered task families {unregistered}; "
                          f"registered: {list(available_tasks())}")

    methods = dict(_require(composed, "methods", name))
    unregistered = [m for m in methods if m not in available_methods()]
    if unregistered:
        raise ConfigError(f"campaign {name!r}: unregistered methods {unregistered}; "
                          f"registered: {list(available_methods())}")

    _, shared_fixed = _split_hp(composed.get("hp"), f"campaign {name!r} hp")
    method_fixed: dict[str, dict[str, Any]] = {}
    for method, block in methods.items():
        axes, fixed = _split_hp(block, f"campaign {name!r} methods.{method}")
        method_fixed[method] = {**{k: v for k, v in shared_fixed.items() if k not in axes}, **fixed}

    data = dict(_require(composed, "data", name))
    n_values = [int(x) for x in _axis({"n": data.get("n_values")}, "n", name)]
    if max(n_values) > int(data["n_max"]):
        raise ConfigError(
            f"campaign {name!r}: data.n_values max {max(n_values)} exceeds n_max {data['n_max']}"
        )

    target_overrides = {
        "sigma_sq": float(data["sigma_sq"]),
        "target_variance": float(data["target_variance"]),
    }

    task_catalog = resolve_task_catalog(_require(composed, "task_catalog", name),
                                        (config_dir / f"{BASE_CONFIG}.yaml").resolve())

    overrides = dict(composed.get("workload_overrides") or {})
    for key, block in overrides.items():
        if ":" not in key or not isinstance(block, dict):
            raise ConfigError(f"campaign {name!r}: workload_overrides[{key!r}] "
                              "must map 'family:preset' to a dict")

    return {
        "version": 1,
        "name": name,
        "model": size,
        "workload_overrides": overrides,
        "task_catalog_resolved": task_catalog,
        "_config_path": str(config_dir / f"{name}.yaml"),
        "defaults": {
            "arch": arch,
            "training": _require(composed, "training", name),
            "data": data,
            "n_max": int(data["n_max"]),
            "test_evaluation": _require(composed, "test_evaluation", name),
            "retain": _require(composed, "retain", name),
            "connectivity": _require(composed, "connectivity", name),
            "device": _require(composed, "device", name),
            "torch_threads": _require(composed, "torch_threads", name),
        },
        "methods": {m: {"fixed": method_fixed[m]} for m in methods},
        "tasks": {family: {"preset": preset, "overrides": target_overrides}
                  for family, preset in tasks.items()},
        "dispatch_profile": profile,
        "dispatch": dispatch_cfg,
        "_axes": {**{key: _axis(composed, key, name) for key in AXES},
                  "n": n_values, "tasks": list(tasks), "methods": list(methods)},
    }


def ensure_campaign_datasets(
    campaign: Mapping[str, Any],
    cells: list[dict[str, Any]],
    datasets_root: str | Path,
    *,
    dry: bool = False,
) -> int:
    """Build each master dataset this campaign needs and stamp `dataset_dir` on the cells."""
    data = campaign["defaults"]["data"]
    directories: dict[tuple, Path] = {}
    for raw in cells:
        key = (raw["task"], raw["d"], raw["function_seed"], raw["dataset_seed"])
        if key not in directories:
            _, cell = resolve_cell(raw, campaign)
            target = get_task(raw["task"]).draw(cell, int(raw["function_seed"]))
            directories[key] = ensure_dataset(
                datasets_root, target, raw["dataset_seed"],
                n_max=int(campaign["defaults"]["n_max"]),
                val_size=int(data["val_size"]),
                dev_test_size=int(data["development_test_size"]),
                test_size=int(data["test_size"]),
                test_audit_size=int(data["test_audit_size"]),
                diag_size=int(data["diagnostic_size"]),
                dry=dry,
            )
        raw["dataset_dir"] = str(directories[key])
    return len(directories)


# ---------------------------------------------------------------- dispatch --

# The scheduler dispatch of the campaigns (job scripts, queue accounting) is not part of this archive.


def _load(name: str, overrides: list[str] | None):
    composed, config_dir = compose_campaign(name, overrides)
    return composed, build_campaign_dict(composed, config_dir, name)


def _workloads(campaign: dict) -> list[dict[str, Any]]:
    axes = campaign["_axes"]
    return [{"task": task, "preset": campaign["tasks"][task]["preset"],
             "d": int(d), "n": int(n)}
            for task in axes["tasks"] for d in axes["d"] for n in axes["n"]]


def _dev_workloads(campaign: dict) -> list[dict[str, Any]]:
    """Representative development workloads for calibration: n_min / n_mid / n_max."""
    axes = campaign["_axes"]
    ns = sorted({int(n) for n in axes["n"]})
    dev = sorted({ns[0], ns[len(ns) // 2], ns[-1]})
    return [{"task": task, "preset": campaign["tasks"][task]["preset"],
             "d": int(d), "n": n}
            for task in axes["tasks"] for d in axes["d"] for n in dev]


def _hpo_conf(composed: dict, name: str) -> dict[str, Any]:
    conf = composed.get("hpo")
    if not isinstance(conf, dict) or not conf.get("initial_spaces"):
        raise ConfigError(f"campaign {name!r} has no hpo.initial_spaces block; "
                          "see configs/smoke.yaml")
    return {
        "trials": int(conf.get("trials", 64)),
        "pilot_trials": int(conf.get("pilot_trials", 16)),
        "pilot_max_rounds": int(conf.get("pilot_max_rounds", 5)),
        "seed": int(conf.get("seed", 0)),
        "function_seed": int(conf.get("function_seed", 0)),
        "dataset_seed": int(conf.get("dataset_seed", 0)),
        "boundary_margin": float(conf.get("boundary_margin", 0.1)),
        "spaces": dict(conf["initial_spaces"]),
    }


# ------------------------------------------------------------------- spaces --

def _parse_space(space: dict, context: str) -> tuple[dict[str, Any], list[tuple]]:
    """Split a space into fixed scalars and (key, low, high, log, hard) dimensions.

    ``hard`` marks bounds that are theory-fixed: the dimension is still
    searched, but a hard edge is never widened, contracted, or gated.
    ``hard: true`` fixes both edges; ``hard: [low]`` fixes one (e.g. a penalty
    range whose low edge stands in for "off" by design).
    """
    fixed: dict[str, Any] = {}
    dims: list[tuple[str, float, float, bool, frozenset]] = []
    for key, value in dict(space).items():
        if isinstance(value, dict):
            low, high = float(value["low"]), float(value["high"])
            log = bool(value.get("log", False))
            if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
                raise ConfigError(f"{context}.{key}: require finite low < high")
            if log and low <= 0.0:
                raise ConfigError(f"{context}.{key}: log scale requires low > 0")
            hard = value.get("hard", False)
            if isinstance(hard, bool):
                hard_edges = frozenset(("low", "high")) if hard else frozenset()
            else:
                hard_edges = frozenset(str(edge) for edge in hard)
            if not hard_edges <= {"low", "high"}:
                raise ConfigError(f"{context}.{key}: hard must be true or a subset of [low, high]")
            dims.append((key, low, high, log, hard_edges))
        else:
            fixed[key] = value
    if not dims:
        raise ConfigError(f"{context}: no searchable dimensions (all values are scalars)")
    return fixed, dims


def _unit_position(value: float, low: float, high: float, log: bool) -> float:
    """A value's coordinate in the space's sampling geometry, 0 at low, 1 at high."""
    if log:
        return (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    return (value - low) / (high - low)


def _from_unit(u: float, low: float, high: float, log: bool) -> float:
    if log:
        return float(np.exp(np.log(low) + u * (np.log(high) - np.log(low))))
    return float(low + u * (high - low))


def _widened(low: float, high: float, log: bool, edge: str) -> tuple[float, float]:
    """Bounds after widening one edge: x10 per log edge, +50% span linear."""
    if log:
        return (low / 10.0, high) if edge == "low" else (low, high * 10.0)
    span = high - low
    return (low - 0.5 * span, high) if edge == "low" else (low, high + 0.5 * span)


def _sample_space(space: dict, trials: int, seed: int, context: str) -> list[dict[str, Any]]:
    """``trials`` scrambled-Sobol points; scalars in the space pass through fixed."""
    fixed, dims = _parse_space(space, context)
    unit = qmc.Sobol(d=len(dims), scramble=True, seed=seed).random(trials)
    configurations = []
    for row in unit:
        hp = dict(fixed)
        for (key, low, high, log, _hard), u in zip(dims, row):
            hp[key] = _from_unit(float(u), low, high, log)
        configurations.append(hp)
    return configurations
