"""Validation and resolution of ArityBench run configurations.

Operates on plain mappings; ``campaign.py`` owns the YAML.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


class ConfigError(ValueError):
    """Raised when a YAML or resolved run configuration is invalid."""


def _mapping(value: Any, context: str) -> dict[str, Any]:
    """Coerce an optional mapping to a dict, or fail with context."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a mapping")
    return dict(value)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``; lists replace wholesale."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_task_catalog(catalog: Mapping[str, Any], source: str | Path) -> dict[str, Any]:
    """Merge the ``task_catalog`` defaults into every preset and validate each."""
    catalog = _mapping(catalog, "task_catalog")
    if int(catalog.get("version", 0)) != 1:
        raise ConfigError("task_catalog.version must be 1")
    defaults = _mapping(catalog.get("defaults"), "task_catalog defaults")
    tasks = _mapping(catalog.get("tasks"), "task_catalog tasks")
    if not tasks:
        raise ConfigError("task_catalog must define at least one task")

    normalized: dict[str, Any] = {}
    for family, family_cfg in tasks.items():
        family_cfg = _mapping(family_cfg, f"tasks.{family}")
        presets = _mapping(family_cfg.get("presets"), f"tasks.{family}.presets")
        if not presets:
            raise ConfigError(f"tasks.{family} must define at least one preset")
        normalized_presets = {}
        for preset_name, preset_cfg in presets.items():
            resolved = _deep_merge(defaults, _mapping(preset_cfg, f"{family}.{preset_name}"))
            _validate_task_cell(family, resolved, f"tasks.{family}.presets.{preset_name}",
                                require_d=False)
            normalized_presets[str(preset_name)] = resolved
        normalized[str(family)] = {"presets": normalized_presets}

    return {
        "version": 1,
        "tasks": normalized,
        "_config_path": str(source),
    }


def resolve_task_preset(
    catalog: Mapping[str, Any],
    family: str,
    preset: str,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one preset with overrides applied."""
    try:
        base = catalog["tasks"][family]["presets"][preset]
    except KeyError as exc:
        available = sorted(catalog.get("tasks", {}).get(family, {}).get("presets", {}))
        if family not in catalog.get("tasks", {}):
            raise ConfigError(
                f"unknown task family {family!r}; available: {sorted(catalog.get('tasks', {}))}"
            ) from exc
        raise ConfigError(
            f"unknown preset {preset!r} for {family}; available: {available}"
        ) from exc
    resolved = _deep_merge(base, _mapping(overrides, "task overrides"))
    _validate_task_cell(family, resolved, f"resolved task {family}/{preset}", require_d=False)
    return resolved


def resolve_cell(raw_spec: Mapping[str, Any], campaign: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Resolve ``(preset, cell)`` with ``d`` and ``sigma_sq`` injected."""
    raw = _mapping(raw_spec, "raw run spec")
    family = raw.get("task")
    selection = _mapping(campaign.get("tasks", {}).get(family), f"campaign task {family}")
    if not selection:
        raise ConfigError(f"task {family!r} is not enabled by this campaign")

    preset = str(raw.get("task_preset", selection["preset"]))
    overrides = _deep_merge(
        _mapping(selection.get("overrides"), "campaign task overrides"),
        _mapping(raw.get("cell"), "run cell overrides"),
    )
    cell = resolve_task_preset(campaign["task_catalog_resolved"], family, preset, overrides=overrides)

    d = raw.get("d", campaign.get("defaults", {}).get("d"))
    if d is None:
        raise ConfigError("run spec requires d")
    cell["d"] = int(d)

    sigma_sq = raw.get("sigma_sq", cell.get("sigma_sq"))
    if sigma_sq is None:
        raise ConfigError("sigma_sq must be defined by the task preset or run spec")
    cell["sigma_sq"] = float(sigma_sq)
    return preset, cell


def resolve_run_spec(raw_spec: Mapping[str, Any], campaign: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one self-contained run spec, ready to save as ``spec.json``."""
    raw = _mapping(raw_spec, "raw run spec")
    family = raw.get("task")
    method = raw.get("method")
    if not isinstance(family, str) or not isinstance(method, str):
        raise ConfigError("raw run spec requires string 'task' and 'method'")
    if method not in campaign["methods"]:
        raise ConfigError(f"method {method!r} is not enabled by this campaign")

    preset, cell = resolve_cell(raw, campaign)

    resolved = _deep_merge(campaign["defaults"], raw)
    # Per-workload profile overrides (amendment 5): a campaign may pin a
    # different architecture (and, where amended, training budget) for one
    # "family:preset" workload; everything else inherits the common profile.
    override = _mapping(campaign.get("workload_overrides", {}),
                        "workload_overrides").get(f"{family}:{preset}")
    if override:
        resolved = _deep_merge(resolved, _mapping(override,
                               f"workload_overrides['{family}:{preset}']"))
    resolved["task"] = family
    resolved["task_preset"] = preset
    resolved["cell"] = cell
    resolved["d"] = int(cell["d"])
    resolved["sigma_sq"] = float(cell["sigma_sq"])

    method_cfg = _mapping(campaign["methods"][method], f"methods.{method}")
    hp = {}
    hp.update(_mapping(campaign.get("optimizer", {}).get("fixed"), "optimizer.fixed"))
    hp.update(_mapping(method_cfg.get("fixed"), f"methods.{method}.fixed"))
    hp.update(_mapping(raw.get("hp"), "run hp"))
    resolved["hp"] = hp
    resolved["config_provenance"] = {
        "campaign": campaign["_config_path"],
        "task_catalog": campaign["task_catalog_resolved"]["_config_path"],
    }

    validate_resolved_run_spec(resolved)
    return resolved


def _validate_task_cell(family: str, cell: Mapping[str, Any], context: str,
                        *, require_d: bool) -> None:
    """Check one task cell against its family's declared contract."""
    from .registry import TASKS, load_builtin_registrations

    load_builtin_registrations()
    cls = TASKS.get(str(family))
    if cls is None:
        raise ConfigError(
            f"{context}: unknown task family {family!r}; registered: {sorted(TASKS)}"
        )
    try:
        cls.validate_cell(cell, require_d=require_d)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{context}: {exc}") from exc


def _validate_hp_keys(method: str, hp: Mapping[str, Any]) -> None:
    """Reject unknown hp keys for methods that declare what they consume.

    Every hp dict merges the searched dimensions, the campaign's fixed
    arguments, and the envelope ``lr`` into one free mapping; methods read
    fixed arguments with defaulting lookups, so a misspelled key would
    otherwise run silently at the code default (the historical
    ``p_open_init`` mislabeling failure class). Methods with
    ``hp_keys = None`` (legacy arms) keep the permissive behavior.
    """
    from .registry import METHODS, load_builtin_registrations

    load_builtin_registrations()
    cls = METHODS.get(str(method))
    declared = getattr(cls, "hp_keys", None) if cls is not None else None
    if declared is None:
        return
    unknown = set(hp) - set(declared) - {"lr"}
    if unknown:
        raise ConfigError(
            f"spec.hp for method {method!r} carries unknown keys "
            f"{sorted(unknown)}; declared keys are "
            f"{sorted(set(declared) | {'lr'})}"
        )


def validate_resolved_run_spec(spec: Mapping[str, Any]) -> None:
    """Validate the self-contained contract consumed by the run executor."""
    required = {
        "task", "method", "cell", "d", "sigma_sq", "function_seed", "dataset_seed", "n", "n_max",
        "hp", "arch", "training", "data", "dataset_dir",
    }
    missing = required - set(spec)
    if missing:
        raise ConfigError(f"resolved run spec missing keys: {sorted(missing)}")

    cell = _mapping(spec["cell"], "spec.cell")
    if int(cell.get("d", spec["d"])) != int(spec["d"]):
        raise ConfigError("spec.cell.d and spec.d disagree")
    if float(cell.get("sigma_sq", spec["sigma_sq"])) != float(spec["sigma_sq"]):
        raise ConfigError("spec.cell.sigma_sq and spec.sigma_sq disagree")
    _validate_task_cell(spec["task"], {**cell, "d": spec["d"], "sigma_sq": spec["sigma_sq"]},
                        "spec.cell", require_d=True)

    hp = _mapping(spec["hp"], "spec.hp")
    if "lr" not in hp or float(hp["lr"]) <= 0:
        raise ConfigError("spec.hp.lr must be positive")
    _validate_hp_keys(spec["method"], hp)
    if int(spec["n"]) < 1 or int(spec["n_max"]) < int(spec["n"]):
        raise ConfigError("require 1 <= n <= n_max")

    arch = _mapping(spec["arch"], "spec.arch")
    if arch.get("act", "relu") != "relu":
        raise ConfigError("ArityBench architectures are pinned to ReLU")
    if int(arch.get("width", 0)) <= 0 or int(arch.get("hidden", -1)) < 0:
        raise ConfigError("spec.arch requires positive width and nonnegative hidden")

    training = _mapping(spec["training"], "spec.training")
    scheduler = _mapping(training.get("scheduler"), "spec.training.scheduler")
    validation = _mapping(training.get("validation"), "spec.training.validation")
    early_stopping = _mapping(training.get("early_stopping"), "spec.training.early_stopping")

    optimizer = training.get("optimizer")
    if isinstance(optimizer, str):
        optimizer = {"type": optimizer}
    optimizer = _mapping(optimizer, "spec.training.optimizer")
    for path, supported, value in (
        ("optimizer.type", "adam", optimizer.get("type")),
        ("scheduler.type", "cosine", scheduler.get("type")),
        ("validation.metric", "nmse", validation.get("metric")),
    ):
        if value != supported:
            raise ConfigError(
                f"spec.training.{path}={value!r} is not implemented; only {supported!r} is"
            )
    betas = tuple(float(b) for b in optimizer.get("betas", (0.9, 0.999)))
    if len(betas) != 2 or not all(0.0 <= b < 1.0 for b in betas):
        raise ConfigError("spec.training.optimizer.betas must be two values in [0, 1)")
    if float(optimizer.get("eps", 1e-8)) <= 0.0:
        raise ConfigError("spec.training.optimizer.eps must be positive")
    if float(optimizer.get("weight_decay", 0.0)) != 0.0:
        raise ConfigError(
            "spec.training.optimizer.weight_decay is pinned to 0.0: weight-space "
            "regularization is the object of study"
        )

    for key in ("max_steps", "batch_size"):
        if int(training.get(key, 0)) <= 0:
            raise ConfigError(f"spec.training.{key} must be positive")
    if int(validation.get("eval_every", 0)) <= 0:
        raise ConfigError("spec.training.validation.eval_every must be positive")
    frac = float(scheduler.get("lr_final_frac", -1))
    if not 0.0 <= frac <= 1.0:
        raise ConfigError("spec.training.scheduler.lr_final_frac must lie in [0, 1]")
    if not isinstance(validation.get("restore_best_checkpoint"), bool):
        raise ConfigError("spec.training.validation.restore_best_checkpoint must be a boolean")

    if not isinstance(early_stopping.get("enabled"), bool):
        raise ConfigError("spec.training.early_stopping.enabled must be a boolean")
    if early_stopping["enabled"]:
        missing = {
            "min_frac_elapsed", "max_lr_frac", "patience_evals", "min_rel_improvement",
        } - set(early_stopping)
        if missing:
            raise ConfigError(
                f"spec.training.early_stopping is enabled but missing {sorted(missing)}"
            )

    device_spec = str(spec.get("device", "cuda"))
    if device_spec != "auto":
        try:
            torch.device(device_spec)
        except (TypeError, RuntimeError) as exc:
            raise ConfigError(f"invalid spec.device {device_spec!r}") from exc

    data = _mapping(spec["data"], "spec.data")
    for key in ("val_size", "development_test_size", "test_size",
                "test_audit_size", "diagnostic_size", "n_max"):
        if int(data.get(key, 0)) <= 0:
            raise ConfigError(f"spec.data.{key} must be positive")
    if not str(spec["dataset_dir"]).strip():
        raise ConfigError("spec.dataset_dir must name the saved master-dataset directory")