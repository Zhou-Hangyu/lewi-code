"""Training and single-run execution — the worker side of the pipeline.

``python -m csfrbench.train --run-dir <dir>`` trains the one cell described by
the immutable ``<dir>/spec.json`` and writes the ``run.json`` shard back; the
control side (``csfrbench.pipeline``) decides which cells exist and dispatches
one such call per SLURM manifest line.
"""
from __future__ import annotations

import hashlib
import json
import itertools
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics
from .configs import validate_resolved_run_spec
from .datasets import iterate_batches, load_splits, make_loader
from .registry import get_method, get_task
from .utils.seeds import derive


def _resolve_device(requested=None) -> torch.device:
    """Resolve a run device, defaulting strictly to CUDA.

    Use ``device: auto`` to permit CPU fallback or ``device: cpu`` to force a
    CPU run. The strict CUDA default prevents an incorrectly dispatched GPU run
    from silently consuming CPU resources.
    """
    requested = "cuda" if requested is None else str(requested).lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Check that the SLURM job requested a GPU and that this environment "
            "contains a CUDA-enabled PyTorch build."
        )
    return device


EVAL_BATCH_SIZE = 8192

# Which held-out pools each policy may touch. The frozen final pool stays sealed until
# the method and search protocol are frozen; `test_audit` is an independent pool of the
# same size, reported alongside `test` as a fixed-realization sensitivity check.
EVAL_POOLS = {
    "forbidden": (),
    "development": ("development_test",),
    "final": ("development_test", "test", "test_audit"),
}


def _interval_train_nmse(interval) -> float | None:
    """Training NMSE over the minibatches since the last evaluation, or None if idle."""
    count = interval["count"]
    if count == 0:
        return None
    mean = interval["sum"] / count
    variance = max(0.0, interval["sumsq"] / count - mean * mean)
    value = (interval["sse"] / count) / (variance + 1e-12)
    return value if math.isfinite(value) else None


def _pointwise_sq_errors(model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Per-point squared errors and targets over a loader, in evaluation order.

    ``metrics.nmse``'s value equals ``errors.sum() / (sq.sum() - sum**2/m)``
    on these arrays; audits recompute it on arbitrary index subsets.
    """
    model_device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    errors, targets = [], []
    try:
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(model_device, non_blocking=True)
                y_batch = y_batch.to(model_device, non_blocking=True)
                prediction = model(X_batch)
                if prediction.shape != y_batch.shape:
                    raise ValueError(
                        f"prediction shape {tuple(prediction.shape)} does not "
                        f"match target {tuple(y_batch.shape)}"
                    )
                y64 = y_batch.to(torch.float64)
                errors.append((prediction.to(torch.float64) - y64).square()
                              .reshape(-1).cpu().numpy())
                targets.append(y64.reshape(-1).cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(errors), np.concatenate(targets)


def _nmse_from_errors(errors: np.ndarray, targets: np.ndarray) -> float:
    denominator = targets @ targets - targets.sum() ** 2 / targets.size
    return float(errors.sum() / denominator)


def train(
    model,
    method,
    hp,
    train_dataset,
    val_dataset,
    tconf,
    *,
    stochastic_seed: int,
    batch_seed: int,
    eval_seed: int | None = None,
    record_val_errors: bool = False,
):
    """Train under ``tconf`` (the spec's ``training`` block) and select a checkpoint.

    Mutates ``model`` in place and returns it in eval mode. Test data are never
    accessed here. A nonfinite gradient norm or loss ends the run as ``diverged``
    before the offending update is applied, rather than raising.

    Every evaluation loader carries its own generator (``eval_seed``; the
    batch seed stands in when a caller passes none): iterating a DataLoader
    without one draws its base seed from the global torch RNG, which used to
    shift the training-noise stream of stochastic-forward methods by the
    evaluation cadence (CSFRBench v1.0).
    """
    scheduler_conf = tconf.get("scheduler") or {}
    validation_conf = tconf.get("validation") or {}
    optimizer_conf = tconf.get("optimizer", "adam")
    if isinstance(optimizer_conf, str):
        optimizer_conf = {"type": optimizer_conf}
    for path, supported, value in (
        ("optimizer.type", "adam", optimizer_conf.get("type", "adam")),
        ("scheduler.type", "cosine", scheduler_conf.get("type", "cosine")),
        ("validation.metric", "nmse", validation_conf.get("metric", "nmse")),
    ):
        if value != supported:
            raise ValueError(f"training.{path}={value!r} is not implemented; only {supported!r} is")

    max_steps = int(tconf["max_steps"])
    batch_size = int(tconf["batch_size"])
    eval_every = int(validation_conf["eval_every"])
    restore_best = bool(validation_conf["restore_best_checkpoint"])
    learning_rate = float(hp["lr"])
    lr_final_frac = float(scheduler_conf["lr_final_frac"])

    if max_steps <= 0 or eval_every <= 0 or batch_size <= 0:
        raise ValueError("max_steps, eval_every, and batch_size must be positive")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("hp['lr'] must be finite and positive")
    if not 0.0 <= lr_final_frac <= 1.0:
        raise ValueError("lr_final_frac must lie in [0, 1]")

    n_train = len(train_dataset)
    if n_train == 0 or len(val_dataset) == 0:
        raise ValueError("training and validation sets must be nonempty")
    if batch_size > n_train:
        raise ValueError(
            f"batch_size={batch_size} exceeds n_train={n_train}; choose a batch size "
            "no larger than the smallest training split"
        )

    device = next(model.parameters()).device

    # Reset after build(): training randomness is independent of construction.
    torch.manual_seed(int(stochastic_seed))

    adam_betas = tuple(float(b) for b in optimizer_conf.get("betas", (0.9, 0.999)))
    adam_eps = float(optimizer_conf.get("eps", 1e-8))
    if float(optimizer_conf.get("weight_decay", 0.0)) != 0.0:
        raise ValueError(
            "training.optimizer.weight_decay is pinned to 0.0: weight-space "
            "regularization is the object of study"
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=adam_betas,
        eps=adam_eps,
        weight_decay=0.0,
    )
    if any(group.get("weight_decay", 0.0) != 0.0 for group in optimizer.param_groups):
        raise RuntimeError("Adam weight decay must be exactly 0.0")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=learning_rate * lr_final_frac,
    )

    grad_clip = tconf.get("grad_clip")
    if grad_clip is not None:
        grad_clip = float(grad_clip)
        if grad_clip <= 0.0 or not math.isfinite(grad_clip):
            raise ValueError("grad_clip must be finite and positive, or None")

    early_stop = tconf.get("early_stopping") or {}
    if not isinstance(early_stop, dict):
        raise TypeError("early_stopping must be a mapping or null")
    if not early_stop.get("enabled", False):
        early_stop = {}
    if early_stop:
        required = {"min_frac_elapsed", "max_lr_frac", "patience_evals"}
        missing = required - set(early_stop)
        if missing:
            raise ValueError(f"early_stopping is enabled but missing keys: {sorted(missing)}")
        if not 0.0 <= float(early_stop["min_frac_elapsed"]) <= 1.0:
            raise ValueError("early_stopping.min_frac_elapsed must lie in [0, 1]")
        if not 0.0 <= float(early_stop["max_lr_frac"]) <= 1.0:
            raise ValueError("early_stopping.max_lr_frac must lie in [0, 1]")
        if int(early_stop["patience_evals"]) <= 0:
            raise ValueError("early_stopping.patience_evals must be positive")
    min_rel_improvement = float(early_stop.get("min_rel_improvement", 1e-4))
    if not 0.0 <= min_rel_improvement < 1.0:
        raise ValueError("early_stopping.min_rel_improvement must lie in [0, 1)")

    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(int(batch_seed))
    train_loader = make_loader(
        train_dataset, batch_size, shuffle=True, drop_last=True, generator=batch_generator
    )
    eval_generator = torch.Generator(device="cpu")
    eval_generator.manual_seed(int(batch_seed if eval_seed is None else eval_seed))
    val_loader = make_loader(val_dataset, EVAL_BATCH_SIZE, generator=eval_generator)

    best_val_nmse = float("inf")
    best_step = 0
    best_state = None
    patience_best = float("inf")
    stale_evals = 0
    val_curve: list[tuple[int, float, float | None]] = []
    val_error_rows: list[np.ndarray] = []
    val_targets: np.ndarray | None = None
    interval = {"sse": 0.0, "sum": 0.0, "sumsq": 0.0, "count": 0}
    steps_run = 0
    num_clipped = 0
    grad_norm_max = None
    diverged_step = None
    diverged_reason = None

    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    batches = enumerate(itertools.islice(iterate_batches(train_loader), max_steps), start=1)
    for step, (input_batch, target_batch) in batches:
        optimizer.zero_grad(set_to_none=True)

        input_batch = input_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)
        prediction = model(input_batch)
        if prediction.shape != target_batch.shape:
            raise ValueError(
                f"prediction shape {tuple(prediction.shape)} does not match target "
                f"shape {tuple(target_batch.shape)} at step {step}"
            )

        task_loss = F.mse_loss(prediction, target_batch)
        with torch.no_grad():
            targets = target_batch.to(torch.float64)
            interval["sse"] += task_loss.detach().item() * target_batch.numel()
            interval["sum"] += targets.sum().item()
            interval["sumsq"] += targets.square().sum().item()
            interval["count"] += target_batch.numel()
        penalty = method.penalty(model, hp)
        if penalty is None:
            loss = task_loss
        else:
            if not torch.is_tensor(penalty):
                raise TypeError("method.penalty() must return None or a scalar tensor")
            if penalty.ndim != 0:
                raise ValueError(
                    f"method.penalty() must return a scalar; received {tuple(penalty.shape)}"
                )
            if penalty.device != task_loss.device:
                raise ValueError("penalty and task loss must be on the same device")
            loss = task_loss + penalty

        if not bool(torch.isfinite(loss).item()):
            diverged_step, diverged_reason = step, "nonfinite_loss"
            print(
                f"DIVERGED: nonfinite loss at step {step}: "
                f"task={task_loss.detach().item()}, total={loss.detach().item()}",
                flush=True,
            )
            break

        loss.backward()

        try:
            grad_norm = torch.nn.utils.get_total_norm(
                [p.grad for p in model.parameters() if p.grad is not None],
                norm_type=2.0,
                error_if_nonfinite=True,
            )
        except RuntimeError as exc:
            diverged_step, diverged_reason = step, "nonfinite_grad_norm"
            print(f"DIVERGED: nonfinite gradient norm at step {step}: {exc}", flush=True)
            break

        grad_norm_max = (
            grad_norm if grad_norm_max is None else torch.maximum(grad_norm_max, grad_norm)
        )
        if grad_clip is not None:
            torch.nn.utils.clip_grads_with_norm_(model.parameters(), grad_clip, grad_norm)
            num_clipped += int(float(grad_norm) > grad_clip)

        optimizer.step()
        method.after_optimizer_step(model, optimizer, step=step, total_steps=max_steps)
        scheduler.step()
        steps_run = step

        if step % eval_every == 0 or step == max_steps:
            if record_val_errors:
                point_errors, point_targets = _pointwise_sq_errors(model, val_loader)
                if val_targets is None:
                    val_targets = point_targets
                val_error_rows.append(point_errors.astype(np.float32))
                val_nmse = _nmse_from_errors(point_errors, point_targets)
            else:
                val_nmse = float(metrics.nmse(model, val_loader))
            if not math.isfinite(val_nmse):
                raise FloatingPointError(
                    f"nonfinite validation NMSE at step {step}: {val_nmse}"
                )
            val_curve.append((step, val_nmse, _interval_train_nmse(interval)))
            interval = {"sse": 0.0, "sum": 0.0, "sumsq": 0.0, "count": 0}

            if val_nmse < patience_best * (1.0 - min_rel_improvement):
                patience_best = val_nmse
                stale_evals = 0
            else:
                stale_evals += 1

            if val_nmse < best_val_nmse:
                best_val_nmse = val_nmse
                best_step = step
                if restore_best:
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    }

            if (
                early_stop
                and step >= float(early_stop["min_frac_elapsed"]) * max_steps
                and scheduler.get_last_lr()[0]
                < float(early_stop["max_lr_frac"]) * learning_rate
                and stale_evals >= int(early_stop["patience_evals"])
            ):
                break

    if diverged_step is not None and not val_curve:
        val_nmse = float(metrics.nmse(model, val_loader))
        if math.isfinite(val_nmse):
            val_curve.append((steps_run, val_nmse, _interval_train_nmse(interval)))
            best_val_nmse, best_step = val_nmse, steps_run
            if restore_best:
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }

    if not val_curve and diverged_step is None:
        raise RuntimeError("no validation evaluation was performed")

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wallclock_s = time.perf_counter() - start_time

    result = {
        "best_val_nmse": best_val_nmse if math.isfinite(best_val_nmse) else None,
        "best_step": best_step,
        "steps_run": steps_run,
        "examples_seen": steps_run * batch_size,
        "best_near_cap": int(best_step >= 0.9 * max_steps),
        "exhausted_budget": int(steps_run == max_steps),
        "val_curve": val_curve,
        "wallclock_s": wallclock_s,
        "diverged": int(diverged_step is not None),
        "grad_norm_max": float(grad_norm_max) if grad_norm_max is not None else 0.0,
    }
    if diverged_step is not None:
        result["diverged_step"] = int(diverged_step)
        result["diverged_reason"] = diverged_reason
    if grad_clip is not None:
        result["grad_clip_fraction"] = num_clipped / max(steps_run, 1)
    if record_val_errors and val_error_rows:
        result["val_sq_errors"] = np.stack(val_error_rows)
        result["val_targets"] = val_targets.astype(np.float32)
    return result


def execute_experiment_run(spec: dict, run_dir):
    """Execute, evaluate, and persist one self-contained benchmark run."""
    validate_resolved_run_spec(spec)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(spec.get("torch_threads", 4)))
    device = _resolve_device(spec.get("device"))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    run_start = time.perf_counter()

    test_policy = spec.get("test_evaluation", "forbidden")
    if test_policy not in EVAL_POOLS:
        raise ValueError(f"test_evaluation must be one of {sorted(EVAL_POOLS)}")

    task_family = get_task(spec["task"])
    target = task_family.draw(spec["cell"], int(spec["function_seed"]))
    method = get_method(spec["method"])

    pair_key = (target.fingerprint, int(spec["n"]), spec["dataset_seed"])
    seed_replicate = int(spec.get("seed_replicate", 0))
    if seed_replicate:
        # Replicate 0 keeps the historical hashes; nonzero draws fresh, paired
        # training randomness at a fixed instance (v2 seed-decomposition audit).
        pair_key = (*pair_key, seed_replicate)
    seeds = {
        "initialization": derive(*pair_key, "initialization"),
        "stochastic": derive(*pair_key, "stochastic"),
        "batch_order": derive(*pair_key, "batch_order"),
        "gate_init": derive(*pair_key, "gate_init"),
        "evaluation": derive(*pair_key, "evaluation"),
    }

    splits = load_splits(spec["dataset_dir"], n=int(spec["n"]),
                         verify_hashes=bool(spec["data"].get("verify_hashes", True)))
    if splits["target_fingerprint"] != target.fingerprint:
        raise RuntimeError(
            f"dataset {spec['dataset_dir']} was built for target "
            f"{splits['target_fingerprint']}, but this spec reproduces {target.fingerprint}"
        )

    train_dataset = splits["training"].to(device)
    val_dataset = splits["validation"].to(device)

    if int(spec["training"]["batch_size"]) > int(splits["n_train"]):
        raise ValueError(
            f"batch_size={spec['training']['batch_size']} exceeds n_train={splits['n_train']}"
        )

    model_config = {
        "width": int(spec["arch"]["width"]),
        "hidden": int(spec["arch"]["hidden"]),
        "hp": spec["hp"],
        # Gates draw from a stream distinct from the backbone weights so paired
        # dense/gated models share raw weights (consumed by l0hc gate seeding).
        "auxiliary_init_seed": int(seeds["gate_init"]),
    }
    torch.manual_seed(int(seeds["initialization"]))
    model = method.build(model_config, target).to(device)

    retain = spec.get("retain") or {}
    init_checkpoint_path = None
    if retain.get("init_pt", True):
        # The exact pre-training initialization: the probe baseline the
        # evaluation protocol requires (A^init), saved before any update.
        init_checkpoint_path = run_dir / "init.pt"
        _atomic_torch_save(model.state_dict(), init_checkpoint_path)

    training_metrics = train(
        model,
        method,
        spec["hp"],
        train_dataset,
        val_dataset,
        spec["training"],
        stochastic_seed=int(seeds["stochastic"]),
        batch_seed=int(seeds["batch_order"]),
        eval_seed=int(seeds["evaluation"]),
        record_val_errors=bool(spec.get("audit_val_errors", False)),
    )
    if model.training:
        raise RuntimeError("train() must return the selected model in evaluation mode")

    diverged = bool(training_metrics.get("diverged"))

    test_metrics = {}
    if not diverged:
        eval_generator = torch.Generator(device="cpu")
        eval_generator.manual_seed(int(seeds["evaluation"]))
        for pool in EVAL_POOLS[test_policy]:
            pool_errors, pool_targets = _pointwise_sq_errors(
                model, make_loader(splits[pool], EVAL_BATCH_SIZE,
                                   generator=eval_generator)
            )
            test_metrics[f"{pool}_nmse"] = _nmse_from_errors(pool_errors, pool_targets)
            # Monte-Carlo profile of the pool estimate (v2 test-pool re-check):
            # NMSE on nested prefixes of the fixed evaluation order.
            for prefix in (8192, 16384, 32768):
                if prefix < pool_targets.size:
                    test_metrics[f"{pool}_nmse_m{prefix}"] = _nmse_from_errors(
                        pool_errors[:prefix], pool_targets[:prefix]
                    )

    shared_diagnostics: dict = {}
    method_diagnostics: dict = {}
    if not diverged:
        shared_diagnostics = metrics.structure(method, model) if method.weight_graph else {}
        method_diagnostics = _validate_scalar_metrics(
            method.extra_diagnostics(model), source="method.extra_diagnostics"
        )
        duplicate_keys = set(shared_diagnostics) & set(method_diagnostics)
        if duplicate_keys:
            raise ValueError(f"duplicate diagnostic keys: {sorted(duplicate_keys)}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    compute_wallclock_s = time.perf_counter() - run_start

    scalar_training_metrics = {
        key: value for key, value in training_metrics.items()
        if key not in ("val_curve", "val_sq_errors", "val_targets")
    }

    gpu_metrics = {}
    if device.type == "cuda":
        gpu_metrics = {
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_runtime": torch.version.cuda,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }

    run_metrics = {
        "device": str(device),
        **gpu_metrics,
        "n_train": int(splits["n_train"]),
        "n_val": int(splits["n_val"]),
        "target_id": splits["target_id"],
        "max_steps": int(spec["training"]["max_steps"]),
        "batch_size": int(spec["training"]["batch_size"]),
        **test_metrics,
        **scalar_training_metrics,
        **shared_diagnostics,
        **method_diagnostics,
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "n_trainable_params": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "compute_wallclock_s": compute_wallclock_s,
    }
    if diverged:
        run_metrics = {
            key: (None if isinstance(value, float) and not math.isfinite(value) else value)
            for key, value in run_metrics.items()
        }

    artifacts = []
    if init_checkpoint_path is not None:
        artifacts.append(_artifact(init_checkpoint_path, "init_checkpoint", "pt"))

    if "val_sq_errors" in training_metrics:
        # Per-checkpoint per-point validation errors: the v2 selection audits
        # (trial budget, validation size) re-run the whole selection on index
        # subsets of this array. Steps column pairs rows to val_curve entries.
        errors_path = run_dir / "val_errors.npz"
        _atomic_numpy_savez(
            errors_path,
            sq_errors=training_metrics["val_sq_errors"],
            targets=training_metrics["val_targets"],
            steps=np.asarray([s for s, _v, _t in training_metrics["val_curve"]],
                             dtype=np.int64),
        )
        artifacts.append(_artifact(errors_path, "val_errors", "npz"))

    if retain.get("best_pt", True):
        # The selected (best-validation) checkpoint the evaluation consumes.
        checkpoint_path = run_dir / "best.pt"
        _atomic_torch_save(model.state_dict(), checkpoint_path)
        artifacts.append(_artifact(checkpoint_path, "checkpoint", "pt"))

    if spec.get("connectivity"):
        _emit_connectivity(
            run_dir,
            model,
            target,
            splits["diagnostic"].tensors[0],
            {"n_train": int(splits["n_train"]), **test_metrics},
            artifacts,
            fmt=spec.get("connectivity_format", "pdf"),
        )

    # The validation curve travels inside the shard (one record per run);
    # ingest exports it as parquet list columns.
    val_curve = [[int(step), float(value), None if train_nmse is None else float(train_nmse)]
                 for step, value, train_nmse in training_metrics["val_curve"]]

    _write_shard(
        spec=spec,
        run_dir=run_dir,
        metrics=run_metrics,
        target=target,
        seeds=seeds,
        n_train=int(splits["n_train"]),
        artifacts=artifacts,
        val_curve=val_curve,
        diverged=diverged,
    )
    return run_metrics


def _validate_scalar_metrics(values, *, source: str) -> dict:
    """Coerce a diagnostic mapping to finite numeric scalars."""
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise TypeError(f"{source} must return a dictionary")

    normalized = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError(f"{source} keys must be strings")
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"{source}[{key!r}] must be scalar")
            value = value.detach().cpu().item()
        elif isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, int):
            normalized[key] = int(value)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise FloatingPointError(f"{source}[{key!r}] is nonfinite: {value}")
            normalized[key] = float(value)
        else:
            raise TypeError(f"{source}[{key!r}] must be a numeric scalar")
    return normalized


def _write_shard(*, spec, run_dir, metrics, target, seeds, n_train, artifacts,
                 val_curve, diverged=False):
    """Write the runstore shard (``run.json``) for one finished run."""
    from runstore import (
        STATUS_COMPLETED,
        STATUS_FAILED,
        RunRecord,
        SHARD_SCHEMA_VERSION,
        capture_provenance,
        now_iso,
    )

    run_spec = json.loads(json.dumps(spec, default=str))
    run_spec["target"] = target.to_dict()
    run_spec["seeds"] = {name: int(seed) for name, seed in seeds.items()}
    run_id = hashlib.blake2b(
        json.dumps(run_spec, sort_keys=True, separators=(",", ":")).encode(),
        digest_size=16,
    ).hexdigest()

    record = RunRecord(
        run_id=run_id,
        run_label=(
            f'{spec["task"]}__{spec["method"]}__N{spec["n"]}__fs{spec["function_seed"]}'
        ),
        schema_version=SHARD_SCHEMA_VERSION,
        method_id=spec["method"],
        task_id=spec["task"],
        namespace="csfrbench",
        n_train=int(n_train),
        seed=int(seeds["initialization"]),
        task_seed=int(spec["function_seed"]),
        sweep_id=spec.get("sweep_id"),
        spec=run_spec,
        metrics=metrics,
        status=STATUS_FAILED if diverged else STATUS_COMPLETED,
        finished_at=now_iso(),
        source="shard",
        run_dir=str(run_dir),
        artifacts=artifacts,
        val_curve=val_curve,
        **capture_provenance(),
    )
    record.write_shard(run_dir)


def _artifact(path: Path, kind: str, fmt: str) -> dict:
    """Describe one on-disk artifact for the shard."""
    return {
        "kind": kind,
        "uri": f"file://{path.resolve()}",
        "bytes": path.stat().st_size,
        "sha256": None,
        "format": fmt,
        "framework": None,
    }


def _atomic_numpy_savez(path: Path, **arrays) -> None:
    """``np.savez`` via a same-directory temporary file and atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _atomic_torch_save(obj, path: Path) -> None:
    """``torch.save`` via a same-directory temporary file, fsync, and atomic
    rename, so a killed worker can never leave a truncated checkpoint under
    the final name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _emit_connectivity(run_dir, model, target, X, scalars, artifacts, fmt="pdf"):
    """Best-effort connectivity figure from the already selected model."""
    try:
        import dagnet

        output = Path(run_dir) / f"connectivity.{fmt}"
        # dagnet.extract assumes a CPU model (numpy conversions + CPU target eval);
        # move the selected model + sample to CPU for this one-off post-training figure.
        extracted = dagnet.extract(model.cpu(), target, X.cpu(), scalars=scalars)
        dagnet.render(extracted, str(output))
        artifacts.append(_artifact(output, "figure", fmt))
    except Exception as exc:
        print(f"[connectivity] figure skipped: {exc!r}")

def main(argv=None) -> None:
    """Worker entry point: train the cell described by ``<run-dir>/spec.json``.

    Idempotence lives at the dispatch level — the manifest loop skips a dir
    that already has ``run.json``; calling this directly re-runs the cell.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--spec", default=None, help="defaults to <run-dir>/spec.json")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    spec_path = Path(args.spec) if args.spec else run_dir / "spec.json"
    execute_experiment_run(json.loads(spec_path.read_text()), run_dir)


if __name__ == "__main__":
    main()
