"""The training protocol and the execution of one unit.

One unit is one training run: a target instance, a training-set size, a method
and one hyperparameter configuration. Every method trains under the same
protocol:

* Adam with ``beta1 = 0.9``, ``beta2 = 0.999``, ``eps = 1e-8``, no weight decay;
* batch size 32, minibatches drawn without replacement per epoch, last partial batch dropped;
* cosine decay of the learning rate to 1% of its initial value over the step budget;
* no warm-up, no gradient clipping, no early stopping;
* 64 validation checkpoints at equal distances; the checkpoint with the lowest
  validation NMSE is kept and evaluated once on the test pool.

A run whose loss or gradient norm is not finite stops before that update and is
recorded as diverged. A diverged run has no test value and is never selected.

The five random streams of a unit (initialization, training noise, minibatch
order, method state, evaluation) have seeds derived from the target fingerprint,
the training-set size and the dataset seed. They do not depend on the method or
on the hyperparameters, so methods are compared on the same initial weights and
the same minibatch order.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, default_collate

from . import __version__, datasets, metrics
from .methods import get_method
from .seeds import derive

BATCH_SIZE = 32
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
LR_FINAL_FRACTION = 0.01
EVAL_BATCH_SIZE = 8192
STREAMS = ("initialization", "stochastic", "batch_order", "gate_init", "evaluation")


class TensorPool(TensorDataset):
    """Tensors in memory; a minibatch is one indexing operation."""

    def __getitems__(self, indices):
        return tuple(tensor[indices] for tensor in self.tensors)


def _collate(batch):
    return batch if isinstance(batch, tuple) else default_collate(batch)


def make_loader(dataset, batch_size: int, *, shuffle: bool = False, drop_last: bool = False,
                generator: torch.Generator | None = None) -> DataLoader:
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, drop_last=drop_last,
                      generator=generator, num_workers=0, collate_fn=_collate)


def _cycle(loader: DataLoader):
    while True:
        yield from loader


def _total_norm(gradients) -> torch.Tensor:
    norms = [torch.linalg.vector_norm(g, 2.0) for g in gradients]
    return torch.linalg.vector_norm(torch.stack(norms), 2.0)


def unit_seeds(fingerprint: str, n: int, dataset_seed: int, seed_replicate: int = 0) -> dict[str, int]:
    """The seeds of the five random streams of a unit.

    ``seed_replicate = 0`` gives the seeds of the paper's experiments. Another
    value gives new training randomness on the same target and data.
    """
    key: tuple = (fingerprint, int(n), int(dataset_seed))
    if int(seed_replicate):
        key = (*key, int(seed_replicate))
    return {name: int(derive(*key, name)) for name in STREAMS}


def _interval_nmse(interval) -> float | None:
    if interval["count"] == 0:
        return None
    mean = interval["sum"] / interval["count"]
    variance = max(0.0, interval["sumsq"] / interval["count"] - mean * mean)
    value = (interval["sse"] / interval["count"]) / (variance + 1e-12)
    return value if math.isfinite(value) else None


def train(model, method, hp: dict, train_set, val_set, *, max_steps: int, eval_every: int, stochastic_seed: int,
          batch_seed: int, eval_seed: int, stop_after_evals: int | None = None) -> dict[str, Any]:
    """Train ``model`` in place and leave it at its best validation checkpoint, in evaluation mode.

    ``stop_after_evals`` ends the run after that number of validation
    checkpoints. It is a tool for tests: the learning-rate schedule still spans
    ``max_steps``, so the checkpoints equal those of the complete run.
    """
    max_steps, eval_every = int(max_steps), int(eval_every)
    learning_rate = float(hp["lr"])
    if max_steps <= 0 or eval_every <= 0:
        raise ValueError("max_steps and eval_every must be positive")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("the learning rate must be finite and positive")
    if len(train_set) < BATCH_SIZE or len(val_set) == 0:
        raise ValueError(f"the training set must hold at least {BATCH_SIZE} examples and the validation set one")

    device = next(model.parameters()).device
    torch.manual_seed(int(stochastic_seed))             # the training noise does not depend on the construction
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=ADAM_BETAS, eps=ADAM_EPS,
                                 weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps,
                                                           eta_min=learning_rate * LR_FINAL_FRACTION)
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(int(batch_seed))
    train_loader = make_loader(train_set, BATCH_SIZE, shuffle=True, drop_last=True, generator=batch_generator)
    eval_generator = torch.Generator(device="cpu")
    eval_generator.manual_seed(int(eval_seed))
    val_loader = make_loader(val_set, EVAL_BATCH_SIZE, generator=eval_generator)

    best_val, best_step, best_state = float("inf"), 0, None
    val_curve: list[tuple[int, float, float | None]] = []
    interval = {"sse": 0.0, "sum": 0.0, "sumsq": 0.0, "count": 0}
    steps_run = 0
    grad_norm_max = None
    diverged_step = diverged_reason = None

    model.train()
    start = time.perf_counter()
    for step, (X, y) in enumerate(itertools.islice(_cycle(train_loader), max_steps), start=1):
        optimizer.zero_grad(set_to_none=True)
        X, y = X.to(device), y.to(device)
        prediction = model(X)
        if prediction.shape != y.shape:
            raise ValueError(f"prediction shape {tuple(prediction.shape)} != target shape {tuple(y.shape)}")
        task_loss = F.mse_loss(prediction, y)
        with torch.no_grad():
            targets = y.to(torch.float64)
            interval["sse"] += task_loss.detach().item() * y.numel()
            interval["sum"] += targets.sum().item()
            interval["sumsq"] += targets.square().sum().item()
            interval["count"] += y.numel()
        penalty = method.penalty(model, hp)
        loss = task_loss if penalty is None else task_loss + penalty
        if not bool(torch.isfinite(loss).item()):
            diverged_step, diverged_reason = step, "nonfinite_loss"
            break
        loss.backward()
        grad_norm = _total_norm([p.grad for p in model.parameters() if p.grad is not None])
        if not bool(torch.isfinite(grad_norm).item()):
            diverged_step, diverged_reason = step, "nonfinite_grad_norm"
            break
        grad_norm_max = grad_norm if grad_norm_max is None else torch.maximum(grad_norm_max, grad_norm)

        optimizer.step()
        method.after_step(model, optimizer, step=step, total_steps=max_steps)
        scheduler.step()
        steps_run = step

        if step % eval_every == 0 or step == max_steps:
            val_nmse = float(metrics.nmse(model, val_loader))
            if not math.isfinite(val_nmse):
                raise FloatingPointError(f"the validation NMSE is not finite at step {step}")
            val_curve.append((step, val_nmse, _interval_nmse(interval)))
            interval = {"sse": 0.0, "sum": 0.0, "sumsq": 0.0, "count": 0}
            if val_nmse < best_val:
                best_val, best_step = val_nmse, step
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            if stop_after_evals is not None and len(val_curve) >= int(stop_after_evals):
                break

    if diverged_step is not None and not val_curve:     # diverged before the first checkpoint
        val_nmse = float(metrics.nmse(model, val_loader))
        if math.isfinite(val_nmse):
            val_curve.append((steps_run, val_nmse, _interval_nmse(interval)))
            best_val, best_step = val_nmse, steps_run
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    result = {
        "best_val_nmse": best_val if math.isfinite(best_val) else None,
        "best_step": best_step,
        "steps_run": steps_run,
        "val_curve": [[int(s), float(v), None if t is None else float(t)] for s, v, t in val_curve],
        "wallclock_s": time.perf_counter() - start,
        "diverged": int(diverged_step is not None),
        "grad_norm_max": float(grad_norm_max) if grad_norm_max is not None else 0.0,
        "complete": int(steps_run == max_steps),
    }
    if diverged_step is not None:
        result["diverged_step"] = int(diverged_step)
        result["diverged_reason"] = diverged_reason
    return result


def build_model(method_name: str, target, *, width: int, hidden: int, hp: dict, n: int, dataset_seed: int | None = None,
                seed_replicate: int = 0):
    """Build the model of a unit at its initialization. Returns ``(method, model, seeds)``."""
    dataset_seed = int(target.function_seed if dataset_seed is None else dataset_seed)
    seeds = unit_seeds(target.fingerprint, n, dataset_seed, seed_replicate)
    method = get_method(method_name)
    method.check_hp(hp)
    cfg = {"width": int(width), "hidden": int(hidden), "hp": dict(hp), "auxiliary_seed": seeds["gate_init"]}
    torch.manual_seed(seeds["initialization"])
    return method, method.build(cfg, target), seeds


def run_unit(target, *, method: str, hp: dict, n: int, width: int, hidden: int, max_steps: int, eval_every: int,
             n_max: int | None = None, dataset_seed: int | None = None, seed_replicate: int = 0,
             test_pools: tuple[str, ...] = ("test",), data: dict | None = None, keep_states: bool = False,
             threads: int = 1, stop_after_evals: int | None = None, device: str = "cpu") -> dict[str, Any]:
    """Train one unit and evaluate its selected checkpoint on the test pool.

    ``data`` may hold a dataset of :func:`csfrbench.datasets.make_dataset`; else
    the pools are drawn here. With ``keep_states`` the result holds the state of
    the model at the initialization (``init_state``) and at the selected
    checkpoint (``best_state``).

    The records of the paper were made on a CPU with one thread. Another
    ``device`` or thread count trains the same protocol; its results agree with
    the records statistically, not bit for bit.
    """
    torch.set_num_threads(int(threads))
    device = torch.device(device)
    n = int(n)
    dataset_seed = int(target.function_seed if dataset_seed is None else dataset_seed)
    if data is None:
        data = datasets.make_dataset(target, int(n_max or n), dataset_seed=dataset_seed,
                                     pools=("train", "val", *test_pools))
    if data["meta"]["fingerprint"] != target.fingerprint or data["meta"]["dataset_seed"] != dataset_seed:
        raise ValueError("the dataset belongs to another target instance or dataset seed")
    if data["X_train"].shape[0] < n:
        raise ValueError(f"the training pool holds {data['X_train'].shape[0]} examples; n = {n}")

    model_method, model, seeds = build_model(method, target, width=width, hidden=hidden, hp=hp, n=n,
                                             dataset_seed=dataset_seed, seed_replicate=seed_replicate)
    model = model.to(device)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} if keep_states else None
    start = time.perf_counter()
    outcome = train(model, model_method, hp,
                    TensorPool(data["X_train"][:n].to(device), data["y_train"][:n].to(device)),
                    TensorPool(data["X_val"].to(device), data["y_val"].to(device)),
                    max_steps=max_steps, eval_every=eval_every, stochastic_seed=seeds["stochastic"],
                    batch_seed=seeds["batch_order"], eval_seed=seeds["evaluation"],
                    stop_after_evals=stop_after_evals)

    result: dict[str, Any] = {
        "target_id": target.target_id, "fingerprint": target.fingerprint, "method": method, "hp": dict(hp),
        "n": n, "d": target.d, "width": int(width), "hidden": int(hidden), "max_steps": int(max_steps),
        "eval_every": int(eval_every), "dataset_seed": dataset_seed, "seed_replicate": int(seed_replicate),
        "seeds": seeds, **outcome,
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if not outcome["diverged"]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seeds["evaluation"])
        for pool in test_pools:
            loader = make_loader(TensorPool(data[f"X_{pool}"].to(device), data[f"f_{pool}"].to(device)),
                                 EVAL_BATCH_SIZE, generator=generator)
            errors, targets = metrics.pointwise_squared_errors(model, loader)
            result[f"{pool}_nmse"] = metrics.nmse_from_errors(errors, targets)
        result["diagnostics"] = {k: (v.item() if torch.is_tensor(v) else v)
                                 for k, v in model_method.diagnostics(model).items()}
    result["unit_wallclock_s"] = time.perf_counter() - start
    result["environment"] = environment()
    if keep_states:
        result["init_state"] = init_state
        result["best_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return result


def environment() -> dict[str, str]:
    import scipy
    return {"csfrbench": __version__, "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "python": platform.python_version(), "machine": platform.machine(), "system": platform.system(),
            "threads": str(torch.get_num_threads())}


def write_json(path: str | Path, payload: dict) -> None:
    """Write a JSON file through a temporary file and a rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
