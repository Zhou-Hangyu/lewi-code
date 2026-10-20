"""Engine for 09-swm-lewi: cells, the frozen trainer call, HPO and final runners.

Everything method-specific comes from ``v05_port`` (canonical v0.5 builders);
everything task-specific from ``swm_tasks``. This module owns only the common
learner contract and the evaluation.

Common learner contract (goal doc Phase 1; frozen in ``results/contract.json``):
  backbone      MLP width 32, 4 hidden ReLU layers, vector delta head
  optimizer     Adam(0.9, 0.999, eps 1e-8), weight decay 0
  budget        32,000 optimizer steps
  schedule      cosine to 1% of the initial lr, no warmup
  checkpoint    every 500 steps, retrospective best validation checkpoint
  early stop    none
  target        standardized next-state delta in feature space
  HPO           32 scrambled-Sobol trials, plain validation argmin
Every arm gets exactly this; only intrinsic method hyperparameters differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from v05_port import (  # noqa: F401
    COMPETITIVE, LEWI_ARMS, REGISTRY_KEY, STUDY, build_model, hp_defaults,
    method_of, structural_params,
)
from swm_tasks import (  # noqa: F401
    SEED0, TASKS, derive, episode_slices, feature_projectors, featurize,
    load_corpus, project_features_,
)

from csfrbench.train import train  # noqa: E402  (path set up by v05_port)

RESULTS = STUDY / "results"

# ---- frozen common contract ------------------------------------------------
WIDTH = 32
HIDDEN = 4
MAX_STEPS = 32_000
EVAL_EVERY = 500
BATCH = 32
N_TRIALS = 32
N_FINAL_SEEDS = 16

HORIZONS = (1, 2, 4, 8, 16, 32)
SELECT_HORIZONS = (1, 2, 4, 8, 16)
H_REF = 16
VAL_WINDOW_CAP = 2_048
TEST_WINDOW_CAP = 4_096
NMSE_CAP = 1e6

#: diagnostic grid (Phase 3 step 1); the claim grid is frozen in N_GRID below
DIAG_GRID = (512, 2048, 8192, 32768, 131072)
N_GRID = (512, 1024, 2048, 4096, 8192, 16384, 32768)

ARMS = COMPETITIVE + LEWI_ARMS


def contract() -> dict:
    return {"width": WIDTH, "hidden": HIDDEN, "max_steps": MAX_STEPS,
            "eval_every": EVAL_EVERY, "batch_size": BATCH,
            "optimizer": "adam(0.9,0.999,eps=1e-8,wd=0)",
            "schedule": "cosine to 1% of lr, no warmup",
            "checkpoint": "retrospective best validation checkpoint",
            "early_stopping": False, "hpo_trials": N_TRIALS,
            "selection": "mean validation rollout NMSE over h in "
                         f"{SELECT_HORIZONS}",
            "target": "standardized next-state delta in feature space",
            "n_grid": list(N_GRID), "seed0": SEED0}


# ---------------------------------------------------------------- seeds

def trial_seeds(task: str, n: int, tag: str = "hpo") -> dict:
    return {k: derive(SEED0, "swm", task, n, tag, k)
            for k in ("init", "aux", "stochastic", "batch")}


def final_seeds(task: str, n: int, s: int, stream: str = "final") -> dict:
    """Paired instance seeds. ``stream`` selects the seed bank: "final" is the
    held-out bank used for every claim, "devfinal" is a disjoint bank used for
    development confirmation so the held-out bank is never touched before the
    global freeze. The method name never enters the derivation, so instance s
    of two arms shares its data order and initialization draw."""
    return {k: derive(SEED0, "swm", task, n, stream, s, k)
            for k in ("init", "aux", "stochastic", "batch")}


def sobol_configs(task: str, arm: str, n: int, space: dict,
                  trials: int = N_TRIALS, tag: str = "") -> list[dict]:
    from scipy.stats import qmc

    dims = [(k, v) for k, v in space.items() if isinstance(v, (tuple, list))]
    fixed = {k: v for k, v in space.items() if not isinstance(v, (tuple, list))}
    if not dims:
        return [dict(fixed)] * trials
    sampler = qmc.Sobol(d=len(dims), scramble=True,
                        seed=derive(SEED0, "swm", task, arm, n, "sobol", tag))
    configs = []
    for row in sampler.random(trials):
        hp = dict(fixed)
        for (key, (low, high, log)), u in zip(dims, row):
            hp[key] = (float(np.exp(np.log(low) + u * (np.log(high) - np.log(low))))
                       if log else float(low + u * (high - low)))
        configs.append(hp)
    return configs


# ---------------------------------------------------------------- cell data

def split_episodes(task: str, n_eps: int) -> dict:
    """Episode-level 80/10/10 split. No transition ever appears in two splits."""
    order = np.random.default_rng(derive(SEED0, "swm", task, "split")).permutation(n_eps)
    n_val = max(1, int(round(0.10 * n_eps)))
    n_test = max(1, int(round(0.10 * n_eps)))
    return {"train": order[: n_eps - n_val - n_test],
            "val": order[n_eps - n_val - n_test: n_eps - n_test],
            "test": order[n_eps - n_test:]}


def transitions_of(task, corpus, ep_indices):
    """(X = [features, action], Y = feature delta) over the given episodes."""
    xs, ys = [], []
    slices = episode_slices(corpus)
    for e in ep_indices:
        (oa, ob), (aa, ab) = slices[int(e)]
        states, acts = corpus["obs"][oa:ob], corpus["act"][aa:ab]
        assert len(acts) == len(states) - 1, (len(acts), len(states))
        feats = featurize(task, states)
        xs.append(np.concatenate([feats[:-1], acts], axis=-1))
        ys.append(feats[1:] - feats[:-1])
    return np.concatenate(xs), np.concatenate(ys)


def budget_index(task: str, pool_size: int, n: int) -> np.ndarray:
    """Nested random subset of size n from the training-transition pool.

    Nesting: one permutation per task, prefixes of it are the budgets, so
    every smaller budget is a subset of every larger one. Drawing across all
    training episodes (rather than taking an episode prefix) keeps the
    sample-size axis meaning 'n transitions' rather than 'n/H episodes', and
    keeps small budgets from being dominated by a handful of neighbouring
    frames inside two or three episodes.
    """
    order = np.random.default_rng(derive(SEED0, "swm", task, "pool")).permutation(pool_size)
    return order[: min(n, pool_size)]


def rollout_windows(task, corpus, ep_indices, cap, seed_key, h_max):
    slices = episode_slices(corpus)
    starts = []
    for e in ep_indices:
        (oa, ob), _ = slices[int(e)]
        limit = (ob - oa - 1) - h_max
        starts.extend((int(e), t) for t in range(max(limit, 0)))
    rng = np.random.default_rng(derive(SEED0, "swm", task, "windows", seed_key))
    if len(starts) > cap:
        starts = [starts[i] for i in sorted(rng.choice(len(starts), cap, replace=False))]
    s0, acts, fut = [], [], []
    for e, t in starts:
        (oa, _ob), (aa, _ab) = slices[int(e)]
        feats = featurize(task, corpus["obs"][oa + t: oa + t + h_max + 1])
        s0.append(feats[0])
        acts.append(corpus["act"][aa + t: aa + t + h_max])
        fut.append(feats[1:])
    return (np.asarray(s0), np.asarray(acts, dtype=np.float64), np.asarray(fut))


_DIM_CACHE: dict = {}


def informative_dims(task: str, corpus=None) -> np.ndarray:
    """Boolean mask over feature dimensions that actually move.

    Several tasks carry state components that never change (tworoom's goal and
    room geometry, manipulator's unused object slots). Their prediction error
    and their variance are both zero, so they contribute a hard 0 to a per-
    dimension NMSE average and silently deflate every arm's score by the same
    factor -- which leaves rankings intact but makes the absolute numbers, and
    therefore the task-health thresholds, meaningless. The mask is computed
    once per task from the corpus, before any split or model, so it is
    identical for every arm and every cell.
    """
    if task in _DIM_CACHE:
        return _DIM_CACHE[task]
    corpus = corpus if corpus is not None else load_corpus(task)
    feats = featurize(task, corpus["obs"])
    ptr = corpus["ep_ptr"]
    # within-episode deltas only: a dimension that is constant inside every
    # episode is trivially predictable (predict zero change) no matter how much
    # it varies between episodes, and carries no information about the dynamics
    moves = np.zeros(feats.shape[1], dtype=bool)
    for i in range(len(ptr) - 1):
        block = feats[int(ptr[i]):int(ptr[i + 1])]
        if len(block) > 1:
            moves |= np.abs(np.diff(block, axis=0)).max(0) > 1e-8
    if not moves.any():
        moves = np.ones(feats.shape[1], dtype=bool)
    _DIM_CACHE[task] = moves
    return moves


_CELL_CACHE: dict = {}


def make_cell_data(task: str, n: int, corpus_name: str | None = None):
    import torch
    from torch.utils.data import TensorDataset

    key = (task, n, corpus_name)
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    corpus = load_corpus(task, corpus_name)
    n_eps = len(corpus["ep_ptr"]) - 1
    splits = split_episodes(task, n_eps)
    X_pool, Y_pool = transitions_of(task, corpus, splits["train"])
    idx = budget_index(task, len(X_pool), n)
    X_train, Y_train = X_pool[idx], Y_pool[idx]
    X_val, Y_val = transitions_of(task, corpus, splits["val"])

    mu_x, sd_x = X_train.mean(0), np.maximum(X_train.std(0), 1e-8)
    mu_y, sd_y = Y_train.mean(0), np.maximum(Y_train.std(0), 1e-8)
    norm = {"mu_x": mu_x, "sd_x": sd_x, "mu_y": mu_y, "sd_y": sd_y}

    def ds(X, Y):
        return TensorDataset(torch.tensor((X - mu_x) / sd_x, dtype=torch.float32),
                             torch.tensor((Y - mu_y) / sd_y, dtype=torch.float32))

    h_max = max(HORIZONS)
    cell = {
        "train": ds(X_train, Y_train), "val": ds(X_val, Y_val),
        "test_ds": ds(*transitions_of(task, corpus, splits["test"])),
        "norm": norm, "n_actual": int(len(X_train)), "pool_size": int(len(X_pool)),
        "obs_dim": int(corpus["meta"]["obs_dim"]),
        "keep_dims": informative_dims(task, corpus),
        "val_windows": rollout_windows(task, corpus, splits["val"],
                                       VAL_WINDOW_CAP, "val", h_max),
        "test_windows": rollout_windows(task, corpus, splits["test"],
                                        TEST_WINDOW_CAP, "test", h_max),
        "splits": {k: v.tolist() for k, v in splits.items()},
    }
    _CELL_CACHE.clear()
    _CELL_CACHE[key] = cell
    return cell


# ---------------------------------------------------------------- evaluation

def rollout_nmse(task: str, model, windows, norm, obs_dim, horizons=HORIZONS,
                 keep=None) -> dict:
    import torch

    s0, acts, fut = windows
    groups = feature_projectors(task, obs_dim)
    mu_x, sd_x, mu_y, sd_y = (norm[k] for k in ("mu_x", "sd_x", "mu_y", "sd_y"))
    cur = s0.copy()
    out = {}
    model.eval()
    with torch.no_grad():
        for h in range(1, max(horizons) + 1):
            xa = np.concatenate([cur, acts[:, h - 1]], axis=-1)
            dn = model(torch.tensor((xa - mu_x) / sd_x, dtype=torch.float32))
            cur = cur + (dn.double().numpy() * sd_y + mu_y)
            if not np.isfinite(cur).all():
                cur = np.nan_to_num(cur, nan=1e9, posinf=1e9, neginf=-1e9)
            project_features_(cur, groups)
            if h in horizons:
                err = cur - fut[:, h - 1]
                per_dim = (err ** 2).mean(0) / np.maximum(fut[:, h - 1].var(0), 1e-8)
                sel = per_dim if keep is None else per_dim[keep]
                out[h] = {"nmse": float(min(sel.mean(), NMSE_CAP)),
                          "per_dim": np.minimum(per_dim, NMSE_CAP).round(6).tolist()}
    return out


def one_step_nmse(model, dataset, keep=None) -> dict:
    import torch

    X, Y = dataset.tensors
    model.eval()
    with torch.no_grad():
        pred = model(X)
    err = (pred - Y).double().numpy()
    y = Y.double().numpy()
    per_dim = (err ** 2).mean(0) / np.maximum(y.var(0), 1e-8)
    sel = per_dim if keep is None else per_dim[keep]
    return {"nmse": float(min(sel.mean(), NMSE_CAP)),
            "per_dim": np.minimum(per_dim, NMSE_CAP).round(6).tolist()}


def selection_score(rollout: dict) -> float:
    return float(np.mean([rollout[h]["nmse"] for h in SELECT_HORIZONS]))


# ---------------------------------------------------------------- training

def tconf(batch: int = BATCH, max_steps: int = MAX_STEPS,
          eval_every: int = EVAL_EVERY) -> dict:
    max_steps = int(os.environ.get("SWM_SMOKE_MAX_STEPS", max_steps))
    return {
        "optimizer": {"type": "adam", "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.0},
        "grad_clip": None, "batch_size": int(batch), "max_steps": max_steps,
        "scheduler": {"type": "cosine", "lr_final_frac": 0.01},
        "validation": {"eval_every": min(eval_every, max_steps), "metric": "nmse",
                       "restore_best_checkpoint": True},
        "early_stopping": {"enabled": False},
    }


def train_one(task: str, n: int, arm: str, hp: dict, seeds: dict, *,
              width=WIDTH, hidden=HIDDEN, batch=BATCH, max_steps=MAX_STEPS,
              cell=None, corpus_name=None, builder=None, method=None):
    import torch

    cell = cell or make_cell_data(task, n, corpus_name)
    in_dim = cell["train"].tensors[0].shape[1]
    out_dim = cell["train"].tensors[1].shape[1]
    hp = dict(hp_defaults(arm) | dict(hp))
    torch.manual_seed(seeds["init"])
    builder = builder or build_model
    model = builder(arm, in_dim, out_dim, hp, seeds["aux"], width, hidden)
    method = method or method_of(arm)
    result = train(model, method, dict(hp), cell["train"], cell["val"],
                   tconf(batch, max_steps), stochastic_seed=seeds["stochastic"],
                   batch_seed=seeds["batch"])
    val_roll = rollout_nmse(task, model, cell["val_windows"], cell["norm"],
                            cell["obs_dim"], keep=cell["keep_dims"])
    result["val_rollout"] = {str(h): v["nmse"] for h, v in val_roll.items()}
    result["val_rollout_score"] = selection_score(val_roll)
    # Capacity diagnostic: training-set fit of the selected checkpoint. A high
    # train NMSE with a high test NMSE is capacity or optimization failure; a
    # low train NMSE with a high test NMSE is a data-limited regime.
    result["train_one_step_nmse"] = one_step_nmse(
        model, cell["train"], cell["keep_dims"])["nmse"]
    result["informative_dims"] = int(cell["keep_dims"].sum())
    result["feature_dims"] = int(cell["keep_dims"].size)
    result["hp"] = dict(hp)
    result["n_actual"] = cell["n_actual"]
    result["params"] = int(sum(p.numel() for p in model.parameters()))
    result["structural_params"] = structural_params(arm, model)
    try:
        result["diagnostics"] = {k: float(v) for k, v in
                                 method.extra_diagnostics(model).items()}
    except Exception:  # noqa: BLE001
        result["diagnostics"] = {}
    result.pop("val_errors", None)
    return result, model, cell


# ---------------------------------------------------------------- runners

def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
    claim = path.with_suffix(".claim")
    if claim.exists():
        claim.unlink(missing_ok=True)


CLAIM_STALE_SECONDS = 5400


def claim_or_skip(path) -> bool:
    """Public claim for scripts that are not cell runners.

    Every job in a dynamically dispatched fleet walks the same work list, so a
    unit with no claim is run by every job that reaches it. The HPO and final
    runners claim their own output; the probe, Markov, matched-rate and
    dependency scripts use this instead.
    """
    path = Path(path)
    if path.exists():
        return False
    return _claim(path)


def release_claim(path) -> None:
    claim = Path(path).with_suffix(".claim")
    if claim.exists():
        claim.unlink(missing_ok=True)


def _heartbeat(claim: Path, period: int = 600) -> None:
    """Keep a live claim fresh. A cell that runs longer than CLAIM_STALE_SECONDS
    (learned-basis HPO cells on slow nodes) must not be retaken while its job is
    alive; a dead job stops touching its claim and the stale rule applies."""
    import threading
    import time

    def beat():
        while True:
            time.sleep(period)
            try:
                os.utime(claim, None)
            except FileNotFoundError:
                return

    threading.Thread(target=beat, daemon=True).start()


def _claim(path: Path) -> bool:
    """Take an exclusive claim on a cell, so two jobs handed overlapping unit
    lists do not both compute it. A claim older than CLAIM_STALE_SECONDS is
    treated as abandoned (preempted job, crashed node) and can be retaken."""
    import time

    path.parent.mkdir(parents=True, exist_ok=True)
    claim = path.with_suffix(".claim")
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.environ.get("SLURM_JOB_ID", os.getpid())).encode())
        os.close(fd)
        _heartbeat(claim)
        return True
    except FileExistsError:
        try:
            if time.time() - claim.stat().st_mtime > CLAIM_STALE_SECONDS:
                claim.unlink()  # retake through the exclusive create: one winner
                return _claim(path)
        except FileNotFoundError:
            return _claim(path)
        return False


def spaces() -> dict:
    path = RESULTS / "spaces.json"
    raw = json.loads(path.read_text())
    return {arm: {k: (tuple(v) if isinstance(v, list) else v)
                  for k, v in sp.items()} for arm, sp in raw.items()}


#: the cell for the currently running pool. Built once in the parent and
#: inherited by every worker through fork, so the corpus is featurized and the
#: rollout windows are assembled once per cell rather than once per trial --
#: that assembly is a Python loop over thousands of windows and dominated the
#: wall time when each worker repeated it.
_SHARED_CELL = None


def _hpo_worker(args):
    task, arm, n, i, hp, batch, width, hidden, max_steps, corpus, tag = args
    import torch

    torch.set_num_threads(1)
    try:
        result, _model, _cell = train_one(
            task, n, arm, hp, trial_seeds(task, n), width=width, hidden=hidden,
            batch=batch, max_steps=max_steps, corpus_name=corpus,
            cell=_SHARED_CELL)
        result["trial"] = i
        return result
    except Exception as exc:  # noqa: BLE001
        return {"trial": i, "hp": hp, "error": f"{type(exc).__name__}: {exc}",
                "diverged": 1, "val_rollout_score": float("inf"),
                "best_val_nmse": float("inf")}


def winner_key(r: dict):
    return (int(r.get("diverged", 0) or 0), r.get("val_rollout_score", float("inf")),
            r.get("best_val_nmse", float("inf")), r.get("trial", 0))


def run_hpo_cell(task, arm, n, procs, *, batch=BATCH, width=WIDTH, hidden=HIDDEN,
                 max_steps=MAX_STEPS, trials=N_TRIALS, out=None, space=None,
                 corpus=None, tag=""):
    out = out or RESULTS / "hpo" / task / arm / f"n{n}.json"
    if out.exists():
        print(f"[hpo] {out} exists, skipping")
        return
    if not _claim(out):
        print(f"[hpo] {out} claimed by another job, skipping")
        return
    if space is None:
        space = spaces()[arm]
        space = {k: (tuple(v) if isinstance(v, list) else v) for k, v in space.items()}
    configs = sobol_configs(task, arm, n, space, trials, tag)
    global _SHARED_CELL
    _SHARED_CELL = make_cell_data(task, n, corpus)
    jobs = [(task, arm, n, i, hp, batch, width, hidden, max_steps, corpus, tag)
            for i, hp in enumerate(configs)]
    results = sorted(_pool_map(_hpo_worker, jobs, procs), key=lambda r: r["trial"])
    w = min(range(len(results)), key=lambda i: winner_key(results[i]))
    _atomic_write(out, {"task": task, "arm": arm, "n": n, "batch": batch,
                        "width": width, "hidden": hidden, "max_steps": max_steps,
                        "space": {k: list(v) if isinstance(v, tuple) else v
                                  for k, v in space.items()},
                        "winner_trial": w, "winner_hp": results[w]["hp"],
                        "trials": results})
    print(f"[hpo] {out} winner={results[w]['hp']} "
          f"score={results[w].get('val_rollout_score')}")


def _final_worker(args):
    (task, n, arm, hp, s, batch, width, hidden, max_steps, corpus, keep, stream,
     models_dir) = args
    import torch

    torch.set_num_threads(1)
    try:
        result, model, cell = train_one(
            task, n, arm, hp, final_seeds(task, n, s, stream), width=width,
            hidden=hidden, batch=batch, max_steps=max_steps, corpus_name=corpus,
            cell=_SHARED_CELL)
    except Exception as exc:  # noqa: BLE001
        return {"seed_index": s, "hp": hp, "error": f"{type(exc).__name__}: {exc}",
                "diverged": 1, "test_rollout": {}, "test_one_step": {"nmse": NMSE_CAP}}
    result["seed_index"] = s
    result["test_one_step"] = one_step_nmse(model, cell["test_ds"], cell["keep_dims"])
    result["test_rollout"] = {str(h): v["nmse"] for h, v in rollout_nmse(
        task, model, cell["test_windows"], cell["norm"], cell["obs_dim"],
        keep=cell["keep_dims"]).items()}
    result["test_rollout_per_dim"] = {}
    if s in keep:
        path = Path(models_dir) / f"{arm}_s{s}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "hp": hp, "arm": arm, "task": task, "n": n, "seed_index": s,
                    "width": width, "hidden": hidden,
                    "norm": {k: v.tolist() for k, v in cell["norm"].items()},
                    "in_dim": cell["train"].tensors[0].shape[1],
                    "out_dim": cell["train"].tensors[1].shape[1]}, path)
    return result


def run_final_cell(task, n, arm, procs, *, batch=BATCH, width=WIDTH, hidden=HIDDEN,
                   max_steps=MAX_STEPS, out=None, hp=None, corpus=None,
                   seeds=N_FINAL_SEEDS, keep_models=(), stream="final",
                   winners_path=None):
    out = out or RESULTS / "final" / task / f"n{n}" / f"{arm}.json"
    if out.exists():
        print(f"[final] {out} exists, skipping")
        return
    if not _claim(out):
        print(f"[final] {out} claimed by another job, skipping")
        return
    if hp is None:
        path = Path(winners_path) if winners_path else RESULTS / "winners.json"
        winners = json.loads(path.read_text())
        hp = winners[task][arm][str(n)]["hp"]
    global _SHARED_CELL
    _SHARED_CELL = make_cell_data(task, n, corpus)
    # kept models live beside the cell's own result file, whatever tree it is in
    models_dir = str(Path(out).parent / "models")
    jobs = [(task, n, arm, hp, s, batch, width, hidden, max_steps, corpus,
             set(keep_models), stream, models_dir) for s in range(seeds)]
    results = _pool_map(_final_worker, jobs, procs)
    _atomic_write(out, {"task": task, "n": n, "arm": arm, "batch": batch,
                        "width": width, "hidden": hidden, "max_steps": max_steps,
                        "hp": hp, "seed_stream": stream, "instances": results})
    ok = [r for r in results if not r.get("diverged")]
    med = float(np.median([r["test_rollout"][str(H_REF)] for r in ok])) if ok else float("nan")
    print(f"[final] {out}  median h{H_REF} test rollout NMSE = {med:.4f} "
          f"({len(ok)}/{len(results)} clean)")


def allocated_cpus() -> int:
    """Cores this process may use, from SLURM, falling back to the machine."""
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(key)
        if value and value.isdigit():
            return int(value)
    return os.cpu_count() or 12


def _pool_map(fn, jobs, procs):
    """``procs <= 0`` means one worker per allocated core.

    Running one worker per *trial* looked attractive because it leaves no idle
    tail, but oversubscribing 32 single-threaded torch processes onto 12 cores
    measured about five times slower than matching the core count: the loss to
    context switching and to page faults on the forked parent image dwarfs the
    tail it was meant to remove.
    """
    import multiprocessing as mp

    if procs <= 0:
        procs = min(len(jobs), allocated_cpus())
    if procs <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(procs, len(jobs)),
                             mp_context=mp.get_context("fork")) as pool:
        return list(pool.map(fn, jobs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=["hpo", "final"])
    p.add_argument("--task", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--procs", type=int, default=0)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--hidden", type=int, default=HIDDEN)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--seeds", type=int, default=N_FINAL_SEEDS)
    p.add_argument("--keep-models", default="")
    p.add_argument("--trials", type=int, default=N_TRIALS)
    p.add_argument("--out", default="")
    p.add_argument("--spaces", default="")
    p.add_argument("--tag", default="")
    p.add_argument("--stream", default="final")
    p.add_argument("--winners", default="")
    p.add_argument("--corpus", default="",
                   help="corpus name suffix, e.g. 'xl' reads data/<task>_xl.npz")
    a = p.parse_args()
    corpus = f"{a.task}_{a.corpus}" if a.corpus else None
    keep = tuple(int(x) for x in a.keep_models.split(",") if x != "")
    out = Path(a.out) if a.out else None
    space = None
    if a.spaces:
        raw = json.loads(Path(a.spaces).read_text()).get(a.arm)
        if raw is not None:
            space = {k: (tuple(v) if isinstance(v, list) else v)
                     for k, v in raw.items()}
    if a.phase == "hpo":
        run_hpo_cell(a.task, a.arm, a.n, a.procs, batch=a.batch, width=a.width,
                     hidden=a.hidden, max_steps=a.max_steps, trials=a.trials,
                     out=out, space=space, tag=a.tag, corpus=corpus)
    else:
        run_final_cell(a.task, a.n, a.arm, a.procs, batch=a.batch, width=a.width,
                       hidden=a.hidden, max_steps=a.max_steps, seeds=a.seeds,
                       keep_models=keep, out=out, stream=a.stream,
                       winners_path=a.winners or None, corpus=corpus)


if __name__ == "__main__":
    main()
