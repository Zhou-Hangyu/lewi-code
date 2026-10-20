"""Plan and execute runs.

A *unit* is one training run: (setting, workload, arm, target instance,
training-set size, trial). :func:`plan` lists the units of a part of the
benchmark; :func:`run` executes them with a pool of single-thread workers and
writes one JSON line per unit. A unit that is on record in the output folder is
not run again, so an interrupted campaign continues where it stopped.

Output folder::

    runs-<host>-<pid>.jsonl     one line per finished unit (each worker has its own file)
    states/<unit>.pt            with keep_states: the states at the initialization and at the selected checkpoint

On a cluster, give each job the same list of units and its own shard:
``--shard 3/16`` runs the units 3, 19, 35, ... of the list.

Engines: ``auto`` (the default) trains LeWi and Dense units in stacks of many networks with LeWi's fused kernels
(:mod:`csfrbench.stack`: the same protocol, outcomes that agree with the records statistically) and every
other method with the per-network trainer; ``exact`` trains every unit with the per-network trainer, which reproduces
a record bit for bit in the records' environment.
"""
from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Iterable
from multiprocessing import get_context
from pathlib import Path

from . import search, suite, training


def unit_id(u: dict) -> str:
    return f"{u['setting']}.{u['workload']}.{u['arm']}.i{u['instance']}.n{u['n']}.t{int(u['trial']):02d}"


def plan(setting: str, workloads: Iterable[str] | None = None, arms: Iterable[str] | None = None,
         instances: Iterable[int] | None = None, sizes: Iterable[int] | None = None,
         trials: Iterable[int] | None = None) -> list[dict]:
    """The units of a part of a setting; an argument that is None selects everything."""
    units = []
    for name in (list(workloads) if workloads else list(suite.SETTINGS[setting])):
        w = suite.workload(setting, name)
        for arm in (list(arms) if arms else list(w.arms)):
            if arm not in w.arms:
                continue
            for n in (list(sizes) if sizes else list(w.profile.sizes)):
                if int(n) not in w.profile.sizes:
                    continue
                configs = search.configurations(w, arm, int(n))
                for instance in (list(instances) if instances else list(w.instances)):
                    for trial in (list(trials) if trials is not None else range(suite.TRIALS)):
                        units.append({"setting": setting, "workload": name, "arm": arm, "instance": int(instance),
                                      "n": int(n), "trial": int(trial), "hp": configs[int(trial)]})
    return units


def units_of_winners(winners: list[dict]) -> list[dict]:
    """The units that run the selected configurations of a list of winners again."""
    return [{"setting": w["setting"], "workload": w["workload"], "arm": w["arm"], "instance": int(w["instance"]),
             "n": int(w["n"]), "trial": int(w["trial"]),
             "hp": {**(suite.ARMS[w["arm"]].fixed if w["arm"] in suite.ARMS else {}), **w["hp"]}} for w in winners]


def execute(unit: dict, *, keep_states: bool = False, stop_after_evals: int | None = None, data_cache: dict | None = None,
            method: str | None = None, threads: int = 1, device: str = "cpu") -> dict:
    """Run one unit of the benchmark and return its record.

    ``method`` names a registered method for an arm that is not part of the
    suite (a new method); the unit then carries its own ``hp``.
    """
    w = suite.workload(unit["setting"], unit["workload"])
    arm = suite.ARMS.get(unit["arm"])
    hp = dict(unit["hp"]) if "hp" in unit else search.configurations(w, unit["arm"], unit["n"])[int(unit["trial"])]
    if arm is not None:                     # the fixed arguments of an arm of the suite always apply
        conflict = {k for k, v in arm.fixed.items() if k in hp and hp[k] != v}
        if conflict:
            raise ValueError(f"arm {arm.name!r} has the fixed arguments {arm.fixed}; the unit gives {hp}")
        hp = {**arm.fixed, **hp}
    target = data = None
    key = (w.setting, w.name, int(unit["instance"]))
    if data_cache is not None and key in data_cache:
        target, data = data_cache[key]
    if target is None:
        from . import datasets
        target = w.target(int(unit["instance"]))
        data = datasets.make_dataset(target, w.profile.n_max, pools=("train", "val", "test"))
        if data_cache is not None:
            data_cache.clear()              # one dataset in memory per worker
            data_cache[key] = (target, data)
    result = training.run_unit(
        target, method=method or (arm.method if arm else unit["arm"]), hp=hp, n=int(unit["n"]), width=w.profile.width,
        hidden=w.profile.hidden, max_steps=w.profile.max_steps, eval_every=w.profile.eval_every, data=data,
        seed_replicate=int(unit.get("seed_replicate", 0)), keep_states=keep_states, threads=threads,
        stop_after_evals=stop_after_evals, device=device)
    record = {"unit": unit_id(unit), "setting": w.setting, "workload": w.name, "arm": unit["arm"],
              "instance": int(unit["instance"]), "n": int(unit["n"]), "trial": int(unit["trial"]), **result}
    return record


def model_of(run: dict, state: dict | None = None):
    """The target and the model of a run. Returns ``(target, model)``.

    ``run`` holds ``setting``, ``workload``, ``arm``, ``instance``, ``n`` and
    ``hp``: a record of :func:`execute` or an entry of
    :func:`csfrbench.record.checkpoints`. Without ``state`` the model is at its
    initialization. With ``state`` (``run["best_state"]`` of a run with
    ``keep_states``) the model has that state and is in evaluation mode.
    """
    w = suite.workload(run["setting"], run["workload"])
    target = w.target(int(run["instance"]))
    arm = suite.ARMS.get(run["arm"])
    _method, model, _seeds = training.build_model(
        arm.method if arm else run["method"], target, width=w.profile.width, hidden=w.profile.hidden, hp=run["hp"],
        n=int(run["n"]), seed_replicate=int(run.get("seed_replicate", 0)))
    if state is not None:
        model.load_state_dict(state)
        model.eval()
    return target, model


# ------------------------------------------------------------------ pool

_STATE: dict = {}


def _init_worker(out_dir: str, keep_states: bool, method_modules: tuple[str, ...], stop_after_evals) -> None:
    import importlib

    import torch
    torch.set_num_threads(1)                # one thread per worker process, as the records were made
    for module in method_modules:
        importlib.import_module(module)
    _STATE.update(out=Path(out_dir), keep=keep_states, cache={}, stop=stop_after_evals,
                  file=Path(out_dir) / f"runs-{socket.gethostname().split('.')[0]}-{os.getpid()}.jsonl")


def _failure(unit: dict, exc: Exception) -> dict:
    return {"unit": unit_id(unit), **{k: unit[k] for k in ("setting", "workload", "arm", "instance", "n", "trial")},
            "failed": 1, "error": f"{type(exc).__name__}: {exc}"}


def _write(record: dict) -> None:
    import torch
    if _STATE["keep"] and "best_state" in record:
        states = _STATE["out"] / "states"
        states.mkdir(parents=True, exist_ok=True)
        torch.save({"init": record.pop("init_state"), "best": record.pop("best_state"), "unit": record["unit"],
                    "hp": record["hp"]}, states / f"{record['unit']}.pt")
    record.pop("init_state", None)
    record.pop("best_state", None)
    with open(_STATE["file"], "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _work(unit: dict) -> dict:
    t0 = time.time()
    try:
        record = execute(unit, keep_states=_STATE["keep"], data_cache=_STATE["cache"], method=unit.get("method"),
                         stop_after_evals=_STATE["stop"])
    except Exception as exc:  # noqa: BLE001  the campaign continues; the failure is on record
        record = _failure(unit, exc)
    _write(record)
    return {"unit": record["unit"], "seconds": round(time.time() - t0, 1), "failed": int(record.get("failed", 0)),
            "diverged": int(record.get("diverged", 0))}


def _work_stack(units: list[dict]) -> dict:
    from . import stack
    t0 = time.time()
    try:
        records = stack.train_stack(units, keep_states=_STATE["keep"], stop_after_evals=_STATE["stop"])
    except Exception as exc:  # noqa: BLE001  the campaign continues; the failure is on record
        records = [_failure(u, exc) for u in units]
    for record in records:
        _write(record)
    return {"unit": f"stack of {len(units)}: {units[0]['workload']} {units[0]['arm']} ...",
            "seconds": round(time.time() - t0, 1), "failed": sum(int(r.get("failed", 0)) for r in records),
            "diverged": sum(int(r.get("diverged", 0)) for r in records), "count": len(records)}


def read_runs(out_dir: str | Path) -> list[dict]:
    """Every record of an output folder; a unit that is on record twice keeps its last record."""
    records: dict[str, dict] = {}
    for path in sorted(Path(out_dir).glob("runs-*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:          # a line that another process is writing
                continue
            records[r["unit"]] = r
    return list(records.values())


def run(units: list[dict], out_dir: str | Path, *, workers: int = 1, shard: tuple[int, int] | None = None,
        keep_states: bool = False, method_modules: tuple[str, ...] = (), progress: bool = True,
        stop_after_evals: int | None = None, engine: str = "auto", stack_size: int | None = None) -> dict:
    """Execute the units that are not on record in ``out_dir``. Returns a summary.

    ``method_modules`` names importable modules that register new methods; each
    worker imports them. ``stop_after_evals`` ends every run after that number
    of validation checkpoints: a tool to test a pipeline, not a result.
    ``engine`` is ``auto`` (LeWi and Dense in stacks, see :mod:`csfrbench.stack`)
    or ``exact`` (every unit with the per-network trainer); ``stack_size`` sets
    the networks per stack (default: 128 at width 32, 32 at width 64).
    """
    if engine not in ("auto", "exact"):
        raise ValueError("engine must be 'auto' or 'exact'")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if shard is not None:
        index, count = shard
        if not 0 <= index < count:
            raise ValueError(f"shard {index}/{count} is not valid")
        units = units[index::count]
    done = {r["unit"] for r in read_runs(out_dir) if not r.get("failed")}
    todo = [u for u in units if unit_id(u) not in done]
    # order by dataset, so a worker draws each dataset once
    todo.sort(key=lambda u: (u["setting"], u["workload"], u["instance"], u["n"], u["arm"], u["trial"]))
    summary = {"units": len(units), "on_record": len(units) - len(todo), "run": 0, "failed": 0, "diverged": 0}
    if not todo:
        return summary
    start = time.time()
    if engine == "exact":
        stacks, single = [], todo
    else:
        from . import stack
        stacks, single = stack.stacks_of(todo, stack_size)
    context = get_context("spawn")
    done_count = 0
    with context.Pool(int(workers), initializer=_init_worker,
                      initargs=(str(out_dir), bool(keep_states), tuple(method_modules), stop_after_evals)) as pool:
        jobs = [(_work_stack, s) for s in stacks] + [(_work, u) for u in single]
        for r in pool.imap_unordered(_dispatch, jobs, chunksize=1):
            count = int(r.get("count", 1))
            done_count += count
            summary["run"] += count
            summary["failed"] += r["failed"]
            summary["diverged"] += r["diverged"]
            if progress:
                print(f"[{done_count}/{len(todo)}] {r['unit']} {r['seconds']} s"
                      + (f" {r['failed']} FAILED" if r["failed"] else "")
                      + (f" {r['diverged']} diverged" if r["diverged"] else ""), flush=True)
    summary["seconds"] = round(time.time() - start)
    return summary


def _dispatch(job):
    function, argument = job
    return function(argument)
