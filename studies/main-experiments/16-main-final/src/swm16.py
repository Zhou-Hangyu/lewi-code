"""SWM-16: the definitive external validation of the paper (external/PLAN.md; intake/swm-rerun.md).

Sixteen world-model tasks, nine arms, T = 32 scrambled-Sobol trials and 16 paired final seeds per
(task, size, arm), every arm built through the CSFRBench v1.1 package under the campaign's readout rule
(14-main/src/swm_v10.py pins the package and the fixed hyperparameters; 09-swm-lewi's harness trains and
evaluates; both are read-only). One training run = one unit, claimed one at a time by pool workers on
free_qos packs, so a preempted pack loses only the units in flight.

    python src/swm16.py units [--mode final|audit]        pending units (trials, then the finals whose winner exists)
    python src/swm16.py pool <deadline_epoch_s> [--mode M]  claim-and-run units until the deadline minus one unit
    python src/swm16.py submit --packs K --nodes N [--mode M] [--cores 24]   free_qos packs of N nodes (24 workers per node)
    python src/swm16.py feed [--packs K --nodes N --mode M] keep K packs live until nothing is pending
    python src/swm16.py status [--mode M]                   counts per task / arm
    python src/swm16.py select [--mode M]                   write every winner whose 32 trials are complete
    python src/swm16.py read                                external/analysis: cells, curves, contrasts, tallies, integrity
    python src/swm16.py audit-read                          the audit decision (external/analysis/audit_*.csv, audit_decision.json)

Records: external/results/<mode>/hpo/<task>/n<N>/<arm>/t<tt>.json (one trial), .../<arm>.json (the winner),
external/results/<mode>/final/<task>/n<N>/<arm>/s<ss>.json (one seed). The audit writes under
external/results/audit/<setting>/ with 8 trials and 4 "devfinal" seeds per cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
STUDY = HERE.parents[1]
REPO = HERE.parents[4]
M14 = REPO / "studies" / "main-experiments" / "14-main" / "src"
S09 = REPO / "studies" / "main-experiments" / "09-swm-lewi"
EXTERNAL = STUDY / "external"
RESULTS = EXTERNAL / "results"
LOGS = EXTERNAL / "logs"
ANALYSIS = EXTERNAL / "analysis"
CONTRACT_PATH = EXTERNAL / "contract.json"
PY = os.environ.get("SWM_PYTHON", sys.executable)
JOB_PREFIX = "swm16"
USER_JOB_CAP = 46                     # the user's job cap with a margin for the interactive job and small things
DEFAULT_PACKS = 25
H_REF = 16
NMSE_FLOOR, NMSE_CAP = 1e-12, 1e6
ARMS = ("dense", "l1", "l2", "l0hc", "rigl", "fiedler", "bimt", "erb", "lewi-v1")
LABEL = {"dense": "Dense", "l1": "L1", "l2": "L2", "l0hc": "L0", "rigl": "RigL", "fiedler": "Fiedler", "bimt": "BIMT",
         "erb": "neuron positions", "lewi-v1": "LeWi"}
SPACE_KEY = {"lewi-v1": "lewi"}       # LeWi searches 09's lewi learning-rate space (its only number), as lane X did
TASKS = ["acrobot", "ballincup", "cartpole", "cheetah", "finger", "hopper", "ogbmaze", "pendulum", "pendulumctl",
         "pointmaze", "reacher", "tworoom", "walker", "pusht", "humanoid", "ogbcube"]
TASKS_ADDED = ["manipulator", "piecewise", "quadruped"]   # PLAN amendment 3 (2026-09-15): the registry tasks the 256x4 gate admits (gate256.json)
XL_GRID = [8192, 16384, 32768, 65536, 131072, 262144, 524288]
SMALL_GRID = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
TRIALS, SEEDS, STREAM = 32, 16, "final"
AUDIT = {
    "settings": {"S0": {"width": 128, "hidden": 4, "max_steps": 128000}, "S1": {"width": 128, "hidden": 4, "max_steps": 256000},
                 "S2a": {"width": 256, "hidden": 4, "max_steps": 128000}, "S2b": {"width": 256, "hidden": 4, "max_steps": 256000}},
    "cells": {"pusht": [8192, 131072], "humanoid": [8192, 131072], "ogbcube": [1024, 16384], "cheetah": [8192, 131072], "finger": [8192, 131072]},
    "arms": ["dense", "lewi-v1"], "trials": 8, "seeds": 4, "stream": "devfinal", "rule_threshold": -0.0414,
}
GATE = {                              # 09's task-health gate (protocol section 6, 09/src/gate.py) applied at this contract
    "extra": ["manipulator", "mountaincar", "ogbscene", "piecewise", "quadruped"],   # the registry tasks the 128x4 gate excluded
    "trials": 16,                     # 09's capacity-cell protocol (results/calib_v2/capacity)
    "wide_factor": 2,                 # "the contract width against twice the width, same budget"
    "stream": "hpo",
}
_ENV_READY = False


# ---------------------------------------------------------------- the harness (09 through 14-main's v1.1 gate)

def harness():
    """Import 14-main's lane-X module (the v1.1 package first on the path, the fixed hp pins, the Erb shim) and 09's
    engine; both read-only. Every training run goes through lewibench.train_one."""
    global _ENV_READY
    if not _ENV_READY:
        for p in (str(M14), str(S09 / "src")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import swm_v10  # noqa: F401  (pins the package and the harness; asserts the v1.1 snapshot)
        _ENV_READY = True
    import lewibench
    import swm_tasks
    import v05_port
    return lewibench, swm_tasks, v05_port


def contract() -> dict:
    if CONTRACT_PATH.exists():
        return json.loads(CONTRACT_PATH.read_text())
    return {"frozen": False}


def corpus_of(task: str) -> str | None:
    c = contract()
    if task == "ogbcube":
        return "ogbcube_xl" if c.get("ogbcube_xl") else None
    return f"{task}_xl"


def grid_of(task: str) -> list[int]:
    c = contract()
    return list(c.get("grids", {}).get(task) or (XL_GRID if corpus_of(task) else SMALL_GRID))


def spaces() -> dict:
    raw = json.loads((S09 / "results" / "spaces_final_v2.json").read_text())
    return {arm: {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw[SPACE_KEY.get(arm, arm)].items()} for arm in ARMS}


# ---------------------------------------------------------------- units

def root(mode: str, setting: str | None = None) -> Path:
    return RESULTS / mode / setting if setting else RESULTS / mode


def gate_grid(task: str) -> list[int]:
    return grid_of(task) if task in TASKS else XL_GRID


def gate_corpus(task: str) -> str | None:
    return corpus_of(task) if task in TASKS else f"{task}_xl"


def cells_of(mode: str) -> list[dict]:
    """Every (setting, task, n, arm) cell of a mode with its constants."""
    out = []
    if mode == "capacity":
        c = contract()
        w, h, st = c["width"], c["hidden"], c["max_steps"]
        for task in TASKS + GATE["extra"]:
            grid = gate_grid(task)
            cells = [(grid[-1], w * GATE["wide_factor"])]                  # the capacity reference: twice the width, same budget
            if task not in TASKS:                                          # a task outside the campaign also needs its contract cells
                cells += [(grid[0], w), (grid[-1], w)]
            for n, width in cells:
                out.append(dict(mode=mode, setting=f"w{width}", task=task, n=int(n), arm="dense", trials=GATE["trials"], seeds=0,
                                stream=GATE["stream"], corpus=gate_corpus(task), width=width, hidden=h, max_steps=st))
        return out
    if mode == "audit":
        for setting, cfg in AUDIT["settings"].items():
            for task, ns in AUDIT["cells"].items():
                for n in ns:
                    for arm in AUDIT["arms"]:
                        out.append(dict(mode=mode, setting=setting, task=task, n=int(n), arm=arm, trials=AUDIT["trials"], seeds=AUDIT["seeds"],
                                        stream=AUDIT["stream"], corpus=corpus_of(task), **cfg))
        return out
    c = contract()
    if not c.get("frozen"):
        raise SystemExit("external/contract.json is not frozen: run the audit first (PLAN.md section 3)")
    for task in TASKS + TASKS_ADDED:
        for n in grid_of(task):
            for arm in ARMS:
                out.append(dict(mode=mode, setting=None, task=task, n=int(n), arm=arm, trials=TRIALS, seeds=SEEDS, stream=STREAM,
                                corpus=corpus_of(task), width=c["width"], hidden=c["hidden"], max_steps=c["max_steps"]))
    return out


def cell_dir(cell: dict, kind: str) -> Path:
    return root(cell["mode"], cell["setting"]) / kind / cell["task"] / f"n{cell['n']}" / cell["arm"]


def winner_path(cell: dict) -> Path:
    return root(cell["mode"], cell["setting"]) / "hpo" / cell["task"] / f"n{cell['n']}" / f"{cell['arm']}.json"


def _have(folder: Path) -> set:
    return set(os.listdir(folder)) if folder.exists() else set()


def pending_units(mode: str, only_cell=None) -> list[dict]:
    """Task by task: the trial units of the task's cells, then the final units of its cells with a winner (a task
    completes before the next one starts, so a read at any time holds whole tasks); within a task the longest
    units first (short packs: a long unit fits only at the start of a worker's pack)."""
    by_task: dict = {}
    for cell in cells_of(mode):
        if only_cell and (cell["task"], cell["n"], cell.get("setting")) != only_cell:
            continue
        by_task.setdefault(cell["task"], []).append(cell)
    units = []
    for cells in by_task.values():
        trials, finals = [], []
        for cell in cells:
            hdir = cell_dir(cell, "hpo")
            wp = winner_path(cell)
            if not wp.exists():                                # a selected cell needs no trial (its records may be archived)
                have = _have(hdir)
                missing = [t for t in range(cell["trials"]) if f"t{t:02d}.json" not in have]
                if missing:
                    trials += [dict(cell, kind="hpo", index=t, out=hdir / f"t{t:02d}.json") for t in missing]
                    continue
                select_cell(cell)
            if not wp.exists():
                continue
            fdir = cell_dir(cell, "final")
            have = _have(fdir)
            finals += [dict(cell, kind="final", index=s, out=fdir / f"s{s:02d}.json") for s in range(cell["seeds"])
                       if f"s{s:02d}.json" not in have]
        trials.sort(key=lambda u: -unit_seconds(u))
        finals.sort(key=lambda u: -unit_seconds(u))
        units += trials + finals
    return units


RESERVE_BASE = {"dense": 900, "lewi-v1": 1280, "l1": 760, "l2": 1200, "bimt": 850, "rigl": 1040, "erb": 1280, "l0hc": 1280, "fiedler": 1020}
RESERVE_DEFAULT = 1280                            # seconds at 128 x 4 / 128k steps, x WIDTH_FACTOR at width 256: above the finals' measured
WIDTH_FACTOR = {128: 1.0, 256: 1.9, 512: 4.8}     # measured dense at n = 524,288: 17 min at 256 x 4, 43 min at 512 x 4 (2026-09-14)                                # p90 at 256 x 4 (acrobot: LeWi / neuron positions / L0 36 min, L2 34, RigL 29,
                                                  # Fiedler 25, BIMT 23, L1 19; Dense 25 in the audit); every unit ends before the deadline


def unit_seconds(cell: dict) -> int:
    """The reserve before the deadline: no unit starts with less than this left (about the p90 of a unit of the arm,
    scaled by the step budget and by WIDTH_FACTOR at width 256)."""
    base = RESERVE_BASE.get(cell["arm"], RESERVE_DEFAULT)
    return int(base * cell["max_steps"] / 128000 * WIDTH_FACTOR.get(cell["width"], 1.0))


# ---------------------------------------------------------------- running one unit

_CELL_CACHE: dict = {}


def cell_data(lewibench, task: str, n: int, corpus: str | None):
    key = (task, n, corpus)
    if key not in _CELL_CACHE:
        _CELL_CACHE.clear()
        _CELL_CACHE[key] = lewibench.make_cell_data(task, n, corpus)
    return _CELL_CACHE[key]


def split_hash(cell_obj) -> str:
    h = hashlib.sha256()
    for k in ("train", "val", "test"):
        h.update(k.encode() + json.dumps(cell_obj["splits"][k]).encode())
    h.update(b"n_actual" + str(cell_obj["n_actual"]).encode())
    return h.hexdigest()[:16]


def init_open_edges(v05_port, arm: str, in_dim: int, out_dim: int, hp: dict, seeds: dict, width: int, hidden: int) -> dict:
    """The masked transitions' open-edge counts at initialization and the redraw count (LeWi's connected-initialization
    rule); built exactly as train_one builds the model (same seeds), so the trained model of the unit had this start."""
    import torch
    torch.manual_seed(seeds["init"])
    model = v05_port.build_model(arm, in_dim, out_dim, hp, seeds["aux"], width, hidden)
    out = {"readout_redraws": getattr(model, "lewi_readout_redraws", None), "open_edges_init": []}
    layers = getattr(model, "structural_layers", None)
    for layer in (layers() if callable(layers) else []):
        mask = getattr(layer, "event_mask", None)
        if mask is not None:
            out["open_edges_init"].append(int(mask.sum().item()))
    return out


FLUSH_ENV = "SWM16_FLUSH_L2_TRIALS"


def flush_denormal(u: dict) -> bool:
    """Subnormal floats make a strong-penalty L2 unit (lambda above about 0.1) 6 to 11 times slower, past the free_qos
    hour; a pack sent with --flush-l2-trials runs the L2 trial units under torch.set_flush_denormal(True), and the
    finals of a cell whose winner trial was flushed share its numerics; recorded per unit.
    Every other arm never flushes."""
    if os.environ.get(FLUSH_ENV) != "1" or u["arm"] != "l2":
        return False
    if u["kind"] == "hpo":
        return True
    w = json.loads(winner_path(u).read_text())
    return bool((w.get("trials_summary") or [{}] * TRIALS)[w["winner_trial"]].get("flush_denormal"))


def run_unit(u: dict) -> dict:
    lewibench, swm_tasks, v05_port = harness()
    import torch
    torch.set_num_threads(1)
    flush = flush_denormal(u)
    torch.set_flush_denormal(flush)
    t0 = time.time()
    cell_obj = cell_data(lewibench, u["task"], u["n"], u["corpus"])
    space = spaces()[u["arm"]]
    if u["kind"] == "hpo":
        configs = lewibench.sobol_configs(u["task"], u["arm"], u["n"], space, u["trials"], u["mode"])
        hp = configs[u["index"]]
        seeds = lewibench.trial_seeds(u["task"], u["n"])
    else:
        hp = dict(json.loads(winner_path(u).read_text())["winner_hp"])
        seeds = lewibench.final_seeds(u["task"], u["n"], u["index"], u["stream"])
    in_dim = cell_obj["train"].tensors[0].shape[1]
    out_dim = cell_obj["train"].tensors[1].shape[1]
    full_hp = dict(v05_port.hp_defaults(u["arm"]) | dict(hp))
    max_steps = min(int(u["max_steps"]), int(os.environ.get("SWM16_SMOKE_STEPS", 10 ** 9)))   # a smoke test caps the budget; never a record
    rec = {"mode": u["mode"], "setting": u["setting"], "task": u["task"], "n": u["n"], "arm": u["arm"], "kind": u["kind"],
           "index": u["index"], "hp": hp, "hp_full": full_hp, "seeds": seeds, "stream": u["stream"] if u["kind"] == "final" else "hpo",
           "width": u["width"], "hidden": u["hidden"], "max_steps": max_steps, "batch": lewibench.BATCH, "corpus": u["corpus"] or u["task"],
           "split_hash": split_hash(cell_obj), "n_actual": cell_obj["n_actual"], "pool_size": cell_obj["pool_size"],
           "package": os.environ.get("CSFR_PKG") or str(Path(sys.modules["csfrbench"].__file__).parent), "job": os.environ.get("SLURM_JOB_ID"),
           "flush_denormal": flush, "host": os.uname().nodename, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))}
    try:
        rec["init"] = init_open_edges(v05_port, u["arm"], in_dim, out_dim, full_hp, seeds, u["width"], u["hidden"])
        result, model, _ = lewibench.train_one(u["task"], u["n"], u["arm"], hp, seeds, width=u["width"], hidden=u["hidden"],
                                               batch=lewibench.BATCH, max_steps=max_steps, cell=cell_obj, corpus_name=u["corpus"])
        rec.update(result)
        if u["kind"] == "final":
            rec["test_one_step"] = lewibench.one_step_nmse(model, cell_obj["test_ds"], cell_obj["keep_dims"])
            rec["test_rollout"] = {str(h): v["nmse"] for h, v in lewibench.rollout_nmse(
                u["task"], model, cell_obj["test_windows"], cell_obj["norm"], cell_obj["obs_dim"], keep=cell_obj["keep_dims"]).items()}
        rec["error"] = None
    except Exception as exc:  # noqa: BLE001
        rec.update({"error": f"{type(exc).__name__}: {exc}", "diverged": 1, "val_rollout_score": float("inf"), "best_val_nmse": float("inf"),
                    "test_rollout": {}, "test_one_step": {"nmse": NMSE_CAP}})
    rec["wall_s"] = time.time() - t0
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return rec


def write_record(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")          # two workers selecting the same winner do not share it
    tmp.write_text(json.dumps(rec, default=_json_default))
    os.replace(tmp, path)
    claim = path.with_suffix(".claim")
    if claim.exists():
        claim.unlink(missing_ok=True)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
        return None
    raise TypeError(str(type(o)))


def select_cell(cell: dict) -> bool:
    """The winner of a complete HPO cell (09's key: no divergence, lowest validation rollout score, lowest one-step
    validation NMSE, lowest trial)."""
    lewibench, _, _ = harness()
    hdir = cell_dir(cell, "hpo")
    recs = []
    for t in range(cell["trials"]):
        p = hdir / f"t{t:02d}.json"
        if not p.exists():
            return False
        r = json.loads(p.read_text())
        r["trial"] = t
        r.setdefault("diverged", 0)
        r["val_rollout_score"] = r.get("val_rollout_score") if r.get("val_rollout_score") is not None else float("inf")
        r["best_val_nmse"] = r.get("best_val_nmse") if r.get("best_val_nmse") is not None else float("inf")
        recs.append(r)
    w = min(range(len(recs)), key=lambda i: lewibench.winner_key(recs[i]))
    out = {"task": cell["task"], "n": cell["n"], "arm": cell["arm"], "setting": cell["setting"], "trials": cell["trials"],
           "space": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spaces()[cell["arm"]].items()},
           "winner_trial": w, "winner_hp": recs[w]["hp"], "winner_val_rollout_score": recs[w]["val_rollout_score"],
           "winner_best_val_nmse": recs[w]["best_val_nmse"], "winner_best_step": recs[w].get("best_step"),
           "diverged_trials": int(sum(int(bool(r.get("diverged"))) for r in recs)),
           "trials_summary": [{k: r.get(k) for k in ("trial", "hp", "val_rollout_score", "best_val_nmse", "best_step", "diverged", "wall_s", "flush_denormal")} for r in recs],
           "selected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_record(winner_path(cell), out)
    return True


def select(mode: str) -> None:
    n = sum(select_cell(c) for c in cells_of(mode) if not winner_path(c).exists())
    print(f"{n} winners written")


# ---------------------------------------------------------------- the pool worker

def pool(deadline: float, mode: str) -> None:
    lewibench, _, _ = harness()
    last = None
    done = 0
    while True:
        units = pending_units(mode)
        if not units:
            print("nothing pending", flush=True)
            return
        # keep the cell the worker already holds in memory when it still has units; else the first pending unit
        same = [u for u in units if last and (u["task"], u["n"], u["corpus"]) == last]
        order = same + [u for u in units if u not in same]
        now = time.time()
        fits = [u for u in order if now <= deadline - unit_seconds(u)]   # a unit that cannot end before the deadline is skipped
        if not fits:
            print(f"deadline: {done} units done, no pending unit fits", flush=True)
            return
        started = False
        for u in fits:
            if u["out"].exists() or not lewibench._claim(u["out"]):
                continue
            rec = run_unit(u)
            write_record(u["out"], rec)
            done += 1
            last = (u["task"], u["n"], u["corpus"])
            started = True
            print(f"{time.strftime('%H:%M:%S')} {u['kind']} {u['task']} n={u['n']} {u['arm']} #{u['index']} "
                  f"{'diverged' if rec.get('diverged') else ''}{'ftz ' if rec.get('flush_denormal') else ''}{rec['wall_s']:.0f}s", flush=True)
            break
        if not started:
            time.sleep(60)          # everything pending is claimed by others or waits for a winner


# ---------------------------------------------------------------- free_qos packs

# The queue verbs of the runner (job packs, their tending and status) are not part of this archive; `one`, `select`
# and the record readers below run on one machine.


def log10_nmse(rec: dict, key: str = "test_rollout") -> float:
    v = (rec.get(key) or {}).get(str(H_REF))
    if v is None or rec.get("diverged") or rec.get("error"):
        v = NMSE_CAP
    return float(math.log10(max(float(v), NMSE_FLOOR)))


def student_t(d) -> dict:
    import numpy as np
    from scipy.stats import t as st
    d = np.asarray(d, dtype=float)
    n = int(d.size)
    m = float(d.mean())
    half = float(st.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("inf")
    return {"mean": m, "lo": m - half, "hi": m + half, "n": n}


def aulc_log(grid: list[int], values: list[float]) -> float:
    import numpy as np
    x = np.log2(np.asarray(grid, dtype=float))
    y = np.asarray(values, dtype=float)
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def audit_read() -> dict:
    """The audit decision (PLAN.md section 3): paired mean log10 rollout-16 test ratio between settings over the
    cells x arms on the devfinal seeds, the majority rule, and the checkpoint positions."""
    import numpy as np
    import pandas as pd
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cells_of("audit"):
        fdir = cell_dir(c, "final")
        wp = winner_path(c)
        for s in range(c["seeds"]):
            p = fdir / f"s{s:02d}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            rows.append(dict(setting=c["setting"], task=c["task"], n=c["n"], arm=c["arm"], seed=s, log10=log10_nmse(r), diverged=int(bool(r.get("diverged"))),
                             best_step=r.get("best_step"), max_steps=c["max_steps"], frac=(r.get("best_step") or 0) / c["max_steps"],
                             wall_s=r.get("wall_s"), winner_trial=json.loads(wp.read_text())["winner_trial"] if wp.exists() else None))
    df = pd.DataFrame(rows, columns=["setting", "task", "n", "arm", "seed", "log10", "diverged", "best_step", "max_steps", "frac", "wall_s", "winner_trial"])
    df.to_csv(ANALYSIS / "audit_instances.csv", index=False)
    if df.empty:
        print("no audit final instance yet")
        return {}
    decision = {"read_local": time.strftime("%Y-%m-%d %H:%M", time.localtime()), "rule": "adopt the larger setting only if the paired mean log10 test "
                "rollout-16 ratio over cells x arms <= -0.0414 and better on a majority of cells (09 contract v2)", "comparisons": {}}
    comps = [("S0", "S1", "budget 128k -> 256k at 128x4"), ("S0", "S2a", "width 128 -> 256 at 128k"), ("S1", "S2b", "width 128 -> 256 at 256k")]
    for a, b, what in comps:
        A = df[df.setting == a].set_index(["task", "n", "arm", "seed"]).log10
        B = df[df.setting == b].set_index(["task", "n", "arm", "seed"]).log10
        idx = A.index.intersection(B.index)
        if len(idx) == 0:
            continue
        d = (B.loc[idx] - A.loc[idx])
        per_cell = d.groupby(level=["task", "n", "arm"]).mean()
        cells_better = int((per_cell < 0).sum())
        adopt = bool(d.mean() <= AUDIT["rule_threshold"] and cells_better > len(per_cell) / 2)
        decision["comparisons"][f"{a}->{b}"] = {"what": what, "pairs": int(len(idx)), "mean_dlog10": float(d.mean()), **student_t(d.values),
                                                "cells": int(len(per_cell)), "cells_better": cells_better, "adopt": adopt,
                                                "per_cell": {f"{t}/n{n}/{arm}": float(v) for (t, n, arm), v in per_cell.items()}}
    pos = df.groupby(["setting"]).agg(frac_mean=("frac", "mean"), frac_median=("frac", "median"), share_last10=("frac", lambda x: float((x >= 0.9).mean())),
                                      diverged=("diverged", "sum"), wall_median_s=("wall_s", "median"), instances=("log10", "size")).reset_index()
    pos.to_csv(ANALYSIS / "audit_positions.csv", index=False)
    decision["positions"] = pos.to_dict("records")
    (ANALYSIS / "audit_decision.json").write_text(json.dumps(decision, indent=1) + "\n")
    print(json.dumps({k: v for k, v in decision.items() if k != "comparisons"}, indent=1))
    for k, v in decision["comparisons"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "per_cell"})
    return decision


def read() -> dict:
    """The final read: cells, curves, paired contrasts, tallies, integrity (external/analysis/*.csv, read.json)."""
    import numpy as np
    import pandas as pd
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    c = contract()
    inst, integrity = [], []
    for cell in cells_of("final"):
        fdir, hdir = cell_dir(cell, "final"), cell_dir(cell, "hpo")
        recs = [json.loads((fdir / f"s{s:02d}.json").read_text()) for s in range(cell["seeds"]) if (fdir / f"s{s:02d}.json").exists()]
        hashes = {r.get("split_hash") for r in recs}
        wp = winner_path(cell)
        w = json.loads(wp.read_text()) if wp.exists() else {}
        n_tr = len(w["trials_summary"]) if w.get("trials_summary") else sum((hdir / f"t{i:02d}.json").exists() for i in range(cell["trials"]))   # trial records may be archived
        for r in recs:
            init = r.get("init") or {}
            inst.append(dict(task=cell["task"], n=cell["n"], arm=cell["arm"], seed=r["index"], log10=log10_nmse(r), nmse=(r.get("test_rollout") or {}).get(str(H_REF)),
                             one_step=(r.get("test_one_step") or {}).get("nmse"), diverged=int(bool(r.get("diverged"))), error=int(bool(r.get("error"))),
                             best_step=r.get("best_step"), frac=(r.get("best_step") or 0) / cell["max_steps"], wall_s=r.get("wall_s"),
                             flush=int(bool(r.get("flush_denormal"))),
                             split_hash=r.get("split_hash"), readout_redraws=init.get("readout_redraws"),
                             min_open_edges_init=min(init.get("open_edges_init") or [np.nan]) if init.get("open_edges_init") else np.nan,
                             params=r.get("params"), structural_params=r.get("structural_params"),
                             pin_width=r.get("width"), pin_hidden=r.get("hidden"), pin_steps=r.get("max_steps"), pin_batch=r.get("batch"),
                             pin_package=r.get("package"), pin_readout=(r.get("hp") or {}).get("readout")))
        integrity.append(dict(task=cell["task"], n=cell["n"], arm=cell["arm"], trials=n_tr, trials_expected=cell["trials"], finals=len(recs),
                              finals_expected=cell["seeds"], split_hashes=len(hashes), diverged_trials=w.get("diverged_trials"),
                              diverged_finals=int(sum(int(bool(r.get("diverged"))) for r in recs)), errors=int(sum(int(bool(r.get("error"))) for r in recs)),
                              winner_trial=w.get("winner_trial"), winner_hp=json.dumps(w.get("winner_hp")), winner_best_step=w.get("winner_best_step"),
                              flush_trials=int(sum(int(bool(t.get("flush_denormal"))) for t in w.get("trials_summary") or [])),
                              flush_winner=int(bool((w.get("trials_summary") or [{}] * TRIALS)[w.get("winner_trial") or 0].get("flush_denormal"))),
                              flush_finals=int(sum(int(bool(r.get("flush_denormal"))) for r in recs))))
    df = pd.DataFrame(inst)
    df.to_csv(ANALYSIS / "instances.csv", index=False)
    pd.DataFrame(integrity).to_csv(ANALYSIS / "integrity_cells.csv", index=False)
    if df.empty:
        print("no final records yet")
        return {}
    # curves: geometric mean over seeds (10^mean log10) with the t band of the log values
    cur = []
    for (task, n, arm), g in df.groupby(["task", "n", "arm"]):
        st = student_t(g.log10.values)
        cur.append(dict(task=task, n=n, arm=arm, label=LABEL[arm], seeds=len(g), gmean=10 ** st["mean"], glo=10 ** st["lo"], ghi=10 ** st["hi"],
                        log10_mean=st["mean"], log10_lo=st["lo"], log10_hi=st["hi"], median_nmse=float(g.nmse.median()) if g.nmse.notna().any() else np.nan,
                        diverged=int(g.diverged.sum())))
    curves = pd.DataFrame(cur)
    curves.to_csv(ANALYSIS / "curves.csv", index=False)
    # AULC per (task, arm, seed) over the task's grid where every size is present for that seed
    au = []
    for (task, arm), g in df.groupby(["task", "arm"]):
        grid = grid_of(task)
        by = g.set_index(["seed", "n"]).log10
        for s in sorted(g.seed.unique()):
            if all((s, n) in by.index for n in grid):
                au.append(dict(task=task, arm=arm, seed=s, aulc=aulc_log(grid, [by.loc[(s, n)] for n in grid])))
    aulc = pd.DataFrame(au)
    aulc.to_csv(ANALYSIS / "aulc_seeds.csv", index=False)
    gate = json.loads((ANALYSIS / "gate256.json").read_text()) if (ANALYSIS / "gate256.json").exists() else {}
    claim_bearing = [t for t in TASKS + TASKS_ADDED if t in set(gate.get("claim_bearing") or TASKS)]   # PLAN amendment 3: the gated suite
    con, tally = [], {ref: [0, 0, 0] for ref in ARMS if ref != "lewi-v1"}
    tally_cb = {ref: [0, 0, 0] for ref in ARMS if ref != "lewi-v1"}
    for task in TASKS + TASKS_ADDED:
        a = aulc[(aulc.task == task) & (aulc.arm == "lewi-v1")].set_index("seed").aulc
        for ref in ARMS:
            if ref == "lewi-v1":
                continue
            b = aulc[(aulc.task == task) & (aulc.arm == ref)].set_index("seed").aulc
            idx = a.index.intersection(b.index)
            if len(idx) < 2:
                continue
            d = (a.loc[idx] - b.loc[idx]).values
            st = student_t(d)
            verdict = "win" if st["hi"] < 0 else "loss" if st["lo"] > 0 else "tie"
            tally[ref][{"win": 0, "tie": 1, "loss": 2}[verdict]] += 1
            if task in claim_bearing:
                tally_cb[ref][{"win": 0, "tie": 1, "loss": 2}[verdict]] += 1
            con.append(dict(task=task, reference=ref, label=LABEL[ref], contrast="LeWi - " + LABEL[ref], **st, seeds_favouring_lewi=int((d < 0).sum()),
                            verdict=verdict, complete=bool(len(idx) == SEEDS), claim_bearing=bool(task in claim_bearing)))
    contrasts = pd.DataFrame(con)
    contrasts.to_csv(ANALYSIS / "contrasts.csv", index=False)
    # the strongest non-LeWi baseline per task (lowest mean AULC over seeds)
    strongest = {}
    for task, g in aulc[aulc.arm != "lewi-v1"].groupby("task"):
        m = g.groupby("arm").aulc.mean()
        strongest[task] = {"arm": str(m.idxmin()), "label": LABEL[str(m.idxmin())], "mean_aulc": float(m.min())}
    # boundary occupancy of the winners (within 5 per cent of a log-space edge)
    occ = []
    for cell in cells_of("final"):
        wp = winner_path(cell)
        if not wp.exists():
            continue
        w = json.loads(wp.read_text())
        for k, v in w["space"].items():
            if not isinstance(v, list):
                continue
            lo, hi = float(v[0]), float(v[1])
            x = float(w["winner_hp"][k])
            pos = (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo)) if v[2] else (x - lo) / (hi - lo)
            occ.append(dict(task=cell["task"], n=cell["n"], arm=cell["arm"], parameter=k, position=pos, at_low_edge=int(pos <= 0.05), at_high_edge=int(pos >= 0.95)))
    occd = pd.DataFrame(occ)
    occd.to_csv(ANALYSIS / "boundary_occupancy.csv", index=False)
    record = {"read_local": time.strftime("%Y-%m-%d %H:%M", time.localtime()), "contract": c, "arms": list(ARMS), "labels": LABEL, "tasks": TASKS + TASKS_ADDED,
              "tasks_original": TASKS, "tasks_added": TASKS_ADDED, "claim_bearing": claim_bearing,
              "excluded": {t: (next((r.get("reason") for r in gate.get("reports", []) if r.get("task") == t), "")) for t in TASKS + TASKS_ADDED if t not in claim_bearing},
              "grids": {t: grid_of(t) for t in TASKS + TASKS_ADDED}, "trials": TRIALS, "seeds": SEEDS, "h_ref": H_REF,
              "tally": {ref: dict(zip(("wins", "ties", "losses"), v)) for ref, v in tally.items()},
              "tally_claim_bearing": {ref: dict(zip(("wins", "ties", "losses"), v)) for ref, v in tally_cb.items()}, "strongest_baseline": strongest,
              "coverage": {"cells_with_winner": int(sum(winner_path(cl).exists() for cl in cells_of("final"))), "cells": len(cells_of("final")),
                           "final_instances": int(len(df)), "final_expected": len(cells_of("final")) * SEEDS,
                           "diverged_finals": int(df.diverged.sum()), "errors": int(df.error.sum())},
              "boundary": {arm: {p: {"low": int(g.at_low_edge.sum()), "high": int(g.at_high_edge.sum()), "cells": int(len(g))}
                                 for p, g in gg.groupby("parameter")} for arm, gg in occd.groupby("arm")} if len(occd) else {},
              "checkpoint_positions": {"share_last10": float((df.frac >= 0.9).mean()), "median_frac": float(df.frac.median()),
                                       "by_task": {t: {"share_last10": float((g.frac >= 0.9).mean()), "median_frac": float(g.frac.median()), "instances": int(len(g))}
                                                   for t, g in df.groupby("task")},
                                       "note": "under the cosine schedule the annealed final iterate is often the best validation checkpoint, so a position near the end "
                                               "does not by itself show a binding horizon; the horizon test is the audit's paired doubling of the step budget (PLAN.md sections 3 and 6)"},
              "pins": {"records": int(len(df)),
                       "off_contract": int(((df.pin_width != c.get("width")) | (df.pin_hidden != c.get("hidden")) | (df.pin_steps != c.get("max_steps"))).sum()),
                       "missing": int(df[["pin_width", "pin_hidden", "pin_steps", "pin_batch", "pin_package"]].isna().any(axis=1).sum()),
                       "batch": sorted(int(b) for b in df.pin_batch.dropna().unique()),
                       "packages": sorted(str(x) for x in df.pin_package.dropna().unique()),
                       "readout_not_structural": int((df.pin_readout.notna() & (df.pin_readout != "structural")).sum()),
                       "readout_recorded": int(df.pin_readout.notna().sum()),
                       "flush_denormal_trials": int(sum(i["flush_trials"] for i in integrity)),
                       "flush_denormal_trial_arms": sorted({i["arm"] for i in integrity if i["flush_trials"]}),
                       "flush_denormal_winners": int(sum(i["flush_winner"] for i in integrity)),
                       "flush_denormal_finals": int(df.flush.sum()),
                       "flush_denormal_final_cells": int(sum(1 for i in integrity if i["flush_finals"])),
                       "flush_denormal_mixed_cells": int(sum(1 for i in integrity if (i["flush_finals"] and not i["flush_winner"])
                                                            or (i["flush_winner"] and i["flush_finals"] != i["finals"])))},
              "lewi_init": {"records_with_init": int(df[df.arm == "lewi-v1"].min_open_edges_init.notna().sum()),
                            "min_open_edges": float(df[df.arm == "lewi-v1"].min_open_edges_init.min()) if (df.arm == "lewi-v1").any() else None,
                            "max_redraws": float(df[df.arm == "lewi-v1"].readout_redraws.max()) if (df.arm == "lewi-v1").any() else None}}
    (ANALYSIS / "read.json").write_text(json.dumps(record, indent=1, default=_json_default) + "\n")
    print(json.dumps({k: record[k] for k in ("tally", "coverage", "checkpoint_positions", "lewi_init")}, indent=1))
    return record


def flush_split() -> dict:
    """LeWi versus L2 per cell, split by whether the L2 cell's winner trial was flushed (PLAN amendment 2): the flag marks
    the cells where a strong penalty is L2's best configuration. external/analysis/flush_split.{csv,json}. The split
    covers the 16 original tasks: the tasks added under amendment 3 ran every L2 trial flushed from the start, so
    their 21 L2 cells have no unflushed configuration to split against."""
    import numpy as np
    import pandas as pd
    from scipy import stats
    inst = pd.read_csv(ANALYSIS / "instances.csv")
    integ = pd.read_csv(ANALYSIS / "integrity_cells.csv")
    inst, integ = inst[inst.task.isin(TASKS)], integ[integ.task.isin(TASKS)]
    flag = integ[integ.arm == "l2"].set_index(["task", "n"]).flush_winner
    rows = []
    for (task, n), g in inst[inst.arm.isin(["lewi-v1", "l2"])].groupby(["task", "n"]):
        a = g[g.arm == "lewi-v1"].set_index("seed").log10
        b = g[g.arm == "l2"].set_index("seed").log10
        idx = a.index.intersection(b.index)
        d = (a.loc[idx] - b.loc[idx]).values
        st = student_t(d)
        rows.append(dict(task=task, n=n, flushed_winner=int(flag.loc[(task, n)]), seeds=len(d), mean=st["mean"], lo=st["lo"], hi=st["hi"],
                         seeds_favouring_lewi=int((d < 0).sum()), verdict="win" if st["hi"] < 0 else "loss" if st["lo"] > 0 else "tie"))
    df = pd.DataFrame(rows).sort_values(["task", "n"])
    df.to_csv(ANALYSIS / "flush_split.csv", index=False)
    out = {}
    for f, g in df.groupby("flushed_winner"):
        out["flushed" if f else "unflushed"] = dict(cells=int(len(g)), wins=int((g.verdict == "win").sum()), ties=int((g.verdict == "tie").sum()),
                                                   losses=int((g.verdict == "loss").sum()), win_rate=float((g.verdict == "win").mean()),
                                                   mean_margin=float(g["mean"].mean()), median_margin=float(g["mean"].median()),
                                                   seeds_favouring_lewi=float(g.seeds_favouring_lewi.mean() / SEEDS))
    gf, gu = df[df.flushed_winner == 1], df[df.flushed_winner == 0]
    out["fisher_p_win_rate"] = float(stats.fisher_exact([[int((gf.verdict == "win").sum()), int((gf.verdict != "win").sum())],
                                                         [int((gu.verdict == "win").sum()), int((gu.verdict != "win").sum())]])[1])
    out["mannwhitney_p_margin"] = float(stats.mannwhitneyu(gf["mean"], gu["mean"], alternative="two-sided")[1])
    con = pd.read_csv(ANALYSIS / "contrasts.csv")
    con = con[con.reference == "l2"].set_index("task")
    out["by_task"] = {t: dict(flushed_cells=int(g.flushed_winner.sum()), cells=int(len(g)), task_verdict=str(con.loc[t].verdict), task_mean=float(con.loc[t]["mean"]),
                              cell_wins=int((g.verdict == "win").sum()), cell_losses=int((g.verdict == "loss").sum())) for t, g in df.groupby("task")}
    (ANALYSIS / "flush_split.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "by_task"}, indent=1))
    return out


# ---------------------------------------------------------------- main

def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("verb")
    p.add_argument("arg", nargs="?")
    p.add_argument("--mode", default="final", choices=["final", "audit", "capacity"])
    p.add_argument("--packs", type=int, default=DEFAULT_PACKS)
    p.add_argument("--nodes", type=int, default=4)
    p.add_argument("--cores", type=int, default=24)
    p.add_argument("--wall", default="00:59:00")
    p.add_argument("--flush-l2-trials", action="store_true", help="the L2 trial units run under torch.set_flush_denormal(True), recorded per unit")
    p.add_argument("--shape", default="fixed", choices=["fixed", "auto"], help="auto: packs shaped to the live free-core distribution (pack_shape)")
    p.add_argument("--every", type=int, default=240)
    a = p.parse_args(argv)
    if a.verb == "units":
        for u in pending_units(a.mode):
            print(u["kind"], u.get("setting") or "", u["task"], u["n"], u["arm"], u["index"])
    elif a.verb == "pool":
        pool(float(a.arg), a.mode)
    elif a.verb == "submit":
        submit(a.packs, a.nodes, a.cores, a.mode, a.wall, flush_l2_trials=a.flush_l2_trials, shape=a.shape)
    elif a.verb == "feed":
        feed(a.packs, a.nodes, a.cores, a.mode, a.wall, every=a.every, flush_l2_trials=a.flush_l2_trials, shape=a.shape)
    elif a.verb == "tend":
        tend(a.mode)
    elif a.verb == "status":
        status(a.mode)
    elif a.verb == "select":
        select(a.mode)
    elif a.verb == "read":
        read()
    elif a.verb == "audit-read":
        audit_read()
    elif a.verb == "flush-split":
        flush_split()
    elif a.verb == "one":                     # a smoke test: the first pending unit (of the arm named by `arg`), in this process
        u = next(x for x in pending_units(a.mode) if not a.arg or x["arm"] == a.arg)
        rec = run_unit(u)
        print(json.dumps({k: rec.get(k) for k in ("task", "n", "arm", "kind", "index", "best_val_nmse", "val_rollout_score", "test_rollout", "diverged", "error", "wall_s", "init", "split_hash", "package")}, default=_json_default))
    else:
        raise SystemExit(f"unknown verb {a.verb}")


if __name__ == "__main__":
    main()
