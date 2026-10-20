"""Verification of an installation against the records of the paper's experiments.

Every check compares a value that this installation computes with the value on
record. A check has one of four results:

``exact``     the value equals the record bit for bit
``close``     the value agrees with the record within the tolerance of the check
``FAIL``      the value does not agree with the record
``skipped``   the check does not apply

The records were made on Linux x86-64 with PyTorch 2.11, NumPy 2.4 and one
thread. In that environment every check is ``exact``. With another processor,
library version or thread count, floating-point results can differ in the last
digits; a training run amplifies such a difference with the number of steps.
A ``close`` result is then correct. A ``FAIL`` is an error.

The amplification depends on the configuration, mainly on the learning rate.
So a training run is compared at two horizons:

* *the first three optimizer steps*, with the tolerance ``TOL_FIRST_STEPS``.
  The reference values are those of this package in the environment of the
  records (the campaign recorded a value every 1,000 or 2,000 steps);
* *the first validation checkpoints* of the record. Each recorded run of the
  package holds its measured ``rounding_sensitivity``: the largest relative
  change of the validation NMSE at the first three checkpoints when every
  trainable value moves by one unit in the last place at the initialization.
  A run with a sensitivity of at most ``STABLE`` is a stable run: its
  checkpoints are compared with the tolerance ``TOL_FIRST_CHECKPOINTS``. A
  run that is not stable, and the end of every complete run, is compared only
  for a gross difference (``TOL_FACTOR``).

Levels:

``quick``      constants, search configurations, selection and AULC tables, values of the paper,
               recorded states, a sample of targets and the first checkpoints of five runs:
               LeWi and Dense in both settings, LeWi on the depth-5 tree (about two minutes)
``standard``   every target of both settings, every recorded state, the first checkpoints of a
               run of every method in both settings and on both backbones, linear probes,
               counterfactual pairs
``full``       ``standard`` and complete training runs of LeWi, Dense and GraphOracle(all) in both
               settings, a complete run of LeWi on the depth-5 tree, DAS fits of a recorded run
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from . import analysis, das, datasets, probes, record, runner, search, suite, targets, training

RECORD_KEY = {"Xtr_master": "X_train", "ytr_master": "y_train", "ftr_master": "f_train", "Xva": "X_val",
              "yva": "y_val", "fva": "f_val", "Xdv": "X_dev", "fdv": "f_dev", "Xte": "X_test", "fte": "f_test",
              "Xta": "X_test_audit", "fta": "f_test_audit", "Xdg": "X_diagnostic", "fdg": "f_diagnostic"}

TOL_CONFIGURATION = 1e-12       # relative: search configurations
TOL_TABLE = 1e-10               # absolute: AULC and its statistics, computed from recorded runs
TOL_NORMALIZATION = 1e-9        # relative: normalization constants of a target
TOL_EVALUATION = 1e-4           # relative: NMSE of a recorded state (one forward pass in float32)
TOL_FIRST_CHECKPOINTS = 2e-2    # relative: validation NMSE at the first checkpoints of a stable run
TOL_PROBE = 1e-3                # absolute: probe R^2 of a recorded state
TOL_FIRST_STEPS = 1e-2          # relative: validation NMSE after each of the first optimizer steps
STABLE = 1e-4                   # largest rounding sensitivity of a stable run
LEARNS = 0.8                    # a run learns in its first checkpoints: validation NMSE at the third one
TOL_FACTOR = 3.0                # ratio: a run that is not stable, or the test NMSE of a complete run
TOL_DAS = 0.1                   # absolute: test R^2 of a variable of a DAS fit


@dataclass
class Result:
    check: str
    status: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.status:8s} {self.check}" + (f"  ({self.detail})" if self.detail else "")


def _relative(a: float, b: float) -> float:
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def factor(a: float, b: float) -> float:
    """The ratio of the larger to the smaller of two positive values."""
    if not (a > 0 and b > 0):
        return math.inf
    return max(a / b, b / a)


def _grade(name: str, exact: bool, close: bool, detail: str = "") -> Result:
    return Result(name, "exact" if exact else ("close" if close else "FAIL"), detail)


def is_stable(run: dict) -> bool:
    """A recorded run whose first checkpoints do not move by more than ``STABLE`` with a change at the rounding level."""
    return run.get("rounding_sensitivity", math.inf) <= STABLE


def learns(run: dict) -> bool:
    """A recorded run whose validation NMSE at the third checkpoint is at most ``LEARNS``."""
    return len(run["val_curve"]) >= 3 and run["val_curve"][2][1] <= LEARNS


# ------------------------------------------------------------------ constants


def check_suite() -> list[Result]:
    """The settings hold the workloads, methods, instances and grids of the paper."""
    out = []
    expected = {"primary": (11088, 354816), "nuisance": (12096, 387072)}
    for setting, workloads in suite.SETTINGS.items():
        conditions = sum(len(w.arms) * len(w.instances) * len(w.profile.sizes) for w in workloads.values())
        ok = (conditions, conditions * suite.TRIALS) == expected[setting] and len(workloads) == 8
        out.append(_grade(f"suite/{setting}: 8 workloads, {expected[setting][0]} conditions, "
                          f"{expected[setting][1]} runs", ok, False))
    p, d = suite.COMMON, suite.DEEP
    ok = ((p.width, p.hidden, p.max_steps, p.eval_every, p.sizes[0], p.sizes[-1]) == (32, 4, 64000, 1000, 512, 32768)
          and (d.width, d.hidden, d.max_steps, d.eval_every, d.sizes[0], d.sizes[-1]) == (64, 8, 128000, 2000, 2048, 131072)
          and p.max_steps // p.eval_every == 64 and d.max_steps // d.eval_every == 64
          and (training.BATCH_SIZE, training.ADAM_BETAS, training.ADAM_EPS, training.LR_FINAL_FRACTION)
          == (32, (0.9, 0.999), 1e-8, 0.01) and suite.TRIALS == 32
          and (datasets.VAL_SIZE, datasets.TEST_SIZE, datasets.DIAGNOSTIC_SIZE) == (4096, 65536, 8192))
    out.append(_grade("suite/protocol constants of Section 6.1 and Appendix C.11", ok, False))
    dims = {w.name: (w.d, w.active) for w in suite.SETTINGS["primary"].values()}
    ok = all(v == (8, 8) for k, v in dims.items() if k != "depth5_tree") and dims["depth5_tree"] == (16, 16) \
        and all(w.d == 32 for w in suite.SETTINGS["nuisance"].values())
    out.append(_grade("suite/input dimensions: d = 8 (16 on the depth-5 tree); d = 32 with nuisance inputs", ok, False))
    return out


#: Table 2 of the paper: the frozen search spaces of the primary setting, as printed (three significant digits)
TABLE2 = {
    "p8b": {"dense": {"lr": (0.00143, 0.261)}, "l1": {"lr": (1.57e-6, 0.349), "lam": (1e-7, 670)},
            "l2": {"lr": (2.16e-5, 0.648), "lam": (1e-7, 1.13e3)}, "l0": {"lr": (2.67e-4, 0.0443), "lam": (1e-7, 157)},
            "rigl": {"lr": (1.01e-5, 0.234), "density": (0.0309, 1)}, "bimt": {"lr": (1.18e-5, 0.125), "lam": (1e-7, 0.453)},
            "fiedler": {"lr": (3.47e-5, 0.268), "delta": (1e-7, 2.83e8)}, "lewi": {"lr": (4.58e-6, 0.0999)},
            "go1": {"lr": (0.00161, 0.149)}, "go2": {"lr": (0.00200, 0.0799)}, "go_all": {"lr": (0.00200, 0.0799)}},
    "pos8p": {"positions": {"lr": (1.28e-5, 0.431)}},
    "p0": {"dense": {"lr": (7.70e-4, 0.475)}, "l1": {"lr": (2.72e-4, 0.457), "lam": (1e-7, 6.28e11)},
           "l2": {"lr": (1.93e-5, 0.115), "lam": (1e-7, 7.26e3)}, "l0": {"lr": (2.46e-4, 0.00839), "lam": (1e-7, 75.5)},
           "rigl": {"lr": (1.00e-5, 0.0121), "density": (0.0155, 0.720)},
           "bimt": {"lr": (9.88e-7, 0.265), "lam": (1e-7, 1.35e3)},
           "fiedler": {"lr": (5.78e-6, 0.0334), "delta": (1e-7, 2.66e7)}, "positions": {"lr": (4.20e-7, 0.194)},
           "lewi": {"lr": (7.92e-7, 0.0712)}, "go1": {"lr": (5.28e-4, 0.0301)}, "go2": {"lr": (0.00102, 0.0256)},
           "go_all": {"lr": (0.00125, 0.0190)}},
}


def check_spaces() -> list[Result]:
    """The frozen search spaces of the primary setting equal Table 2 of the paper to its printed digits."""
    bad = []
    count = 0
    for space_set, arms in TABLE2.items():
        for arm, dims in arms.items():
            have = {d["name"]: d for d in search.space(space_set, arm)}
            if set(have) != set(dims):
                bad.append(f"{space_set}/{arm}: dimensions {sorted(have)}")
                continue
            for name, (low, high) in dims.items():
                count += 1
                for printed, value in ((low, have[name]["low"]), (high, have[name]["high"])):
                    if float(f"{value:.3g}") != float(f"{printed:.3g}"):
                        bad.append(f"{space_set}/{arm}/{name}: {value:.4g} is printed as {printed:.4g}")
                if not have[name]["log"]:
                    bad.append(f"{space_set}/{arm}/{name}: not logarithmic")
    return [_grade(f"spaces/Table 2 of the paper: {count} intervals", not bad, False, "; ".join(bad[:4]))]


def check_configurations(settings=("primary", "nuisance")) -> list[Result]:
    """The 32 configurations of every (workload, method, size) equal the recorded ones."""
    out = []
    for setting in settings:
        recorded = record.trials(setting)
        exact = close = True
        worst = 0.0
        count = 0
        for w in suite.SETTINGS[setting].values():
            for arm in w.arms:
                for n in w.profile.sizes:
                    mine, theirs = search.configurations(w, arm, n), recorded[(w.name, arm, n)]
                    count += 1
                    for a, b in zip(mine, theirs):
                        if a == b:
                            continue
                        exact = False
                        if set(a) != set(b):
                            close = False
                            continue
                        for k in a:
                            if isinstance(a[k], float) or isinstance(b[k], float):
                                r = _relative(float(a[k]), float(b[k]))
                                worst = max(worst, r)
                                close &= r <= TOL_CONFIGURATION
                            else:
                                close &= a[k] == b[k]
        out.append(_grade(f"search/{setting}: {count} lists of 32 configurations", exact, close,
                          "" if exact else f"largest relative difference {worst:.2e}"))
    return out


def check_freeze_rule() -> list[Result]:
    """The construction rule of a search space on hand-computed examples."""
    start = [{"name": "lr", "low": 1e-5, "high": 1.0, "log": True, "hard": []},
             {"name": "lam", "low": 1e-7, "high": 1e3, "log": True, "hard": ["low"]}]
    selected = [{"lr": 10 ** (-3 + 2 * i / 15), "lam": 10 ** (-9 + 8 * i / 15)} for i in range(16)]
    got = search.freeze(start, selected)
    # lr: span 2 decades, padded by 0.25 decades on each side; lam: low edge clipped at the hard boundary
    ok = (abs(math.log10(got[0]["low"]) + 3.25) < 1e-12 and abs(math.log10(got[0]["high"]) + 0.75) < 1e-12
          and got[1]["low"] == 1e-7 and abs(math.log10(got[1]["high"]) - 0.0) < 1e-12)
    return [_grade("search/construction rule of a search space (25% of the span in log space)", ok, False)]


# ------------------------------------------------------------------ analysis


def check_analysis(settings=("primary", "nuisance")) -> list[Result]:
    """Selection and AULC tables from the recorded runs equal the recorded tables and the paper."""
    out = []
    instances, tables, comparisons = {}, {}, {}
    for setting in settings:
        winners = analysis.select_winners(record.unit_rows(setting))
        ledger = {(r["workload"], r["arm"], r["instance"], r["n"]): r for r in record.winners(setting)}
        same = len(winners) == len(ledger) and all(
            (w["trial"], w["best_val_nmse"], w["test_nmse"], w["best_step"])
            == tuple(ledger[(w["workload"], w["arm"], w["instance"], w["n"])][k]
                     for k in ("trial", "best_val_nmse", "test_nmse", "best_step")) for w in winners)
        out.append(_grade(f"analysis/{setting}: selection over {len(ledger) * suite.TRIALS} recorded runs gives the "
                          f"{len(ledger)} recorded winners", same, False))

        sizes = {w.name: w.profile.sizes for w in suite.SETTINGS[setting].values()}
        inst = analysis.instance_aulc(record.winners(setting), sizes)
        rec_inst = {(r["workload"], r["arm"], r["instance"]): r["aulc"] for r in record.aulc_instances(setting)}
        diffs = [abs(r["aulc"] - rec_inst[(r["workload"], r["arm"], r["instance"])]) for r in inst]
        ok = len(inst) == len(rec_inst)
        out.append(_grade(f"analysis/{setting}: AULC of {len(rec_inst)} learning curves", ok and max(diffs) == 0.0,
                          ok and max(diffs) <= TOL_TABLE, f"largest difference {max(diffs):.2e}"))

        table = analysis.aulc_table(inst)
        rec_table = {(r["workload"], r["arm"]): r for r in record.aulc(setting)}
        worst = max(max(abs(r[k] - rec_table[(r["workload"], r["arm"])][k]) for k in ("mean", "lo", "hi"))
                    for r in table if math.isfinite(r["mean"]) and abs(r["mean"]) < 1e3)
        ok = len(table) == len(rec_table)
        out.append(_grade(f"analysis/{setting}: mean AULC and 95% interval of {len(rec_table)} (workload, method) cells",
                          ok and worst == 0.0, ok and worst <= TOL_TABLE, f"largest difference {worst:.2e}"))

        cons = analysis.contrasts(inst)
        rec_cons = {(r["workload"], r["arm"]): r for r in record.contrasts(setting)}
        ok = len(cons) == len(rec_cons)
        worst = max(max(abs(r[k] - rec_cons[(r["workload"], r["arm"])][k]) for k in ("mean", "lo", "hi"))
                    for r in cons if abs(r["mean"]) < 1e3) if ok else float("inf")
        wins = all(r["wins"] == rec_cons[(r["workload"], r["arm"])]["wins"] for r in cons) if ok else False
        out.append(_grade(f"analysis/{setting}: {len(rec_cons)} paired comparisons LeWi - method", ok and wins and worst == 0.0,
                          ok and wins and worst <= TOL_TABLE, f"largest difference {worst:.2e}"))
        instances[setting], tables[setting], comparisons[setting] = inst, table, cons

    if "primary" in tables:
        out.extend(_check_paper(tables["primary"], comparisons["primary"]))
    if len(tables) == 2:
        out.extend(_check_paper_nuisance(instances))
    return out


def _check_paper(table: list[dict], cons: list[dict]) -> list[Result]:
    paper = record.paper_tables()
    mean = {(r["workload"], r["arm"]): r for r in table}
    con = {(r["workload"], r["arm"]): r for r in cons}
    bad, count = [], 0
    blocks = dict(paper["table3"].items())
    blocks = {k: v for k, v in blocks.items() if isinstance(v, dict)}
    blocks.update({k: v["aulc"] for k, v in paper["table4"].items()})
    for workload, arms in blocks.items():
        for arm, (m, h) in arms.items():
            count += 1
            r = mean[(workload, arm)]
            if f"{r['mean']:.3f}" != f"{m:.3f}" or f"{r['half_width']:.3f}" != f"{h:.3f}":
                bad.append(f"{workload}/{arm}: {r['mean']:.3f}±{r['half_width']:.3f} is printed as {m:.3f}±{h:.3f}")
    out = [_grade(f"paper/Tables 3 and 4: {count} values of mean AULC ± half-width", not bad, False, "; ".join(bad[:4]))]
    bad, count = [], 0
    for workload, block in paper["table4"].items():
        for arm, (m, lo, hi, wins) in block["lewi_minus"].items():
            count += 1
            r = con[(workload, arm)]
            if any(f"{a:+.3f}" != f"{b:+.3f}" for a, b in ((r["mean"], m), (r["lo"], lo), (r["hi"], hi))) \
                    or r["wins"] != wins:
                bad.append(f"{workload}/{arm}: {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] ({r['wins']})")
    out.append(_grade(f"paper/Table 4: {count} paired comparisons with interval and wins", not bad, False,
                      "; ".join(bad[:4])))
    bad = []
    for workload, percent in paper["text"]["lewi_below_dense_percent"].items():
        value = 100 * (1 - mean[(workload, "lewi")]["mean"] / mean[(workload, "dense")]["mean"])
        if round(value) != percent:
            bad.append(f"{workload}: {value:.1f}% is printed as {percent}%")
    for arm, percent in paper["text"]["gap_closed_percent_depth4_tree"].items():
        value = 100 * analysis.gap_closed(table, "depth4_tree", arm)
        if round(value) != percent:
            bad.append(f"gap closed by {arm}: {value:.1f}% is printed as {percent}%")
    below = ", ".join(f"{v}%" for v in paper["text"]["lewi_below_dense_percent"].values())
    gaps = " and ".join(f"{v}%" for v in paper["text"]["gap_closed_percent_depth4_tree"].values())
    out.append(_grade(f"paper/Section 6.2: LeWi below Dense by {below}; gap closed {gaps}", not bad, False, "; ".join(bad)))
    return out


def _check_paper_nuisance(instances: dict[str, list[dict]]) -> list[Result]:
    """Section 6.4: with nuisance inputs the advantage of LeWi over Dense increases on the seven 8-input workloads."""
    paper = record.paper_tables()["text"]["nuisance_inputs"]
    difference = {}
    for setting, inst in instances.items():
        aulc = {(r["workload"], r["arm"], r["instance"]): r["aulc"] for r in inst}
        for workload in paper["lewi_advantage_over_dense_increases"]:
            w = suite.workload(setting, workload)
            difference[(setting, workload)] = np.asarray(
                [aulc[(workload, "lewi", i)] - aulc[(workload, "dense", i)] for i in w.instances])
    bad = []
    for workload in paper["lewi_advantage_over_dense_increases"]:
        p, n = difference[("primary", workload)], difference[("nuisance", workload)]
        if not (n.mean() < p.mean() and np.median(n) < np.median(p)):
            bad.append(f"{workload}: {p.mean():+.3f} with every input active, {n.mean():+.3f} with nuisance inputs")
    for setting, better in paper["better_on_the_high_arity_node"].items():
        d = difference[(setting, "high_arity_node")]
        if not (d > 0 if better == "dense" else d < 0).all():
            bad.append(f"high-arity node, {setting}: {better} is not better on every instance")
    return [_grade("paper/Section 6.4: LeWi - Dense decreases with nuisance inputs on the 7 workloads with 8 active inputs",
                   not bad, False, "; ".join(bad[:4]))]


def check_search_audit() -> list[Result]:
    """Appendix C.12: the fraction of the selected values within 10% of a boundary of their search space."""
    paper = record.paper_tables()["search_audit"]
    values = defaultdict(list)
    dimension = {}
    for setting in suite.SETTINGS:
        for w in record.winners(setting):
            workload = suite.workload(setting, w["workload"])
            for dim in search.space(workload.space_set_of(w["arm"]), w["arm"]):
                key = (setting, workload.profile.name, w["arm"], dim["name"])
                values[key].append(w["hp"][dim["name"]])
                dimension[key] = dim
    audit = {key: search.boundary_occupancy(dimension[key], values[key], paper["margin"]) for key in values}
    near = {key: a["at_low"] + a["at_high"] for key, a in audit.items()}
    primary = [key for key in audit if key[0] == "primary"]
    medians = [audit[key]["median_position"] for key in primary]
    got = {"primary_largest_occupancy": round(max(near[key] for key in primary), 2),
           "primary_median_position": [round(min(medians), 2), round(max(medians), 2)],
           "primary_lewi_lr_percent": {b: round(100 * near[("primary", b, "lewi", "lr")]) for b in ("32x4", "64x8")},
           "nuisance_64x8_lam_percent": {a: round(100 * near[("nuisance", "64x8", a, "lam")]) for a in ("bimt", "l1")}}
    bad = [f"{k}: {got[k]} is printed as {paper[k]}" for k in got if got[k] != paper[k]]
    return [_grade("paper/Appendix C.12: search-space audit (0.17; 0.31 to 0.76; LeWi 2% and 3%; BIMT 33%, L1 26%)",
                   not bad, False, "; ".join(bad))]


# ------------------------------------------------------------------ targets


def check_targets(sample: int | None = 16) -> list[Result]:
    """Fingerprint, normalization and the tensors of the datasets of the recorded target instances."""
    items = sorted(record.targets().items())
    if sample:
        items = items[:: max(1, len(items) // sample)][:sample]
    exact = close = True
    notes = []
    tensors = 0
    for key, rec in items:
        presets = targets.FAMILIES[rec["family"]][1]
        preset = next((p for p, kw in presets.items()
                       if len(presets) == 1 or tuple(kw.get("profile", ())) == tuple(rec["layer_profile"])))
        target = targets.draw(rec["family"], preset, rec["d"], rec["instance"])
        if target.fingerprint != rec["fingerprint"] or target.target_id != rec["target_id"]:
            exact = close = False
            notes.append(f"{key}: fingerprint {target.fingerprint}")
            continue
        mean = float(np.asarray(target.node_mu[-1]).reshape(-1)[0])
        std = float(np.asarray(target.node_sd[-1]).reshape(-1)[0])
        if (mean, std) != (rec["output_mean"], rec["output_std"]):
            exact = False
            if abs(mean - rec["output_mean"]) > 1e-9 or _relative(std, rec["output_std"]) > TOL_NORMALIZATION:
                close = False
                notes.append(f"{key}: normalization")
        data = datasets.make_dataset(target, rec["n_max"], dataset_seed=rec["dataset_seed"], pools=datasets.POOLS)
        for rk, nk in RECORD_KEY.items():
            tensors += 1
            if datasets.tensor_sha256(data[nk]) != rec["sha256"][rk]:
                exact = False
                x = data[nk]
                # without the recorded values, a tensor that differs is checked for the properties of its pool
                if nk.startswith("X_") and (float(x.min()) < -1.0 or float(x.max()) > 1.0):
                    close = False
                    notes.append(f"{key}: {nk} outside [-1, 1]")
                if nk.startswith("f_") and nk != "f_train" and abs(float(x.double().var()) - 1.0) > 0.05:
                    close = False
                    notes.append(f"{key}: variance of {nk} is {float(x.double().var()):.3f}")
    return [_grade(f"targets/{len(items)} datasets: fingerprint, normalization, SHA-256 of {tensors} tensors",
                   exact, close, "; ".join(notes[:3]))]


def _sha(arrays) -> str:
    import hashlib
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


LOCAL_NAME = {"single_function": "single_function", "fourier_single": "single_fourier",
              "high_arity_stress": "high_arity_stress", "high_arity_node": "high_arity_node"}


def check_target_values(sample: int | None = None) -> list[Result]:
    """Graph, normalization, values, intermediates and gradients of the targets on a fixed input."""
    items = sorted(record.target_values().items())
    if sample:
        items = items[:: max(1, len(items) // sample)][:sample]
    exact = close = True
    notes = []
    for key, want in items:
        name, d, instance, sigma = key.split("|")
        d, instance = int(d), int(instance)
        if float(sigma) == 0.0:
            target = suite.local_function(LOCAL_NAME[name], instance)
        else:
            w = suite.workload("primary", name)
            target = targets.draw(w.family, w.preset, d, instance)
        X = np.random.default_rng(12345).uniform(-1.0, 1.0, size=(4096, d)).astype(np.float32)
        structure = (target.fingerprint == want["fingerprint"]
                     and [int(i) for i in np.ravel(target.active_raw)] == want["active"]
                     and [list(p) for p in target.graph.node_parents] == want["node_parents"]
                     and list(target.graph.readout_indices) == want["readout"]
                     and target.topology_hash == want["topology_hash"])
        if not structure:
            exact = close = False
            notes.append(f"{key}: graph or fingerprint")
            continue
        f = target.evaluate(X)
        mu = [float(np.asarray(v).reshape(-1)[0]) for v in target.node_mu]
        sd = [float(np.asarray(v).reshape(-1)[0]) for v in target.node_sd]
        same = (_sha([f]) == want["f"] and _sha([target.intermediates(X)]) == want["intermediates"]
                and _sha([target.gradients(X[:256])]) == want["gradients"] and mu == want["node_mu"]
                and sd == want["node_sd"] and [float(target.out_mu), float(target.out_sd)] == want["out"])
        if same:
            continue
        exact = False
        near = (np.allclose(f[:8], want["f_head"], rtol=0, atol=1e-5) and np.allclose(mu, want["node_mu"], atol=1e-9)
                and np.allclose(sd, want["node_sd"], rtol=TOL_NORMALIZATION))
        if not near:
            close = False
            notes.append(f"{key}: values")
    return [_grade(f"targets/{len(items)} target instances: graph, normalization, values, intermediates, gradients",
                   exact, close, "; ".join(notes[:3]))]


# ------------------------------------------------------------------ recorded states


def _model_of(run: dict, which: str | None = None):
    return runner.model_of(run, None if which is None else run[which])


def check_states(limit: int | None = None) -> list[Result]:
    """Recorded states: the initialization is rebuilt, the selected state gives the recorded NMSE."""
    runs = sorted(record.checkpoints().items())
    if limit:
        runs = runs[:: max(1, len(runs) // limit)][:limit]
    out = []
    init_exact = init_close = eval_exact = eval_close = True
    notes = []
    torch.set_num_threads(1)
    for key, run in runs:
        target, model = _model_of(run)
        if training.unit_seeds(target.fingerprint, run["n"], target.function_seed) != run["seeds"]:
            init_exact = init_close = False
            notes.append(f"{key}: seeds")
        state = model.state_dict()
        if set(state) != set(run["init"]):
            init_exact = init_close = False
            notes.append(f"{key}: keys of the state")
            continue
        for k, v in state.items():
            if not torch.equal(v, run["init"][k]):
                init_exact = False
                if not torch.allclose(v.double(), run["init"][k].double(), rtol=1e-5, atol=1e-6):
                    init_close = False
                    notes.append(f"{key}: initial {k}")
        if sum(p.numel() for p in model.parameters()) != run["n_params"]:
            init_exact = init_close = False
            notes.append(f"{key}: number of parameters")

        model.load_state_dict(run["best"])
        model.eval()
        data = datasets.make_dataset(target, 32, pools=("val", "test"))
        val = training.metrics.nmse(model, training.make_loader(
            training.TensorPool(data["X_val"], data["y_val"]), training.EVAL_BATCH_SIZE))
        errors, targets = training.metrics.pointwise_squared_errors(model, training.make_loader(
            training.TensorPool(data["X_test"], data["f_test"]), training.EVAL_BATCH_SIZE))
        test = training.metrics.nmse_from_errors(errors, targets)
        if val != run["best_val_nmse"] or _relative(test, run["test_nmse"]) > 1e-12:
            eval_exact = False
            if _relative(val, run["best_val_nmse"]) > TOL_EVALUATION or _relative(test, run["test_nmse"]) > TOL_EVALUATION:
                eval_close = False
                notes.append(f"{key}: NMSE {val!r} {test!r}")
    arms = sorted({r["arm"] for _k, r in runs})
    out.append(_grade(f"states/{len(runs)} recorded runs of {len(arms)} methods: the initialization is rebuilt",
                      init_exact, init_close, "; ".join(notes[:3])))
    out.append(_grade(f"states/{len(runs)} recorded selected states give the recorded validation and test NMSE",
                      eval_exact, eval_close, "; ".join(notes[:3])))
    return out


# ------------------------------------------------------------------ training


def check_first_steps(runs: list[str] | None = None) -> list[Result]:
    """The validation and training NMSE after each of the first optimizer steps of recorded runs."""
    recorded = record.curves()
    keys = [k for k in (runs if runs is not None else sorted(recorded)) if "first_steps" in recorded[k]]
    if not keys:
        return [Result("training/first optimizer steps", "skipped", "no run of the selection has recorded first steps")]
    exact = close = True
    worst = 0.0
    notes = []
    for key in keys:
        run = recorded[key]
        w = suite.workload(run["setting"], run["workload"])
        want = [list(r) for r in run["first_steps"]]
        result = training.run_unit(w.target(run["instance"]), method=suite.ARMS[run["arm"]].method, hp=run["hp"],
                                   n=run["n"], width=w.profile.width, hidden=w.profile.hidden,
                                   max_steps=w.profile.max_steps, eval_every=1, stop_after_evals=len(want),
                                   test_pools=())
        got = result["val_curve"]
        if got == want:
            continue
        exact = False
        difference = max((max(_relative(a[1], b[1]), _relative(a[2], b[2])) for a, b in zip(got, want)),
                         default=math.inf) if len(got) == len(want) else math.inf
        worst = max(worst, difference)
        if difference > TOL_FIRST_STEPS:
            close = False
            notes.append(f"{key.replace('|', ' ')}: {difference:.2e}")
    arms = sorted({recorded[k]["arm"] for k in keys})
    return [_grade(f"training/first 3 optimizer steps of {len(keys)} runs of {len(arms)} methods", exact, close,
                   "; ".join(notes[:3]) if notes else ("" if exact else f"largest relative difference {worst:.2e}"))]


def first_step_runs() -> list[str]:
    """One recorded run with first optimizer steps for every method, setting and backbone, the shortest first."""
    seen, keys = set(), []
    for key, run in sorted(record.curves().items(), key=lambda kv: (kv[1]["n"], kv[0])):
        tag = (run["setting"], run["arm"], suite.workload(run["setting"], run["workload"]).profile.name)
        if "first_steps" in run and tag not in seen:
            seen.add(tag)
            keys.append(key)
    return keys


def check_training(runs: list[str] | None = None, checkpoints: int = 2, complete: bool = False) -> list[Result]:
    """Training runs against the recorded validation curves.

    With ``complete`` the run uses its whole budget and the selected checkpoint,
    its state and its test NMSE are compared too.
    """
    recorded = record.curves()
    keys = runs if runs is not None else sorted(recorded)
    out = []
    for key in keys:
        run = recorded[key]
        w = suite.workload(run["setting"], run["workload"])
        start = time.time()
        result = training.run_unit(w.target(run["instance"]), method=suite.ARMS[run["arm"]].method, hp=run["hp"],
                                   n=run["n"], width=w.profile.width, hidden=w.profile.hidden,
                                   max_steps=w.profile.max_steps, eval_every=w.profile.eval_every,
                                   keep_states=complete, stop_after_evals=None if complete else checkpoints)
        mine, theirs = result["val_curve"], run["val_curve"]
        k = len(theirs) if complete else min(checkpoints, len(theirs))
        first = min(k, checkpoints)
        exact = mine[:k] == [list(r) for r in theirs[:k]]
        worst = max((_relative(a[1], b[1]) for a, b in zip(mine[:first], theirs[:first])), default=math.inf)
        ratio = max((factor(a[1], b[1]) for a, b in zip(mine[:first], theirs[:first])), default=math.inf)
        stable = is_stable(run)
        close = len(mine) >= first and (worst <= TOL_FIRST_CHECKPOINTS if stable else ratio <= TOL_FACTOR)
        detail = f"{time.time() - start:.0f} s" + ("" if exact else f"; largest relative difference {worst:.2e}"
                                                   + ("" if stable else " (the run is sensitive to rounding)"))
        name = f"training/{key.replace('|', ' ')}: "
        if not complete:
            out.append(_grade(name + f"first {k} validation checkpoints", exact, close, detail))
            continue
        test = result.get("test_nmse")
        if test is None:
            out.append(Result(name + f"complete run of {w.profile.max_steps} steps", "FAIL", detail + "; the run diverged"))
            continue
        same_state = all(torch.equal(result["best_state"][n], run["best"][n]) for n in run.get("best", {}))
        exact = (exact and same_state and result["best_step"] == run["best_step"]
                 and result["best_val_nmse"] == run["best_val_nmse"] and _relative(test, run["test_nmse"]) <= 1e-12)
        close = close and factor(test, run["test_nmse"]) <= TOL_FACTOR
        detail += f"; test NMSE {test:.6g}, on record {run['test_nmse']:.6g}"
        out.append(_grade(name + f"complete run of {w.profile.max_steps} steps", exact, close, detail))
    return out


def one_run_per_method(setting: str | None = None, arms: tuple[str, ...] | None = None, deep: bool = False) -> list[str]:
    """One recorded run of the package for every method (and setting): a stable run that learns, else a stable run,
    else the shortest run.

    ``deep`` selects the runs of the depth-5 tree (width 64, eight hidden layers).
    """
    seen, keys = set(), []
    order = sorted(record.curves().items(), key=lambda kv: (not (is_stable(kv[1]) and learns(kv[1])),
                                                            not is_stable(kv[1]), kv[1]["n"], kv[0]))
    for key, run in order:
        tag = (run["setting"], run["arm"])
        w = suite.workload(run["setting"], run["workload"])
        if tag in seen or (setting and run["setting"] != setting) or (arms and run["arm"] not in arms) \
                or (w.profile is suite.DEEP) != deep:
            continue
        seen.add(tag)
        keys.append(key)
    return keys


# ------------------------------------------------------------------ diagnostics


def check_probes(limit: int | None = None) -> list[Result]:
    """Linear probes of recorded states against the recorded probe matrices."""
    diag = {k: v for k, v in sorted(record.diagnostics().items()) if "probe" in v}
    keys = list(diag)[:limit] if limit else list(diag)
    exact = close = True
    worst = 0.0
    for key in keys:
        run = record.checkpoints()[key]
        for which in ("best", "init"):
            target, model = _model_of(run, which)
            X = datasets.make_dataset(target, 32, pools=("diagnostic",))["X_diagnostic"]
            got = probes.probe(model, target, X)
            want = diag[key]["probe"][f"r2_{which}"]
            if got["intermediates"] != list(diag[key]["probe"]["intermediates"]) or got["r2"].shape != want.shape:
                exact = close = False
                continue
            if not np.array_equal(got["r2"], want):
                exact = False
                # a probe with R^2 far below 0 has no stable value; compare where the record is above -1
                mask = want > -1.0
                difference = float(np.abs(got["r2"] - want)[mask].max())
                worst = max(worst, difference)
                close &= difference <= TOL_PROBE
    return [_grade(f"probes/{len(keys)} recorded runs, initialization and selected state", exact, close,
                   "" if exact else f"largest difference {worst:.2e}")]


def check_pairs(limit: int = 3) -> list[Result]:
    """The counterfactual pairs have the properties of the protocol."""
    ok = True
    for w in list(suite.SETTINGS["primary"].values())[:limit]:
        target = w.target(6000)
        p = das.pairs(target, "das_val")
        m = len(das.variables(target))
        counts = np.bincount(p["var_idx"], minlength=m)
        ok &= p["Xb"].shape == (m * das.VAL_PAIRS, target.d) and bool((counts == das.VAL_PAIRS).all())
        # an intervention with the source equal to the base gives the target itself
        same = das.counterfactual_output(target, p["Xb"][:256], p["Xb"][:256], das.variables(target)[0]["node"])
        ok &= bool(np.array_equal(same, target.evaluate(p["Xb"][:256])))
    return [_grade(f"das/counterfactual pairs of {limit} targets: balance, shape and identity intervention", ok, False)]


def check_das(limit: int = 1, layers=(1,), seeds=(0,)) -> list[Result]:
    """DAS fits of recorded states against the recorded fits."""
    diag = {k: v for k, v in sorted(record.diagnostics().items()) if "das" in v}
    out = []
    for key in list(diag)[:limit]:
        run = record.checkpoints()[key]
        target, model = _model_of(run, "best")
        data = das.pools(target)
        for fit in diag[key]["das"]:
            if fit["layer"] not in layers or fit["seed"] not in seeds:
                continue
            start = time.time()
            got = das.fit_layer(model, target, fit["layer"], fit["seed"], data)
            exact = all(list(got[k]) == list(fit[k]) for k in ("test_per_variable", "val_per_variable",
                                                              "no_edit_per_variable", "dims")) \
                and got["best_step"] == fit["best_step"]
            worst = max(abs(a - b) for a, b in zip(got["test_per_variable"], fit["test_per_variable"]))
            out.append(_grade(f"das/{key.replace('|', ' ')}: layer {fit['layer']}, restart {fit['seed']}", exact,
                              worst <= TOL_DAS, f"{time.time() - start:.0f} s"
                              + ("" if exact else f"; largest difference of the test R^2 {worst:.2e}")))
    return out


# ------------------------------------------------------------------ levels


def run(level: str = "quick", progress=print) -> list[Result]:
    """Run the checks of a level and return the results."""
    if level not in ("quick", "standard", "full"):
        raise ValueError("level must be quick, standard or full")
    torch.set_num_threads(1)
    steps = [check_suite, check_spaces, check_freeze_rule, check_configurations, check_analysis, check_search_audit]
    lewi_deep = one_run_per_method("primary", ("lewi",), deep=True)
    if level == "quick":
        short = one_run_per_method(arms=("lewi", "dense")) + lewi_deep
        steps += [lambda: check_target_values(40), lambda: check_targets(16), lambda: check_states(12),
                  lambda: check_first_steps(first_step_runs()),
                  lambda: check_training(short), lambda: check_probes(2), lambda: check_pairs(2)]
    else:
        steps += [check_target_values, lambda: check_targets(None), check_states, check_first_steps,
                  lambda: check_training(one_run_per_method() + one_run_per_method(deep=True)), check_probes,
                  lambda: check_pairs(8)]
    if level == "full":
        steps += [lambda: check_training(one_run_per_method(arms=("lewi", "dense", "go_all")) + lewi_deep, complete=True),
                  lambda: check_das(1, layers=(1, 2, 3, 4), seeds=(0, 1, 2))]
    results: list[Result] = []
    for step in steps:
        for r in step():
            results.append(r)
            if progress:
                progress(str(r))
    return results


def summary(results: list[Result]) -> dict[str, int]:
    out = {"exact": 0, "close": 0, "FAIL": 0, "skipped": 0}
    for r in results:
        out[r.status] += 1
    return out
