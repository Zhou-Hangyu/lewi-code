"""From runs to the reported tables.

The unit of analysis is the target instance. For each workload and method:

1. *selection*: per target instance and training-set size, the configuration
   with the lowest validation NMSE among the 32 runs (:func:`csfrbench.search.select`);
2. *learning curve*: the test NMSE of the selected run at every size of the grid;
3. *AULC*: the area under the learning curve of each instance
   (:func:`csfrbench.metrics.aulc`);
4. *report*: the arithmetic mean of the AULC over the instances with a two-sided
   95% Student-t interval, and for a comparison of two methods the mean of the
   paired differences per instance with its interval.

A comparison whose interval contains zero is inconclusive. No p-value is reported.

A run is a mapping with the keys ``workload``, ``arm``, ``instance``, ``n``,
``trial``, ``best_val_nmse``, ``test_nmse`` and ``diverged``.
"""
from __future__ import annotations

import csv
import gzip
import io
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import metrics, search, suite

OFF_SCALE = 10.0        # a selected run with a test NMSE above this value is off scale (disclosed)
DEAD = 1.0              # a selected run with a validation NMSE at or above this value did not learn (disclosed)


def select_winners(runs: list[dict], trials: int = suite.TRIALS) -> list[dict]:
    """One selected run per (workload, arm, instance, n) with a complete set of trials."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["workload"], r["arm"], int(r["instance"]), int(r["n"]))].append(r)
    winners = []
    for key in sorted(groups):
        best = search.select(groups[key], trials)
        if best is None:
            continue
        winners.append({**best, "trials": len(groups[key]),
                        "trials_diverged": sum(int(r.get("diverged", 0)) for r in groups[key])})
    return winners


def instance_aulc(winners: list[dict], sizes: dict[str, tuple[int, ...]] | None = None) -> list[dict]:
    """The AULC of every (workload, arm, instance) that has a selected run at every size of the grid.

    ``sizes`` maps a workload to its grid; by default the grid is the set of sizes on record for the workload.
    """
    by_curve: dict[tuple, dict[int, float]] = defaultdict(dict)
    grid: dict[str, set[int]] = defaultdict(set)
    for w in winners:
        if w.get("test_nmse") is None:
            continue
        by_curve[(w["workload"], w["arm"], int(w["instance"]))][int(w["n"])] = float(w["test_nmse"])
        grid[w["workload"]].add(int(w["n"]))
    rows = []
    for (workload, arm, instance), curve in sorted(by_curve.items()):
        ns = sorted(sizes[workload]) if sizes and workload in sizes else sorted(grid[workload])
        if any(n not in curve or not math.isfinite(curve[n]) for n in ns):
            continue
        values = [curve[n] for n in ns]
        rows.append({"workload": workload, "arm": arm, "instance": instance, "aulc": metrics.aulc(ns, values),
                     "off_scale": int(sum(v > OFF_SCALE for v in values)), "curve": dict(zip(ns, values))})
    return rows


def aulc_table(instances: list[dict]) -> list[dict]:
    """Mean AULC per (workload, arm) with its 95% Student-t interval over the target instances."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in instances:
        groups[(r["workload"], r["arm"])].append(r["aulc"])
    return [{"workload": wl, "arm": arm, **metrics.t_interval(values)} for (wl, arm), values in sorted(groups.items())]


def contrasts(instances: list[dict], method: str = "lewi") -> list[dict]:
    """Paired differences ``AULC(method) - AULC(arm)`` per workload; a negative value favours ``method``."""
    table: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in instances:
        table[r["workload"]][r["arm"]][int(r["instance"])] = r["aulc"]
    rows = []
    for workload, arms in sorted(table.items()):
        if method not in arms:
            continue
        for arm in (a for a in suite.ARM_ORDER if a in arms and a != method):
            shared = sorted(set(arms[method]) & set(arms[arm]))
            if not shared:
                continue
            d = np.asarray([arms[method][i] - arms[arm][i] for i in shared])
            rows.append({"workload": workload, "method": method, "arm": arm, **metrics.t_interval(d),
                         "wins": int((d < 0).sum()), "instances": len(shared)})
    return rows


def gap_closed(table: list[dict], workload: str, arm: str, lower: str = "dense", upper: str = "go_all") -> float:
    """The share of the AULC gap between ``lower`` and ``upper`` that ``arm`` closes."""
    mean = {r["arm"]: r["mean"] for r in table if r["workload"] == workload}
    return (mean[lower] - mean[arm]) / (mean[lower] - mean[upper])


def learning_curves(winners: list[dict]) -> list[dict]:
    """The test NMSE over the target instances per (workload, arm, n).

    ``mean``, ``lo``, ``hi``: the arithmetic mean and its 95% Student-t interval.
    ``geometric_mean``, ``log_lo``, ``log_hi``: the mean and the 95% Student-t
    interval in log-NMSE space, as NMSE values. Use them on a logarithmic axis.
    """
    groups: dict[tuple, list[float]] = defaultdict(list)
    for w in winners:
        if w.get("test_nmse") is not None:
            groups[(w["workload"], w["arm"], int(w["n"]))].append(float(w["test_nmse"]))
    rows = []
    for (workload, arm, n), values in sorted(groups.items()):
        v = np.asarray(values)
        linear = metrics.t_interval(v)
        log = metrics.t_interval(np.log10(np.clip(v, 1e-12, None)))
        rows.append({"workload": workload, "arm": arm, "n": n, "instances": int(v.size), "mean": linear["mean"],
                     "lo": linear["lo"], "hi": linear["hi"], "median": float(np.median(v)),
                     "geometric_mean": 10 ** log["mean"], "log_lo": 10 ** log["lo"], "log_hi": 10 ** log["hi"],
                     "off_scale": int((v > OFF_SCALE).sum())})
    return rows


def disclosures(winners: list[dict]) -> list[dict]:
    """Per (workload, arm): selected runs that did not learn, that are off scale, and diverged trials."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for w in winners:
        groups[(w["workload"], w["arm"])].append(w)
    return [{"workload": wl, "arm": arm, "selected": len(g),
             "dead": sum(float(w["best_val_nmse"]) >= DEAD for w in g),
             "off_scale": sum(w.get("test_nmse") is not None and float(w["test_nmse"]) > OFF_SCALE for w in g),
             "diverged_trials": sum(int(w.get("trials_diverged", 0)) for w in g)}
            for (wl, arm), g in sorted(groups.items())]


# ------------------------------------------------------------------ files


def read_runs(path: str | Path) -> list[dict]:
    """Read runs from a CSV file (plain or gzip) or from JSON lines."""
    import json
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        text = handle.read()
    if path.name.endswith((".jsonl", ".jsonl.gz")):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        out: dict = dict(row)
        for key in ("instance", "n", "trial", "best_step", "diverged", "trials", "trials_diverged"):
            if out.get(key) not in (None, ""):
                out[key] = int(float(out[key]))
        for key in ("best_val_nmse", "test_nmse"):
            out[key] = float(out[key]) if out.get(key) not in (None, "") else None
        if out.get("hp"):
            out["hp"] = json.loads(out["hp"])
        rows.append(out)
    return rows


def format_table(table: list[dict], digits: int = 3) -> str:
    """A text table: one row per arm, one column per workload, ``mean +- half width``."""
    workloads = [w for w in suite.SETTINGS["primary"] if any(r["workload"] == w for r in table)]
    cell = {(r["workload"], r["arm"]): f"{r['mean']:.{digits}f}±{r['half_width']:.{digits}f}" for r in table}
    arms = [a for a in suite.ARM_ORDER if any(r["arm"] == a for r in table)]
    width = max(len(w) for w in workloads) + 2
    lines = ["".join([f"{'method':18s}"] + [f"{w:>{width}s}" for w in workloads])]
    for arm in arms:
        lines.append("".join([f"{suite.ARMS[arm].paper_name:18s}"]
                             + [f"{cell.get((w, arm), '-'):>{width}s}" for w in workloads]))
    return "\n".join(lines)
