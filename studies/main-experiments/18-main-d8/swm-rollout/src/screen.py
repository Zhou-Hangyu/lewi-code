"""Screen of the sealed SWM-16 records (16-main-final/external/results/final; read only).

Per (task, arm, n) cell: the winner trial and its validation values, the validation and test values of the 16
finals, the fragility index (validation only), exploded finals, and the interior bump of the rollout curve.

    python screen.py harvest     tables/units_final.csv, tables/units_trial.csv (one pass over the records)
    python screen.py cells       tables/cells_screen.csv, tables/bumps.csv, tables/screen_by_arm.csv
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
STUDY = HERE.parents[1]
REPO = HERE.parents[5]
SEALED = REPO / "studies/main-experiments/16-main-final/external/results/final"
TABLES = STUDY / "tables"
ARMS = ("dense", "l1", "l2", "l0hc", "rigl", "fiedler", "bimt", "erb", "lewi-v1")
EXPLODED = 1.0            # log10: a final whose value is above 10 x the median of its cell's 16 finals


def _cell(wp: Path):
    finals, trials = [], []
    w = json.loads(wp.read_text())
    task, n, arm = w["task"], int(w["n"]), w["arm"]
    for t in w["trials_summary"]:
        hp = t["hp"]
        trials.append(dict(task=task, n=n, arm=arm, trial=t["trial"], winner=int(t["trial"] == w["winner_trial"]),
                           lr=hp.get("lr"), hp=json.dumps(hp, sort_keys=True), val_score=t["val_rollout_score"],
                           val_one_step=t["best_val_nmse"], best_step=t["best_step"], diverged=t["diverged"]))
    for fp in sorted((SEALED / "final" / task / f"n{n}" / arm).glob("s*.json")):
        r = json.loads(fp.read_text())
        vr, tr, d = r.get("val_rollout") or {}, r.get("test_rollout") or {}, r.get("diagnostics") or {}
        finals.append(dict(task=task, n=n, arm=arm, seed=r["index"], lr=r["hp"].get("lr"), val_score=r.get("val_rollout_score"),
                           val_one_step=r.get("best_val_nmse"), val_h1=vr.get("1"), val_h16=vr.get("16"),
                           test_one_step=(r.get("test_one_step") or {}).get("nmse"), test_h1=tr.get("1"), test_h4=tr.get("4"),
                           test_h8=tr.get("8"), test_h16=tr.get("16"), test_h32=tr.get("32"), best_step=r.get("best_step"), diverged=r.get("diverged"),
                           density=d.get("masked_boundary_density"), readout_open=d.get("readout_open_frac")))
    return finals, trials


def harvest() -> None:
    from concurrent.futures import ThreadPoolExecutor
    finals, trials = [], []
    paths = sorted((SEALED / "hpo").glob("*/n*/*.json"))
    with ThreadPoolExecutor(48) as ex:                       # the read is latency bound (one small file per record)
        for f, t in ex.map(_cell, paths):
            finals += f
            trials += t
    TABLES.mkdir(exist_ok=True)
    pd.DataFrame(finals).to_csv(TABLES / "units_final.csv", index=False)
    pd.DataFrame(trials).to_csv(TABLES / "units_trial.csv", index=False)
    print(len(finals), "finals,", len(trials), "trials")


def log10(x):
    return np.log10(np.clip(np.asarray(x, dtype=float), 1e-12, 1e6))


def cells() -> None:
    f = pd.read_csv(TABLES / "units_final.csv")
    t = pd.read_csv(TABLES / "units_trial.csv")
    for c in ("val_score", "val_one_step"):
        t[c] = t[c].fillna(1e6)
    rows = []
    for (task, arm, n), g in f.groupby(["task", "arm", "n"]):
        tt = t[(t.task == task) & (t.arm == arm) & (t.n == n)].sort_values("val_score")
        w = tt[tt.winner == 1].iloc[0]
        lv, lt = log10(g.val_score.fillna(1e6)), log10(g.test_h16.fillna(1e6))
        rows.append(dict(
            task=task, arm=arm, n=n, seeds=len(g), winner_trial=int(w.trial), winner_lr=w.lr,
            trial_val_score=w.val_score, trial_val_one_step=w.val_one_step,
            finals_val_gmean=10 ** lv.mean(), finals_val_median=float(np.median(g.val_score.fillna(1e6))), finals_val_max=float(g.val_score.fillna(1e6).max()),
            fragility=lv.mean() - math.log10(max(w.val_score, 1e-12)),                    # validation only
            fragility_median=float(np.median(lv)) - math.log10(max(w.val_score, 1e-12)),
            finals_worse_than_trial=int((lv > math.log10(max(w.val_score, 1e-12))).sum()),
            val_sd_log10=lv.std(ddof=1), exploded_val=int((lv > np.median(lv) + EXPLODED).sum()),
            test_gmean=10 ** lt.mean(), test_median=float(np.median(g.test_h16.fillna(1e6))), test_max=float(g.test_h16.fillna(1e6).max()),
            test_sd_log10=lt.std(ddof=1), exploded_test=int((lt > np.median(lt) + EXPLODED).sum()),
            test_one_step_gmean=10 ** log10(g.test_one_step).mean(),
            runner_up_gap=float(log10(tt.val_score.iloc[1]) - log10(tt.val_score.iloc[0])),
            trials_within_10pct=int((log10(tt.val_score) <= log10(tt.val_score.iloc[0]) + math.log10(1.1)).sum()),
        ))
    c = pd.DataFrame(rows).sort_values(["task", "arm", "n"]).reset_index(drop=True)
    # interior bump of the curve (log10 geometric mean of the 16-step rollout test NMSE): above BOTH neighbours
    c["bump"] = np.nan
    c["val_bump"] = np.nan
    for (task, arm), g in c.groupby(["task", "arm"]):
        g = g.sort_values("n")
        m, v = np.log10(g.test_gmean.values), np.log10(g.finals_val_gmean.values)
        for i in range(1, len(g) - 1):
            c.loc[g.index[i], "bump"] = m[i] - max(m[i - 1], m[i + 1])
            c.loc[g.index[i], "val_bump"] = v[i] - max(v[i - 1], v[i + 1])
    c.to_csv(TABLES / "cells_screen.csv", index=False)
    b = c[(c.bump > 0)].sort_values("bump", ascending=False)
    b.to_csv(TABLES / "bumps.csv", index=False)
    by_arm = c.groupby("arm").agg(cells=("n", "size"), interior=("bump", "count"), bumps_gt_0p05=("bump", lambda x: int((x > 0.05).sum())),
                                  bumps_gt_0p10=("bump", lambda x: int((x > 0.10).sum())), bumps_gt_0p20=("bump", lambda x: int((x > 0.20).sum())),
                                  cells_with_exploded_test=("exploded_test", lambda x: int((x > 0).sum())), exploded_test_finals=("exploded_test", "sum"),
                                  cells_with_exploded_val=("exploded_val", lambda x: int((x > 0).sum())), exploded_val_finals=("exploded_val", "sum"),
                                  fragility_median=("fragility", "median"), fragility_p90=("fragility", lambda x: float(np.quantile(x, 0.9))),
                                  fragile_gt_0p15=("fragility", lambda x: int((x > 0.15).sum())), fragile_gt_0p30=("fragility", lambda x: int((x > 0.30).sum())))
    by_arm.reindex(ARMS).to_csv(TABLES / "screen_by_arm.csv")
    print(by_arm.reindex(ARMS).to_string())


if __name__ == "__main__":
    {"harvest": harvest, "cells": cells}[sys.argv[1]]()
