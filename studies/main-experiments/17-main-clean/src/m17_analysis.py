"""17-main-clean analysis: units -> ledger -> curves, AULC, paired contrasts.

Conventions follow 16-main-final: AULC is the trapezoid of test NMSE over
log2 n normalized by the window (lower is better); intervals are 95%
Student-t over target instances; contrasts are paired per-instance
differences, LeWi minus the arm, so negative favours LeWi. A winner with
sealed test NMSE above 10 is off scale and its cell reports the median; a
winner with best validation NMSE at or above 1 is dead; a diverged unit (no test value at its best
step) is never the winner, as in the freeze. Both are disclosed, with the diverged trials per arm.
A curve point whose instances contain outliers (Hampel identifier on log10
NMSE, k = 3) shows the median of the remaining instances and names the
excluded ones (decision 2026-09-17); the ledger and AULC keep every value.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import m17 as M  # noqa: E402

OFF_SCALE = 10.0
DEAD_VAL = 1.0
ANALYSIS = M.STUDY / "analysis"
LEDGER = M.STUDY / "ledger"


def tci(values) -> dict:
    from scipy.stats import t as student_t
    v = np.asarray(list(values), dtype=float)
    n = v.size
    mean = float(v.mean()) if n else float("nan")
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    half = float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
    return {"mean": mean, "ci95_lo": mean - half, "ci95_hi": mean + half, "sd": sd,
            "n_instances": int(n), "median": float(np.median(v)) if n else float("nan"),
            "n_negative": int((v < 0).sum()), "n_positive": int((v > 0).sum())}


HAMPEL_K = 3.0            # Hampel identifier: |x - median| > k * 1.4826 * MAD on log10 NMSE
MAD_FLOOR = 0.15          # ... with the scale floored at 0.15 decades, so an instance within a
                          # factor 2.8 of the median is never an outlier, however tight the rest


def outliers(x: np.ndarray) -> np.ndarray:
    """Boolean mask of the outlying instances at one curve point (Hampel identifier,
    Davies & Gather 1993, on the log10 scale)."""
    med = np.median(x)
    scale = max(1.4826 * float(np.median(np.abs(x - med))), MAD_FLOOR)
    return np.abs(x - med) > HAMPEL_K * scale


def logstats(values, labels=None) -> dict:
    """Geometric mean and 95% t-band over instances on the log10 scale.

    A point with outlying instances (decision 2026-09-17) reports the median of the
    remaining instances as its value, the t half-width of those instances about it as
    its band, and names the excluded instances; the ledger keeps every value.
    """
    from scipy.stats import t as student_t
    v = np.asarray(list(values), dtype=float)
    v = np.clip(v, 1e-12, None)
    x = np.log10(v)
    out = outliers(x) if x.size >= 4 else np.zeros(x.size, dtype=bool)
    keep = x[~out]

    def half_of(z):
        return (float(student_t.ppf(0.975, z.size - 1) * z.std(ddof=1) / math.sqrt(z.size))
                if z.size > 1 else float("nan"))
    if out.any():
        m, half = float(np.median(keep)), half_of(keep)
    else:
        m, half = float(x.mean()), half_of(x)
    names = list(labels) if labels is not None else [str(i) for i in range(x.size)]
    return {"gmean": 10 ** m, "glo": 10 ** (m - half), "ghi": 10 ** (m + half),
            "off_scale": int((v > OFF_SCALE).sum()),
            "n_outliers": int(out.sum()),
            "outliers": "; ".join(f"{names[i]}={v[i]:.3g}" for i in np.flatnonzero(out))}


TRIALS_PER_CONDITION = 32       # the protocol's panel size; see winners()


def winners(panel: str) -> pd.DataFrame:
    """One row per (cell, arm, fs, n): the validation-selected trial."""
    src = M.RESULTS / f"units_{panel}.parquet"
    if not src.exists():
        return pd.DataFrame()
    df = pd.read_parquet(src)
    df = df[df["val"].notna()]
    if df.empty:
        return df
    # Only conditions whose full T = 32 panel is on record. A winner drawn from a
    # partial panel is the minimum of fewer draws, so it is systematically worse than
    # the winner the protocol specifies - and the shortfall is not spread evenly. With
    # main at 78% the completion ran from 100% of tree16's conditions down to 5% of
    # ftree16's lewi-v1-rgm, so comparing arms across it would have penalised exactly
    # the arm under test. A condition is either the pre-registered estimand or it is
    # absent; dropping it also removes its (cell, arm, instance) group from the AULC,
    # which needs every n anyway.
    counts = df.groupby(["cell", "arm", "fs", "n"])["val"].transform("size")
    df = df[counts >= TRIALS_PER_CONDITION]
    if df.empty:
        return df
    key = ["cell", "arm", "fs", "n"]
    trials = df.groupby(key)["val"].size()
    diverged = df.groupby(key)["diverged"].sum()
    # A diverged unit is on record but has no test value at its best step, so it cannot be
    # the winner (the freeze applies the same rule); the condition's trial count keeps it.
    df = df[df["diverged"] == 0]
    if df.empty:
        return df
    idx = df.groupby(key)["val"].idxmin()
    w = df.loc[idx].copy()
    w["dead"] = (w["val"] >= DEAD_VAL).astype(int)
    w["off_scale"] = (w["test"] > OFF_SCALE).astype(int)
    conds = pd.MultiIndex.from_frame(w[key])
    w["trials"] = trials.reindex(conds).to_numpy()
    w["diverged_trials"] = diverged.reindex(conds).to_numpy()
    return w.sort_values(["cell", "arm", "fs", "n"]).reset_index(drop=True)


def aulc_instances(w: pd.DataFrame, ns: list[int]) -> pd.DataFrame:
    """Trapezoid of test NMSE over log2 n, normalized by the window width."""
    grid = np.log2(np.asarray(sorted(ns), dtype=float))
    span = grid[-1] - grid[0]
    rows = []
    if span <= 0:                 # a single-n panel (the capacity audits): no curve to integrate
        for (cell, arm, fs), g in w.groupby(["cell", "arm", "fs"]):
            y = g["test"].to_numpy(dtype=float)
            if y.size != 1 or not np.isfinite(y[0]):
                continue
            rows.append({"cell": cell, "arm": arm, "fs": int(fs), "aulc": float(y[0]),
                         "complete": 1, "off_scale": int(y[0] > OFF_SCALE)})
        return pd.DataFrame(rows)
    for (cell, arm, fs), g in w.groupby(["cell", "arm", "fs"]):
        g = g.set_index("n").reindex(sorted(ns))
        y = g["test"].to_numpy(dtype=float)
        if np.isnan(y).any():
            continue
        rows.append({"cell": cell, "arm": arm, "fs": int(fs),
                     "aulc": float(np.trapezoid(y, grid) / span),
                     "complete": 1, "off_scale": int((y > OFF_SCALE).sum())})
    return pd.DataFrame(rows)


def build(panel: str, ns: list[int] | None = None, out: str | None = None) -> dict:
    ns = ns or M.PANELS[panel]["ns"]
    out_dir = ANALYSIS / (out or panel)
    out_dir.mkdir(parents=True, exist_ok=True)
    w = winners(panel)
    if w.empty:
        print(f"{panel}: no winners yet")
        return {}
    LEDGER.mkdir(parents=True, exist_ok=True)
    w.to_csv(LEDGER / f"winners_{panel}.csv", index=False)

    curves = []
    for (cell, arm, n), g in w.groupby(["cell", "arm", "n"]):
        s = logstats(g["test"], [f"fs{int(f)}" for f in g["fs"]])
        curves.append({"cell": cell, "arm": arm, "paper_name": M.PAPER.get(arm, arm), "n": int(n),
                       "instances": int(len(g)), **s,
                       "median_nmse": float(g["test"].median()),
                       "dead": int(g["dead"].sum()),
                       "mean_best_frac": float((g["best_step"] /
                                                M.PROFILES[M.PANELS[panel]["profile"]]["max_steps"]).mean())})
    pd.DataFrame(curves).to_csv(out_dir / "curves.csv", index=False)

    inst = aulc_instances(w, ns)
    inst.to_csv(out_dir / "aulc_instances.csv", index=False)
    if inst.empty:
        # No instance has a winner at every size yet (a panel's first harvest): the curves and the
        # ledger are written, the AULC tables wait. Grouping an empty frame raised KeyError here.
        inst = pd.DataFrame(columns=["cell", "arm", "fs", "aulc", "complete", "off_scale"])

    rows = []
    for (cell, arm), g in inst.groupby(["cell", "arm"]):
        rows.append({"cell": cell, "arm": arm, "paper_name": M.PAPER.get(arm, arm),
                     **{f"aulc_{k}": v for k, v in tci(g["aulc"]).items()},
                     "off_scale_instances": int((g["off_scale"] > 0).sum())})
    aulc = pd.DataFrame(rows)
    if not aulc.empty:
        aulc = aulc.sort_values(["cell", "aulc_mean"])
    aulc.to_csv(out_dir / "aulc.csv", index=False)

    pivot = inst.pivot_table(index=["cell", "fs"], columns="arm", values="aulc") if not inst.empty else pd.DataFrame()
    cons = []
    for cell in sorted(inst["cell"].unique()):
        arms = [a for a in M.ARM_ORDER if a in pivot.columns]
        if M.LEWI not in arms:
            continue
        sub = pivot.loc[cell] if cell in pivot.index.get_level_values(0) else None
        if sub is None:
            continue
        for ref in arms:
            if ref == M.LEWI:
                continue
            pair = sub[[M.LEWI, ref]].dropna()
            if pair.empty:
                continue
            d = (pair[M.LEWI] - pair[ref]).to_numpy()
            cons.append({"cell": cell, "arm": M.LEWI, "ref": ref,
                         "contrast": f"LeWi - {M.PAPER.get(ref, ref)}", **tci(d)})
        if "noreg" in sub.columns and "graph_oracle" in sub.columns:
            pair = sub[["graph_oracle", "noreg"]].dropna()
            if not pair.empty:
                cons.append({"cell": cell, "arm": "graph_oracle", "ref": "noreg",
                             "contrast": "GO(all) - Dense",
                             **tci((pair["graph_oracle"] - pair["noreg"]).to_numpy())})
    pd.DataFrame(cons).to_csv(out_dir / "contrasts.csv", index=False)

    disc = []
    for (cell, arm), g in w.groupby(["cell", "arm"]):
        disc.append({"cell": cell, "arm": arm, "paper_name": M.PAPER.get(arm, arm),
                     "winners": int(len(g)), "dead": int(g["dead"].sum()),
                     "off_scale": int(g["off_scale"].sum()),
                     "diverged_trials": int(g["diverged_trials"].sum()),
                     "median_trials": float(g["trials"].median())})
    pd.DataFrame(disc).to_csv(out_dir / "disclosures.csv", index=False)

    summary = {"panel": panel, "generated_local": M.now_local(), "generated_utc": M.now_utc(),
               "profile": M.PROFILES[M.PANELS[panel]["profile"]], "window": list(ns),
               "cells": sorted(w["cell"].unique()),
               "arms_complete": {c: {a: int(len(g)) for a, g in gg.groupby("arm")}
                                 for c, gg in inst.groupby("cell")},
               "units": int(len(pd.read_parquet(M.RESULTS / f"units_{panel}.parquet")))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"{panel}: {len(w)} winners, {len(inst)} instance AULCs -> {out_dir.relative_to(M.STUDY)}")
    return summary


def boundary_occupancy(panel: str) -> pd.DataFrame:
    """Fraction of selected winners sitting on an edge of their search space."""
    w = winners(panel)
    if w.empty:
        return pd.DataFrame()
    rows = []
    spaces_set = M.PANELS[panel]["spaces"]
    for (cell, arm), g in w.groupby(["cell", "arm"]):
        try:
            sp = M.space(arm, spaces_set)
        except SystemExit:
            continue
        for key, dom in sp.items():
            if not isinstance(dom, dict) or "low" not in dom:
                continue
            vals = pd.to_numeric(
                g["hp_json"].map(lambda s, k=key: json.loads(s).get(k)), errors="coerce").dropna()
            if vals.empty:
                continue
            lo, hi = float(dom["low"]), float(dom["high"])
            tol = (math.log10(hi) - math.log10(lo)) * 0.02 if dom.get("log") else (hi - lo) * 0.02
            at_lo = (np.log10(vals) <= math.log10(lo) + tol) if dom.get("log") else (vals <= lo + tol)
            at_hi = (np.log10(vals) >= math.log10(hi) - tol) if dom.get("log") else (vals >= hi - tol)
            rows.append({"cell": cell, "arm": arm, "parameter": key, "winners": int(len(vals)),
                         "at_low_edge": float(at_lo.mean()), "at_high_edge": float(at_hi.mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for panel in sys.argv[1:]:
        build(panel)
        bo = boundary_occupancy(panel)
        if not bo.empty:
            (ANALYSIS / panel).mkdir(parents=True, exist_ok=True)
            bo.to_csv(ANALYSIS / panel / "boundary_occupancy.csv", index=False)
