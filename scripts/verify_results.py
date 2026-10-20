#!/usr/bin/env python3
"""Recompute the paper's numbers from the packaged records and compare them with the manuscript.

    python scripts/verify_results.py [--quick] [--write DIR]        (CSFR_ROOT=DIR selects another archive root)

Sections (each check prints PASS / FAIL / INFO with the values it compared):
  A  the winner rule: the validation-selected winner of every (setting, arm, instance, size) of the primary d = 8 campaign is
     recomputed from its per-trial unit table (32 scrambled-Sobol trials per cell; the input rule's re-run tables for the
     repaired LeWi cells) and compared with the ledger
  B  AULC per instance, the mean with its 95% Student-t interval per (setting, arm), the paired contrasts and win counts,
     recomputed from the ledger and compared with the aggregate tables; the main text's percentages and gap shares;
     Tables 3 and 4 of the draft
  C  the nuisance-input setting (d = 32): the winner rule and the paired LeWi - Dense contrasts behind Figure 6(a)
  D  Figure 5: the probe and counterfactual R^2 cell means at n = 2^14 recomputed from the per-run tables; section 6.3's gap shares
  E  wiring alignment: Table 5 and the ratio to the controls (Figures 6b and 17)
  F  learned connection density: Appendix D.5 and Figure 19
  G  computational cost: Table 6 against the cost record (the exact implementation, one CPU thread)
  H  world model: the adopted per-seed record rebuilt from the record as run and the three adopted layers, the AULC and the
     paired contrasts per task, the win / tie / loss tallies of the abstract, section 6.5, Figure 7, Figure 21 and E.3; Table 7
  I  the frozen search spaces of Table 2 against the space files
  J  the boundary audit of Appendix C.12
  K  protocol constants: trials per cell, instances, grids, budgets, LeWi's constants and the input rule
--quick limits A to one setting and skips C and the raw part of D (about one minute); the full pass takes a few minutes.
The exit status is 1 when a check fails. The checks that fail against the draft on purpose are listed in the README
(section "Known differences between the draft and the records").
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("CSFR_ROOT", HERE.parent)).resolve()
sys.path.insert(0, str(HERE))
from paper_claims import CLAIMS, TABLE3, TABLE4, TABLE5, TABLE6, TABLE7  # noqa: E402

S18 = ROOT / "studies/main-experiments/18-main-d8"
S17 = ROOT / "studies/main-experiments/17-main-clean"
S16 = ROOT / "studies/main-experiments/16-main-final"
FIG = ROOT / "studies/communication/paper/figures"
D8 = FIG / "data-paper-d8"
OUT8 = FIG / "out-paper/d8"
LEWI = "lewi-v1-rgm"
WINDOW5 = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
WINDOW = [512, 1024, 2048, 4096, 8192, 16384, 32768]
ARM_NAME = {"noreg": "Dense", "l1": "l1", "l2": "l2", "l0hc": "L0", "rigl": "RigL", "bimt": "BIMT", "fiedler": "Fiedler",
            "erb_positions": "Neuron positions", "go1": "GraphOracle(1)", "go2": "GraphOracle(2)", "graph_oracle": "GraphOracle(all)",
            "support_oracle": "GraphOracle(0)", LEWI: "LeWi"}
RESULTS: list[tuple[str, str, str, str]] = []
KNOWN = {"Table 5 (precision, recall, F1 at n = 2^15) equals the structure tables",
         "mean F1 for edges originating at target intermediates is 0.471 (main text)"}   # the draft predates the last rebuild of the structure tables (README, known differences)


def check(section: str, ok, name: str, detail: str = "") -> bool:
    status = "PASS" if ok is True else "FAIL" if ok is False else "INFO"
    if status == "FAIL" and name in KNOWN:
        status = "KNOWN"
    RESULTS.append((section, status, name, detail))
    print(f"[{section}] {status:4s} {name}" + (f"  |  {detail}" if detail else ""), flush=True)
    return ok is True


def t_interval(x) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    if x.size < 2:
        return m, m, m
    h = float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size))
    return m, m - h, m + h


def aulc(ns, ys) -> float:
    x = np.log2(np.asarray(ns, dtype=float))
    y = np.asarray(ys, dtype=float)
    return float(np.trapezoid(y, x) / (x[-1] - x[0])) if hasattr(np, "trapezoid") else float(np.trapz(y, x) / (x[-1] - x[0]))


def close(a, b, tol=1e-9) -> bool:
    a, b = float(a), float(b)
    return abs(a - b) <= tol or abs(a - b) <= 1e-9 * max(abs(a), abs(b))


def window_of(cell: str) -> list[int]:
    return WINDOW5 if cell.startswith("tree5") else WINDOW


# ---------------------------------------------------------------- A. the winner rule
def winners_from_units(units: pd.DataFrame) -> pd.DataFrame:
    u = units.drop_duplicates(["cell", "arm", "fs", "n", "trial"]).copy()
    counts = u.groupby(["cell", "arm", "fs", "n"]).trial.transform("size")
    u = u[(counts >= 32) & u.val.notna() & (u.diverged.fillna(0).astype(int) == 0)]
    u = u.sort_values(["cell", "arm", "fs", "n", "val", "trial"], kind="stable")
    return u.groupby(["cell", "arm", "fs", "n"], sort=False).head(1).reset_index(drop=True)


def repaired_winners(sets: dict[str, dict[str, str]]) -> pd.DataFrame:
    parts = []
    for name, cell_of in sets.items():
        p = S18 / "input-cutoff/tables" / f"repair_{name}_units.csv"
        if not p.exists():
            continue
        u = pd.read_csv(p)
        u = u[~u.control.astype(bool)].copy()
        u["cell"] = u.cell.map(lambda c: cell_of.get(c, c))
        u["arm"] = LEWI
        parts.append(u)
    if not parts:
        return pd.DataFrame(columns=["cell", "arm", "fs", "n", "trial", "val", "test"])
    return winners_from_units(pd.concat(parts, ignore_index=True))


def compare_ledger(recomputed: pd.DataFrame, ledger: pd.DataFrame, section: str, label: str, only_cells=None) -> None:
    led = ledger.rename(columns={"function_seed": "fs", "best_val_nmse": "lval", "test_nmse": "ltest", "trial": "ltrial"})
    if only_cells is not None:
        led = led[led.cell.isin(only_cells)]
        recomputed = recomputed[recomputed.cell.isin(only_cells)]
    m = led.merge(recomputed[["cell", "arm", "fs", "n", "trial", "val", "test"]], on=["cell", "arm", "fs", "n"], how="left")
    missing = int(m.trial.isna().sum())
    same_trial = (m.ltrial == m.trial)
    same_val = np.isclose(m.lval.astype(float), m.val.astype(float), rtol=0, atol=1e-9)
    same_test = np.isclose(m.ltest.astype(float), m.test.astype(float), rtol=0, atol=1e-9)
    ok = int((same_trial & same_val & same_test).sum())
    check(section, ok == len(m) and missing == 0, f"{label}: recomputed winners equal the ledger",
          f"{ok} of {len(m)} rows equal (trial, validation NMSE, test NMSE); {missing} ledger rows without a recomputed winner")
    if ok != len(m):
        bad = m[~(same_trial & same_val & same_test)].head(5)
        print(bad[["cell", "arm", "fs", "n", "ltrial", "trial", "lval", "val"]].to_string())


def section_A(quick: bool) -> pd.DataFrame:
    ledger = pd.read_csv(S18 / "ledger/d8/winners.csv")
    panels = ["main8b"] if quick else ["main8b", "go8b", "pos8b", "rand8b", "rand8bpos"]
    parts = []
    for p in panels:
        f = S17 / "results" / f"units_{p}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=["cell", "arm", "fs", "n", "trial", "val", "test", "diverged"]))
    if not quick and (S17 / "results/units_main.parquet").exists():
        main = pd.read_parquet(S17 / "results/units_main.parquet", columns=["cell", "arm", "fs", "n", "trial", "val", "test", "diverged"])
        main = main[main.cell == "tree16"].copy()
        main["cell"] = "tree5d16"
        parts.append(main)
    units = pd.concat(parts, ignore_index=True)
    win = winners_from_units(units)
    sets = {"in5": {}, "in5_main": {"tree16": "tree5d16"}, "in5_rand8b": {}}
    rep = repaired_winners(sets)
    if len(rep):
        key = ["cell", "arm", "fs", "n"]
        win = pd.concat([win.merge(rep[key], on=key, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge"),
                         rep[win.columns.intersection(rep.columns)]], ignore_index=True)
    cells = ["tree8"] if quick else sorted(set(units.cell))
    pairs = set(zip(win.cell, win.arm))
    ledger = ledger[[(c, a) in pairs for c, a in zip(ledger.cell, ledger.arm)]]
    compare_ledger(win, ledger, "A", "d = 8" + (" (tree8 only, --quick)" if quick else " (the cells and arms of the shipped unit table)"), cells)
    check("A", True, "the winner rule", "per (setting, arm, instance, size): 32 trials complete, no divergence, lowest validation NMSE, lowest trial index on ties; "
          f"LeWi rows of the {len(rep)} cells changed by the input rule come from the rule's 32-trial re-runs")
    return win


# ---------------------------------------------------------------- B. AULC aggregates
def instance_aulc(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell, arm, fs), g in ledger.groupby(["cell", "arm", "function_seed"]):
        w = window_of(cell)
        g = g.set_index("n").reindex(w)
        if g.test_nmse.isna().any():
            continue
        rows.append(dict(cell=cell, arm=arm, fs=fs, aulc=aulc(w, g.test_nmse.to_numpy())))
    return pd.DataFrame(rows)


def section_B(tag: str, ledger_path: Path, contract: Path, claims: bool) -> pd.DataFrame:
    ledger = pd.read_csv(ledger_path)
    ia = instance_aulc(ledger)
    agg = pd.read_csv(contract / "aulc.csv")
    con = pd.read_csv(contract / "contrasts.csv")
    n_ok = n_all = 0
    for (cell, arm), g in ia.groupby(["cell", "arm"]):
        m, lo, hi = t_interval(g.aulc)
        row = agg[(agg.cell == cell) & (agg.arm == arm)]
        if len(row) != 1:
            continue
        n_all += 1
        n_ok += close(m, row.iloc[0]["mean"]) and close(lo, row.iloc[0].ci95_lo) and close(hi, row.iloc[0].ci95_hi)
    check("B", n_ok == n_all and n_all > 0, f"{tag}: mean AULC and 95% t interval per (setting, arm) equal the aggregate table", f"{n_ok} of {n_all}")
    c_ok = c_all = 0
    contrasts = {}
    for cell, g in ia.groupby("cell"):
        piv = g.pivot(index="fs", columns="arm", values="aulc")
        if LEWI not in piv:
            continue
        for arm in piv.columns:
            if arm == LEWI:
                continue
            d = (piv[LEWI] - piv[arm]).dropna()
            m, lo, hi = t_interval(d)
            wins = int((d < 0).sum())
            contrasts[(cell, arm)] = (m, lo, hi, wins, len(d))
            row = con[(con.cell == cell) & (con.arm == LEWI) & (con.ref == arm)]
            if len(row) != 1:
                continue
            c_all += 1
            c_ok += close(m, row.iloc[0]["mean"]) and close(lo, row.iloc[0].ci95_lo) and close(hi, row.iloc[0].ci95_hi) and int(row.iloc[0].wins) == wins
    check("B", c_ok == c_all and c_all > 0, f"{tag}: paired LeWi - arm contrasts (mean, interval, wins) equal the contrast table", f"{c_ok} of {c_all}")
    means = ia.groupby(["cell", "arm"]).aulc.mean()
    if claims:
        for cell, want in (("tree8", 0.20), ("dag8", 0.34), ("rand8", 0.27), ("tree5d16", 0.31)):
            red = (means[(cell, "noreg")] - means[(cell, LEWI)]) / means[(cell, "noreg")]
            check("B", round(red, 2) == want, f"LeWi reduces mean AULC by {int(want * 100)}% on {cell} (main text)", f"{red:.4f}")
        for cell, ref, want_w, want_n in (("rand8", "noreg", 32, 32), ("tree5d16", "noreg", 16, 16)):
            m, lo, hi, w, n = contrasts[(cell, ref)]
            check("B", w == want_w and n == want_n, f"LeWi better than Dense on {want_w} of {want_n} instances ({cell})", f"{w} of {n}; mean {m:.4f} [{lo:.4f}, {hi:.4f}]")
        learned = ["noreg", "l1", "l2", "l0hc", "rigl", "bimt", "fiedler", "erb_positions"]
        all16 = all(contrasts[("tree5d16", a)][3] == 16 for a in learned if ("tree5d16", a) in contrasts)
        check("B", all16, "depth-5 tree: LeWi better than every learned baseline on all 16 instances", ", ".join(f"{ARM_NAME[a]} {contrasts[('tree5d16', a)][3]}/16" for a in learned if ("tree5d16", a) in contrasts))
        gap = (means[("tree8", "noreg")] - means[("tree8", LEWI)]) / (means[("tree8", "noreg")] - means[("tree8", "graph_oracle")])
        gap_l0 = (means[("tree8", "noreg")] - means[("tree8", "l0hc")]) / (means[("tree8", "noreg")] - means[("tree8", "graph_oracle")])
        gg = pd.read_csv(contract / "go_gap.csv")
        check("B", round(gap, 2) == 0.51 and close(gap, gg[gg.cell == "tree8"].share_closed_mean.iloc[0], 1e-9), "LeWi closes 51% of the Dense-to-GraphOracle(all) gap on the depth-4 tree", f"{gap:.4f}")
        check("B", round(gap_l0, 2) == 0.26, "L0 closes 26% of that gap", f"{gap_l0:.4f}")
        # Table 3 and Table 4 of the draft
        bad = []
        for arm, cells in TABLE3.items():
            for cell, (mv, hv) in cells.items():
                if (cell, arm) not in means.index:
                    bad.append(f"{arm}/{cell}: no rows")
                    continue
                g = ia[(ia.cell == cell) & (ia.arm == arm)].aulc
                m, lo, hi = t_interval(g)
                if round(m, 3) != mv or round(m - lo, 3) != hv:
                    bad.append(f"{arm}/{cell}: draft {mv}±{hv}, record {m:.3f}±{m - lo:.3f}")
        check("B", not bad, "Table 3 (mean AULC ± half-width, six fixed settings) equals the recomputation", "; ".join(bad) if bad else f"{sum(len(v) for v in TABLE3.values())} cells")
        bad = []
        for cell, arms in TABLE4.items():
            for arm, vals in arms.items():
                g = ia[(ia.cell == cell) & (ia.arm == arm)].aulc
                m, lo, hi = t_interval(g)
                if round(m, 3) != vals[0] or round(m - lo, 3) != vals[1]:
                    bad.append(f"{arm}/{cell} AULC: draft {vals[0]}±{vals[1]}, record {m:.3f}±{m - lo:.3f}")
                if len(vals) > 2 and (cell, arm) in contrasts:
                    cm, clo, chi, w, n = contrasts[(cell, arm)]
                    # the draft prints AULC(LeWi) - AULC(method)
                    if round(cm, 3) != vals[2] or round(clo, 3) != vals[3] or round(chi, 3) != vals[4] or w != vals[5]:
                        bad.append(f"{arm}/{cell} contrast: draft {vals[2]} [{vals[3]}, {vals[4]}] ({vals[5]}), record {cm:.3f} [{clo:.3f}, {chi:.3f}] ({w})")
        check("B", not bad, "Table 4 (depth-5 tree and sampled depth-4 DAGs) equals the recomputation", "; ".join(bad) if bad else "every cell")
    return ia


# ---------------------------------------------------------------- C. d = 32
def section_C() -> None:
    ledger = pd.read_csv(S18 / "ledger/d32s/winners.csv")
    parts = []
    for p in ("main32s", "main32t"):
        f = S17 / "results" / f"units_{p}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=["cell", "arm", "fs", "n", "trial", "val", "test", "diverged"]))
    if parts:
        units = pd.concat(parts, ignore_index=True)
        units["cell"] = units.cell.map(lambda c: "tree5d32" if c == "tree5x32" else c + "32s")
        win = winners_from_units(units)
        compare_ledger(win, ledger, "C", "d = 32")
    else:
        check("C", None, "d = 32 winner rule", "the per-trial unit tables of the nuisance-input campaign are not in this archive (the ledger is the record); the contrasts are checked below")
    ia8 = instance_aulc(pd.read_csv(S18 / "ledger/d8/winners.csv"))
    ia32 = instance_aulc(ledger)
    rows = []
    for c8, c32 in (("tree8", "tree32s"), ("dag8", "dag32s"), ("ftree8", "ftree32s"), ("flat8", "flat32s"), ("dense8", "dense32s"), ("fdag8", "fdag32s"), ("rand8", "rand32s")):
        d8 = ia8[ia8.cell == c8].pivot(index="fs", columns="arm", values="aulc")
        d32 = ia32[ia32.cell == c32].pivot(index="fs", columns="arm", values="aulc")
        m8 = float((d8[LEWI] - d8["noreg"]).mean())
        m32 = float((d32[LEWI] - d32["noreg"]).mean())
        rows.append((c8, m8, m32))
    ok = all(m32 < m8 for _, m8, m32 in rows)
    check("C", ok, "LeWi's advantage over Dense grows on all seven settings with 24 nuisance inputs (Figure 6a)",
          "; ".join(f"{c}: {m8:+.3f} -> {m32:+.3f}" for c, m8, m32 in rows))


# ---------------------------------------------------------------- D. Figure 5
def section_D(quick: bool) -> None:
    agg = pd.read_csv(OUT8 / "learned_computational_structure_n16384.csv")
    cells = agg[agg.panel == "cells"]
    level = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3}

    def val(block, method, k, layer):
        r = cells[(cells.block == block) & (cells.method == method) & (cells.series == f"intermediate {k}") & (cells.layer.astype(str).isin([layer, layer.lstrip("h")]))]
        return float(r.value.iloc[0]) if len(r) == 1 else float("nan")

    def share(block, ks, layer):
        s = []
        for k in ks:
            d, l, g = val(block, "noreg", k, layer), val(block, LEWI, k, layer), val(block, "graph_oracle", k, layer)
            s.append((l - d) / (g - d))
        return float(np.mean(s))
    s23 = share("linear", [5, 6], "h3")
    s34 = share("linear", [7], "h4")
    check("D", 0.67 <= s23 <= 0.83, "probe R^2: LeWi closes roughly three quarters of the Dense-to-GraphOracle(all) gap for level-2 intermediates in layer 3", f"share {s23:.3f}")
    check("D", 0.58 <= s34 <= 0.76, "probe R^2: about two thirds of the gap for the level-3 intermediate in layer 4", f"share {s34:.3f}")
    c11 = share("causal", [1, 2, 3, 4], "h1")
    c23 = share("causal", [5, 6], "h3")
    check("D", c11 >= 0.85, "counterfactual R^2: level-1 intermediates in layer 1, LeWi closes nearly all of the gap", f"share {c11:.3f}")
    l23 = np.mean([val("causal", LEWI, k, "h3") for k in (5, 6)])
    g23 = np.mean([val("causal", "graph_oracle", k, "h3") for k in (5, 6)])
    check("D", abs(l23 - g23) <= 0.05, "counterfactual R^2: level-2 intermediates in layer 3 reach approximately GraphOracle(all)", f"LeWi {l23:.3f}, GraphOracle(all) {g23:.3f} (share {c23:.3f})")
    if quick:
        return
    # the raw per-run tables: the figure numbers the intermediates of a level per instance by the target's node numbering, so the
    # per-cell means are not comparable without that numbering; the per-level means are (a mean over the level's intermediates)
    depth = agg[agg.panel == "by_depth"]

    def level_val(block, method, lvl, layer):
        r = depth[(depth.block == block) & (depth.method == method) & (depth.series == f"Level {lvl}") & (depth.layer.astype(str).isin([layer, layer.lstrip("h")]))]
        return float(r.value.iloc[0]) if len(r) == 1 else float("nan")
    pv = pd.read_csv(D8 / "learned_structure/probe_values.csv")
    pv = pv[(pv.task == "tree_functional_anova") & (pv.n == 16384) & pv.method.isin(["noreg", LEWI, "graph_oracle"]) & pv.layer.isin(["h1", "h2", "h3", "h4"])].copy()
    pv["lvl"] = pv.node_depth.astype(int)
    n_rows = len(pv)
    means = pv.groupby(["method", "lvl", "layer"]).r2_trained.mean()
    bad = sum(0 if close(m, level_val("linear", method, lvl, layer), 1e-9) else 1 for (method, lvl, layer), m in means.items())
    check("D", bad == 0 and len(means) == 36, "Figure 5 (left): the 36 per-level probe R^2 means recomputed from the per-run probe table (16 selected runs x 7 intermediates x 4 layers)",
          f"{len(means) - bad} of {len(means)} equal; {n_rows} probe rows")
    cf = pd.read_csv(D8 / "causal_nopatch/cells.csv")
    cf = cf[(cf.fit_list == "joint7") & (cf.task == "tree_functional_anova") & (cf.n == 16384) & cf.method.isin(["noreg", LEWI, "graph_oracle"])].copy()
    cf["k"] = cf["var"].str.extract(r"z(\d+)").astype(int) + 1
    cf["lvl"] = cf.k.map(level)
    cf["layer_name"] = "h" + cf.layer.astype(int).astype(str)
    idx = cf.groupby(["method", "function_seed", "k", "layer_name"]).val_r2.idxmax()
    best = cf.loc[idx]
    cmeans = best.groupby(["method", "lvl", "layer_name"]).test_r2.mean()
    bad = sum(0 if close(m, level_val("causal", method, lvl, layer), 1e-6) else 1 for (method, lvl, layer), m in cmeans.items())
    check("D", bad == 0 and len(cmeans) == 36, "Figure 5 (right): the 36 per-level counterfactual R^2 means recomputed from the per-fit table (best restart by validation, 16 instances)",
          f"{len(cmeans) - bad} of {len(cmeans)} equal; {len(best)} fits")


# ---------------------------------------------------------------- E. wiring alignment
def section_E() -> None:
    wa = pd.read_csv(S18 / "structure/tables/wiring_alignment.csv")
    wa = wa[wa.n == 32768]
    fam = {"tree": "tree", "dag": "dag", "ftree": "ftree", "rand": "rand"}
    bad = []
    trained = wa[wa.condition == "trained"].set_index("family")
    for key, want in TABLE5.items():
        r = trained.loc[fam[key]]
        got = (r.edge_precision_mean, r.edge_recall_mean, r.edge_f1_mean, r.higher_precision_mean, r.higher_recall_mean, r.higher_f1_mean, r.decidable_recall_mean)
        if any(round(g, 3) != w for g, w in zip(got, want)):
            bad.append(f"{key}: draft {want}, record {tuple(round(g, 3) for g in got)}")
    check("E", not bad, "Table 5 (precision, recall, F1 at n = 2^15) equals the structure tables", "; ".join(bad) if bad else "four rows")
    f1 = {c: float(wa[wa.condition == c].edge_f1_mean.mean()) for c in ("trained", "init", "rewired", "shuffled")}
    ratios = {c: f1["trained"] / f1[c] for c in ("init", "rewired", "shuffled")}
    check("E", all(v > 3 for v in ratios.values()), "trained LeWi's edge F1 is more than three times the controls' (mean over the four settings)",
          f"trained {f1['trained']:.3f}; " + ", ".join(f"{c} {f1[c]:.3f} (x{r:.1f})" for c, r in ratios.items()))
    per_setting = all((wa[wa.condition == "trained"].set_index("family").edge_f1_mean / wa[wa.condition == c].set_index("family").edge_f1_mean > 3).all() for c in ("init", "rewired", "shuffled"))
    per = {c: (wa[wa.condition == "trained"].set_index("family").edge_f1_mean / wa[wa.condition == c].set_index("family").edge_f1_mean).round(2).to_dict() for c in ("init", "rewired", "shuffled")}
    check("E", None, "the ratio per setting and control (the main text states the mean over the four settings)", str(per))
    mean_higher = float(trained.higher_f1_mean.mean())
    check("E", round(mean_higher, 3) == 0.471, "mean F1 for edges originating at target intermediates is 0.471 (main text)", f"record {mean_higher:.3f}")


# ---------------------------------------------------------------- F. density
def section_F() -> None:
    s = pd.read_csv(S18 / "density/tables/summary.csv").set_index(["setting", "cell"])
    init_ok = all(0.60 <= v <= 0.62 for v in s.density_init_overall_mean)
    check("F", init_ok, "initialization density 0.61 in every setting", f"{s.density_init_overall_mean.min():.3f} .. {s.density_init_overall_mean.max():.3f}")
    comp8 = s.loc[[("d8", c) for c in ("tree8", "dag8", "ftree8", "flat8", "fdag8", "rand8")]].density_end_overall_mean
    check("F", 0.285 <= comp8.min() and comp8.max() <= 0.405, "all-active compositional settings: end density 0.29-0.40", f"{comp8.min():.3f} .. {comp8.max():.3f}")
    d8dense = float(s.loc[("d8", "dense8")].density_end_overall_mean)
    check("F", abs(d8dense - float(s.loc[("d8", "dense8")].density_init_overall_mean)) < 0.02, "high-arity control stays near its initialization density (d = 8)", f"{d8dense:.3f}")
    comp32 = s.loc[[("d32s", c) for c in ("tree32s", "dag32s", "ftree32s", "flat32s", "fdag32s", "rand32s")]].density_end_overall_mean
    check("F", 0.235 <= comp32.min() and comp32.max() <= 0.315, "with nuisance inputs: 0.24-0.31", f"{comp32.min():.3f} .. {comp32.max():.3f}")
    check("F", round(float(s.loc[("d32s", "dense32s")].density_end_overall_mean), 2) == 0.42, "high-arity control with nuisance inputs: 0.42", f"{float(s.loc[('d32s', 'dense32s')].density_end_overall_mean):.3f}")
    t5 = [float(s.loc[("d8", "tree5d16")].density_end_overall_mean), float(s.loc[("d32s", "tree5d32")].density_end_overall_mean)]
    check("F", all(round(v, 2) == 0.42 for v in t5), "depth-5 tree near 0.42 in both conditions", f"{t5[0]:.3f}, {t5[1]:.3f}")
    inp = s.loc["d32s"].density_end_input_mean
    check("F", 0.065 <= inp.min() and inp.max() <= 0.295, "input transition 0.07-0.29 with nuisance inputs", f"{inp.min():.3f} .. {inp.max():.3f}")
    b = pd.read_csv(S18 / "density/tables/by_size.csv")
    t = b[(b.setting == "d8") & (b.cell == "tree5d16")].set_index("n").density_end_overall_mean
    check("F", round(float(t[2048]), 2) == 0.58 and round(float(t[131072]), 2) == 0.38, "depth-5 tree falls from 0.58 at n = 2^11 to 0.38 at n = 2^17", f"{float(t[2048]):.3f} -> {float(t[131072]):.3f}")


# ---------------------------------------------------------------- G. cost
def section_G() -> None:
    raw = pd.read_csv(S18 / "overhead-fast/tables/overhead_fast.csv")
    case_of = {"32x4": "32x4", "64x8": "64x8-d16"}
    bad = []
    pct = {}
    for bb, want in TABLE6.items():
        rows = raw[(raw.case == case_of[bb]) & (raw.impl == "exact") & (raw.threads == 1) & (raw.device == "cpu")]
        dense = rows[rows.arm == "noreg-exact"]
        lewi = rows[rows.arm == "lewi-k1-exact"]
        if len(dense) != 1 or len(lewi) != 1:
            bad.append(f"{bb}: expected one exact one-thread CPU row per arm in overhead_fast.csv, found {len(dense)} and {len(lewi)}")
            continue
        d, l = dense.iloc[0], lewi.iloc[0]
        pct[bb] = (float(l.step_ms_median) / float(d.step_ms_median) - 1) * 100
        got = (round(float(d.step_ms_median), 3), round(float(l.step_ms_median), 3), int(round(pct[bb])), int(d.n_params), int(l.n_params),
               int(round(float(d.full_wall_s_median))), int(round(float(l.full_wall_s_median))),
               round(float(d.inference_model_ms), 2), round(float(l.inference_model_ms), 2),
               int(round(float(d.peak_rss_mb))), int(round(float(l.peak_rss_mb))))
        if got != want:
            bad.append(f"{bb}: manuscript {want}, record {got}")
    check("G", not bad, "Table 6 (step time, overhead, parameters, budget, inference, peak memory) equals the exact one-thread CPU rows of the cost record",
          "; ".join(bad) if bad else "both backbones (overhead_fast.csv, arms noreg-exact and lewi-k1-exact)")
    check("G", bool(pct) and round(min(pct.values())) == 107 and round(max(pct.values())) == 140, "training-step time 107-140% higher than Dense (main text, D.7)",
          ", ".join(f"{k} +{v:.1f}%" for k, v in pct.items()))
    extra = {bb: (want[4] - want[3]) / want[3] for bb, want in TABLE6.items()}
    check("G", 0.05 <= min(extra.values()) and max(extra.values()) <= 0.125, "LeWi adds 5-12% more parameters", ", ".join(f"{k} {v:.3f}" for k, v in extra.items()))


# ---------------------------------------------------------------- H. world model
def wm_record() -> tuple[pd.DataFrame, list[str], dict]:
    ana = S16 / "external/analysis"
    read = json.loads((ana / "read.json").read_text())
    counted = list(read["claim_bearing"])
    inst = pd.read_csv(ana / "instances.csv")
    inst = inst[inst.task.isin(counted)][["task", "n", "arm", "seed", "nmse"]].copy()
    key = ["task", "n", "arm", "seed"]
    layers = {}
    p = S18 / "swm-in5/tables/instances_in5.csv"
    if p.exists():
        in5 = pd.read_csv(p)
        in5 = in5[in5.role == "rule"].copy()
        in5["arm"] = "lewi-v1"
        layers["in5"] = in5[key + ["nmse"]]
    p = S18 / "swm-rollout/tables/repaired_instances.csv"
    if p.exists():
        rp = pd.read_csv(p)
        rp = rp[rp.role == "flagged"]
        layers["resel"] = rp[key + ["nmse"]]
    p = ROOT / "results/csfrbench/swmin5/resel-in5/manipulator/n262144/lewi-v1/s04.json"
    if p.exists():
        rec = json.loads(p.read_text())
        tr = rec.get("test_rollout") or {}
        v = None
        for k in ("16", "h16", 16):
            if isinstance(tr, dict) and str(k) in {str(x) for x in tr}:
                v = tr[str(k)] if str(k) in tr else tr[k]
                break
        if isinstance(v, dict):
            v = v.get("nmse", v.get("mean"))
        if v is not None:
            layers["resel_in5_seed4"] = pd.DataFrame([dict(task="manipulator", n=262144, arm="lewi-v1", seed=4, nmse=float(v))])
    p = S18 / "swm-lrx/tables/finals.csv"
    if p.exists():
        lx = pd.read_csv(p)
        layers["lrx"] = lx[lx.task.isin(counted)][key + ["nmse"]]
    for name, lay in layers.items():
        inst = inst.merge(lay.rename(columns={"nmse": "new"}), on=key, how="left")
        n_sub = int(inst.new.notna().sum())
        inst["nmse"] = inst.new.where(inst.new.notna(), inst.nmse)
        inst = inst.drop(columns="new")
        check("H", True, f"layer {name} substituted", f"{n_sub} final records")
    return inst, counted, read


def section_H() -> None:
    inst, counted, read = wm_record()
    y = np.log10(np.clip(inst.nmse.astype(float), 1e-12, 1e6))
    inst = inst.assign(log10=y)
    rows = []
    for (task, arm, seed), g in inst.groupby(["task", "arm", "seed"]):
        g = g.sort_values("n")
        grid = read["grids"][task] if isinstance(read.get("grids"), dict) else sorted(g.n)
        if list(g.n) != list(grid):
            continue
        rows.append(dict(task=task, arm=arm, seed=seed, aulc=aulc(g.n.to_numpy(), g.log10.to_numpy())))
    a = pd.DataFrame(rows)
    paired = pd.read_csv(D8 / "world_model_transfer_all/paired.csv")
    paired = paired[(paired.n == "aulc") & paired.task.isin(counted)]
    refs = ["dense", "l1", "l2", "l0hc", "rigl", "fiedler", "bimt", "erb"]
    tally = {}
    n_ok = n_all = 0
    for task, g in a.groupby("task"):
        piv = g.pivot(index="seed", columns="arm", values="aulc")
        for ref in refs:
            if ref not in piv or "lewi-v1" not in piv:
                continue
            d = (piv["lewi-v1"] - piv[ref]).dropna()
            m, lo, hi = t_interval(d)
            verdict = "win" if hi < 0 else "loss" if lo > 0 else "tie"
            tally.setdefault(ref, {"win": 0, "tie": 0, "loss": 0})[verdict] += 1
            row = paired[(paired.task == task) & (paired.reference == ref)]
            if len(row) == 1:
                n_all += 1
                r = row.iloc[0]
                n_ok += close(m, r.dlog10_mean, 1e-9) and close(lo, r.dlog10_lo, 1e-9) and close(hi, r.dlog10_hi, 1e-9) and r.verdict == verdict
    check("H", n_ok == n_all and n_all == 136, "paired AULC contrasts (mean, interval, verdict) recomputed per task and reference equal paired.csv", f"{n_ok} of {n_all} (17 tasks x 8 references)")
    seven = ["dense", "l1", "l2", "l0hc", "rigl", "fiedler", "bimt"]
    W = sum(tally[r]["win"] for r in seven)
    T = sum(tally[r]["tie"] for r in seven)
    L = sum(tally[r]["loss"] for r in seven)
    check("H", (W, T, L) == (99, 6, 14), "seven principal baselines: 99 wins, 6 ties, 14 losses of 119", f"{W}-{T}-{L}")
    W8, T8, L8 = W + tally["erb"]["win"], T + tally["erb"]["tie"], L + tally["erb"]["loss"]
    check("H", (W8, T8, L8) == (116, 6, 14), "with learned neuron positions: 116 wins, 6 ties, 14 losses of 136", f"{W8}-{T8}-{L8}")
    want = dict(dense=(16, 0, 1), l1=(15, 0, 2), l2=(13, 0, 4), l0hc=(16, 1, 0), rigl=(14, 1, 2), fiedler=(13, 3, 1), bimt=(12, 1, 4), erb=(17, 0, 0))
    bad = [f"{r}: {tally[r]['win']}-{tally[r]['tie']}-{tally[r]['loss']} (draft {w[0]}-{w[1]}-{w[2]})" for r, w in want.items() if (tally[r]["win"], tally[r]["tie"], tally[r]["loss"]) != w]
    check("H", not bad, "per-baseline win-tie-loss counts of Appendix E.3 / Figure 21", "; ".join(bad) if bad else ", ".join(f"{r} {w[0]}-{w[1]}-{w[2]}" for r, w in want.items()))
    check("H", tally["dense"]["win"] == 16, "LeWi improves over Dense on 16 of 17 tasks", f"{tally['dense']['win']} of 17")
    # Table 7
    t7 = pd.read_csv(OUT8 / "world_model_hpo_table.csv")
    spaces = json.loads((ROOT / "studies/main-experiments/09-swm-lewi/results/spaces_final_v2.json").read_text())
    lrx = json.loads((S18 / "swm-lrx/manifests/spaces_lrx.json").read_text())
    bad = []
    for _, r in t7.iterrows():
        want_lo, want_hi = TABLE7.get((r.arm, r.key), (None, None))
        if want_lo is not None and (float(f"{r.low:.3g}") != float(f"{want_lo:.3g}") or float(f"{r.high:.3g}") != float(f"{want_hi:.3g}")):
            bad.append(f"{r.arm}/{r.key}: draft [{want_lo:.3g}, {want_hi:.3g}], table [{r.low:.3g}, {r.high:.3g}]")
    check("H", not bad, "Table 7 (world-model search spaces) equals the draft", "; ".join(bad) if bad else f"{len(t7)} rows")
    src_ok = True
    for _, r in t7.iterrows():
        if r.arm in ("dense", "l1") and r.key == "lr":
            sp = lrx.get("arms", {}).get(r.arm, {}).get("expanded", {})
            sp = sp.get("lr") if isinstance(sp, dict) else sp
        else:
            sp = (spaces.get(r.arm) or spaces.get({"lewi-v1": "lewi"}.get(r.arm, r.arm)) or {}).get(r.key)
        if isinstance(sp, dict):
            lo, hi = sp.get("low", sp.get("lo")), sp.get("high", sp.get("hi"))
        elif isinstance(sp, (list, tuple)) and len(sp) >= 2:
            lo, hi = sp[0], sp[1]
        else:
            lo = hi = None
        if lo is None or not (close(lo, r.low, 1e-12) and close(hi, r.high, 1e-12)):
            src_ok = False
    check("H", src_ok, "Table 7's ranges equal the frozen space files (spaces_final_v2.json, spaces_lrx.json)", "")


# ---------------------------------------------------------------- I. search spaces
def section_I() -> None:
    t2 = pd.read_csv(OUT8 / "search_space_table.csv")
    doc = json.loads((S17 / "manifests/search_spaces.json").read_text())
    bad = 0
    n = 0
    for _, r in t2.iterrows():
        src = Path(str(r.source_file))
        key = "start" if src.parent.name == "tree_depth4" else "start8" if src.parent.name == "hpo_spaces" else src.parent.name
        p = (doc.get(key) or {}).get(src.stem, {}).get(r.parameter) if isinstance(doc.get(key), dict) else None
        if not isinstance(p, dict):
            continue
        n += 1
        if not (close(p.get("low"), r.low, 1e-12) and close(p.get("high"), r.high, 1e-12)):
            bad += 1
            print(f"      {r.arm} {r.backbone} {r.parameter}: frozen space [{p.get('low')}, {p.get('high')}] table [{r.low}, {r.high}]")
    check("I", bad == 0 and n > 0, "Table 2's ranges equal the frozen space files (manifests/search_spaces.json)", f"{n - bad} of {n} rows checked against their frozen space")
    want = {("lewi-v1-rgm", "32x4", "lr"): (4.58e-6, 0.0999), ("lewi-v1-rgm", "64x8", "lr"): (7.92e-7, 0.0712), ("noreg", "32x4", "lr"): (0.00143, 0.261),
            ("noreg", "64x8", "lr"): (7.70e-4, 0.475), ("l0hc", "32x4", "lam"): (1e-7, 157.0), ("bimt", "64x8", "lam"): (1e-7, 1.35e3)}
    bad = []
    main = t2[t2.block == "main"]
    for (arm, bb, par), (lo, hi) in want.items():
        r = main[(main.arm == arm) & (main.backbone == bb) & (main.parameter == par)]
        if len(r) != 1:
            bad.append(f"{arm}/{bb}/{par}: {len(r)} rows")
            continue
        if float(f"{r.low.iloc[0]:.3g}") != float(f"{lo:.3g}") or float(f"{r.high.iloc[0]:.3g}") != float(f"{hi:.3g}"):
            bad.append(f"{arm}/{bb}/{par}: table [{r.low.iloc[0]:.3g}, {r.high.iloc[0]:.3g}] draft [{lo:.3g}, {hi:.3g}]")
    check("I", not bad, "spot check of six Table 2 ranges against the draft", "; ".join(bad) if bad else "six ranges")


# ---------------------------------------------------------------- J. boundary audit
def section_J() -> None:
    b = pd.read_csv(S18 / "analysis/hpo_boundary/boundary_occupancy.csv")
    mx = float(b.frac_within_10pct_of_boundary.max())
    med = b.median_normalized_position
    check("J", round(mx, 2) == 0.17, "C.12: largest boundary occupancy 0.17 at d = 8", f"{mx:.3f}")
    check("J", round(float(med.min()), 2) == 0.31 and round(float(med.max()), 2) == 0.76, "C.12: median normalized positions 0.31-0.76", f"{med.min():.3f} .. {med.max():.3f}")
    lewi = b[(b.method == LEWI) & (b.parameter == "lr")].set_index("backbone").frac_within_10pct_of_boundary
    got = {k: round(float(v), 2) for k, v in lewi.items()}
    check("J", got.get("32x4") == 0.02 and any(v == 0.03 for k, v in got.items() if k != "32x4"), "C.12: LeWi 2% (32x4) and 3% (depth-5) near a boundary", str(got))
    d = pd.read_csv(S18 / "analysis/hpo_boundary/boundary_occupancy_d32s.csv")
    d5 = d[d.set == "p32t"].set_index(["method", "parameter"]).frac_within_10pct_of_boundary
    check("J", round(float(d5[("bimt", "lam")]), 2) == 0.33 and round(float(d5[("l1", "lam")]), 2) == 0.26, "C.12: depth-5 nuisance setting BIMT 33% and l1 26%",
          f"BIMT {float(d5[('bimt', 'lam')]):.3f}, l1 {float(d5[('l1', 'lam')]):.3f}")


# ---------------------------------------------------------------- K. protocol constants
def section_K(quick: bool) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "studies/main-experiments/12-extra/src"))
    sys.path.insert(0, str(S18 / "input-cutoff/src"))
    from csfrbench.methods.cmask import CODE_INIT
    from csfrbench.methods.lewi_v1 import LeWiV1
    import cutoff_methods
    r = int(getattr(LeWiV1, "code_dim", 3))
    check("K", CODE_INIT == 0.5 and r == 3 and 2 * r * CODE_INIT ** 2 == 1.5, "LeWi constants: r = 3, sigma_e = 0.5, R^2 = 2 r sigma_e^2 = 1.5", f"CODE_INIT {CODE_INIT}, code_dim {r}")
    check("K", 5 in cutoff_methods.MIN_OPEN, "input rule: an input keeps at least five open first-transition connections at initialization (lewi-v1-rgm-in5)", str(cutoff_methods.MIN_OPEN))
    plan = json.loads((S17 / "manifests/main8b_plan.json").read_text())
    text = json.dumps(plan)
    check("K", '"max_steps": 64000' in text or "64000" in text, "main8b plan: 64,000 steps", "")
    plan5 = json.loads((S17 / "manifests/main32t_plan.json").read_text())
    check("K", "128000" in json.dumps(plan5), "main32t plan: 128,000 steps", "")
    led = pd.read_csv(S18 / "ledger/d8/winners.csv")
    inst = led.groupby("cell").function_seed.nunique()
    check("K", all(inst[c] == 16 for c in ("tree8", "dag8", "ftree8", "flat8", "dense8", "fdag8", "tree5d16")) and inst["rand8"] == 32,
          "16 instances per fixed setting, 32 sampled depth-4 DAGs", inst.to_dict().__repr__())
    grid = led.groupby("cell").n.unique().map(lambda a: sorted(int(x) for x in a))
    check("K", all(grid[c] == WINDOW for c in ("tree8", "rand8")) and grid["tree5d16"] == WINDOW5, "size grids 2^9..2^15 and 2^11..2^17", "")
    check("K", (led.trials == 32).all(), "32 trials per (setting, arm, instance, size) in the ledger", f"{int((led.trials == 32).sum())} of {len(led)} rows")
    if not quick:
        u = pd.read_parquet(S17 / "results/units_main8b.parquet", columns=["cell", "arm", "fs", "n", "trial"])
        cnt = u.drop_duplicates().groupby(["cell", "arm", "fs", "n"]).trial.nunique()
        check("K", (cnt == 32).all(), "every cell of the d = 8 unit table holds 32 distinct trials", f"{int((cnt == 32).sum())} of {len(cnt)} cells")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--write", type=Path, default=None, help="write the recomputed tables (winners, AULC, contrasts, world-model AULC) here")
    a = ap.parse_args(argv)
    print(f"root: {ROOT}")
    sections = [("A", lambda: section_A(a.quick)), ("B", lambda: section_B("d = 8", S18 / "ledger/d8/winners.csv", S18 / "analysis/d8/contract", True)),
                ("B32", lambda: section_B("d = 32", S18 / "ledger/d32s/winners.csv", S18 / "analysis/d32s/contract", False)),
                ("C", section_C if not a.quick else None), ("D", lambda: section_D(a.quick)), ("E", section_E), ("F", section_F), ("G", section_G),
                ("H", section_H), ("I", section_I), ("J", section_J), ("K", lambda: section_K(a.quick))]
    outputs = {}
    for name, fn in sections:
        if fn is None:
            print(f"[{name}] skipped (--quick)")
            continue
        try:
            outputs[name] = fn()
        except Exception as exc:  # noqa: BLE001
            check(name, False, f"section {name} raised", repr(exc))
    if a.write:
        a.write.mkdir(parents=True, exist_ok=True)
        if isinstance(outputs.get("A"), pd.DataFrame):
            outputs["A"].to_csv(a.write / "winners_d8_recomputed.csv", index=False)
        if isinstance(outputs.get("B"), pd.DataFrame):
            outputs["B"].to_csv(a.write / "aulc_instances_d8_recomputed.csv", index=False)
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\n{len(RESULTS)} checks: {sum(r[1] == 'PASS' for r in RESULTS)} pass, {len(fails)} fail, {sum(r[1] == 'KNOWN' for r in RESULTS)} known differences, {sum(r[1] == 'INFO' for r in RESULTS)} info")
    for s, _, name, detail in (r for r in RESULTS if r[1] == "KNOWN"):
        print(f"  KNOWN [{s}] {name}: {detail}")
    for s, _, name, detail in fails:
        print(f"  FAIL [{s}] {name}: {detail}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
