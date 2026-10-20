"""The mechanism ablations and the sensitivity sweeps of the final LeWi on the adopted arm's initialization (intake/analysis.md
sections 3 and 5; decision 2026-09-22: every LeWi result of the paper under the same input and readout rejection-sampling rules):
read 17-main-clean's amendment-13 panels against the adopted LeWi of this study, paired per instance, into analysis/ablations/
and the figure data of Figure 6(c) and the appendix.

    python src/ablations.py            # tables -> analysis/ablations/*.csv, summary.json; figure data -> figures/data-final-clean/fig6/

Panels (17-main-clean amendment 13, src/abl_in5_methods.py: every variant and sweep arm built from lewi-v1-rgm-in5):
  abl8-in5      the four mechanism variants on the six d = 8 workloads (the sampled graphs on 32 instances), full grid
  abl32s-in5    the four variants on the tree and the reuse DAG at d = 32 on the twin's seeds (6000-6015), full grid
  abl8r-in5, abl8rad-in5, abl8seed-in5   the sweeps of r, the radius and the seed replicates, reduced grid, in full
  abl8init-in5  the initialization scale, reduced grid
abl8-in5 and abl8init-in5 ran the cells whose initialization the rule changes (17-main-clean manifests/abl_in5_cells.csv, the
input-cutoff census) plus two control cells per workload, and every cell of the sampled graphs' extension instances 6020-6035
(the record's abl8 ran 6000-6015); on every other cell the in5 build equals the record's unit bit for bit, and the record's row
(abl8, abl8init) stands. The controls are compared with the record here (summary.json "checks"). The reference of every
contrast is the adopted LeWi (ledger/d8, ledger/d32s; analysis/<setting>/contract/aulc_instances.csv).

Tables:
  contrasts.csv        per workload and variant: AULC(LeWi) - AULC(variant) over the full grid, paired per instance,
                       mean, 95% Student-t, median, wins (instances where the variant is worse), n_pairs
  contrasts_by_n.csv   the same per size (test NMSE)
  aulc.csv             every arm's AULC (mean, 95% t) per workload, LeWi and the variants
  contrasts_d32.csv, contrasts_d32_by_n.csv, aulc_d32.csv   the same in the nuisance-input setting on the twin's seeds
  sensitivity_{r,rad,init,seed}.csv   the sweeps against the adopted LeWi on the reduced grid (2^11 / 2^13 / 2^15, 6000-6007)
  curves.csv           mean test NMSE against n of Dense, LeWi and the variants (the alternative of panel (c))
  density.csv          per workload and arm over the winners: densities, dead units, redraws, the learning-rate trial spread
                       (from the run records on disk; the record's rows kept in abl8 / abl8init are archived and carry none)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "17-main-clean" / "src"))
import final as F  # noqa: E402
import m17 as M  # noqa: E402

OUT = F.STUDY / "analysis" / "ablations"
FIG = F.FIGURES / "data-final-clean" / "fig6"
VARIANTS = list(M.ABL_ARMS)
CELLS = list(M.ABL_CELLS)
CELL_LABEL = {"tree8": "tree", "ftree8": "Fourier tree", "dag8": "reuse DAG", "flat8": "flat modules", "dense8": "high-arity node", "rand8": "sampled depth-4 DAGs",
              "tree": "tree (d = 32)", "dag": "reuse DAG (d = 32)"}
CELLS_D32 = list(getattr(M, "ABL32_CELLS", []))
VARIANT_LABEL = {"lewi-abl-soft": "soft-only (no mask)", "lewi-abl-fixed": "fixed embeddings", "lewi-abl-random": "random sparse mask", "lewi-abl-ste": "straight-through mask",
                 "lewi-mask": "mask-only forward"}
# the mask-only forward of the two LeWi studies (17-main-clean amendment 14, panel lms8: the same cells, grid, instances and trial
# quantiles; its own frozen range) joins the d = 8 contrasts as a fifth variant once its panel is read (Figure 6(c) shows it in the
# straight-through arm's place, decision 2026-09-22; the appendix extension keeps both)
EXTRA_VARIANTS = {"lewi-mask": "lms8"}
IN5_PANEL = {"abl8": "abl8-in5", "abl8r": "abl8r-in5", "abl8rad": "abl8rad-in5", "abl8init": "abl8init-in5", "abl8seed": "abl8seed-in5", "abl32": "abl32s-in5"}
RESTRICTED = ("abl8", "abl8init")                      # the record stands on the cells the rule leaves alone
STUDY_CELL = {"tree": "tree32s", "dag": "dag32s"}       # the harness's d = 32 cells -> this study's
WIN_COLS = ["unit", "root", "cell", "arm", "fs", "n", "trial", "val", "test", "best_step", "diverged"]
CHECKS: dict = {}
_WINNERS: dict = {}


def aulc_of(ns, vals) -> float:
    from csfrbench.metrics import aulc
    return float(aulc(list(ns), list(vals)))


def manifest() -> pd.DataFrame:
    return pd.read_csv(F.M17 / "manifests" / "abl_in5_cells.csv")


def harness_winners(panel: str) -> pd.DataFrame:
    p = F.M17 / "ledger" / f"winners_{panel}.csv"
    if not p.exists():
        return pd.DataFrame(columns=WIN_COLS)
    w = pd.read_csv(p)
    w["root"] = panel
    return w.reindex(columns=WIN_COLS)


def variant_winners(record_panel: str) -> pd.DataFrame:
    """The in5 panel's winners; on a restricted panel the record's rows stay on the cells the rule leaves alone, and the
    control cells are compared with the record (CHECKS)."""
    if record_panel in _WINNERS:
        return _WINNERS[record_panel]
    in5 = harness_winners(IN5_PANEL[record_panel])
    if record_panel not in RESTRICTED:
        CHECKS[record_panel] = {"panel": IN5_PANEL[record_panel], "in5_rows": int(len(in5))}
        _WINNERS[record_panel] = in5
        return in5
    rec = harness_winners(record_panel)
    man = manifest()
    key = ["cell", "fs", "n"]
    rerun = pd.MultiIndex.from_frame(man[man.kind != "control"][key])
    ctrl = pd.MultiIndex.from_frame(man[man.kind == "control"][key])
    k5, kr = pd.MultiIndex.from_frame(in5[key]), pd.MultiIndex.from_frame(rec[key])
    merged = pd.concat([rec[~kr.isin(rerun)], in5[k5.isin(rerun)]], ignore_index=True)
    a = in5[k5.isin(ctrl)].set_index(key + ["arm"])[["trial", "val", "test"]]
    b = rec[kr.isin(ctrl)].set_index(key + ["arm"])[["trial", "val", "test"]]
    j = a.join(b, lsuffix="_in5", rsuffix="_rec", how="inner")
    equal = int(((j.trial_in5 == j.trial_rec) & (j.val_in5 == j.val_rec) & (j.test_in5 == j.test_rec)).sum())
    CHECKS[record_panel] = {"panel": IN5_PANEL[record_panel], "in5_rows": int(k5.isin(rerun).sum()), "record_rows": int((~kr.isin(rerun)).sum()),
                            "control_conditions": int(len(j)), "controls_equal_to_record": equal}
    _WINNERS[record_panel] = merged
    return merged


def instances_of(w: pd.DataFrame, ns: list[int]) -> pd.DataFrame:
    """AULC per (cell, arm, instance) over the sizes ns; complete = every size on record; off_scale = sizes above the threshold."""
    rows = []
    for (cell, arm, fs), g in w[w.n.isin(ns)].groupby(["cell", "arm", "fs"]):
        g = g.sort_values("n")
        complete = bool(g.n.nunique() == len(ns))
        rows.append(dict(cell=cell, arm=arm, fs=int(fs), aulc=aulc_of(g.n, g.test) if complete else np.nan, complete=complete,
                         off_scale=int((g.test > F.OFF_SCALE).sum())))
    return pd.DataFrame(rows, columns=["cell", "arm", "fs", "aulc", "complete", "off_scale"])


def lewi_reference(setting: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The adopted LeWi of a setting (d8 | d32s): its winners (fs, val, test, dir = the checkpoint on record) and its per-instance
    AULC from the study's contract tables, both under the harness's cell names."""
    w = pd.read_csv(F.STUDY / "ledger" / setting / "winners.csv")
    w = w[w.arm == F.LEWI].rename(columns={"function_seed": "fs", "best_val_nmse": "val", "test_nmse": "test"}).copy()
    w["dir"] = [str(F.REPO / c) if isinstance(c, str) and c else "" for c in w.checkpoint]
    inst = pd.read_csv(F.STUDY / "analysis" / setting / "contract" / "aulc_instances.csv")
    inst = inst[inst.arm == F.LEWI].rename(columns={"function_seed": "fs"})[["cell", "arm", "fs", "aulc"]].copy()
    dis = pd.read_csv(F.STUDY / "analysis" / setting / "contract" / "disclosures.csv")
    off = dis[(dis.arm == F.LEWI) & dis.off_scale.astype(bool)].groupby(["cell", "function_seed"]).size()
    inst["off_scale"] = [int(off.get((c, int(f)), 0)) for c, f in zip(inst.cell, inst.fs)]
    inst["complete"] = True
    if setting == "d32s":
        back = {v: k for k, v in STUDY_CELL.items()}
        for t in (w, inst):
            t["cell"] = t.cell.map(lambda c: back.get(c, c))
    return w[["cell", "arm", "fs", "n", "trial", "val", "test", "root", "unit", "dir"]], inst


def paired(a: pd.Series, b: pd.Series) -> dict:
    idx = a.index.intersection(b.index)
    d = (a.loc[idx] - b.loc[idx]).to_numpy(dtype=float)
    st = F.tci(d) if len(d) else {"mean": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan, "median": np.nan}
    return {**st, "wins": int((d < 0).sum()), "n_pairs": int(len(d))}


def contrasts(setting: str = "d8") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """AULC(LeWi) - AULC(variant) per workload, paired per instance over the full grid; the same per size."""
    record_panel, cells = ("abl8", CELLS) if setting == "d8" else ("abl32", CELLS_D32)
    wr, ref = lewi_reference(setting)
    wa = variant_winners(record_panel)
    variants = list(VARIANTS)
    if setting == "d8":
        for v, panel in EXTRA_VARIANTS.items():
            extra = harness_winners(panel)
            extra = extra[extra.arm == v]
            if len(extra):
                wa = pd.concat([wa, extra], ignore_index=True)
                variants.append(v)
    abl = instances_of(wa, list(M.WINDOW8B))
    rows, by_n, au = [], [], []
    for cell in cells:
        lewi = ref[(ref.cell == cell) & (ref.arm == F.LEWI)].set_index("fs").aulc
        st = F.tci(lewi) if len(lewi) else None
        if st:
            au.append({**st, "cell": cell, "workload": CELL_LABEL[cell], "arm": F.LEWI, "label": "LeWi (the adopted arm)", "n_instances": len(lewi)})
        for v in variants:
            x = abl[(abl.cell == cell) & (abl.arm == v)]
            x = x[x.complete.astype(bool)]          # an instance enters once every size of its curve is on record
            if x.empty:
                continue
            x = x.set_index("fs").aulc
            stx = F.tci(x)
            au.append({**stx, "cell": cell, "workload": CELL_LABEL[cell], "arm": v, "label": VARIANT_LABEL[v], "n_instances": len(x)})
            pr = paired(lewi, x)
            ref_c, abl_c = ref[(ref.cell == cell) & (ref.arm == F.LEWI)], abl[(abl.cell == cell) & (abl.arm == v)]
            off_inst = int((ref_c.off_scale > 0).sum()) + int((abl_c.off_scale > 0).sum())
            dead_l = int((wr[wr.cell == cell].val >= F.DEAD_VAL).sum())
            dead_v = int((wa[(wa.cell == cell) & (wa.arm == v)].val >= F.DEAD_VAL).sum())
            off = off_inst > 0 or any(abs(float(t)) > F.OFF_SCALE for t in (pr["mean"], pr["ci95_lo"], pr["ci95_hi"]) if np.isfinite(t))
            rows.append({**pr, "cell": cell, "workload": CELL_LABEL[cell], "variant": v, "label": VARIANT_LABEL[v], "contrast": f"LeWi - {VARIANT_LABEL[v]}", "off_scale": int(off),
                         "off_scale_instances": off_inst, "dead_winners_lewi": dead_l, "dead_winners_variant": dead_v})
            a = wr[wr.cell == cell].set_index(["fs", "n"]).test
            b = wa[(wa.cell == cell) & (wa.arm == v)].set_index(["fs", "n"]).test
            for n in sorted(set(b.index.get_level_values("n"))):
                pa, pb = a[a.index.get_level_values("n") == n].droplevel("n"), b[b.index.get_level_values("n") == n].droplevel("n")
                by_n.append({**paired(pa, pb), "cell": cell, "workload": CELL_LABEL[cell], "variant": v, "label": VARIANT_LABEL[v], "n": int(n)})
    return pd.DataFrame(rows), pd.DataFrame(by_n), pd.DataFrame(au)


def mask_stats(run_dir: Path) -> dict:
    """From the run's own retained states, in ONE scope (every transition, the readout included): the connection density of
    the binary masks at initialization (init.pt) and at the selected checkpoint (best.pt), and the share of hidden units with
    no open incoming or no open outgoing edge at the selected checkpoint. For the soft-only arm these describe the would-be
    mask (1[gate > 1/2]), which its forward pass does not apply."""
    import torch
    out = {}
    for name, key in (("init.pt", "init_density"), ("best.pt", "final_density")):
        f = run_dir / name
        if not f.exists():
            continue
        sd = torch.load(f, map_location="cpu", weights_only=True)
        masks = [v for k, v in sd.items() if k.endswith("event_mask")]          # [h1 x in, h2 x h1, ..., out x h_last]
        out[key] = float(sum(float(m.sum()) for m in masks) / sum(m.numel() for m in masks))
        if name == "best.pt":
            dead = total = 0
            for k in range(len(masks) - 1):
                incoming, outgoing = masks[k].sum(dim=1), masks[k + 1].sum(dim=0)
                dead += int(((incoming == 0) | (outgoing == 0)).sum())
                total += int(incoming.numel())
            out["dead_hidden_share"] = dead / total if total else float("nan")
    return out


def _run_dir(r) -> Path:
    d = getattr(r, "dir", "")
    return Path(d) if isinstance(d, str) and d else F.M17_RUNS / r.root / r.unit


def density() -> pd.DataFrame:
    """Per workload and arm over the winners: initial and final density, dead units, rejected initializations, the seed spread."""
    rows = []
    lewi8, _ = lewi_reference("d8")
    lewi32, _ = lewi_reference("d32s")
    sets = [("adopted LeWi (d = 8)", lewi8[lewi8.cell.isin(CELLS + M.SENS_CELLS)]),
            ("abl8-in5", variant_winners("abl8")), ("abl8r-in5", variant_winners("abl8r")), ("abl8rad-in5", variant_winners("abl8rad")),
            ("abl8init-in5", variant_winners("abl8init")), ("abl8seed-in5", variant_winners("abl8seed")),
            ("adopted LeWi (d = 32)", lewi32[lewi32.cell.isin(CELLS_D32)]), ("abl32s-in5", variant_winners("abl32"))]
    for panel, w in sets:
        if w.empty:
            continue
        parts = []
        for root in sorted(set(w.root.dropna())):
            up = F.M17 / "results" / f"units_{root}.parquet"
            if up.exists():
                parts.append(pd.read_parquet(up, columns=["cell", "arm", "fs", "n", "test"]))
        u = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["cell", "arm", "fs", "n", "test"])
        for (cell, arm), g in w.groupby(["cell", "arm"]):
            dens0, redraw, dead, final, dead_h, dead_w, with_records = [], [], [], [], [], 0, 0
            for r in g.itertuples():
                rd = _run_dir(r)
                dead_w += int(float(r.val) >= F.DEAD_VAL)
                if (rd / "run.json").exists():
                    with_records += 1
                    m = json.load(open(rd / "run.json"))["metrics"]
                    dead.append(1.0 - m.get("first_layer_active_fraction", np.nan) if "first_layer_active_fraction" in m else np.nan)
                    redraw.append(m.get("readout_redraws", np.nan))
                try:
                    ms = mask_stats(rd)
                except Exception:  # noqa: BLE001
                    ms = {}
                if "init_density" in ms:
                    dens0.append(ms["init_density"])
                if "final_density" in ms:
                    final.append(ms["final_density"]); dead_h.append(ms["dead_hidden_share"])
            spread = np.nan
            if len(u):
                keys = pd.MultiIndex.from_frame(g[["fs", "n"]].astype(int))
                uu = u[(u.cell == cell) & (u.arm == arm)]
                uu = uu[pd.MultiIndex.from_frame(uu[["fs", "n"]].astype(int)).isin(keys)]
                s = uu.groupby(["fs", "n"]).test.agg(lambda t: float(np.nanstd(np.log10(np.clip(t, 1e-12, None)))))
                spread = float(s.mean()) if len(s) else np.nan
            rows.append(dict(panel=panel, cell=cell, workload=CELL_LABEL.get(cell, cell), arm=arm, label=M.PAPER.get(arm, arm), winners=len(g),
                             init_density=float(np.nanmean(dens0)) if dens0 else np.nan, final_density=float(np.nanmean(final)) if final else np.nan,
                             dead_first_layer_share=float(np.nanmean(dead)) if dead else np.nan,
                             dead_hidden_share=float(np.nanmean(dead_h)) if dead_h else np.nan, redraws_mean=float(np.nanmean(redraw)) if redraw else np.nan,
                             dead_winners=dead_w, lr_trial_spread_log10_test=spread, winners_with_records=with_records))
    return pd.DataFrame(rows)


def sensitivity(record_panel: str, arms: dict, key: str, record_value) -> pd.DataFrame:
    """The reduced-grid AULC of every sweep arm against the adopted LeWi's on the same instances and sizes (recomputed from the
    ledger's winners over the reduced grid)."""
    w = variant_winners(record_panel)
    ref, _ = lewi_reference("d8")
    if w.empty:
        return pd.DataFrame()
    rows = []
    for cell in M.SENS_CELLS:
        r = ref[(ref.cell == cell) & ref.n.isin(M.SENS_NS) & ref.fs.isin(M.SENS_COHORT)]
        lewi = r.groupby("fs").apply(lambda g: aulc_of(g.sort_values("n").n, g.sort_values("n").test))
        st = F.tci(lewi)
        st["dead_winners"] = int((r.val >= F.DEAD_VAL).sum())
        rows.append({**st, "cell": cell, "workload": CELL_LABEL[cell], "arm": F.LEWI, key: record_value, "label": f"{key} = {record_value} (the adopted arm)", "n_instances": len(lewi),
                     "diff_mean": 0.0, "diff_lo": 0.0, "diff_hi": 0.0, "diff_median": 0.0, "wins": np.nan, "n_pairs": len(lewi)})
        for arm, val in arms.items():
            x = w[(w.cell == cell) & (w.arm == arm)]
            if x.empty:
                continue
            full = x.groupby("fs").n.nunique()
            x = x[x.fs.isin(full[full == len(M.SENS_NS)].index)]
            a = x.groupby("fs").apply(lambda g: aulc_of(g.sort_values("n").n, g.sort_values("n").test))
            stx = F.tci(a)
            pr = paired(lewi, a)
            stx["dead_winners"] = int((x.val >= F.DEAD_VAL).sum())
            rows.append({**stx, "cell": cell, "workload": CELL_LABEL[cell], "arm": arm, key: val, "label": M.PAPER.get(arm, arm), "n_instances": len(a),
                         "diff_mean": pr["mean"], "diff_lo": pr["ci95_lo"], "diff_hi": pr["ci95_hi"], "diff_median": pr["median"], "wins": pr["wins"], "n_pairs": pr["n_pairs"]})
    return pd.DataFrame(rows)


def curves() -> pd.DataFrame:
    """Mean test NMSE against n with its 95% t interval over instances (and the geometric mean with the t band of log10
    NMSE beside it) of Dense, the adopted LeWi and the four variants on the core workloads: the main-text alternative of panel (c)."""
    led = pd.read_csv(F.STUDY / "ledger" / "d8" / "winners.csv").rename(columns={"function_seed": "fs", "test_nmse": "test"})
    ref = led[led.arm.isin(["noreg", F.LEWI])][["cell", "arm", "fs", "n", "test"]]
    var = variant_winners("abl8")[["cell", "arm", "fs", "n", "test"]]
    rows = []
    for w in (ref, var):
        w = w[w.cell.isin(CELLS)]
        for (cell, arm, n), g in w.groupby(["cell", "arm", "n"]):
            if g.fs.nunique() < 16:
                continue                                  # a point enters once all 16 instances are on record
            raw = g.test.to_numpy(dtype=float)
            lg = np.log10(np.clip(raw, 1e-12, None))
            st, sa = F.tci(lg), F.tci(raw)               # the figures draw the arithmetic mean (the curve standard, decision 2026-09-18)
            rows.append(dict(cell=cell, workload=CELL_LABEL[cell], arm=arm, label=M.PAPER.get(arm, arm), n=int(n), mean=sa["mean"], lo=sa["ci95_lo"], hi=sa["ci95_hi"],
                             gmean=10 ** st["mean"], glo=10 ** st["ci95_lo"], ghi=10 ** st["ci95_hi"],
                             median=float(np.median(g.test)), n_instances=int(g.fs.nunique())))
    return pd.DataFrame(rows)


def configs() -> dict:
    out = {"generated_local": F.now_local(), "arms": {}, "panels": {}}
    for a in VARIANTS + list(M.SENS_R_ARMS) + list(M.SENS_RAD_ARMS) + list(getattr(M, "SENS_INIT_ARMS", {})) + list(getattr(M, "SENS_SEED_ARMS", {})):
        out["arms"][a] = {"label": M.PAPER.get(a, a), "method_id": M.ARM_METHOD.get(a, a), "fixed_hp": M.ARM_HP.get(a, {}), "spec_fields": getattr(M, "ARM_SPEC", {}).get(a, {}), "space_file": M.SPACE_FILE_C.get(a), "trial_key": M.TRIAL_KEY.get(a, a)}
    for rec, pn in IN5_PANEL.items():
        if pn in M.PANELS:
            pc = M.PANELS[pn]
            out["panels"][pn] = {"record_panel": rec, "cells": pc["cells"], "arm_method": pc.get("arm_method", {}), "cohort": [int(pc["cohort"][0]), int(pc["cohort"][-1])],
                                 "ns": list(pc["ns"]), "profile": dict(M.PROFILES[pc["profile"]]), "spaces": pc["spaces"], "retain": bool(pc.get("retain")),
                                 "restricted_to_manifest": bool(pc.get("pairs")), "note": pc.get("note"),
                                 "reference": "the adopted LeWi (18-main-d8 ledger/d32s)" if rec == "abl32" else "the adopted LeWi (18-main-d8 ledger/d8)"}
    return out


def feature_selection(write: bool = True) -> pd.DataFrame:
    """Figure 6(a)'s data (figures/data-final-clean/fig6/feature_selection.csv): LeWi minus Dense, L0 and GO(all), paired per instance,
    from the study's contract tables - analysis/d32s/contract (the nuisance-input setting on the twin's instances, d = 32; decision
    2026-09-22, before that analysis/contract, the record's instances) and analysis/d8/contract (the clean setting; LeWi under the
    adopted input-cutoff rule since 2026-09-20), so both settings run on the instances 6000-6015 (6020-6035 added on the sampled graphs).
    off_scale marks a pair with an off-scale winner (a test NMSE above 10) in the cell's disclosures, where the figure reads the median."""
    rows = []
    for setting, cdir, cells in (("nuisance (d = 32)", F.STUDY / "analysis" / "d32s" / "contract", F.D32S_CELLS),
                                 ("clean (d = 8)", F.STUDY / "analysis" / "d8" / "contract", F.D8_CELLS)):
        con, dis = pd.read_csv(cdir / "contrasts.csv"), pd.read_csv(cdir / "disclosures.csv")
        for ref in ("noreg", "l0hc", "graph_oracle"):
            for cell in cells:
                r = con[(con.cell == cell) & (con.arm == F.LEWI) & (con.ref == ref)]
                if r.empty:
                    continue
                r = r.iloc[0]
                off = int(bool(len(dis[(dis.cell == cell) & dis.arm.isin([F.LEWI, ref]) & dis.off_scale.astype(bool)])))
                rows.append(dict(setting=setting, workload=F.CELL[cell]["task"], cell=cell, ref=ref, mean=r["mean"], lo=r.ci95_lo, hi=r.ci95_hi,
                                 median=r["median"], wins=int(r.wins), n_pairs=int(r.n_pairs), off_scale=off))
    df = pd.DataFrame(rows)
    if write:
        FIG.mkdir(parents=True, exist_ok=True)
        df.to_csv(FIG / "feature_selection.csv", index=False)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    settings = os.environ.get("ABLATIONS_SETTINGS", "d8,d32s").split(",")   # d8 alone while the twin's panel runs
    feature_selection()
    con, by_n, au = contrasts("d8")
    F.write(OUT, "contrasts.csv", con)
    F.write(OUT, "contrasts_by_n.csv", by_n)
    F.write(OUT, "aulc.csv", au)
    if len(con):
        con.to_csv(FIG / "ablations.csv", index=False)
    con32 = pd.DataFrame()
    if CELLS_D32 and "d32s" in settings:
        con32, by_n32, au32 = contrasts("d32s")
        F.write(OUT, "contrasts_d32.csv", con32)
        F.write(OUT, "contrasts_d32_by_n.csv", by_n32)
        F.write(OUT, "aulc_d32.csv", au32)
        if len(con32):
            con32.to_csv(FIG / "ablations_d32.csv", index=False)
    elif CELLS_D32:
        for stale in (OUT / "contrasts_d32.csv", OUT / "contrasts_d32_by_n.csv", OUT / "aulc_d32.csv", FIG / "ablations_d32.csv"):
            stale.unlink(missing_ok=True)
        print("  d32s skipped (ABLATIONS_SETTINGS): the twin's panel is not read yet; its outputs are removed until the full read")
    for record_panel, arms, key, value, name in (("abl8r", M.SENS_R_ARMS, "r", 3, "sensitivity_r"), ("abl8rad", M.SENS_RAD_ARMS, "radius", 1.0, "sensitivity_rad"),
                                                  ("abl8init", getattr(M, "SENS_INIT_ARMS", {}), "init_scale", 1.0, "sensitivity_init"),
                                                  ("abl8seed", getattr(M, "SENS_SEED_ARMS", {}), "seed_replicate", 0, "sensitivity_seed")):
        s = sensitivity(record_panel, arms, key, value)
        F.write(OUT, f"{name}.csv", s)
        s.to_csv(FIG / f"{name}.csv", index=False)
    cur = curves()
    F.write(OUT, "curves.csv", cur)
    if len(cur):
        cur.to_csv(FIG / "ablation_curves.csv", index=False)
    F.jdump(OUT / "configs.json", configs())
    dens = density()
    F.write(OUT, "density.csv", dens)
    dens.to_csv(FIG / "density.csv", index=False)
    summ = {"generated_local": F.now_local(), "variants": {v: VARIANT_LABEL[v] for v in VARIANTS}, "cells": CELLS, "reference": "the adopted LeWi (ledger/d8, ledger/d32s)",
            "panels": IN5_PANEL, "checks": CHECKS, "settings": settings,
            "contrasts": {f"{r.cell}/{r.variant}": dict(mean=r["mean"], lo=r.ci95_lo, hi=r.ci95_hi, median=r["median"], wins=int(r.wins), n_pairs=int(r.n_pairs)) for _, r in con.iterrows()} if len(con) else {},
            "contrasts_d32": {f"{r.cell}/{r.variant}": dict(mean=r["mean"], lo=r.ci95_lo, hi=r.ci95_hi, median=r["median"], wins=int(r.wins), n_pairs=int(r.n_pairs)) for _, r in con32.iterrows()} if len(con32) else {},
            "complete": {v: (int((con.variant == v).sum()) if len(con) else 0) for v in VARIANTS}}
    F.jdump(OUT / "summary.json", summ)
    for k, v in CHECKS.items():
        print(f"  {k}: {v}")
    if len(con):
        print(con[["workload", "label", "mean", "ci95_lo", "ci95_hi", "wins", "n_pairs"]].to_string())


if __name__ == "__main__":
    main()
