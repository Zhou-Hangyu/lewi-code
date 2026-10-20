"""The reader of the swm-lrx re-run (2026-09-23): the re-run of Dense and L1 in the expanded learning-rate spaces against the
record. Writes tables/{trials,selections,occupancy,aulc,contrasts,verdict_changes}.csv and tables/read.json; partial data is read as far as
it goes (a cell enters the AULC tables only with its 16 finals on record).
Amendment 1 (the second stage): a cell whose selection2.json says changed enters the AULC tables with its resel-final-s<s>.json in place of
the first-stage finals; tables/stage2.csv carries every evaluated group's trigger values and the second-stage selections.

The reference the re-run is compared against ("prior" in the tables; "adopted" before the adoption of 2026-09-24) is the record as adopted
on 2026-09-23: the record as run (16-main-final/external/analysis/instances.csv) with the 30 LeWi final seeds the input rule in5 re-ran
(swm-in5/tables/instances_in5.csv) and the 480 finals of the 30 cells the validation-only re-selection changed (swm-rollout/tables/
repaired_instances.csv; Manipulator LeWi 262,144 seed 4 from its in5 record) substituted; the record as run is read beside it.
Adoption (2026-09-24; README): the re-run's finals are the adopted record's Dense and L1
arms on the counted tasks - tables/finals.csv (one row per final: task, n, arm, seed, the test rollout NMSE at 16 steps, the one-step test
NMSE, diverged, error, the selected learning rate) is what figures/src/harvest_swm16.py substitutes; the "updated" columns are the adopted
record's contrasts. Statistic: the record's own (swm16.read / the harvester's
contrasts_of): per seed the trapezoid over log2 n of log10 rollout NMSE at 16 steps (diverged at the cap 1e6, floor 1e-12) divided
by the log2 n range; paired differences over the 16 seeds with a two-sided 95% Student-t interval; win / loss when the interval
excludes zero. Positions: (log10 lr - log10 low) / (log10 high - log10 low) in the expanded space; the record's winners are mapped
into the same coordinates from analysis/boundary_occupancy.csv; near an edge = within 10% (the boundary audit's rule; 5% beside it);
material = more than a quarter near an edge or a median outside [0.10, 0.90].

    python src/lrx_read.py            (.venv: pandas, scipy)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student

HERE = Path(__file__).resolve()
STUDY = HERE.parents[1]
REPO = HERE.parents[5]
D18 = REPO / "studies/main-experiments/18-main-d8"
RECORD = REPO / "studies/main-experiments/16-main-final/external/analysis"
OUT = REPO / "results/csfrbench/swmlrx/lrx"
TABLES = STUDY / "tables"
MANIFEST = json.loads((STUDY / "manifests/spaces_lrx.json").read_text())
SEED4 = REPO / "results/csfrbench/swmin5/resel-in5/manipulator/n262144/lewi-v1/s04.json"
ARMS, SEEDS, TRIALS = ["dense", "l1"], 16, 32
CAP, FLOOR = 1e6, 1e-12
EDGE, EDGE5 = 0.10, 0.05


def t_interval(d: np.ndarray) -> tuple[float, float, float]:
    m = float(np.mean(d))
    half = float(student.ppf(0.975, len(d) - 1) * np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return m, m - half, m + half


def verdict(lo: float, hi: float) -> str:
    return "win" if hi < 0 else "loss" if lo > 0 else "tie"


def log10_nmse(nmse, diverged) -> np.ndarray:
    v = np.where(np.asarray(diverged, dtype=float) > 0, CAP, np.asarray(nmse, dtype=float))
    return np.log10(np.clip(v, FLOOR, None))


def position(lr: float, low: float, high: float) -> float:
    return (math.log10(lr) - math.log10(low)) / (math.log10(high) - math.log10(low))


def prior_and_sealed() -> tuple[pd.DataFrame, pd.DataFrame]:
    sealed = pd.read_csv(RECORD / "instances.csv")
    rr = pd.read_csv(D18 / "swm-in5/tables/instances_in5.csv")
    rr = rr[(rr.role == "rule") & (rr.state == "done")][["task", "n", "seed", "nmse", "diverged"]].rename(columns={"nmse": "nmse_in5", "diverged": "div_in5"})
    m = sealed.merge(rr, on=["task", "n", "seed"], how="left")
    hit = m.nmse_in5.notna() & (m.arm == "lewi-v1")
    assert int(hit.sum()) == 30, int(hit.sum())
    m = m.assign(nmse=m.nmse.where(~hit, m.nmse_in5), diverged=m.diverged.where(~hit, m.div_in5)).drop(columns=["nmse_in5", "div_in5"])
    rp = pd.read_csv(D18 / "swm-rollout/tables/repaired_instances.csv")
    rp = rp[rp.role == "flagged"][["task", "n", "arm", "seed", "nmse", "diverged"]].rename(columns={"nmse": "nmse_resel", "diverged": "div_resel"})
    m2 = m.merge(rp, on=["task", "n", "arm", "seed"], how="left")
    hit2 = m2.nmse_resel.notna()
    assert int(hit2.sum()) == 480, int(hit2.sum())
    if SEED4.exists():
        rec = json.loads(SEED4.read_text())
        sel = (m2.task == "manipulator") & (m2.n == 262144) & (m2.arm == "lewi-v1") & (m2.seed == 4)
        m2.loc[sel, "nmse_resel"] = float((rec.get("test_rollout") or {}).get("16"))
        m2.loc[sel, "div_resel"] = int(rec.get("diverged") or 0)
    prior = m2.assign(nmse=m2.nmse.where(~hit2, m2.nmse_resel), diverged=m2.diverged.where(~hit2, m2.div_resel)).drop(columns=["nmse_resel", "div_resel"])
    return prior, sealed


def aulc_table(inst: pd.DataFrame, grids: dict) -> dict[tuple[str, str, int], float]:
    v = inst.assign(l10=log10_nmse(inst.nmse, inst.diverged))
    out = {}
    for (task, arm, seed), g in v.groupby(["task", "arm", "seed"]):
        if task not in grids:
            continue
        g = g.sort_values("n")
        if list(g.n) != list(grids[task]) or g.l10.isna().any():
            continue
        x = np.log2(np.asarray(grids[task], dtype=float))
        out[(task, arm, int(seed))] = float(np.trapezoid(g.l10.to_numpy(), x) / (x[-1] - x[0]))
    return out


def rerun_tables(counted: list[str], grids: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """trials (from the cells' selection.json), selections (one row per selected cell) and the re-run's per-seed finals."""
    rec_pos = pd.read_csv(RECORD / "boundary_occupancy.csv")
    rec_pos = rec_pos[(rec_pos.parameter == "lr") & rec_pos.arm.isin(ARMS)].set_index(["task", "n", "arm"]).position
    trials, sels, finals = [], [], []
    for task in counted:
        for n in grids[task]:
            for arm in ARMS:
                root = OUT / task / f"n{n}" / arm
                sp = root / "selection.json"
                if not sp.exists():
                    continue
                s = json.loads(sp.read_text())
                a = MANIFEST["arms"][arm]
                low_r, low_x, high = a["lr_low_record"], a["lr_low_expanded"], a["lr_high"]
                for t in s["trials_summary"]:
                    trials.append(dict(task=task, n=n, arm=arm, trial=t["trial"], lr=t["hp"]["lr"], lam=t["hp"].get("lam"), val_rollout_score=t["val_rollout_score"],
                                       best_val_nmse=t["best_val_nmse"], best_step=t["best_step"], diverged=t["diverged"], error=t["error"], wall_s=t["wall_s"],
                                       position_expanded=position(t["hp"]["lr"], low_x, high), below_record_low=bool(t["hp"]["lr"] < low_r)))
                lr = s["winner_hp"]["lr"]
                rp = rec_pos.get((task, n, arm), np.nan)
                span_r, span_x = math.log10(high) - math.log10(low_r), math.log10(high) - math.log10(low_x)
                rec_lr = 10 ** (math.log10(low_r) + rp * span_r) if pd.notna(rp) else np.nan
                s2p = root / "selection2.json"
                s2 = json.loads(s2p.read_text()) if s2p.exists() else None
                changed = bool(s2 and s2.get("changed"))
                prefix = "resel-final-s" if changed else "final-s"
                if changed:
                    lr = s2["selected_hp"]["lr"]
                fin = {}
                for sd in range(SEEDS):
                    fp = root / f"{prefix}{sd:02d}.json"
                    if fp.exists():
                        d = json.loads(fp.read_text())
                        fin[sd] = d
                        finals.append(dict(task=task, n=n, arm=arm, seed=sd, nmse=(d.get("test_rollout") or {}).get("16"), one_step=(d.get("test_one_step") or {}).get("nmse"),
                                           diverged=int(d.get("diverged") or 0), error=d.get("error"), lr=lr, wall_s=d.get("wall_s")))
                sels.append(dict(task=task, n=n, arm=arm, counted=True, winner_trial=(s2["selected_trial"] if changed else s["winner_trial"]), first_stage_trial=s["winner_trial"],
                                 stage2_selected=bool(s2), stage2_changed=changed, lr=lr, lam=(s2["selected_hp"].get("lam") if changed else s["winner_hp"].get("lam")),
                                 val_rollout_score=s["winner_val_rollout_score"], best_val_nmse=s["winner_best_val_nmse"], diverged_trials=s["diverged_trials"],
                                 position_expanded=position(lr, low_x, high), position_record_space=position(lr, low_r, high),
                                 below_record_low=bool(lr < low_r), near_low_10=bool(position(lr, low_x, high) <= EDGE), near_low_5=bool(position(lr, low_x, high) <= EDGE5),
                                 near_high_10=bool(position(lr, low_x, high) >= 1 - EDGE), record_lr=rec_lr, record_position=rp,
                                 record_position_in_expanded=((rp * span_r + (span_x - span_r)) / span_x) if pd.notna(rp) else np.nan,
                                 lr_ratio_to_record=(lr / rec_lr) if pd.notna(rp) else np.nan, finals_done=len(fin),
                                 finals_diverged=int(sum(int(d.get("diverged") or 0) for d in fin.values())), finals_errors=int(sum(1 for d in fin.values() if d.get("error"))),
                                 selected_utc=s["selected_utc"]))
    stage2 = []
    for task in counted:
        for arm in ARMS:
            gp = OUT / task / "stage2" / f"{arm}.json"
            if not gp.exists():
                continue
            g = json.loads(gp.read_text())
            for n_str, r in g["cells"].items():
                s2p = OUT / task / f"n{int(n_str)}" / arm / "selection2.json"
                s2 = json.loads(s2p.read_text()) if s2p.exists() else {}
                stage2.append(dict(task=task, n=int(n_str), arm=arm, winner_trial=r["winner_trial"], winner_val_score=r["winner_val_score"], finals_val_gmean=r["finals_val_gmean"],
                                   fragility=r["fragility"], exploded_val=r["exploded_val"], val_bump_rule=r["val_bump_rule"], fragile=r["fragile"], flagged=r["flagged"],
                                   stage2_selected=bool(s2), selected_trial=s2.get("selected_trial"), selected_rank=s2.get("selected_rank"), changed=s2.get("changed"),
                                   selected_lr=(s2.get("selected_hp") or {}).get("lr"), candidates=json.dumps([dict(trial=c["trial"], lr=c["hp"]["lr"], gmean=round(10 ** c["mean_log10_val_score"], 6)) for c in s2.get("candidates", [])])))
    pd.DataFrame(stage2).to_csv(TABLES / "stage2.csv", index=False)
    return pd.DataFrame(trials), pd.DataFrame(sels), pd.DataFrame(finals)


def occupancy(sels: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        g = sels[sels.arm == arm]
        if not len(g):
            continue
        z = g.position_expanded.to_numpy(float)
        n_low, n_high = int((z <= EDGE).sum()), int((z >= 1 - EDGE).sum())
        med = float(np.median(z))
        zr = g.record_position.dropna().to_numpy(float)
        zrx = g.record_position_in_expanded.dropna().to_numpy(float)
        tr = trials[trials.arm == arm]
        rows.append(dict(arm=arm, cells=len(g), cells_total=len(sels[sels.arm == arm]) if len(sels) else 0,
                         near_low_10=n_low, near_low_10_share=n_low / len(g), near_low_5_share=float((z <= EDGE5).mean()),
                         near_high_10=n_high, near_high_10_share=n_high / len(g), median_position=med, material=bool(4 * (n_low + n_high) > len(g) or med < 0.10 or med > 0.90),
                         below_record_low=int(g.below_record_low.sum()), below_record_low_share=float(g.below_record_low.mean()),
                         record_near_low_10_share=float((zr <= EDGE).mean()) if len(zr) else np.nan, record_near_low_5_share=float((zr <= EDGE5).mean()) if len(zr) else np.nan,
                         record_median_position=float(np.median(zr)) if len(zr) else np.nan, record_median_position_in_expanded=float(np.median(zrx)) if len(zrx) else np.nan,
                         record_material=bool(4 * int((zr <= EDGE).sum() + (zr >= 1 - EDGE).sum()) > len(zr) or np.median(zr) < 0.10 or np.median(zr) > 0.90) if len(zr) else None,
                         trial_draws=len(tr), trial_draws_below_record_low_share=float(tr.below_record_low.mean()) if len(tr) else np.nan,
                         median_lr_ratio_to_record=float(g.lr_ratio_to_record.median()) if g.lr_ratio_to_record.notna().any() else np.nan,
                         lr_low_record=MANIFEST["arms"][arm]["lr_low_record"], lr_low_expanded=MANIFEST["arms"][arm]["lr_low_expanded"], lr_high=MANIFEST["arms"][arm]["lr_high"]))
    return pd.DataFrame(rows)


def main() -> None:
    read = json.loads((RECORD / "read.json").read_text())
    counted = list(read["claim_bearing"])
    grids = {t: [int(x) for x in read["grids"][t]] for t in counted}
    prior, sealed = prior_and_sealed()
    trials, sels, finals = rerun_tables(counted, grids)
    TABLES.mkdir(exist_ok=True)
    trials.to_csv(TABLES / "trials.csv", index=False)
    sels.to_csv(TABLES / "selections.csv", index=False)
    finals.to_csv(TABLES / "finals.csv", index=False)
    occ = occupancy(sels, trials) if len(sels) else pd.DataFrame()
    occ.to_csv(TABLES / "occupancy.csv", index=False)
    au_a, au_s = aulc_table(prior, grids), aulc_table(sealed, grids)
    rerun = pd.concat([prior[~prior.arm.isin(ARMS)], finals[["task", "n", "arm", "seed", "nmse", "diverged", "error"]]]) if len(finals) else prior[~prior.arm.isin(ARMS)]
    au_r = aulc_table(rerun, grids)
    # per-task AULC of the re-run arms against the prior and the record as run
    au_rows, con_rows = [], []
    for task in counted:
        for arm in ARMS:
            r = [au_r.get((task, arm, s)) for s in range(SEEDS)]
            if any(v is None for v in r):
                continue
            a = np.array([au_a[(task, arm, s)] for s in range(SEEDS)]); sd_ = np.array([au_s[(task, arm, s)] for s in range(SEEDS)]); r = np.array(r)
            dm, dlo, dhi = t_interval(r - a); sm, slo, shi = t_interval(r - sd_)
            sc = sels[(sels.task == task) & (sels.arm == arm)]
            au_rows.append(dict(task=task, arm=arm, aulc_rerun=float(r.mean()), aulc_prior=float(a.mean()), aulc_sealed=float(sd_.mean()),
                                cells_below_old_bound=int(sc.below_record_low.sum()), cells_stage2_changed=int(sc.stage2_changed.sum()) if 'stage2_changed' in sc else 0,
                                d_rerun_minus_prior=dm, d_lo=dlo, d_hi=dhi, d_verdict=("lower" if dhi < 0 else "higher" if dlo > 0 else "same"),
                                d_rerun_minus_sealed=sm, d_sealed_lo=slo, d_sealed_hi=shi, seeds_lower=int((r < a).sum()),
                                diverged_finals=int(finals[(finals.task == task) & (finals.arm == arm)].diverged.sum())))
        for arm in ARMS:
            keys = [(task, "lewi-v1", s) for s in range(SEEDS)]
            if not all(k in au_a for k in keys) or not all((task, arm, s) in au_r for s in range(SEEDS)):
                continue
            lw = np.array([au_a[k] for k in keys]); lws = np.array([au_s[k] for k in keys])
            d_new = lw - np.array([au_r[(task, arm, s)] for s in range(SEEDS)])
            d_ad = lw - np.array([au_a[(task, arm, s)] for s in range(SEEDS)])
            d_se = lws - np.array([au_s[(task, arm, s)] for s in range(SEEDS)])
            m, lo, hi = t_interval(d_new); ma, loa, hia = t_interval(d_ad); ms, los, his = t_interval(d_se)
            con_rows.append(dict(task=task, reference=arm, updated_mean=m, updated_lo=lo, updated_hi=hi, updated_better=int((d_new < 0).sum()), updated_verdict=verdict(lo, hi),
                                 prior_mean=ma, prior_lo=loa, prior_hi=hia, prior_verdict=verdict(loa, hia),
                                 sealed_mean=ms, sealed_lo=los, sealed_hi=his, sealed_verdict=verdict(los, his),
                                 changed_vs_prior=bool(verdict(lo, hi) != verdict(loa, hia)), changed_vs_sealed=bool(verdict(lo, hi) != verdict(los, his))))
    au = pd.DataFrame(au_rows); con = pd.DataFrame(con_rows)
    au.to_csv(TABLES / "aulc.csv", index=False); con.to_csv(TABLES / "contrasts.csv", index=False)
    chg = con[con.changed_vs_prior | con.changed_vs_sealed] if len(con) else pd.DataFrame()
    chg.to_csv(TABLES / "verdict_changes.csv", index=False)
    tally = {}
    for arm in ARMS:
        g = con[con.reference == arm] if len(con) else pd.DataFrame()
        tally[arm] = {k: ([int((g[f"{k}_verdict"] == v).sum()) for v in ("win", "tie", "loss")] if len(g) else None) for k in ("updated", "prior", "sealed")}
        tally[arm]["tasks"] = int(len(g))
    summary = dict(read_local=time.strftime("%Y-%m-%d %H:%M", time.localtime()), counted_tasks=len(counted), cells_total=sum(len(grids[t]) for t in counted) * len(ARMS),
                   cells_selected=int(len(sels)), cells_with_16_finals=int((sels.finals_done == SEEDS).sum()) if len(sels) else 0, finals_on_record=int(len(finals)),
                   trials_on_record_of_selected_cells=int(len(trials)), rule=MANIFEST["rule"],
                   occupancy={r["arm"]: {k: r[k] for k in ("cells", "near_low_10_share", "near_low_5_share", "near_high_10_share", "median_position", "material",
                                                            "below_record_low", "below_record_low_share", "record_near_low_10_share", "record_median_position",
                                                            "record_median_position_in_expanded", "record_material", "trial_draws_below_record_low_share",
                                                            "median_lr_ratio_to_record")} for r in occ.to_dict("records")} if len(occ) else {},
                   aulc_changes={arm: {k: int((au[au.arm == arm].d_verdict == k).sum()) for k in ("lower", "higher", "same")} for arm in ARMS} if len(au) else {},
                   tally=tally, verdict_changes=chg[["task", "reference", "updated_verdict", "prior_verdict", "sealed_verdict"]].to_dict("records") if len(chg) else [],
                   tasks_with_contrasts={arm: int(len(con[con.reference == arm])) if len(con) else 0 for arm in ARMS},
                   stage2=dict(groups_evaluated=int(len({(t, a) for t in counted for a in ARMS if (OUT / t / "stage2" / f"{a}.json").exists()})),
                               flagged=int(pd.read_csv(TABLES / "stage2.csv").flagged.sum()) if (TABLES / "stage2.csv").stat().st_size > 1 and len(pd.read_csv(TABLES / "stage2.csv")) else 0,
                               changed_cells=int(sels.stage2_changed.sum()) if len(sels) else 0))
    (TABLES / "read.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "rule"}, indent=1, default=float))


if __name__ == "__main__":
    main()
