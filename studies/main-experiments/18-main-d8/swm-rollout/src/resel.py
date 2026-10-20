"""Robust re-selection of fragile cells of the SWM-16 record (18-main-d8/swm-rollout; validation only; one rule for
every arm). the record as run is read only; every record of this study lands under results/csfrbench/swmfix/resel/.

THE RULE (stated in README.md before any unit of it ran):
  trigger     on VALIDATION values of the record as run only, a cell is flagged when both hold:
              (a) fragile winner: the geometric mean of the 16 finals' validation rollout score is at least 1.5 x the
                  winner trial's score (fragility >= log10 1.5), or at least one final's score is above 10 x the median
                  of the cell's 16 finals (an exploded final);
              (b) more data made it worse: the finals' geometric-mean validation score is at least 0.10 log10 above
                  BOTH neighbour sizes (the largest size: above the size before it; the smallest size has no test);
  candidates  the top K = 4 trials of the cell by the record's key (no divergence, validation rollout score,
              one-step validation NMSE, trial index); the record's winner is candidate 1;
  evidence    every candidate is trained on M = 8 seeds of a fresh bank ("resel": never used by any record; the
              method and the candidate never enter the seed derivation, so seed r is paired across candidates);
  selection   the candidate with the lowest mean log10 validation rollout score over the M seeds (a diverged run
              counts at the cap 1e6), then the lowest mean log10 one-step validation NMSE, then the record's rank;
  finals      when the selected candidate is not the record's winner it is trained on the record's 16 paired final
              seeds and tested once per seed; when it is the record's winner the cell keeps its record.
No test value enters the trigger or the selection.

    python resel.py cells                 tables/resel_cells.csv (the flagged cells and the controls, in run order)
    python resel.py pool <deadline> <w>   claim-and-run pending units until the deadline
    python resel.py status                counts per cell
    python resel.py read                  tables/resel_selection.csv, resel_runs.csv, repaired_cells.csv, repaired_instances.csv
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import swmfix  # noqa: E402
from swmfix import swm16  # noqa: E402

STUDY = HERE.parents[1]
TABLES = STUDY / "tables"
ROOT = swmfix.OUT / "resel"
K, M, BANK, SEEDS = 4, 8, "resel", 16
FRAGILITY_MIN, VAL_BUMP_MIN = math.log10(1.5), 0.10
CAP = 1e6
CONTROL_SEED = 20260918
HEADLINE = ("manipulator", 262144, "lewi-v1")
NAMED = [("piecewise", 16384, "lewi-v1")]      # a small bump examined separately; NOT flagged by the trigger (validation bump 0.04); reported apart


def sealed_winner(task: str, n: int, arm: str) -> dict:
    return json.loads((swm16.RESULTS / "final" / "hpo" / task / f"n{n}" / f"{arm}.json").read_text())


_CANDS: dict = {}


def candidates(task: str, n: int, arm: str) -> list[dict]:
    if (task, n, arm) not in _CANDS:
        _CANDS[(task, n, arm)] = _candidates(task, n, arm)
    return _CANDS[(task, n, arm)]


def _candidates(task: str, n: int, arm: str) -> list[dict]:
    w = sealed_winner(task, n, arm)
    key = lambda t: (int(t.get("diverged") or 0), t["val_rollout_score"] if t.get("val_rollout_score") is not None else float("inf"),  # noqa: E731
                     t["best_val_nmse"] if t.get("best_val_nmse") is not None else float("inf"), t["trial"])
    ranked = sorted(w["trials_summary"], key=key)[:K]
    assert ranked[0]["trial"] == w["winner_trial"], (task, n, arm)
    return [dict(rank=i, trial=t["trial"], hp=t["hp"], flush=bool(t.get("flush_denormal")), trial_val_score=t["val_rollout_score"],
                 trial_val_one_step=t["best_val_nmse"]) for i, t in enumerate(ranked)]


def cells() -> None:
    """The flagged cells (the trigger on validation values of tables/cells_screen.csv) and the controls, in run order."""
    import pandas as pd
    c = pd.read_csv(TABLES / "cells_screen.csv")
    c["val_bump_rule"] = c.val_bump
    for (_, _), g in c.groupby(["task", "arm"]):
        g = g.sort_values("n")
        c.loc[g.index[-1], "val_bump_rule"] = np.log10(g.finals_val_gmean.values[-1]) - np.log10(g.finals_val_gmean.values[-2])
    c["fragile"] = (c.fragility >= FRAGILITY_MIN) | (c.exploded_val > 0)
    c["flagged"] = c.fragile & (c.val_bump_rule >= VAL_BUMP_MIN)
    c.to_csv(TABLES / "cells_screen.csv", index=False)
    flagged = c[c.flagged].copy()
    flagged["role"] = "flagged"
    key = lambda r: (0 if (r.task, r.n, r.arm) == HEADLINE else 1 if r.arm == "lewi-v1" else 3 if r.arm != "erb" else 4, -r.val_bump_rule)  # noqa: E731
    flagged["order"] = [key(r) for r in flagged.itertuples()]
    healthy = c[~c.fragile & ~(c.val_bump_rule >= VAL_BUMP_MIN)]
    fixed = [("manipulator", 131072, "lewi-v1"), ("manipulator", 524288, "lewi-v1"), ("manipulator", 262144, "dense")]
    ctrl = [healthy[(healthy.task == t) & (healthy.n == n) & (healthy.arm == a)] for t, n, a in fixed]
    assert all(len(x) == 1 for x in ctrl), "a fixed control is not a healthy cell"
    rng = np.random.default_rng(CONTROL_SEED)
    for arm in swm16.ARMS:                                            # one random healthy cell per arm
        pool_ = healthy[(healthy.arm == arm) & ~healthy.index.isin([x.index[0] for x in ctrl])]
        ctrl.append(pool_.iloc[[int(rng.integers(len(pool_)))]])
    ctrl = pd.concat(ctrl)
    ctrl["role"] = "control"
    ctrl["order"] = [(2, float(i)) for i in range(3)] + [(5, float(i)) for i in range(len(ctrl) - 3)]
    named = pd.concat([c[(c.task == t) & (c.n == n) & (c.arm == a)] for t, n, a in NAMED])
    named = named[~named.flagged].copy()
    named["role"] = "named"
    named["order"] = [(1, 9.0 + i) for i in range(len(named))]
    out = pd.concat([flagged, ctrl, named]).sort_values("order").drop(columns=["order"])
    out["speculative"] = [(int((r.task, r.n, r.arm) == HEADLINE)) for r in out.itertuples()]
    cols = ["task", "n", "arm", "role", "speculative", "winner_trial", "winner_lr", "trial_val_score", "finals_val_gmean", "finals_val_max",
            "fragility", "exploded_val", "val_bump_rule"]
    out[cols].to_csv(TABLES / "resel_cells.csv", index=False)
    print(out[cols].round(4).to_string())
    print(len(flagged), "flagged,", len(ctrl), "controls,", (len(out)) * K * M, "selection units")


def cell_list() -> list[dict]:
    import csv
    with open(TABLES / "resel_cells.csv", newline="") as fh:            # standard library only: the workers' environment has no pandas
        return [dict(r, n=int(r["n"]), speculative=int(r["speculative"] or 0)) for r in csv.DictReader(fh)]


def cell_root(c: dict) -> Path:
    return ROOT / c["task"] / f"n{int(c['n'])}" / c["arm"]


def sel_units(c: dict) -> list[dict]:
    out = []
    for cand in candidates(c["task"], int(c["n"]), c["arm"]):
        for r in range(M):
            out.append(dict(id=f"resel/{c['task']}/n{int(c['n'])}/{c['arm']}/c{cand['trial']:02d}-r{r:02d}", task=c["task"], n=int(c["n"]), arm=c["arm"],
                            kind="custom", index=r, stream=BANK, hp=cand["hp"], test=False, trace=False, flush=cand["flush"], cand=cand["trial"]))
    return out


def final_units(c: dict, trial: int) -> list[dict]:
    cand = next(x for x in candidates(c["task"], int(c["n"]), c["arm"]) if x["trial"] == trial)
    return [dict(id=f"resel/{c['task']}/n{int(c['n'])}/{c['arm']}/final-c{trial:02d}-s{s:02d}", task=c["task"], n=int(c["n"]), arm=c["arm"],
                 kind="custom", index=s, stream="final", hp=cand["hp"], test=True, trace=False, flush=cand["flush"], cand=trial) for s in range(SEEDS)]


def _load(u: dict) -> dict | None:
    p = swmfix.OUT / f"{u['id']}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _log(v) -> float:
    return math.log10(min(max(v if v is not None and np.isfinite(v) else CAP, 1e-12), CAP))


def select(c: dict) -> dict | None:
    """The selection of a cell from its K x M validation records; None while a record is missing."""
    cands = candidates(c["task"], int(c["n"]), c["arm"])
    recs = {u["id"]: _load(u) for u in sel_units(c)}
    if any(r is None for r in recs.values()):
        return None
    rows = []
    for cand in cands:
        rs = [recs[f"resel/{c['task']}/n{int(c['n'])}/{c['arm']}/c{cand['trial']:02d}-r{r:02d}"] for r in range(M)]
        score = [_log(CAP if (r.get("diverged") or r.get("error")) else r.get("val_rollout_score")) for r in rs]
        one = [_log(CAP if (r.get("diverged") or r.get("error")) else r.get("best_val_nmse")) for r in rs]
        rows.append(dict(cand, mean_log10_val_score=float(np.mean(score)), sd_log10_val_score=float(np.std(score, ddof=1)),
                         max_val_score=float(10 ** max(score)), mean_log10_val_one_step=float(np.mean(one)), scores=[round(10 ** s, 6) for s in score]))
    best = min(rows, key=lambda r: (r["mean_log10_val_score"], r["mean_log10_val_one_step"], r["rank"]))
    return dict(task=c["task"], n=int(c["n"]), arm=c["arm"], selected_trial=best["trial"], selected_rank=best["rank"], changed=bool(best["rank"] != 0), candidates=rows)


def pending(cells: list[dict]) -> list[dict]:
    units = []
    for c in cells:
        have = set(os.listdir(cell_root(c))) if cell_root(c).exists() else set()
        sel = [u for u in sel_units(c) if Path(u["id"]).name + ".json" not in have]
        spec = int(c.get("speculative") or 0)
        if sel:
            units += sel
            if spec:                                                  # the finals of every non-record candidate start at once (headline cell)
                for cand in candidates(c["task"], int(c["n"]), c["arm"])[1:]:
                    units += [u for u in final_units(c, cand["trial"]) if Path(u["id"]).name + ".json" not in have]
            continue
        s = select(c)
        if s and s["changed"]:
            units += [u for u in final_units(c, s["selected_trial"]) if Path(u["id"]).name + ".json" not in have]
    return units


def pool(deadline: float, worker: int) -> None:
    lewibench, _, _ = swm16.harness()
    cells = cell_list()
    last, done = None, 0
    while True:
        units = pending(cells)
        if not units:
            print("nothing pending", flush=True)
            return
        blocks: dict = {}
        for u in units:                                               # run order = cell order; inside a cell every worker starts elsewhere
            blocks.setdefault((u["task"], u["n"], u["arm"]), []).append(u)
        rest = []
        for b in blocks.values():
            off = worker % len(b)
            rest += b[off:] + b[:off]
        same = [u for u in rest if last == (u["task"], u["n"])]
        order = same + [u for u in rest if last != (u["task"], u["n"])]
        fits = [u for u in order if time.time() <= deadline - swmfix.RESERVE_S.get(u["arm"], swmfix.RESERVE_DEFAULT_S)]
        if not fits:
            print(f"deadline: {done} units done", flush=True)
            return
        started = False
        for u in fits:
            path = swmfix.OUT / f"{u['id']}.json"
            if path.exists() or not swmfix.claim(path):
                continue
            rec, _, _ = swmfix.run_one(u)
            rec["candidate_trial"] = u["cand"]
            rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            swmfix.write(path, rec)
            done += 1
            last, started = (u["task"], u["n"]), True
            print(f"{time.strftime('%H:%M:%S')} {u['id']} val_score={rec.get('val_rollout_score')} {rec['wall_s']:.0f}s {rec.get('error') or ''}", flush=True)
            break
        if not started:
            time.sleep(60)


# The feeder of the re-selection runs (queue accounting and job scripts) is not part of this archive; the selection,
# the pool of pending units and the readers run on one machine.


def read() -> None:
    """The tables of the rule (run with the repository's .venv: pandas). Test values are opened for the selected
    candidate's finals only."""
    import pandas as pd
    FIGDATA = swmfix.REPO / "studies/communication/paper/figures/data-final-clean/world_model_transfer_all"
    inst = pd.read_csv(swm16.ANALYSIS / "instances.csv")
    runs, sel_rows, rep = [], [], []
    for c in cell_list():
        task, n, arm = c["task"], int(c["n"]), c["arm"]
        for u in sel_units(c):
            r = _load(u)
            if r is not None:
                runs.append(dict(task=task, n=n, arm=arm, role=c["role"], candidate_trial=u["cand"], resel_seed=u["index"], lr=u["hp"].get("lr"),
                                 hp=json.dumps(u["hp"], sort_keys=True), val_rollout_score=r.get("val_rollout_score"), val_one_step=r.get("best_val_nmse"),
                                 best_step=r.get("best_step"), diverged=int(bool(r.get("diverged"))), error=r.get("error"), wall_s=r.get("wall_s")))
        s_ = select(c)
        base = dict(task=task, n=n, arm=arm, role=c["role"], fragility=float(c["fragility"]), exploded_val=int(float(c["exploded_val"])),
                    val_bump=(float(c["val_bump_rule"]) if c["val_bump_rule"] not in ("", None) else np.nan), record_trial=int(c["winner_trial"]))
        if s_ is None:
            done = sum(_load(u) is not None for u in sel_units(c))
            sel_rows.append(dict(base, state=f"selection running ({done}/{K * M})"))
            continue
        row = dict(base, selected_trial=s_["selected_trial"], selected_rank=s_["selected_rank"], changed=int(s_["changed"]))
        for cand in s_["candidates"]:
            k = cand["rank"] + 1
            row.update({f"c{k}_trial": cand["trial"], f"c{k}_lr": cand["hp"].get("lr"), f"c{k}_record_val_score": cand["trial_val_score"],
                        f"c{k}_resel_gmean_val_score": 10 ** cand["mean_log10_val_score"], f"c{k}_resel_sd_log10": cand["sd_log10_val_score"],
                        f"c{k}_resel_max_val_score": cand["max_val_score"]})
        if not s_["changed"]:
            sel_rows.append(dict(row, state="record winner selected again: the cell keeps its record"))
            continue
        finals = [_load(u) for u in final_units(c, s_["selected_trial"])]
        have = [f for f in finals if f is not None]
        if len(have) < SEEDS:
            sel_rows.append(dict(row, state=f"finals running ({len(have)}/{SEEDS})"))
            continue
        old = inst[(inst.task == task) & (inst.n == n) & (inst.arm == arm)].set_index("seed")
        for f in have:
            srow = old.loc[f["index"]]
            rep.append(dict(task=task, n=n, arm=arm, role=c["role"], seed=f["index"], record_trial=int(c["winner_trial"]), selected_trial=s_["selected_trial"],
                            record_log10=float(srow["log10"]), record_nmse=float(srow["nmse"]), record_one_step=float(srow["one_step"]),
                            log10=swm16.log10_nmse(f), nmse=(f.get("test_rollout") or {}).get("16"), one_step=(f.get("test_one_step") or {}).get("nmse"),
                            val_rollout_score=f.get("val_rollout_score"), diverged=int(bool(f.get("diverged"))), hp=json.dumps(f["hp"], sort_keys=True)))
        sel_rows.append(dict(row, state="repaired: 16 finals of the selected candidate"))
    pd.DataFrame(runs).to_csv(TABLES / "resel_runs.csv", index=False)
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(TABLES / "resel_selection.csv", index=False)
    rep = pd.DataFrame(rep)
    rep.to_csv(TABLES / "repaired_instances.csv", index=False)
    print(sel[["task", "n", "arm", "role", "state"]].to_string())
    if rep.empty:
        return
    # before / after per repaired cell (paired over the 16 final seeds) and the figure-data schema
    ba, cells_new = [], []
    for (task, n, arm), g in rep.groupby(["task", "n", "arm"]):
        d = swm16.student_t((g.log10 - g.record_log10).values)
        a, b = swm16.student_t(g.log10.values), swm16.student_t(g.record_log10.values)
        med_a, med_b = np.median(g.log10), np.median(g.record_log10)
        srow_ = sel[(sel.task == task) & (sel.n == n) & (sel.arm == arm)].iloc[0]
        k_sel = int(srow_.selected_rank) + 1
        fresh_rec, fresh_sel = float(srow_["c1_resel_gmean_val_score"]), float(srow_[f"c{k_sel}_resel_gmean_val_score"])
        finals_val_sel = float(10 ** np.log10(g.val_rollout_score.clip(1e-12, 1e6)).mean())
        ba.append(dict(task=task, n=n, arm=arm, role=g.role.iloc[0], record_trial=g.record_trial.iloc[0], selected_trial=g.selected_trial.iloc[0],
                       # validation only: the record winner on the 8 fresh seeds (free of the flagging), the selected candidate on the 8 fresh
                       # seeds (carries the second stage's own winner's curse) and on the 16 final seeds (free of its selection)
                       record_winner_fresh_val=fresh_rec, selected_fresh_val=fresh_sel, selected_finals_val=finals_val_sel,
                       dlog10_val_free=float(np.log10(finals_val_sel) - np.log10(fresh_rec)),
                       record_gmean=10 ** b["mean"], repaired_gmean=10 ** a["mean"], dlog10_mean=d["mean"], dlog10_lo=d["lo"], dlog10_hi=d["hi"],
                       seeds_better=int((g.log10 < g.record_log10).sum()), record_exploded=int((g.record_log10 > med_b + 1).sum()),
                       repaired_exploded=int((g.log10 > med_a + 1).sum()), record_max=float(g.record_nmse.max()), repaired_max=float(g.nmse.max()),
                       record_one_step=float(g.record_one_step.mean()), repaired_one_step=float(g.one_step.mean())))
        cells_new.append(dict(bank=16, task=task, n=n, arm=arm, rollout16_mean=10 ** a["mean"], rollout16_lo=10 ** a["lo"], rollout16_hi=10 ** a["hi"],
                              seeds=len(g), diverged=int(g.diverged.sum()), gmean=10 ** a["mean"], glo=10 ** a["lo"], ghi=10 ** a["hi"], role=g.role.iloc[0]))
    ba = pd.DataFrame(ba)
    ba.to_csv(TABLES / "repaired_before_after.csv", index=False)
    print(ba.round(4).to_string())
    new_all = pd.DataFrame(cells_new)
    new_all.to_csv(TABLES / "repaired_cells.csv", index=False)              # every re-selected cell, figure-data schema + role
    new = new_all[new_all.role == "flagged"].drop(columns=["role"])         # the trial figure data take the FLAGGED cells only:
    rep = rep[rep.role == "flagged"]                                        # controls and the named cell are not the rule's repairs
    cells = pd.read_csv(FIGDATA / "cells.csv")
    key = lambda df: df.task + "|" + df.n.astype(str) + "|" + df.arm  # noqa: E731
    trial = pd.concat([cells[~key(cells).isin(set(key(new)))], new]).sort_values(["task", "n", "arm"])
    trial.to_csv(TABLES / "cells_trial.csv", index=False)                   # the figure's cells.csv with the repaired cells in place (TRIAL)
    # paired.csv (TRIAL): the AULC contrasts of the record's read with the repaired instances in place
    inst2 = inst.set_index(["task", "n", "arm", "seed"]).copy()
    for r in rep.itertuples():
        inst2.loc[(r.task, r.n, r.arm, r.seed), "log10"] = r.log10
    inst2 = inst2.reset_index()
    old_paired = pd.read_csv(FIGDATA / "paired.csv")
    rows = []
    for task in sorted(inst2.task.unique()):
        grid = swm16.grid_of(task)
        au = {}
        for arm, g in inst2[inst2.task == task].groupby("arm"):
            by = g.set_index(["seed", "n"]).log10
            au[arm] = {s_: swm16.aulc_log(grid, [by.loc[(s_, n)] for n in grid]) for s_ in sorted(g.seed.unique()) if all((s_, n) in by.index for n in grid)}
        for ref in swm16.ARMS:
            if ref == "lewi-v1" or ref not in au or "lewi-v1" not in au:
                continue
            idx = sorted(set(au["lewi-v1"]) & set(au[ref]))
            d = np.array([au["lewi-v1"][i] - au[ref][i] for i in idx])
            st = swm16.student_t(d)
            rows.append(dict(bank=16, task=task, n="aulc", reference=ref, dlog10_mean=st["mean"], dlog10_lo=st["lo"], dlog10_hi=st["hi"], better=int((d < 0).sum()),
                             seeds=len(idx), verdict="win" if st["hi"] < 0 else "loss" if st["lo"] > 0 else "tie", complete=bool(len(idx) == SEEDS)))
    paired = pd.DataFrame(rows)
    paired.to_csv(TABLES / "paired_trial.csv", index=False)
    m = paired.merge(old_paired[old_paired.n == "aulc"], on=["task", "reference"], suffixes=("", "_record"))
    chg = m[(m.dlog10_mean - m.dlog10_mean_record).abs() > 1e-9]
    chg[["task", "reference", "dlog10_mean_record", "verdict_record", "dlog10_mean", "dlog10_lo", "dlog10_hi", "verdict"]].to_csv(TABLES / "paired_trial_changes.csv", index=False)
    print(chg[["task", "reference", "dlog10_mean_record", "verdict_record", "dlog10_mean", "verdict"]].round(4).to_string())
    # wins-ties-losses of LeWi per reference, record against trial (all 19 tasks; the 17 counted tasks of the record's gate)
    counted = set(json.loads((FIGDATA / "swm16.json").read_text())["claim_bearing"])
    tally = []
    for ref in [a_ for a_ in swm16.ARMS if a_ != "lewi-v1"]:
        row = {"reference": ref}
        for name, df in (("record", old_paired[old_paired.n == "aulc"]), ("trial", paired)):
            for scope, d_ in (("all19", df[df.reference == ref]), ("counted17", df[(df.reference == ref) & df.task.isin(counted)])):
                row[f"{name}_{scope}"] = "-".join(str(int((d_.verdict == v).sum())) for v in ("win", "tie", "loss"))
        tally.append(row)
    pd.DataFrame(tally).to_csv(TABLES / "tally_trial.csv", index=False)
    print(pd.DataFrame(tally).to_string())


def status() -> None:
    cells = cell_list()
    tot = 0
    for c in cells:
        have = set(os.listdir(cell_root(c))) if cell_root(c).exists() else set()
        ns = sum(1 for u in sel_units(c) if Path(u["id"]).name + ".json" in have)
        nf = sum(1 for h in have if h.startswith("final-") and h.endswith(".json"))
        ncl = sum(1 for h in have if h.endswith(".claim"))
        s = select(c) if ns == K * M else None
        tot += ns + nf
        print(f"{c['task']:12s} n={int(c['n']):7d} {c['arm']:8s} {c['role']:8s} sel {ns:2d}/{K * M} finals {nf:2d} claims {ncl:2d} "
              f"{'-> t%02d (rank %d)%s' % (s['selected_trial'], s['selected_rank'], ' CHANGED' if s['changed'] else '') if s else ''}")
    print("records:", tot, " pending units:", len(pending(cells)))


if __name__ == "__main__":
    verb = sys.argv[1]
    if verb == "cells":
        cells()
    elif verb == "pool":
        pool(float(sys.argv[2]), int(sys.argv[3]))
    elif verb == "feed":
        feed()
    elif verb == "read":
        read()
    elif verb == "status":
        status()
    else:
        raise SystemExit(__doc__)
