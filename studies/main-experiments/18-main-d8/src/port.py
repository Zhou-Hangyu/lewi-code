"""Port the campaign's records (17-main-clean, panel main8n) and the record's depth-5 tree into this study.

    python src/port.py ledger      ledger/winners.csv, ledger/cells.csv: the campaign's winners in the record's schema
                                   + the depth-5 tree re-derived from the raw sources (08-main's ledger, 12-extra's panels)
    python src/port.py contract    analysis/contract/: curves, AULC per instance and per arm, contrasts, GO gap, disclosures
    python src/port.py d8          ledger/d8/, analysis/d8/contract/: the all-active twin (main8b + go8b) in the same schema,
                                   with its depth-5 tree (17-main-clean's main panel, cell tree16 -> tree5d16; final.py)
    python src/port.py d8_posthoc  probes/analysis/d8/, causal/analysis/d8/, embeddings/d8/, analysis/d8/budget/: the twin's
                                   linear probes, Boundless DAS, embedding panels and checkpoint positions
    python src/port.py beside      analysis/contract/vs_record_*.csv, vs_d8_*.csv: the same AULCs and contrasts in the record
                                   (32,000 steps, the same instances) and in the all-active twin (d = 8)
    python src/port.py budget      analysis/budget/: checkpoint positions, and the horizon audit paired with the record
    python src/port.py probes      probes/analysis/: the campaign's linear probes (every arm) in the record's schema
    python src/port.py das         causal/analysis/: the campaign's Boundless DAS (Dense, GO(all), LeWi)
    python src/port.py embeddings  embeddings/: LeWi's embedding panels (instance 7000)
    python src/port.py manifest    manifests/port_manifest.json (every source file: sha256, size)
    python src/port.py all

The record's port functions (16-main-final/src/port.py: sealed_rows, panel_rows, write_ledger, contract_tables)
run here on this study's rows, so every statistic is the record's own computation (csfrbench.metrics).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3] / "src"))
import final as F  # noqa: E402

_spec = importlib.util.spec_from_file_location("port16", F.RECORD / "src" / "port.py")
P16 = importlib.util.module_from_spec(_spec)
sys.modules["port16"] = P16
_spec.loader.exec_module(P16)          # its `import final` resolves to this study's final

use, write = F.use, F.write
LEDGER = F.STUDY / "ledger"
CONTRACT = F.STUDY / "analysis" / "contract"
BUDGET = F.STUDY / "analysis" / "budget"
PROBES = F.STUDY / "probes" / "analysis"
CAUSAL = F.STUDY / "causal" / "analysis"
EMB = F.STUDY / "embeddings"
MAN = F.STUDY / "manifests"
D8_LEDGER = LEDGER / "d8"
D8_CONTRACT = F.STUDY / "analysis" / "d8" / "contract"
D8_BUDGET = F.STUDY / "analysis" / "d8" / "budget"
D8_PROBES = PROBES / "d8"
D8_CAUSAL = CAUSAL / "d8"
D8_EMB = EMB / "d8"
# the suite at d = 8 on the record's instances (decision 2026-09-20; final.PANEL_D8R with pos8r read into it)
D8R_LEDGER = LEDGER / "d8r"
D8R_CONTRACT = F.STUDY / "analysis" / "d8r" / "contract"
D8R_BUDGET = F.STUDY / "analysis" / "d8r" / "budget"
D8R_PAIRS = F.STUDY / "analysis" / "d8r" / "pairs"
D8R_PROBES = PROBES / "d8r"
D8R_CAUSAL = CAUSAL / "d8r"
D8R_EMB = EMB / "d8r"
D8R_CELL_OF = {c: c + "r" for c in F.D8_CELLS}       # the harness's cell name -> this study's
# the nuisance-input setting on the twin's seeds (decision 2026-09-20; final.PANEL_D32S with the depth-5 tree of PANEL_D5S)
D32S_LEDGER = LEDGER / "d32s"
D32S_CONTRACT = F.STUDY / "analysis" / "d32s" / "contract"
D32S_BUDGET = F.STUDY / "analysis" / "d32s" / "budget"
D32S_PAIRS = F.STUDY / "analysis" / "d32s" / "pairs"
D32S_PROBES = PROBES / "d32s"
D32S_CAUSAL = CAUSAL / "d32s"
D32S_EMB = EMB / "d32s"
WINNER_COLS = P16.WINNER_COLS
TREE5_RAW = LEDGER / "tree5_sources.csv"


# ---------------------------------------------------------------- the campaign's winners, in the record's schema

def harness_rows(panel: str, cell_of: dict[str, str]) -> list[dict]:
    """The validation-selected winners of a 17-main-clean panel (ledger/winners_<panel>.csv), with the trial counts
    of every condition read from the panel's unit record and the retained checkpoint located under its run root."""
    src = F.M17 / "ledger" / f"winners_{panel}.csv"
    if not src.exists():
        print(f"  {F.rel(src)} missing: no winners of {panel} yet")
        return []
    w = pd.read_csv(use(src), float_precision="round_trip")      # the harness's values bit for bit (the default parser is one ulp off on some)
    u = pd.read_parquet(use(F.M17 / "results" / f"units_{panel}.parquet"), columns=["cell", "arm", "fs", "n", "val", "diverged"])
    counts = u.groupby(["cell", "arm", "fs", "n"]).agg(trials=("val", "size"), diverged=("diverged", "sum"))
    rows = []
    for r in w.itertuples():
        if r.cell not in cell_of:
            continue
        # a row read into this panel's record from another panel (go8b, pos8b) lives under that panel's run root
        own = r.panel if isinstance(getattr(r, "panel", None), str) and r.panel else panel
        cell = cell_of[r.cell]
        v = F.CELL[cell]
        hp = json.loads(r.hp_json)
        c = counts.loc[(r.cell, r.arm, int(r.fs), int(r.n))]
        d = F.M17_RUNS / own / r.unit
        rows.append(dict(cell=cell, label=v["label"], task=v["task"], preset=v["preset"], sigma_sq=v["sigma"], d=v["d"],
                         arm=r.arm, paper_name=F.LABEL[r.arm], function_seed=int(r.fs), n=int(r.n), trial=int(r.trial),
                         best_val_nmse=float(r.val), test_nmse=float(r.test), test_audit_nmse=np.nan, best_step=int(r.best_step),
                         lr=hp.get("lr"), hp=json.dumps(hp, sort_keys=True), trials=int(c.trials), trials_diverged=int(c.diverged),
                         source="17-main-clean", root=own,
                         unit=f"17-main-clean:{own}/{r.unit}", run_id="",
                         checkpoint=F.rel(d) if F.on_record(d / "best.pt") and F.on_record(d / "init.pt") else ""))     # on disk or in a verified archive
    return rows


def tree5_rows(fresh: bool = False) -> list[dict]:
    """The depth-5 tree from its raw sources (intake/PLAN.md section 4), cached in ledger/tree5_sources.csv."""
    if TREE5_RAW.exists() and not fresh:
        df = pd.read_csv(use(TREE5_RAW))
        return df.to_dict("records")
    rows, _ = P16.sealed_rows()                          # streams 08-main's ledger (~6 min)
    rows = [r for r in rows if r["cell"] == "tree5"]
    rows += P16.panel_rows("w5t5", "tree5", [F.LEWI])
    rows += P16.panel_rows(F.T5B_PANEL, "tree5", F.T5B_ARMS)
    rows += P16.panel_rows("g5", "tree5", ["go1", "go2"])
    df = pd.DataFrame(rows)[WINNER_COLS]
    df["checkpoint"] = df["checkpoint"].fillna("")
    LEDGER.mkdir(parents=True, exist_ok=True)
    df.to_csv(TREE5_RAW, index=False)
    print(f"  {F.rel(TREE5_RAW)}: {len(df)} rows from the raw sources")
    return df.to_dict("records")


def ledger(fresh: bool = False) -> None:
    print("ledger")
    rows = harness_rows(F.PANEL, {c: c for c in F.CAMPAIGN_CELLS})
    rows += tree5_rows(fresh)
    df = pd.DataFrame(rows)[WINNER_COLS]
    df["checkpoint"] = df["checkpoint"].fillna("")
    P16.write_ledger(df, LEDGER)


def d8() -> None:
    print("d8")
    rows = harness_rows(F.PANEL_D8, {c: c for c in F.D8_CELLS})
    rows += harness_rows(F.PANEL_D5, {"tree16": F.D5_CELL})     # the twin's depth-5 tree: 17-main-clean's main panel
    if not rows:
        print("  no all-active winners")
        return
    df = pd.DataFrame(rows)[WINNER_COLS]
    df["checkpoint"] = df["checkpoint"].fillna("")
    df, repairs = adopt_rule(df)
    P16.write_ledger(df, D8_LEDGER)
    write(D8_LEDGER, "repairs.csv", repairs)
    cells = [c for c in F.D8_CELLS_ALL if c in set(df.cell)]
    tabs = P16.contract_tables(df, cells, F.PAIRS)
    for name, t in tabs.items():
        write(D8_CONTRACT, f"{name}.csv", t)
    F.jdump(D8_CONTRACT / "summary.json", summary_of(tabs, cells, F.PANEL_D8))


def adopt_rule(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The input-cutoff rule adopted for the twin's LeWi arm (final.D8_RULE; decision 2026-09-20): in every cell the rule changes, the
    LeWi row becomes the validation-selected winner of the cell's full 32-trial re-run under the rule (final.repair_winners: the
    harness's rule), its learning rate from the re-run's spec, its checkpoint under input-cutoff/runs/repair/; the record's row
    of the cell goes to ledger/d8/repairs.csv beside the adopted values. Every other row is untouched."""
    key = ["cell", "arm", "function_seed", "n"]
    win = F.repair_winners()
    counts = F.repair_units().groupby(["cell", "fs", "n"]).agg(trials=("trial", "size"), diverged=("diverged", "sum"))
    out = df.set_index(key)
    rep = []
    for r in win.itertuples():
        k = (r.cell, F.LEWI, int(r.fs), int(r.n))
        if k not in out.index:
            raise SystemExit(f"adopt_rule: repaired cell {k} has no row in the twin's ledger")
        old = out.loc[k]
        d = F.D8_REPAIR / "runs" / "repair" / r.set / r.unit
        # the unit's arm and learning rate come from the sub-study's unit table (read from the run records; the folders may be archived)
        if r.method != F.D8_RULE_ARM or not (F.on_record(d / "best.pt") and F.on_record(d / "init.pt")):
            raise SystemExit(f"adopt_rule: {d} is not the winner's run under the rule (method {r.method}; checkpoints on record: {F.on_record(d / 'best.pt')})")
        hp = {"lr": float(r.lr)}
        c = counts.loc[(r.cell, int(r.fs), int(r.n))]
        rep.append(dict(cell=r.cell, arm=F.LEWI, function_seed=int(r.fs), n=int(r.n), rule=F.D8_RULE, set=r.set, input_redraws=int(r.input_redraws),
                        record_trial=int(old.trial), record_best_val_nmse=float(old.best_val_nmse), record_test_nmse=float(old.test_nmse),
                        record_best_step=int(old.best_step), record_lr=old.lr, record_unit=old.unit, record_checkpoint=old.checkpoint,
                        trial=int(r.trial), best_val_nmse=float(r.val), test_nmse=float(r.test), best_step=int(r.best_step), lr=hp["lr"],
                        trials=int(c.trials), trials_diverged=int(c.diverged), unit=f"18-main-d8/input-cutoff:repair/{r.set}/{r.unit}", checkpoint=F.rel(d)))
        out.loc[k, ["trial", "best_val_nmse", "test_nmse", "best_step", "lr", "hp", "trials", "trials_diverged", "source", "root", "unit", "run_id", "checkpoint"]] = [
            int(r.trial), float(r.val), float(r.test), int(r.best_step), hp["lr"], json.dumps(hp, sort_keys=True), int(c.trials), int(c.diverged),
            "18-main-d8/input-cutoff", f"repair/{r.set}", f"18-main-d8/input-cutoff:repair/{r.set}/{r.unit}", "", F.rel(d)]
    repairs = pd.DataFrame(rep).sort_values(["cell", "function_seed", "n"]).reset_index(drop=True)
    better = int((repairs.test_nmse < repairs.record_test_nmse).sum())
    print(f"  rule {F.D8_RULE} adopted for LeWi: {len(repairs)} cells re-run under the rule ({repairs.groupby('set').size().to_dict()}); "
          f"{better} lower and {len(repairs) - better} higher in test NMSE than the record's winner; the record's rows in ledger/d8/repairs.csv")
    return out.reset_index()[WINNER_COLS], repairs


def d8r() -> None:
    """The 8-input suite at d = 8 on the record's instances (17-main-clean panel main8r with pos8r read into it): ledger/d8r/
    and analysis/d8r/contract/ in the record's schema; the cells carry the suffix r (final.D8R_CELLS)."""
    print("d8r")
    rows = harness_rows(F.PANEL_D8R, D8R_CELL_OF)
    if not rows:
        print("  no winners of main8r yet")
        return
    df = pd.DataFrame(rows)[WINNER_COLS]
    df["checkpoint"] = df["checkpoint"].fillna("")
    P16.write_ledger(df, D8R_LEDGER)
    cells = [c for c in F.D8R_CELLS if c in set(df.cell)]
    tabs = P16.contract_tables(df, cells, F.PAIRS)
    for name, t in tabs.items():
        write(D8R_CONTRACT, f"{name}.csv", t)
    F.jdump(D8R_CONTRACT / "summary.json", summary_of(tabs, cells, F.PANEL_D8R))


def d8r_posthoc() -> None:
    """The readouts of the suite on the record's instances: 17-main-clean/analysis/main8r/{probes, causal, embeddings}, and the
    checkpoint positions of its winners."""
    print("d8r_posthoc")
    if not (D8R_LEDGER / "winners.csv").exists():
        print("  no d8r ledger yet")
        return
    w = pd.read_csv(D8R_LEDGER / "winners.csv")
    src = F.M17 / "analysis" / F.PANEL_D8R
    _port_probes(src / "probes", D8R_PROBES, w, D8R_CELL_OF)
    _port_das(src / "causal", D8R_CAUSAL, D8R_CELL_OF)
    _port_embeddings(src / "embeddings", D8R_EMB)
    _positions(w, F.D8R_STEPS, D8R_BUDGET)


def d8r_pairs() -> None:
    """analysis/d8r/pairs/: the suite at d = 8 against the campaign at d = 32 paired per instance - the same target functions and
    split seeds, the 24 inert coordinates removed - AULC per (workload, arm) and test NMSE per (workload, arm, n); d = 8 minus
    d = 32, so a negative value means the inert coordinates cost the arm."""
    print("d8r_pairs")
    if not (D8R_CONTRACT / "aulc_instances.csv").exists() or not (CONTRACT / "aulc_instances.csv").exists():
        print("  no instance tables yet")
        return
    a8 = pd.read_csv(D8R_CONTRACT / "aulc_instances.csv")
    a32 = pd.read_csv(CONTRACT / "aulc_instances.csv")
    w8 = pd.read_csv(D8R_LEDGER / "winners.csv")
    w32 = pd.read_csv(LEDGER / "winners.csv")
    rows, by_n = [], []
    for cell in F.D8R_CELLS:
        other = F.D8R_OF_RECORD[cell]
        a = a8[a8.cell == cell].set_index(["arm", "function_seed"]).aulc
        b = a32[a32.cell == other].set_index(["arm", "function_seed"]).aulc
        for arm in [x for x in F.ARMS if x in a.index.get_level_values(0) and x in b.index.get_level_values(0)]:
            idx = a.loc[arm].index.intersection(b.loc[arm].index)
            if len(idx) == 0:
                continue
            d = (a.loc[arm].loc[idx] - b.loc[arm].loc[idx]).values
            rows.append(dict(cell=cell, cell_d32=other, arm=arm, paper_name=F.LABEL[arm], aulc_d8_mean=float(a.loc[arm].loc[idx].mean()),
                             aulc_d32_mean=float(b.loc[arm].loc[idx].mean()), **F.tci(d), n_pairs=len(idx), better_at_d8=int((d < 0).sum())))
            m8 = w8[(w8.cell == cell) & (w8.arm == arm)].set_index(["function_seed", "n"]).test_nmse
            m32 = w32[(w32.cell == other) & (w32.arm == arm)].set_index(["function_seed", "n"]).test_nmse
            for n in F.CELL[cell]["window"]:
                i2 = m8[m8.index.get_level_values("n") == n].index.intersection(m32[m32.index.get_level_values("n") == n].index)
                if len(i2) == 0:
                    continue
                dn = (m8.loc[i2] - m32.loc[i2]).values
                by_n.append(dict(cell=cell, cell_d32=other, arm=arm, paper_name=F.LABEL[arm], n=int(n), nmse_d8_mean=float(m8.loc[i2].mean()),
                                 nmse_d32_mean=float(m32.loc[i2].mean()), **F.tci(dn), n_pairs=len(i2), better_at_d8=int((dn < 0).sum())))
    write(D8R_PAIRS, "aulc_pairs.csv", pd.DataFrame(rows))
    write(D8R_PAIRS, "by_n.csv", pd.DataFrame(by_n))
    F.jdump(D8R_PAIRS / "summary.json", {"generated_local": F.now_local(), "pairs": len(rows),
                                          "note": "d = 8 minus d = 32 paired per instance (the same target function and split seeds; the d = 32 instance carries "
                                                  "24 inert coordinates); mean, 95% Student-t interval and median over the instances both settings have; "
                                                  "better_at_d8 = instances where the d = 8 value is lower. The spaces differ (the twin's refreeze p8b at d = 8; "
                                                  "the campaign's p8n at d = 32), so the difference is the settings', not the dimension's alone."})


def d32s() -> None:
    """The nuisance-input setting on the twin's seeds (17-main-clean panels main32s and main32t): ledger/d32s/ and
    analysis/d32s/contract/ in the record's schema; the cells carry the suffix 32s and the depth-5 tree is tree5d32 (final.D32S_*)."""
    print("d32s")
    rows = harness_rows(F.PANEL_D32S, F.D32S_HARNESS)
    rows += harness_rows(F.PANEL_D5S, F.D5S_HARNESS)
    if not rows:
        print("  no winners of main32s / main32t yet")
        return
    df = pd.DataFrame(rows)[WINNER_COLS]
    df["checkpoint"] = df["checkpoint"].fillna("")
    P16.write_ledger(df, D32S_LEDGER)
    cells = [c for c in F.D32S_CELLS_ALL if c in set(df.cell)]
    tabs = P16.contract_tables(df, cells, F.PAIRS)
    for name, t in tabs.items():
        write(D32S_CONTRACT, f"{name}.csv", t)
    F.jdump(D32S_CONTRACT / "summary.json", summary_of(tabs, cells, F.PANEL_D32S))


def d32s_posthoc() -> None:
    """The readouts of the nuisance-input setting on the twin's seeds: 17-main-clean/analysis/main32s/{probes, causal, embeddings}
    (the seven workloads; the depth-5 tree has no readouts, as the twin's), and the checkpoint positions of every winner."""
    print("d32s_posthoc")
    if not (D32S_LEDGER / "winners.csv").exists():
        print("  no d32s ledger yet")
        return
    w = pd.read_csv(D32S_LEDGER / "winners.csv")
    src = F.M17 / "analysis" / F.PANEL_D32S
    _port_probes(src / "probes", D32S_PROBES, w, F.D32S_HARNESS)
    _port_das(src / "causal", D32S_CAUSAL, F.D32S_HARNESS)
    _port_embeddings(src / "embeddings", D32S_EMB)
    _positions(w, F.D32S_STEPS, D32S_BUDGET)


def d32s_pairs() -> None:
    """analysis/d32s/pairs/: the nuisance-input setting against the twin paired per instance - the same target functions and split
    seeds, 24 inert coordinates added, the twin's adopted LeWi on both sides - AULC per (workload, arm) and test NMSE per (workload,
    arm, n); d = 32 minus d = 8, so a positive value means the inert coordinates cost the arm. The depth-5 tree pairs tree5d32 with
    the twin's tree5d16."""
    print("d32s_pairs")
    if not (D32S_CONTRACT / "aulc_instances.csv").exists() or not (D8_CONTRACT / "aulc_instances.csv").exists():
        print("  no instance tables yet")
        return
    a32 = pd.read_csv(D32S_CONTRACT / "aulc_instances.csv")
    a8 = pd.read_csv(D8_CONTRACT / "aulc_instances.csv")
    w32 = pd.read_csv(D32S_LEDGER / "winners.csv")
    w8 = pd.read_csv(D8_LEDGER / "winners.csv")
    rows, by_n = [], []
    for cell in F.D32S_CELLS_ALL:
        other = F.D32S_OF_TWIN[cell]
        a = a32[a32.cell == cell].set_index(["arm", "function_seed"]).aulc
        b = a8[a8.cell == other].set_index(["arm", "function_seed"]).aulc
        for arm in [x for x in F.ARMS if x in a.index.get_level_values(0) and x in b.index.get_level_values(0)]:
            idx = a.loc[arm].index.intersection(b.loc[arm].index)
            if len(idx) == 0:
                continue
            d = (a.loc[arm].loc[idx] - b.loc[arm].loc[idx]).values
            rows.append(dict(cell=cell, cell_d8=other, arm=arm, paper_name=F.LABEL[arm], aulc_d32_mean=float(a.loc[arm].loc[idx].mean()),
                             aulc_d8_mean=float(b.loc[arm].loc[idx].mean()), **F.tci(d), n_pairs=len(idx), better_at_d32=int((d < 0).sum())))
            m32 = w32[(w32.cell == cell) & (w32.arm == arm)].set_index(["function_seed", "n"]).test_nmse
            m8 = w8[(w8.cell == other) & (w8.arm == arm)].set_index(["function_seed", "n"]).test_nmse
            for n in F.CELL[cell]["window"]:
                i2 = m32[m32.index.get_level_values("n") == n].index.intersection(m8[m8.index.get_level_values("n") == n].index)
                if len(i2) == 0:
                    continue
                dn = (m32.loc[i2] - m8.loc[i2]).values
                by_n.append(dict(cell=cell, cell_d8=other, arm=arm, paper_name=F.LABEL[arm], n=int(n), nmse_d32_mean=float(m32.loc[i2].mean()),
                                 nmse_d8_mean=float(m8.loc[i2].mean()), **F.tci(dn), n_pairs=len(i2), better_at_d32=int((dn < 0).sum())))
    write(D32S_PAIRS, "aulc_pairs.csv", pd.DataFrame(rows))
    write(D32S_PAIRS, "by_n.csv", pd.DataFrame(by_n))
    F.jdump(D32S_PAIRS / "summary.json", {"generated_local": F.now_local(), "pairs": len(rows),
                                           "note": "d = 32 minus d = 8 paired per instance (the same target function and split seeds; the d = 32 instance carries "
                                                   "24 inert coordinates; LeWi is the twin's adopted arm on both sides); mean, 95% Student-t interval and median over "
                                                   "the instances both settings have; better_at_d32 = instances where the d = 32 value is lower. The spaces differ "
                                                   "(each setting's own refreeze on 6016-6019 at its dimension), so the difference is the settings', not the "
                                                   "dimension's alone."})


def d8_posthoc() -> None:
    """The twin's readouts in this record's schema: 17-main-clean/analysis/main8b-in5/{probes, causal, embeddings} - the twin's
    readouts (main8b) with the repaired winners' in the cells the adopted rule changes (m17_posthoc probes-overlay, das-overlay;
    the embeddings run on that panel) - the joint DAS list (the init and indep lists live in the harness's tidy record, read by
    the figure harvester), and the checkpoint positions of the winners."""
    print("d8_posthoc")
    if not (D8_LEDGER / "winners.csv").exists():
        print("  no twin ledger yet")
        return
    w = pd.read_csv(D8_LEDGER / "winners.csv")
    src = F.M17 / "analysis" / F.PANEL_D8_READOUTS
    _port_probes(src / "probes", D8_PROBES, w)
    _port_das(src / "causal", D8_CAUSAL)
    _port_embeddings(src / "embeddings", D8_EMB)
    _positions(w, F.D8_STEPS, D8_BUDGET)


# ---------------------------------------------------------------- contract

def winners() -> pd.DataFrame:
    return pd.read_csv(LEDGER / "winners.csv")


def summary_of(tabs: dict, cells: list[str], panel: str) -> dict:
    summary = {"generated_utc": F.now_utc(), "generated_local": F.now_local(), "panel": panel, "lewi": F.LEWI, "arms": F.ARMS,
               "pairs": F.PAIRS, "windows": {c: F.CELL[c]["window"] for c in cells},
               "cohorts": {c: [F.CELL[c]["cohort"][0], F.CELL[c]["cohort"][-1]] for c in cells},
               "profiles": {c: F.CELL[c]["profile"] for c in cells},
               "off_scale": F.OFF_SCALE, "dead_val": F.DEAD_VAL, "cells": {}}
    for cell in cells:
        a = tabs["aulc"][tabs["aulc"].cell == cell]
        c = tabs["contrasts"][tabs["contrasts"].cell == cell]
        g = tabs["go_gap"][tabs["go_gap"].cell == cell] if len(tabs["go_gap"]) else pd.DataFrame()
        d = tabs["disclosures"][tabs["disclosures"].cell == cell] if len(tabs["disclosures"]) else pd.DataFrame()
        summary["cells"][cell] = {
            "label": F.CELL[cell]["label"], "arms": list(a.arm),
            "aulc": {r.arm: {"mean": r.mean, "lo": r.ci95_lo, "hi": r.ci95_hi, "median": r.median, "n": r.n_instances} for r in a.itertuples()},
            "contrasts": {f"{r.arm} - {r.ref}": {"mean": r.mean, "lo": r.ci95_lo, "hi": r.ci95_hi, "median": r.median,
                                                 "wins": r.wins, "n_pairs": r.n_pairs} for r in c.itertuples()},
            "go_gap": g.iloc[0].to_dict() if len(g) else None,
            "disclosures": {"dead": int(d.dead.sum()) if len(d) else 0, "off_scale": int(d.off_scale.sum()) if len(d) else 0,
                            "cells": d[["arm", "function_seed", "n", "best_val_nmse", "test_nmse"]].to_dict("records") if len(d) else []},
        }
    return summary


def contract() -> None:
    print("contract")
    df = winners()
    cells = [c for c in F.CELLS if c in set(df.cell)]
    tabs = P16.contract_tables(df, cells, F.PAIRS)
    for name, t in tabs.items():
        write(CONTRACT, f"{name}.csv", t)
    F.jdump(CONTRACT / "summary.json", summary_of(tabs, cells, F.PANEL))


# ---------------------------------------------------------------- the record and the twin beside the campaign

def _beside(con: pd.DataFrame, au: pd.DataFrame, other_dir: Path, cell_of: dict[str, str], tag: str) -> None:
    """vs_<tag>_contrasts.csv / vs_<tag>_aulc.csv: every contrast and AULC of the campaign beside the same quantity
    of another setting (the record at 32,000 steps; the all-active twin at d = 8), cell by cell."""
    oc = pd.read_csv(use(other_dir / "contrasts.csv"))
    oa = pd.read_csv(use(other_dir / "aulc.csv"))
    vs = []
    for r in con.itertuples():
        if r.cell not in cell_of:
            continue
        m = oc[(oc.cell == cell_of[r.cell]) & (oc.arm == r.arm) & (oc.ref == r.ref)]
        if len(m) != 1:
            continue
        m = m.iloc[0]
        off = any(abs(float(x)) > F.OFF_SCALE for x in (r.mean, r.ci95_lo, r.ci95_hi, m["mean"], m.ci95_lo, m.ci95_hi))
        vs.append(dict(cell=r.cell, cell_other=cell_of[r.cell], arm=r.arm, ref=r.ref, contrast=r.contrast,
                       mean=r.mean, lo=r.ci95_lo, hi=r.ci95_hi, median=r.median, wins=int(r.wins), n_pairs=int(r.n_pairs),
                       mean_other=m["mean"], lo_other=m.ci95_lo, hi_other=m.ci95_hi, median_other=m["median"], wins_other=int(m.wins), n_pairs_other=int(m.n_pairs),
                       off_scale=int(off), change=(r.median - m["median"]) if off else (r.mean - m["mean"]),
                       share_kept=np.nan if off else ((r.mean / m["mean"]) if m["mean"] else np.nan)))
    write(CONTRACT, f"vs_{tag}_contrasts.csv", pd.DataFrame(vs))
    rows = []
    for r in au.itertuples():
        if r.cell not in cell_of:
            continue
        m = oa[(oa.cell == cell_of[r.cell]) & (oa.arm == r.arm)]
        if len(m) != 1:
            continue
        m = m.iloc[0]
        rows.append(dict(cell=r.cell, cell_other=cell_of[r.cell], arm=r.arm, paper_name=r.paper_name,
                         mean=r.mean, lo=r.ci95_lo, hi=r.ci95_hi, median=r.median, mean_other=m["mean"], lo_other=m.ci95_lo, hi_other=m.ci95_hi,
                         median_other=m["median"], change=r.mean - m["mean"], change_median=r.median - m["median"]))
    write(CONTRACT, f"vs_{tag}_aulc.csv", pd.DataFrame(rows))


def beside() -> None:
    print("beside")
    con = pd.read_csv(CONTRACT / "contrasts.csv")
    au = pd.read_csv(CONTRACT / "aulc.csv")
    _beside(con, au, F.REC_CONTRACT, {c: c for c in F.CAMPAIGN_CELLS}, "record")
    if (D8_CONTRACT / "contrasts.csv").exists():
        _beside(con, au, D8_CONTRACT, {v: k for k, v in F.D8_OF.items()}, "d8")     # incl. tree5 -> tree5d16
    if (D8R_CONTRACT / "contrasts.csv").exists():
        _beside(con, au, D8R_CONTRACT, {v: k for k, v in F.D8R_OF_RECORD.items()}, "d8r")   # the same instances at d = 8
    if (D32S_CONTRACT / "aulc.csv").exists():
        _beside(con, au, D32S_CONTRACT, {v: k for k, v in F.D32S_OF_RECORD.items()}, "d32s")   # the same setting on the twin's instances (tree5 -> tree5d32)


# ---------------------------------------------------------------- training budget

def budget() -> None:
    """Checkpoint positions of the winners, and the horizon audit: the record's AULC at 32,000 steps and this
    campaign's at 64,000 on the same instances, paired per instance (the two settings share their cohorts)."""
    print("budget")
    df = winners()
    steps = {c: (F.STEPS if c != "tree5" else 128000) for c in F.CELLS}
    df = df[df.cell.isin(F.CAMPAIGN_CELLS)].copy()
    if df.empty or not (CONTRACT / "aulc_instances.csv").exists():
        print("  no campaign winners yet")
        return
    _positions(df, steps, BUDGET)
    mine = pd.read_csv(CONTRACT / "aulc_instances.csv")
    rec = pd.read_csv(use(F.REC_CONTRACT / "aulc_instances.csv"))
    hz, hz_n = [], []
    curves_m = pd.read_csv(LEDGER / "winners.csv")
    curves_r = pd.read_csv(use(F.RECORD / "ledger" / "winners.csv"))
    for cell in F.CAMPAIGN_CELLS:
        a = mine[mine.cell == cell].set_index(["arm", "function_seed"]).aulc
        b = rec[rec.cell == cell].set_index(["arm", "function_seed"]).aulc
        for arm in [x for x in F.ARMS if (x in a.index.get_level_values(0)) and (x in b.index.get_level_values(0))]:
            idx = a.loc[arm].index.intersection(b.loc[arm].index)
            if len(idx) == 0:
                continue
            d = (a.loc[arm].loc[idx] - b.loc[arm].loc[idx]).values
            hz.append(dict(cell=cell, arm=arm, paper_name=F.LABEL[arm], horizon_record=F.STEPS_RECORD, horizon=F.STEPS,
                           aulc_record_mean=float(b.loc[arm].loc[idx].mean()), aulc_mean=float(a.loc[arm].loc[idx].mean()),
                           **F.tci(d), n_pairs=len(idx), improved=int((d < 0).sum())))
            wm = curves_m[(curves_m.cell == cell) & (curves_m.arm == arm)].set_index(["function_seed", "n"]).test_nmse
            wr = curves_r[(curves_r.cell == cell) & (curves_r.arm == arm)].set_index(["function_seed", "n"]).test_nmse
            for n in F.CELL[cell]["window"]:
                i2 = wm[wm.index.get_level_values("n") == n].index.intersection(wr[wr.index.get_level_values("n") == n].index)
                if len(i2) == 0:
                    continue
                dn = (wm.loc[i2] - wr.loc[i2]).values
                hz_n.append(dict(cell=cell, arm=arm, paper_name=F.LABEL[arm], n=int(n), **F.tci(dn), n_pairs=len(i2), improved=int((dn < 0).sum())))
    write(BUDGET, "horizon.csv", pd.DataFrame(hz))
    write(BUDGET, "horizon_by_n.csv", pd.DataFrame(hz_n))
    F.jdump(BUDGET / "summary.json", {"generated_local": F.now_local(), "steps": F.STEPS, "steps_record": F.STEPS_RECORD,
                                      "note": "position = best validation checkpoint / budget; horizon = AULC at 64,000 steps minus AULC at 32,000 "
                                              "(the record) paired per instance (the same instances, windows and test pools; the spaces were refrozen "
                                              "at each budget, so the difference is the campaign's, not the budget's alone); negative = the longer budget is better."})


def _positions(df: pd.DataFrame, steps: dict[str, int], folder: Path) -> None:
    """position.csv / position_pooled.csv: the best validation checkpoint as a share of the budget, per cell and arm."""
    df = df[df.cell.isin(steps)].copy()
    df["position"] = df.best_step / df.cell.map(steps)
    pos = []
    for (cell, arm), g in df.groupby(["cell", "arm"]):
        st = F.tci(g.position)
        pos.append(dict(cell=cell, arm=arm, paper_name=F.LABEL[arm], winners=len(g), position_mean=st["mean"], position_lo=st["ci95_lo"],
                        position_hi=st["ci95_hi"], position_median=st["median"], share_in_last_tenth=float((g.position >= 0.9).mean()),
                        share_in_first_half=float((g.position < 0.5).mean())))
    write(folder, "position.csv", pd.DataFrame(pos))
    pooled = []
    for arm, g in df.groupby("arm"):
        st = F.tci(g.position)
        pooled.append(dict(arm=arm, paper_name=F.LABEL[arm], winners=len(g), position_mean=st["mean"], position_lo=st["ci95_lo"],
                           position_hi=st["ci95_hi"], share_in_last_tenth=float((g.position >= 0.9).mean())))
    write(folder, "position_pooled.csv", pd.DataFrame(pooled))


# ---------------------------------------------------------------- post-hoc readouts of the campaign

def _with_record_ids(df: pd.DataFrame, cell_col: str = "cell", arm_col: str = "arm", fs_col: str = "fs") -> pd.DataFrame:
    """The record's ids (task, method, function_seed, paper_name) beside the harness's (cell, arm, fs), where absent."""
    df = df.copy()
    if "task" not in df.columns:
        df.insert(0, "task", df[cell_col].map(lambda c: F.CELL[c]["task"]))
    if "method" not in df.columns:
        df.insert(1, "method", df[arm_col])
    if "function_seed" not in df.columns and fs_col in df.columns:
        df.insert(2, "function_seed", df[fs_col].astype(int))
    df["paper_name"] = df[arm_col].map(F.LABEL)
    return df


def _renamed(df: pd.DataFrame, cell_map: dict[str, str] | None) -> pd.DataFrame:
    """The harness's cell names mapped to this study's where a setting carries its own (the d8r cells)."""
    if cell_map and "cell" in df.columns:
        df = df.copy()
        df["cell"] = df.cell.map(lambda c: cell_map.get(c, c))
    return df


def _port_probes(src: Path, dst: Path, w: pd.DataFrame, cell_map: dict[str, str] | None = None) -> None:
    if not (src / "probe_values.csv").exists():
        print(f"  {F.rel(src)}: no probe tables yet")
        return
    pv = _with_record_ids(_renamed(pd.read_csv(use(src / "probe_values.csv")), cell_map))
    trial = w.set_index(["cell", "arm", "function_seed", "n"]).trial
    pv["hpo_trial"] = [trial.get((c, a, f, n), -1) for c, a, f, n in zip(pv.cell, pv.arm, pv.function_seed, pv.n)]
    write(dst, "probe_values.csv", pv)
    for name in ("probe_summary.csv", "probe_levels.csv"):
        write(dst, name, _with_record_ids(_renamed(pd.read_csv(use(src / name)), cell_map)))
    if (src / "summary.json").exists():
        shutil.copy2(use(src / "summary.json"), dst / "summary_17main.json")


def _port_das(src: Path, dst: Path, cell_map: dict[str, str] | None = None) -> None:
    if not (src / "das_cells.csv").exists():
        print(f"  {F.rel(src)}: no DAS tables yet")
        return
    for name in ("das_cells.csv", "das_star.csv", "das_scores.csv", "das_summary.csv"):
        t = _renamed(pd.read_csv(use(src / name)), cell_map)
        t = _with_record_ids(t, arm_col="method") if "method" in t.columns else _with_record_ids(t)
        t.insert(0, "fit_list", "joint")
        write(dst, name, t)
    con = _renamed(pd.read_csv(use(src / "das_contrasts.csv")), cell_map)
    con.insert(0, "fit_list", "joint")
    con.insert(1, "task", con.cell.map(lambda c: F.CELL[c]["task"]))
    write(dst, "das_contrasts.csv", con)
    if (src / "summary.json").exists():
        shutil.copy2(use(src / "summary.json"), dst / "summary_17main.json")


def _port_embeddings(src: Path, dst: Path) -> None:
    if not (src / "panels.json").exists():
        print(f"  {F.rel(src)}: no embedding panels yet")
        return
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("panels.json", "cells.csv", "positions3d.js", "static.html"):
        if (src / name).exists():
            shutil.copy2(use(src / name), dst / name)
            print(f"  {F.rel(dst / name)} <- {F.rel(src / name)}")


def probes() -> None:
    print("probes")
    _port_probes(F.M17 / "analysis" / F.PANEL / "probes", PROBES, winners())


def das() -> None:
    print("das")
    _port_das(F.M17 / "analysis" / F.PANEL / "causal", CAUSAL)


def embeddings() -> None:
    print("embeddings")
    _port_embeddings(F.M17 / "analysis" / F.PANEL / "embeddings", EMB)


# ---------------------------------------------------------------- manifest

def manifest() -> None:
    print("manifest")
    rows = []
    for relp, p in sorted(F.SOURCES.items()):
        if not p.exists():
            continue
        st = p.stat()
        rows.append({"path": relp, "bytes": st.st_size, "sha256": F.sha256(p) if st.st_size < 2_000_000_000 else None,
                     "mtime_utc": pd.Timestamp(st.st_mtime, unit="s").strftime("%Y-%m-%dT%H:%M:%SZ")})
    prev = MAN / "port_manifest.json"
    old = json.loads(prev.read_text())["sources"] if prev.exists() else []
    seen = {r["path"] for r in rows}
    rows += [r for r in old if r["path"] not in seen]
    F.jdump(prev, {"generated_utc": F.now_utc(), "generated_local": F.now_local(), "sources": sorted(rows, key=lambda r: r["path"])})
    print(f"  {len(rows)} source files")


VERBS = {"ledger": ledger, "contract": contract, "d8": d8, "d8_posthoc": d8_posthoc, "d8r": d8r, "d8r_posthoc": d8r_posthoc, "d8r_pairs": d8r_pairs,
         "d32s": d32s, "d32s_posthoc": d32s_posthoc, "d32s_pairs": d32s_pairs,
         "beside": beside, "budget": budget, "probes": probes, "das": das, "embeddings": embeddings}


def main(argv: list[str]) -> None:
    fresh = "--fresh" in argv
    argv = [a for a in argv if a != "--fresh"]
    verbs = list(VERBS) if argv == ["all"] else [a for a in argv if a != "manifest"]
    for v in verbs:
        VERBS[v](fresh) if v == "ledger" else VERBS[v]()
    manifest()


if __name__ == "__main__":
    main(sys.argv[1:] or ["all"])
