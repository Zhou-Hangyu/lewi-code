"""Constants of 18-main-d8: the record's constants (16-main-final/src/final.py) with this study's paths,
profile, sources and the two settings read beside it.

Every script here imports this module as `final`; the record's port functions (16-main-final/src/port.py)
resolve `final` to it too, so the ledger, the contract tables and the statistics of this study come from
the record's own code applied to this study's rows.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_STUDY = _HERE.parents[1]
_MAIN = _STUDY.parent
_spec = importlib.util.spec_from_file_location("final16", _MAIN / "16-main-final" / "src" / "final.py")
F16 = importlib.util.module_from_spec(_spec)
sys.modules["final16"] = F16
_spec.loader.exec_module(F16)
globals().update({k: v for k, v in vars(F16).items() if not k.startswith("__")})

STUDY = _STUDY
MAIN = _MAIN
RECORD = MAIN / "16-main-final"
M17 = MAIN / "17-main-clean"
M17_RUNS = REPO / "results" / "csfrbench" / "m17" / "runs"
DATA_FINAL = FIGURES / "data-final-clean"
OUT_FINAL = FIGURES / "out-final-clean"
REC_CONTRACT = RECORD / "analysis" / "contract"

PANEL = "main8n"                 # the campaign (17-main-clean DESIGN amendment 8)
PANEL_REFREEZE = "p8n"
PANEL_D8 = "main8b"              # the all-active twin at 64,000 steps (amendment 7, with go8b's rung folded in)
PROFILE = "32x4, 64,000 steps"
STEPS = 64000
STEPS_RECORD = 32000
D8_CELLS = ["tree8", "dag8", "ftree8", "flat8", "dense8", "fdag8", "rand8"]
D8_OF = {"tree8": "tree", "dag8": "dag", "ftree8": "ftree", "flat8": "flat", "dense8": "dense", "fdag8": "fdag", "rand8": "rand"}
D8_COHORT = list(range(6000, 6016))
# The sampled graphs at d = 8 carry 16 more held-out instances since 2026-09-20 (17-main-clean amendment 9: panels rand8b and
# rand8bpos, read into main8b's record), so the cell has 32 graphs, as the record's setting has.
D8_COHORT_RAND = D8_COHORT + list(range(6020, 6036))
D8_EXT_PANELS = ("rand8b", "rand8bpos")
# Neuron positions at d = 8 is panel pos8b (the completeness check of 2026-09-18), read into main8b's record as go8b's rung was.
D8_ARMS = {c: [a for a in F16.ARMS if a != "support_oracle" and not (c in ("flat8", "dense8", "fdag8") and a == "go2")
              and not (c in ("dense8",) and a in ("go1", "graph_oracle")) and not (c in ("flat8",) and a == "go1")
              and not (c in ("tree8", "dag8", "ftree8") and a == "go2")] for c in D8_CELLS}
# GO(1) on the Fourier reuse DAG at d = 8 is panel go8b (amendment 8), read into main8b's record.
D8_ARMS["fdag8"] = [a for a in D8_ARMS["fdag8"] if a != "go2"]

# The campaign's cells carry the record's names, cohorts and windows; only the profile and the source differ.
# The depth-5 tree is the record's own row set, ported.
CELL = {}
for _c in F16.CELLS:
    _v = dict(F16.CELL[_c])
    if _c == "tree5":
        CELL[_c] = dict(_v, source="16-main-final", root="record")
    else:
        CELL[_c] = dict(_v, profile=PROFILE, source="17-main-clean", root=PANEL, panel=PANEL)
for _c in D8_CELLS:
    _v = F16.CELL[D8_OF[_c]]
    CELL[_c] = dict(_v, d=8, label=f"{_v['label']}, d = 8", cohort=(D8_COHORT_RAND if _c == "rand8" else D8_COHORT), profile=PROFILE,
                    source="17-main-clean", root=PANEL_D8, panel=PANEL_D8)
CELLS = list(F16.CELLS)          # tree, dag, ftree, flat, dense, fdag, rand, tree5
CELL_RECORD = list(CELLS)
CAMPAIGN_CELLS = [c for c in CELLS if c != "tree5"]
MECH_CELLS = list(F16.MECH_CELLS)


def cell_of(task: str, preset: str, sigma: float, d: int = 32) -> str | None:
    for c, v in CELL.items():
        if v["task"] == task and v["preset"] == preset and abs(v["sigma"] - float(sigma)) < 1e-9 and v["d"] == d:
            return c
    return None


# The all-active twin's depth-5 tree (decision 2026-09-18): 17-main-clean's main panel (its DESIGN 2-5, the suite at 16 active
# inputs), cell tree16 = the record's depth-5 workload (tree_functional_anova, preset depth4: 16 leaves) with every leaf
# active (d = 16), on the twin's instances 6000-6015, the record's depth-5 profile (64x8, 128,000 steps, n = 2^11 ... 2^17),
# 12 arms (GO(0) is Dense when every coordinate is active), the spaces refrozen in that study (panel p0). Ported here as
# cell tree5d16, the twin's eighth workload, as the record's tree5 is the campaign's.
PANEL_D5 = "main"
D5_CELL = "tree5d16"
CELL[D5_CELL] = dict(F16.CELL["tree5"], d=16, label="tree (depth 5), d = 16 (all 16 leaves active)", cohort=D8_COHORT,
                     source="17-main-clean", root=PANEL_D5, panel=PANEL_D5)
D8_ARMS[D5_CELL] = [a for a in F16.ARMS if a != "support_oracle"]
D8_OF[D5_CELL] = "tree5"
D8_CELLS_ALL = D8_CELLS + [D5_CELL]
D8_STEPS = {c: (STEPS if c != D5_CELL else 128000) for c in D8_CELLS_ALL}
OUT_FINAL_D8 = FIGURES / "out-final-clean" / "d8-64k"
DATA_FINAL_D8 = FIGURES / "data-final-clean-d8"

# The input-cutoff rule adopted for the twin's LeWi arm (decision 2026-09-20; 17-main-clean DESIGN amendment 10): an input with
# fewer than five open first-transition connections at initialization has its position drawn again (rule in5, input-cutoff/). In every
# cell the rule changes (144 of the seven workloads, 18 of the depth-5 tree) the LeWi row is the validation-selected winner of the cell's
# full 32-trial re-run under the rule; every other LeWi cell is the record's bit for bit by the rule's construction. The record's rows of
# the repaired cells: ledger/d8/repairs.csv. The d = 32 setting and the suite on the record's instances (d8r) are not under the rule.
D8_RULE = "in5"
D8_RULE_ARM = f"{LEWI}-{D8_RULE}"                  # the study-local arm the re-runs ran as (input-cutoff/src/cutoff_methods.py): LeWi under the rule
D8_REPAIR = STUDY / "input-cutoff"
D8_REPAIR_SETS = {"in5": {c: c for c in D8_CELLS}, "in5_main": {"tree16": D5_CELL},   # set under runs/repair/, tables/repair_<set>_*.csv -> {harness cell: cell here}
                  "in5_rand8b": {"rand8": "rand8"}}                                       # the sampled graphs on the extension instances 6020-6035 (decision 2026-09-22)
PANEL_D8_READOUTS = "main8b-in5"                   # the harness's winners panel of the adopted readouts (probes, DAS, embeddings): the repaired winners over main8b's


# The archives of the d = 8 campaign's run folders (decision 2026-09-20: archive the campaign's results and free their inodes): every
# tar sits in an archive/ folder beside its tree, and a <name>.list beside the tar holds the tar's entries as repo-relative paths,
# written when the tar was checked against the live root before the root's removal. A checkpoint is on record when it is on disk
# or in such a list; the readouts that need the files (a rebuild of the embeddings, the mechanism figure, the structure records)
# restore the tar first (README, Rebuild).
ARCHIVE_LISTS = (M17_RUNS.parent / "archive", D8_REPAIR / "archive", M17 / "archive")
_ARCHIVED: set | None = None


def archived_paths() -> set:
    global _ARCHIVED
    if _ARCHIVED is None:
        _ARCHIVED = set()
        for folder in ARCHIVE_LISTS:
            for f in sorted(folder.glob("*.list")) if folder.exists() else []:
                _ARCHIVED.update(line.strip().rstrip("/") for line in f.read_text().splitlines() if line.strip())
    return _ARCHIVED


def on_record(path) -> bool:
    """True when the file is on disk or listed in a verified archive (final.ARCHIVE_LISTS)."""
    p = Path(path)
    r = rel(p) if p.is_absolute() else str(p)
    return (REPO / r).exists() or r in archived_paths()


def repair_units(harness_names: bool = False):
    """The 32 trials of every cell re-run under the rule (input-cutoff/tables/repair_<set>_units.csv, the controls left out), one row
    per unit: set, unit, cell, arm, fs, n, trial, val, test, best_step, diverged, input_redraws, record_val, record_test."""
    import pandas as pd
    parts = []
    for name, cell_of in D8_REPAIR_SETS.items():
        u = pd.read_csv(D8_REPAIR / "tables" / f"repair_{name}_units.csv", float_precision="round_trip")
        u = u[~u.control].copy()
        u["set"] = name
        if not harness_names:
            u["cell"] = u.cell.map(cell_of)
        parts.append(u)
    df = pd.concat(parts, ignore_index=True)
    df["arm"] = LEWI
    return df[["set", "unit", "cell", "arm", "fs", "n", "trial", "val", "test", "best_step", "diverged", "input_redraws", "record_val", "record_test", "method", "lr"]]


def repair_winners():
    """The validation-selected winner of every repaired cell under the harness's rule (a complete 32-trial condition, not diverged,
    the lowest validation NMSE, the lowest trial on ties), one row per cell."""
    u = repair_units()
    complete = u.groupby(["cell", "fs", "n"]).trial.transform("size") >= 32
    u = u[complete & u.val.notna() & (u.diverged == 0)]
    return u.sort_values(["cell", "fs", "n", "val", "trial"], kind="stable").groupby(["cell", "fs", "n"], sort=False).head(1).reset_index(drop=True)

# The 8-input suite at d = 8 on the record's own instances (decision 2026-09-20; 17-main-clean amendment 9): the seven workloads on
# the instances of the d = 32 setting (7000-7015; 7000-7031 on the random graphs), the twin's profile, grid, T and spaces,
# every arm of the twin's roster - panel main8r with pos8r (neuron positions) read into it. Its cells carry the twin's names
# with the suffix "r" (tree8r ...), so the three settings sit in one CELL table; the harness's cell names have no suffix.
# It pairs with the campaign per instance: the same target functions and split seeds, the 24 inert coordinates removed.
PANEL_D8R = "main8r"
PANEL_D8R_POS = "pos8r"
D8R_CELLS = [c + "r" for c in D8_CELLS]
D8R_OF = {c + "r": c for c in D8_CELLS}              # -> the harness's (the twin's) cell name
D8R_OF_RECORD = {c + "r": D8_OF[c] for c in D8_CELLS}   # -> the campaign's cell name (the same instances)
for _c in D8_CELLS:
    _v = F16.CELL[D8_OF[_c]]
    CELL[_c + "r"] = dict(_v, d=8, label=f"{_v['label']}, d = 8 on the record's instances", cohort=list(_v["cohort"]), profile=PROFILE,
                          source="17-main-clean", root=PANEL_D8R, panel=PANEL_D8R)
D8R_ARMS = {c + "r": list(D8_ARMS[c]) for c in D8_CELLS}
D8R_STEPS = {c: STEPS for c in D8R_CELLS}
OUT_FINAL_7000 = FIGURES / "out-final-clean-7000"
DATA_FINAL_7000 = FIGURES / "data-final-clean-7000"
# The nuisance-input setting on the twin's seeds (decision 2026-09-20; 17-main-clean amendment 11): the d = 8 campaign repeated
# with 24 inert coordinates on the same instances - the seven workloads at d = 32 (the record's cells: the same families and presets
# as the twin's at d = 8) on the twin's cohort 6000-6015 (6000-6015 and 6020-6035 on the random graphs), the twin's profile, grid and
# T, the record's roster at d = 32 (GO(0) = the active-input oracle), LeWi = the twin's adopted arm (lewi-v1-rgm under rule in5) -
# and the twin's depth-5 tree with 16 inert coordinates (d = 32, the same ambient dimension; the record's depth-5 profile and 13 arms;
# decision 2026-09-21, amendment 12 - a first run at d = 40, panels p40t / main40t, stays on record, superseded). Panels main32s
# (refreeze p32s) and main32t (refreeze p32t). Cells carry the record's names with the suffix "32s" (tree32s ...) and tree5d32; the
# harness's cells are tree ... rand and tree5x32. It pairs with the twin per instance: the same target functions and split seeds,
# inert coordinates added (analysis/d32s/pairs: d = 32 minus d = 8; the depth-5 tree d = 32 minus d = 16).
PANEL_D32S = "main32s"
PANEL_D32S_REFREEZE = "p32s"
PANEL_D5S = "main32t"                                                   # the depth-5 tree at d = 32 (amendment 12)
PANEL_D5S_REFREEZE = "p32t"
PANEL_D40T = "main40t"                                                  # the superseded first run at d = 40 (amendment 11): on record, not in the setting
PANEL_D40T_REFREEZE = "p40t"
D32S_CELLS = [c + "32s" for c in CAMPAIGN_CELLS]                       # tree32s, dag32s, ftree32s, flat32s, dense32s, fdag32s, rand32s
D32S_OF_RECORD = {c + "32s": c for c in CAMPAIGN_CELLS}                # -> the campaign's cell (d = 32 on the record's instances)
D32S_OF_TWIN = {c + "32s": c + "8" for c in CAMPAIGN_CELLS}            # -> the twin's cell (d = 8 on the same instances)
D32S_HARNESS = {c: c + "32s" for c in CAMPAIGN_CELLS}                  # the harness's cell name -> this study's
D5S_CELL = "tree5d32"
D5S_HARNESS = {"tree5x32": D5S_CELL}                                    # the harness's depth-5 cell -> this study's
D40T_CELL = "tree5d40"                                                  # the superseded run's cell (its record: 17-main-clean analysis/main40t)
D32S_CELLS_ALL = D32S_CELLS + [D5S_CELL]
D32S_OF_TWIN[D5S_CELL] = D5_CELL
D32S_OF_RECORD[D5S_CELL] = "tree5"
for _c in CAMPAIGN_CELLS:
    _v = F16.CELL[_c]
    CELL[_c + "32s"] = dict(_v, d=32, label=f"{_v['label']}, d = 32 on the twin's instances", cohort=(D8_COHORT_RAND if _c == "rand" else D8_COHORT),
                            profile=PROFILE, source="17-main-clean", root=PANEL_D32S, panel=PANEL_D32S)
CELL[D5S_CELL] = dict(F16.CELL["tree5"], d=32, label="tree (depth 5), d = 32 on the twin's instances (16 leaves and 16 inert inputs)", cohort=D8_COHORT,
                      source="17-main-clean", root=PANEL_D5S, panel=PANEL_D5S)
D32S_ARMS = {c + "32s": [a for a in F16.ARMS if not (c in ("flat", "dense", "fdag") and a == "go2") and not (c == "dense" and a in ("go1", "graph_oracle"))
                         and not (c == "flat" and a == "go1") and not (c in ("tree", "dag", "ftree") and a == "go2")] for c in CAMPAIGN_CELLS}   # the record's roster at d = 32 (ARMS8N_OF)
D32S_ARMS[D5S_CELL] = list(F16.ARMS)                                   # the record's 13 depth-5 arms
D32S_STEPS = {c: (STEPS if c != D5S_CELL else 128000) for c in D32S_CELLS_ALL}
OUT_FINAL_6000 = FIGURES / "out-final-clean-6000"
DATA_FINAL_6000 = FIGURES / "data-final-clean-6000"
M17_REPORT_D8 = M17 / "report" / "d8-64k"
M17_REPORT_MAIN = M17 / "report"
