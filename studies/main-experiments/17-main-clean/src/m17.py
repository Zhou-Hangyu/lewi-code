"""The CSFRBench campaign definitions: the settings (cells), the arms, the backbones and budgets (profiles), the panels
with their cohorts, grids and frozen search spaces, and the composition of one training unit (plan, trials, _raw).
The campaign's scheduler verbs are not part of this archive; scripts/train_unit.py trains any unit on one machine.
"""

from __future__ import annotations

import fcntl
import datetime
import json
import math
import errno
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve()
STUDY = HERE.parents[1]
MAIN = STUDY.parent
REPO = STUDY.parents[2]
X12_SRC = MAIN / "12-extra" / "src"
for _p in (str(REPO / "src"), str(HERE.parent), str(X12_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The benchmark stack pulls in torch and costs about 40 s to import. Only planning
# and building need it; dispatch, status, harvest and the tender do not, and they
# run every cycle - so it is imported lazily, inside the functions that use it.

RUNS = REPO / "results" / "csfrbench" / "m17" / "runs"
DATASETS = REPO / "results" / "csfrbench" / "m17" / "datasets"
MAN = STUDY / "manifests"
SPACES_OUT = MAN / "spaces"
RESULTS = STUDY / "results"
V11_SPACES = REPO / "studies" / "csfrbench" / "final-design" / "versions" / "v1.1" / "configs" / "hpo_spaces" / "tree_depth4"

# ------------------------------------------------------------------ the suite

CELLS = {
    "tree16":  ("tree_functional_anova", "depth4", 16),
    "dag16":   ("dag16_functional_anova", "default", 16),
    "flat16":  ("flat_functional_anova", "w8", 16),
    "ftree16": ("tree_fourier", "binary_depth4", 16),
    "dense16": ("dense_fourier", "a16", 16),
    "rand16":  ("rand16_functional_anova", "default", 16),
    "tree32":  ("tree_functional_anova", "depth5", 32),        # amendment 5: 32 leaves, all active
    # calibration candidates for the arity-16 Fourier node (DESIGN 7); exactly one
    # becomes `dense16` for the campaign and the rest stay as calibration record.
    "dense16_q24s10": ("dense_fourier", "a16_q24s10", 16),
    "dense16_q48s10": ("dense_fourier", "a16_q48s10", 16),
    "dense16_q48s075": ("dense_fourier", "a16_q48s075", 16),
}
CELL_LABEL = {
    "tree16": "tree (depth 5), 16 leaves", "dag16": "reuse DAG (depth 4), 15 nodes",
    "flat16": "flat modules (8 pairs)", "ftree16": "Fourier tree (depth 5), 16 leaves",
    "dense16": "Fourier dense node (arity 16)",
    "rand16": "random graphs (depth 5), 16 leaves",
    "tree32": "tree (depth 6), 32 leaves",
    "dense16_q24s10": "dense node candidate Q=24, sigma_omega=1.0",
    "dense16_q48s10": "dense node candidate Q=48, sigma_omega=1.0",
    "dense16_q48s075": "dense node candidate Q=48, sigma_omega=0.75",
}
CELL_LEVELS = {"tree16": 4, "dag16": 3, "rand16": 4, "tree32": 5, "flat16": 1, "ftree16": 4, "dense16": 1,
               "dense16_q24s10": 1, "dense16_q48s10": 1, "dense16_q48s075": 1}

ARMS_ALL = ["noreg", "l1", "l2", "l0hc", "lewi-v1-rgm", "rigl", "bimt", "fiedler",
            "erb_positions", "go1", "go2", "graph_oracle"]
AUDIT_ARMS = ["noreg", "l1", "l0hc", "lewi-v1-rgm", "graph_oracle"]
LEWI = "lewi-v1-rgm"
ARMS_OF = {
    "tree16": list(ARMS_ALL),
    "dag16": list(ARMS_ALL),
    "ftree16": list(ARMS_ALL),
    "rand16": list(ARMS_ALL),
    "tree32": ["noreg", "go1", "graph_oracle", "lewi-v1-rgm", "l0hc"],
    "flat16": [a for a in ARMS_ALL if a not in ("go1", "go2")],
    "dense16": [a for a in ARMS_ALL if a not in ("go1", "go2", "graph_oracle")],
}
SUITE = ["tree16", "dag16", "flat16", "ftree16", "dense16"]        # the campaign's five workloads
# The plain-readout campaign runs the suite reuse-DAG first (decision 2026-09-16); the roster
# drops neuron positions, which the authors retired after the 16-active-input campaign.
PLAIN_ORDER = ["dag16", "tree16", "ftree16", "flat16", "dense16"]
DENSE_CANDIDATES = ["dense16", "dense16_q24s10", "dense16_q48s10", "dense16_q48s075"]
# The provisional paper-wide setting (decision 2026-09-16; DESIGN amendment 4): the
# audit profile (64x8, 256,000 steps) on every workload, the grid extended to 2^11..2^18,
# every arm but neuron positions, the random graphs added at 16 leaves, the full cohort.
# Dispatch order: the reuse DAG first (the authors' standing preference), then the new
# workload, then the rest.
SUITE2 = ["dag16", "rand16", "tree16", "ftree16", "flat16", "dense16"]
# The deep test object (decision 2026-09-16; amendment 5): the 32-leaf tree at d = 32 on a
# 128x10 backbone, n = 2^15..2^21, five arms, run after everything else in the setting.
DEEP_CELL = "tree32"
DEEP_ARMS = ["noreg", "go1", "graph_oracle", "lewi-v1-rgm", "l0hc"]
ARMS2_OF = {c: [a for a in ARMS_OF[c] if a != "erb_positions"] for c in SUITE2}

# The pre-registration's own mapping, before any calibration is applied. The reader
# must resolve a candidate through this and never through CELLS: applying a choice
# rewrites CELLS["dense16"], so a reader that looked the preset up there would read
# back the previous run's answer and re-confirm it. A first pass taken on incomplete
# data would then be locked in, and the genuine first candidate could never win.
CELLS_STATIC = dict(CELLS)

# The calibration of DESIGN section 7 chooses the arity-16 Fourier node's shape on
# development seeds 6090-6093 and records it; from then on `dense16` is that preset.
_CAL = STUDY / "manifests" / "calibration.json"
if _CAL.exists():
    _preset = json.loads(_CAL.read_text()).get("dense16_preset")
    if _preset:
        CELLS["dense16"] = ("dense_fourier", _preset, 16)
# The plain-readout arm (decision 2026-09-16). CSFRBench v1.1 made the readout structural -
# penalized, gated or masked like any hidden transition - and `lewi-v1-rgm` is LeWi under
# that rule. `lewi-v1` is the package class itself: the same ball, the same seeds, the same
# knobs, with the readout left a plain linear map (the v1.0 rule 13-main ran). Every other
# arm in this campaign already runs plain, because none of them sets hp["readout"] and
# methods.base.readout_structural defaults to "plain" when the key is absent.
PLAIN_LEWI = "lewi-v1"
# The roster the plain-readout read is ordered and filtered by: the campaign roster with
# the structural LeWi replaced by the plain one and neuron positions dropped. The analysis
# and the report both order arms by a roster and skip anything absent from it, so reading
# the merge with ARMS_ALL alone silently produced no contrasts at all.
PLAIN_ARMS_ALL = [PLAIN_LEWI if a == LEWI else a
                  for a in ARMS_ALL if a != "erb_positions"]

PAPER = {"noreg": "Dense", "l1": "L1", "l2": "L2", "l0hc": "L0", LEWI: "LeWi", "rigl": "RigL",
         "bimt": "BIMT", "fiedler": "Fiedler", "erb_positions": "neuron positions",
         "go1": "GO(1)", "go2": "GO(2)", "graph_oracle": "GO(all)",
         PLAIN_LEWI: "LeWi (plain readout)"}
COLOR = {"noreg": "#000000", "l1": "#1f77b4", "l2": "#74add1", "l0hc": "#313695", LEWI: "#d62728",
         "go1": "#98df8a", "go2": "#66c2a5", "graph_oracle": "#2ca02c", "rigl": "#bcbd22",
         "bimt": "#9467bd", "fiedler": "#7f7f7f", "erb_positions": "#8c564b",
         PLAIN_LEWI: "#e377c2"}
GROUPS = [("reference", ["noreg", "go1", "go2", "graph_oracle"]),
          ("competitive", ["l1", "l2", "l0hc", LEWI]),
          ("tier B: wiring and geometry", ["rigl", "bimt", "fiedler", "erb_positions"])]

ARM_METHOD = {"go1": "graph_oracle", "go2": "graph_oracle"}
ARM_HP = {"go1": {"levels": 1}, "go2": {"levels": 2}}
ARM_SPEC: dict = {}          # per-arm top-level spec fields (18-main-d8: the trainer's `seed_replicate`, fresh paired training randomness)
SPACE_FILE = {"noreg": "noreg", "l1": "l1", "l2": "l2", "l0hc": "l0hc", "rigl": "rigl",
              "bimt": "bimt", "fiedler": "fiedler", "erb_positions": "erb_positions",
              "graph_oracle": "graph_oracle", "go1": "go1", "go2": "go2", LEWI: "lewi-v1",
              PLAIN_LEWI: "lewi-v1"}
TRIAL_KEY = {"go1": "go1", "go2": "go2", LEWI: "lewi-v1", PLAIN_LEWI: "lewi-v1"}

# ------------------------------------------------------------------ profiles

PROFILES = {
    "d5":   dict(width=64, hidden=8, max_steps=128000, eval_every=2000, label="64x8, 128,000 steps"),
    "d5x2": dict(width=64, hidden=8, max_steps=256000, eval_every=4000, label="64x8, 256,000 steps"),
    "w4":   dict(width=64, hidden=4, max_steps=128000, eval_every=2000, label="64x4, 128,000 steps"),
    "w4x2": dict(width=64, hidden=4, max_steps=256000, eval_every=4000, label="64x4, 256,000 steps"),
    "d5x3": dict(width=64, hidden=8, max_steps=384000, eval_every=6000, label="64x8, 384,000 steps"),
    # Amendment 5: four times the setting's budget (steps in proportion to the square of the leaf
    # count, as 32,000 -> 128,000 went from 8 to 16 leaves); eight times (in proportion to n_max)
    # would put a LeWi unit past the 8-hour free_qos wall. 64 checkpoints, as everywhere.
    "w10":  dict(width=128, hidden=10, max_steps=1024000, eval_every=16000, label="128x10, 1,024,000 steps"),
}
WINDOW = [2048, 4096, 8192, 16384, 32768, 65536, 131072]          # 2^11 .. 2^17
WINDOW2 = [4096, 8192, 16384, 32768, 65536, 131072, 262144]       # 2^12 .. 2^18
BIG_N = [262144, 524288]                                          # 2^18, 2^19 (amendment 3)
PANEL_N = [2048, 16384, 131072]
PANEL_N2 = [4096, 32768, 262144]
BUD3_N = [65536, 131072, 262144]          # 2^16 .. 2^18, the 384,000-step audit grid
WINDOW3 = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]   # 2^11 .. 2^18 (amendment 4)
PANEL_N3 = [2048, 32768, 262144]          # 2^11, 2^15, 2^18: the endpoints and the middle of WINDOW3
WINDOW32 = [32768, 65536, 131072, 262144, 524288, 1048576, 2097152]   # 2^15 .. 2^21 (amendment 5)
PANEL_N32 = [32768, 262144, 2097152]      # 2^15, 2^18, 2^21
# The 384,000-step audit compares one arm per family: the two references (Dense and the
# full oracle) and the two learned arms the campaign separates on the reuse DAG.
BUD3_ARMS = ["noreg", "l0hc", LEWI, "graph_oracle"]

COHORT = list(range(6000, 6016))          # the campaign cohort (DESIGN section 3)
COHORT_A = list(range(6000, 6008))        # stage A, and the audit cohort
PANEL_COHORT = [6016, 6017, 6018, 6019]   # the refreeze panels (disjoint)
CAL_COHORT = [6090, 6091, 6092, 6093]     # workload calibration (disjoint)
T = 32

# ------------------------------------------------------------------ panels

PANELS = {
    "cal": dict(cells={"tree16": ["graph_oracle"], "dag16": ["graph_oracle"], "flat16": ["graph_oracle"],
                       "ftree16": ["graph_oracle"], "dense16": ["noreg"]},
                cohort=CAL_COHORT, ns=[2048, 131072], profile="d5", spaces="start",
                note="DESIGN 7: realization check of each workload at the campaign profile"),
    "cal2": dict(cells={c: ["noreg"] for c in
                        ("dense16_q24s10", "dense16_q48s10", "dense16_q48s075")},
                 cohort=CAL_COHORT, ns=[2048, 131072], profile="d5", spaces="start",
                 note="DESIGN 7: the remaining arity-16 dense-node shape candidates"),
    "cal3": dict(cells={c: ["noreg"] for c in ("tree16", "dag16", "flat16", "ftree16")},
                 cohort=CAL_COHORT, ns=[2048, 131072], profile="d5", spaces="start",
                 note="DESIGN 7 / amendment 2: Dense at the grid endpoints, so grid coverage is "
                      "judged on the arm that has to learn the structure, not on the oracle"),
    "p0": dict(cells={c: list(ARMS_ALL) for c in ("tree16", "dag16", "ftree16")},
               cohort=PANEL_COHORT, ns=PANEL_N, profile="d5", spaces="start",
               note="DESIGN 6.2: search-space refreeze panel for the campaign profile"),
    "p1": dict(cells={c: list(AUDIT_ARMS) for c in ("tree16", "dag16")},
               cohort=PANEL_COHORT, ns=PANEL_N2, profile="d5x2", spaces="start",
               note="DESIGN 6.2: refreeze panel for the 256,000-step audit profile"),
    "p2a": dict(cells={"tree16": ["noreg", "graph_oracle"]}, cohort=PANEL_COHORT, ns=[131072],
                profile="w4", spaces="start",
                note="DESIGN 6.2: refreeze panel for the 64x4 / 128,000-step capacity audit"),
    "p2b": dict(cells={"tree16": ["noreg", "graph_oracle"]}, cohort=PANEL_COHORT, ns=[131072],
                profile="w4x2", spaces="start",
                note="DESIGN 6.2: refreeze panel for the 64x4 / 256,000-step capacity audit"),
    "p3": dict(cells={"dag16": list(BUD3_ARMS)}, cohort=PANEL_COHORT, ns=BUD3_N,
               profile="d5x3", spaces="start",
               note="DESIGN 6.2: refreeze panel for the 384,000-step audit profile"),
    "bud3": dict(cells={"dag16": list(BUD3_ARMS)}, cohort=COHORT_A, ns=BUD3_N,
                 profile="d5x3", spaces="p3",
                 note="decision 2026-09-16: budget audit on the reuse DAG at 384,000 steps, "
                      "n = 2^16..2^18, LeWi / Dense / L0 / GO(all) under matched search budgets"),
    "q0": dict(cells={c: [PLAIN_LEWI] for c in ("dag16", "tree16", "ftree16")},
               cohort=PANEL_COHORT, ns=PANEL_N, profile="d5", spaces="start",
               note="DESIGN 6.2 for the plain-readout arm: its own refreeze panel, because a "
                    "space is refrozen on the winners of the arm that will run in it"),
    "plain": dict(cells={c: [PLAIN_LEWI] for c in PLAIN_ORDER}, cohort=COHORT, ns=WINDOW,
                  profile="d5", spaces="q0",
                  note="decision 2026-09-16: the campaign re-run under the plain readout. Only this "
                       "arm changes - every baseline in `main` already ran plain - so the panel "
                       "carries the plain-readout LeWi and is read beside main's baselines"),
    # Read-only: the plain-readout LeWi merged with main's baselines (src/m17_plain.py).
    # It is deliberately absent from ORDER and has no shard directory, so nothing builds
    # or dispatches it; it exists so the analysis can resolve a profile and a window.
    "plainall": dict(cells={c: [PLAIN_LEWI] for c in PLAIN_ORDER}, cohort=COHORT, ns=WINDOW,
                     profile="d5", spaces="q0",
                     note="read-only merge of panel plain with main's baselines"),
    # ---- the provisional setting (DESIGN amendment 4, decision 2026-09-16) ----
    "probe0": dict(cells={c: list(AUDIT_ARMS) for c in ("tree16", "dag16")}, cohort=COHORT_A,
                   ns=WINDOW2, profile="d5x2", spaces="p1", winners="bud", retain=True,
                   note="amendment 4: the budget audit's validation-selected winners re-run under "
                        "their own spec with init.pt and best.pt retained, for the probe, "
                        "causal-probe and embedding preview; no search. The training seeds derive "
                        "from the target, n and the data seed, so a re-run is the sealed winner "
                        "bit for bit"),
    "cal4": dict(cells={"rand16": ["graph_oracle", "noreg"]}, cohort=CAL_COHORT, ns=[2048, 262144],
                 profile="d5x2", spaces="start",
                 note="DESIGN 7 / amendment 4: realization check of the 16-leaf random graphs at "
                      "the setting's profile and grid endpoints"),
    "p4": dict(cells={c: list(ARMS2_OF[c]) for c in SUITE2}, cohort=PANEL_COHORT, ns=PANEL_N3,
               profile="d5x2", spaces="start",
               note="DESIGN 6.2 / amendment 4: refreeze panel for the provisional setting - every "
                    "workload and every arm of the setting, at the endpoints and the middle of its grid"),
    "main2": dict(cells={c: list(ARMS2_OF[c]) for c in SUITE2}, cohort=COHORT, ns=WINDOW3,
                  profile="d5x2", spaces="p4", retain=True,
                  note="amendment 4: the provisional paper-wide setting - six workloads at 16 active "
                       "inputs, 64x8 at 256,000 steps, n = 2^11..2^18, every arm but neuron positions, "
                       "instances 6000-6015, T = 32; init.pt and best.pt retained so the winners carry "
                       "the probes, the causal probes and the embeddings"),
    # ---- the deep test object (DESIGN amendment 5, decision 2026-09-16) ----
    "cal5": dict(cells={DEEP_CELL: ["graph_oracle", "noreg"]}, cohort=CAL_COHORT, ns=[32768, 2097152],
                 profile="w10", spaces="start",
                 note="DESIGN 7 / amendment 5: realization check of the 32-leaf tree at the deep profile "
                      "and the grid endpoints"),
    "p5": dict(cells={DEEP_CELL: list(DEEP_ARMS)}, cohort=PANEL_COHORT, ns=PANEL_N32,
               profile="w10", spaces="start",
               note="DESIGN 6.2 / amendment 5: refreeze panel for the deep profile"),
    "deep": dict(cells={DEEP_CELL: list(DEEP_ARMS)}, cohort=COHORT_A, ns=WINDOW32,
                 profile="w10", spaces="p5", retain=True,
                 note="amendment 5: the 32-leaf tree at d = 32, 128x10 at 1,024,000 steps, n = 2^15..2^21, "
                      "Dense / GO(1) / GO(all) / LeWi / L0, instances 6000-6007, T = 32; run after the setting"),
    "main": dict(cells={c: list(ARMS_OF[c]) for c in SUITE}, cohort=COHORT, ns=WINDOW,
                 profile="d5", spaces="p0",
                 note="DESIGN 2-5: the campaign, five workloads at 16 active inputs"),
    "bud": dict(cells={c: list(AUDIT_ARMS) for c in ("tree16", "dag16")}, cohort=COHORT_A, ns=WINDOW2,
                profile="d5x2", spaces="p1",
                note="DESIGN 10: budget and grid audit, 2^12..2^18 at 256,000 steps"),
    "cap1": dict(cells={"tree16": ["noreg", "graph_oracle"]}, cohort=COHORT_A, ns=[131072],
                 profile="w4", spaces="p2a", note="DESIGN 11: 64x4 capacity audit at 128,000 steps"),
    "cap2": dict(cells={"tree16": ["noreg", "graph_oracle"]}, cohort=COHORT_A, ns=[131072],
                 profile="w4x2", spaces="p2b", note="DESIGN 11: 64x4 capacity audit at 256,000 steps"),
    "cap3": dict(cells={"tree16": ["noreg", "graph_oracle"], "ftree16": ["noreg", "graph_oracle"]},
                 cohort=COHORT_A, ns=BIG_N, profile="w4x2", spaces="p2b",
                 note="DESIGN 11 / amendment 3: is 64x4 enough with abundant data? "
                      "n = 2^18 and 2^19 at 256,000 steps, on the tree and the Fourier tree"),
}
STAGE_A = {"main": COHORT_A}          # panels dispatched in two cohort stages
ORDER = ["probe0", "cal4", "p4", "main2", "cal5", "p5", "deep", "p3", "bud3", "q0", "plain", "cap3", "cal", "cal2",
         "cal3", "p0", "p1", "p2a", "p2b", "main", "bud", "cap1", "cap2"]

# ------------------------------------------------------------------ the 8-input suite (DESIGN amendment 6)
# Decision 2026-09-16 (intake/8-input.md): the record's seven 8-leaf workloads with the ambient
# dimension equal to the active-input count (d = 8) on the record's common backbone (32x4), the training
# budget and the sample-size grid recalibrated for this setting, the spaces refrozen, the campaign rerun
# with the post-hoc readouts; the depth-5 tree stays as it is. Families and presets are the record's.
CELLS.update({
    "tree8": ("tree_functional_anova", "depth3", 8),
    "dag8": ("dag_functional_anova", "default", 8),
    "ftree8": ("tree_fourier", "binary_depth3", 8),
    "flat8": ("flat_functional_anova", "w4", 8),
    "dense8": ("dense_fourier", "frozen_v1", 8),
    "fdag8": ("dag_fourier_v2", "default", 8),
    "rand8": ("rand_functional_anova", "default", 8),
})
CELL_LABEL.update({
    "tree8": "tree (depth 4), 8 leaves", "dag8": "reuse DAG (depth 3), 7 nodes",
    "ftree8": "Fourier tree (depth 4), 8 leaves", "flat8": "flat modules (4 pairs)",
    "dense8": "Fourier dense node (arity 8)", "fdag8": "Fourier reuse DAG (depth 3), 7 nodes",
    "rand8": "random graphs (depth 4), 8 leaves",
})
CELL_LEVELS.update({"tree8": 3, "dag8": 2, "ftree8": 3, "flat8": 1, "dense8": 1, "fdag8": 2, "rand8": 3})
SUITE8 = ["dag8", "tree8", "rand8", "ftree8", "flat8", "fdag8", "dense8"]     # dispatch order: the reuse DAG first
ARMS8_BASE = ["noreg", "l1", "l2", "l0hc", LEWI, "rigl", "bimt", "fiedler"]
# The record's roster at d = 8 minus GO(0) and input L0 (Dense, and a control of the inert-input mechanism,
# when every coordinate is active) and minus neuron positions (retired by the authors with the setting), with
# the GO rungs where the record ran them; none on the dense node, whose masks are all ones.
ARMS8_OF = {"tree8": ARMS8_BASE + ["go1", "graph_oracle"], "dag8": ARMS8_BASE + ["go1", "graph_oracle"],
            "ftree8": ARMS8_BASE + ["go1", "graph_oracle"], "flat8": ARMS8_BASE + ["graph_oracle"],
            "dense8": list(ARMS8_BASE), "fdag8": ARMS8_BASE + ["graph_oracle"],
            "rand8": ARMS8_BASE + ["go1", "go2", "graph_oracle"]}
# The record's common profile (CSFRBench amendment 7) at the candidate budgets: own cosine schedule over the
# full horizon, 64 validation checkpoints each. 32,000 is the record's; 64,000 brackets it from above;
# 128,000 exists for the ordering audit only, in case the rule lands on 64,000.
PROFILES.update({
    "c8k": dict(width=32, hidden=4, max_steps=8000, eval_every=125, label="32x4, 8,000 steps"),
    "c16k": dict(width=32, hidden=4, max_steps=16000, eval_every=250, label="32x4, 16,000 steps"),
    "c32k": dict(width=32, hidden=4, max_steps=32000, eval_every=500, label="32x4, 32,000 steps"),
    "c64k": dict(width=32, hidden=4, max_steps=64000, eval_every=1000, label="32x4, 64,000 steps"),
    "c128k": dict(width=32, hidden=4, max_steps=128000, eval_every=2000, label="32x4, 128,000 steps"),
})
LADDER8 = ["c8k", "c16k", "c32k", "c64k"]
AUDIT8 = {"c8k": "c16k", "c16k": "c32k", "c32k": "c64k", "c64k": "c128k"}     # the doubled budget of the ordering audit
WINDOW8 = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]     # 2^7 .. 2^15 (design decision): the record's grid two doublings down
PANEL_N8 = [128, 2048, 32768]                                        # the endpoints and the middle of WINDOW8
END8 = [128, 32768]
CAL8_COHORT = list(range(6090, 6098))     # eight target functions: the four smoke seeds of DESIGN 3 and four more of the range
AUDIT8_ARMS = ["noreg", "l0hc", LEWI, "graph_oracle"]
COMMON_SPACES = V11_SPACES.parent         # the v1.1 common-profile set: the spaces the record's 8-leaf arms ran in
SPACE_FILE_C = {**SPACE_FILE, "noreg": "dense", "go2": "graph_oracle"}   # Dense's file is `dense` there; GO(2) shares GO(all)'s, as the record ran it
COMMON_SETS = {"start8", "p8", "p8x"}     # space sets under the common file names
# Reserves above the record's measured maxima at 32x4 (LeWi 178 s, positions 181 s at 32,000 steps), scaled by the budget.
_CAL8_CELLS = {c: (["noreg"] if c == "dense8" else ["graph_oracle", "noreg"]) for c in SUITE8}
for _prof in LADDER8:
    PANELS[f"e{_prof[1:]}"] = dict(
        cells={c: list(a) for c, a in _CAL8_CELLS.items()}, cohort=CAL8_COHORT, ns=END8, profile=_prof, spaces="start8",
        note=f"amendment 6: the budget-and-grid calibration at {PROFILES[_prof]['label']} - the reference arm "
             f"(GO(all); Dense on the dense node) and Dense at the grid endpoints on every workload, instances 6090-6097")
# The verdict of the calibration (src/m17_cal8.py -> manifests/calibration8.json) names the profile; the refreeze
# panel, the campaign and the ordering audit exist only once it is written, so nothing can build at a budget
# the rule has not chosen.
_CAL8 = STUDY / "manifests" / "calibration8.json"
PROFILE8 = (json.loads(_CAL8.read_text()).get("profile") if _CAL8.exists() else None)
if PROFILE8:
    PANELS["p8"] = dict(cells={c: [a for a in ARMS8_OF[c] if a != "go2"] for c in SUITE8}, cohort=PANEL_COHORT,
                        ns=PANEL_N8, profile=PROFILE8, spaces="start8",
                        note="DESIGN 6.2 / amendment 6: refreeze panel of the 8-input suite - every workload and every arm "
                             "of the roster (GO(2) shares GO(all)'s file) at the endpoints and the middle of the grid")
    PANELS["main8"] = dict(cells={c: list(ARMS8_OF[c]) for c in SUITE8}, cohort=COHORT, ns=WINDOW8,
                           profile=PROFILE8, spaces="p8", retain=True,
                           note="amendment 6: the 8-input suite - seven workloads at d = 8, 32x4 at the calibrated budget, "
                                "n = 2^7..2^15, the record's roster without GO(0), input L0 and neuron positions, instances "
                                "6000-6015, T = 32; init.pt and best.pt retained for the probes, the causal probes and the embeddings")
    PANELS["p8x"] = dict(cells={c: list(AUDIT8_ARMS) for c in ("tree8", "dag8", "ftree8")}, cohort=PANEL_COHORT,
                         ns=PANEL_N8, profile=AUDIT8[PROFILE8], spaces="start8",
                         note="amendment 6, rule 4: refreeze panel of the audit arms at twice the calibrated budget")
    PANELS["bud8x"] = dict(cells={c: list(AUDIT8_ARMS) for c in ("tree8", "dag8", "ftree8")}, cohort=COHORT_A,
                           ns=WINDOW8, profile=AUDIT8[PROFILE8], spaces="p8x",
                           note="amendment 6, rule 4: the ordering audit - the audit arms over the full grid at twice the "
                                "calibrated budget on instances 6000-6007, read beside main8")
PANELS8 = [f"e{p[1:]}" for p in LADDER8] + ["p8", "main8", "p8x", "bud8x"]
# ---- the 64,000-step setting (DESIGN amendment 7, decision 2026-09-16) ----
# The same campaign at a 64,000-step budget with the record's grid 2^9..2^15 restored; everything else as
# main8 (workloads, roster, backbone, cohort, T, checkpoints retained), a refreeze at this profile first.
WINDOW8B = [512, 1024, 2048, 4096, 8192, 16384, 32768]     # 2^9 .. 2^15, the record's common grid
PANEL_N8B = [512, 4096, 32768]                             # the endpoints and the middle of WINDOW8B
COMMON_SETS.add("p8b")
PANELS["p8b"] = dict(cells={c: [a for a in ARMS8_OF[c] if a != "go2"] for c in SUITE8}, cohort=PANEL_COHORT,
                     ns=PANEL_N8B, profile="c64k", spaces="start8",
                     note="DESIGN 6.2 / amendment 7: refreeze panel of the 64,000-step setting - every workload and "
                          "every arm of the roster (GO(2) shares GO(all)'s file) at 2^9 / 2^12 / 2^15 on 6016-6019")
PANELS["main8b"] = dict(cells={c: list(ARMS8_OF[c]) for c in SUITE8}, cohort=COHORT, ns=WINDOW8B,
                        profile="c64k", spaces="p8b", retain=True,
                        note="amendment 7: the 8-input suite at 64,000 steps - seven workloads at d = 8, 32x4, "
                             "n = 2^9..2^15, the record's roster without GO(0), input L0 and neuron positions, "
                             "instances 6000-6015, T = 32; init.pt and best.pt retained for the post-hoc readouts")
PANELS8B = ["p8b", "main8b"]
# ---- the record's setting at 64,000 steps (DESIGN amendment 8, decision 2026-09-17; 18-main-d8/intake) ----
# The paper's d = 32 setting (16-main-final: the seven 8-leaf workloads with 24 inert coordinates, the record's
# cohort 7000-7015 and its 32 held-out graphs, the 32x4 backbone, n = 2^9..2^15, the record's full roster with
# every GO rung below the target's depth) at the 64,000-step budget of amendment 7, a refreeze at this profile
# first. The cells carry the record's own names, so a row here and a row of the record's contract read the same;
# the datasets are the record's by construction (same family, preset, d, instance and split seeds).
CELLS.update({
    "tree": ("tree_functional_anova", "depth3", 32),
    "dag": ("dag_functional_anova", "default", 32),
    "ftree": ("tree_fourier", "binary_depth3", 32),
    "flat": ("flat_functional_anova", "w4", 32),
    "dense": ("dense_fourier", "frozen_v1", 32),
    "fdag": ("dag_fourier_v2", "default", 32),
    "rand": ("rand_functional_anova", "default", 32),
})
CELL_LABEL.update({
    "tree": "tree (depth 4)", "dag": "reuse DAG (depth 3)", "ftree": "Fourier tree (depth 4)",
    "flat": "flat modules", "dense": "Fourier dense node", "fdag": "Fourier reuse DAG (depth 3)",
    "rand": "held-out random graphs (depth 4)",
})
CELL_LEVELS.update({"tree": 3, "dag": 2, "ftree": 3, "flat": 1, "dense": 1, "fdag": 2, "rand": 3})
SUITE8N = ["dag", "tree", "rand", "ftree", "flat", "fdag", "dense"]     # dispatch order: the reuse DAG first
COHORT8N = list(range(7000, 7016))            # the record's final cohort ...
COHORT8N_RAND = list(range(7000, 7032))       # ... and its 32 sealed held-out graphs
GO0 = "support_oracle"                        # GO(0): the true active-input support (registry id go0; the record's arm name)
SPACE_FILE[GO0] = SPACE_FILE_C[GO0] = "go0"
PAPER[GO0] = "GO(0)"
COLOR[GO0] = "#f781bf"
# The roster order the analyses and reports use; ARMS_ALL itself stays the 16-input rosters' source.
ARM_ORDER = ARMS_ALL[:9] + [GO0] + ARMS_ALL[9:]
ARMS8N_BASE = ARMS8_BASE + ["erb_positions"]
# The GO ladder of each workload: GO(0), then every GO(k) whose masks differ from the full oracle's, then GO(all).
# A rung equal to the full oracle is refused by the method (CSFRBench amendment 12; checked per target on
# 2026-09-17): on the depth-4 trees only the root sits above level 2, so GO(2) is GO(all) there and GO(2) exists
# on the random graphs alone; the flat modules (one level) carry GO(0) and GO(all); the dense node (one node over
# its whole support) GO(0) only. The record ran no GO(1) on the Fourier reuse DAG; that rung is distinct and runs.
ARMS8N_OF = {
    "tree": ARMS8N_BASE + [GO0, "go1", "graph_oracle"],
    "dag": ARMS8N_BASE + [GO0, "go1", "graph_oracle"],
    "ftree": ARMS8N_BASE + [GO0, "go1", "graph_oracle"],
    "flat": ARMS8N_BASE + [GO0, "graph_oracle"],
    "dense": ARMS8N_BASE + [GO0],
    "fdag": ARMS8N_BASE + [GO0, "go1", "graph_oracle"],
    "rand": ARMS8N_BASE + [GO0, "go1", "go2", "graph_oracle"],
}
COMMON_SETS.add("p8n")
PANELS["p8n"] = dict(cells={c: [a for a in ARMS8N_OF[c] if a != "go2"] for c in SUITE8N}, cohort=PANEL_COHORT,
                     ns=PANEL_N8B, profile="c64k", spaces="start8",
                     note="DESIGN 6.2 / amendment 8: refreeze panel of the record's setting at 64,000 steps - every "
                          "workload at d = 32 and every arm of the record's roster (GO(2) shares GO(all)'s file) at "
                          "2^9 / 2^12 / 2^15 on 6016-6019")
PANELS["main8n"] = dict(cells={c: list(ARMS8N_OF[c]) for c in SUITE8N}, cohort=COHORT8N,
                        cell_cohort={"rand": COHORT8N_RAND}, ns=WINDOW8B, profile="c64k", spaces="p8n", retain=True,
                        note="amendment 8: the record's setting at 64,000 steps - seven workloads at d = 32 (8 active "
                             "inputs, 24 inert), 32x4, n = 2^9..2^15, the record's roster with every GO rung below the "
                             "target's depth, instances 7000-7015 (7000-7031 on the random graphs), T = 32; init.pt "
                             "and best.pt retained for the post-hoc readouts")
# The rung amendment 7's campaign left out (decision 2026-09-17: every workload carries its whole ladder): GO(1) on
# the Fourier reuse DAG, the one distinct rung main8b did not run; read into main8b's record once complete
# (src/m17_go8b.py), under main8b's spaces, cohort, grid and budget.
PANELS["go8b"] = dict(cells={"fdag8": ["go1"]}, cohort=COHORT, ns=WINDOW8B, profile="c64k", spaces="p8b", retain=True,
                      note="amendment 8: the GO rung missing from main8b - GO(1) on the Fourier reuse DAG - under "
                           "main8b's spaces, cohort, grid and budget")
PANELS8N = ["p8n", "main8n", "go8b"]


def now_local() -> str:
    import datetime as dt
    from zoneinfo import ZoneInfo
    return dt.datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M")


def now_utc() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ campaign / spaces

_CAMPAIGN = None


def campaign(panel: str | None = None) -> dict:
    global _CAMPAIGN
    if _CAMPAIGN is None:
        import m17_tasks
        import w5_methods
        from csfrbench.pipeline import build_campaign_dict, compose_campaign
        composed, config_dir = compose_campaign("v05")
        c = build_campaign_dict(composed, config_dir, "v05")
        import abl_methods                     # 18-main-d8: the mechanism-ablation arms of the final LeWi
        if str(CUTOFF_SRC) not in sys.path:
            sys.path.append(str(CUTOFF_SRC))
        import cutoff_methods                  # 18-main-d8/input-cutoff: LeWi under the input rules (amendment 11 runs lewi-v1-rgm-in5)
        import abl_in5_methods                 # amendment 13: the ablation arms on the adopted arm's initialization
        import lms_methods                     # amendment 14: the mask-only forward on the adopted arm
        for arm in (*w5_methods.ARMS, *abl_methods.ARMS, *getattr(abl_methods, "INIT_ARMS", {}), *(f"lewi-v1-rgm-in{m}" for m in cutoff_methods.MIN_OPEN), "lewi-v1-rgm-infloor",
                    *abl_in5_methods.ARMS, *lms_methods.ARMS):
            c["methods"].setdefault(arm, {"fixed": {}})
        m17_tasks.register(c)
        _CAMPAIGN = c
    if not panel:
        return _CAMPAIGN
    c = json.loads(json.dumps(_CAMPAIGN))
    prof = PROFILES[PANELS[panel]["profile"]]
    ns = PANELS[panel]["ns"]
    # The v0.5 campaign carries its own workload override for
    # tree_functional_anova:depth4 (64x8 / 128,000 steps). It is replaced here, not
    # inherited: in this study the profile is a property of the panel, the same for
    # every workload, and the capacity audit deliberately runs a different one.
    c["defaults"]["n_max"] = max(ns)
    c["defaults"]["data"] = {**c["defaults"]["data"], "n_max": max(ns), "n_values": list(ns)}
    c["defaults"]["training"] = {
        **c["defaults"]["training"], "max_steps": int(prof["max_steps"]),
        "validation": {**c["defaults"]["training"]["validation"],
                       "eval_every": int(prof["eval_every"])}}
    c["defaults"]["arch"] = {**c["defaults"].get("arch", {}),
                             "width": int(prof["width"]), "hidden": int(prof["hidden"]), "act": "relu"}
    c["workload_overrides"] = {
        f"{task}:{preset}": {"arch": {"width": int(prof["width"]), "hidden": int(prof["hidden"]),
                                      "act": "relu"},
                             "training": {"max_steps": int(prof["max_steps"]),
                                          "validation": {"eval_every": int(prof["eval_every"])}},
                             "n_values": list(ns)}
        for task, preset, _d in CELLS.values()}
    return c


_HPO_SEED = None


def hpo_seed() -> int:
    global _HPO_SEED
    if _HPO_SEED is None:
        from csfrbench.pipeline import compose_campaign
        composed, _ = compose_campaign("v05")
        _HPO_SEED = int(composed["hpo"]["seed"])
    return _HPO_SEED


def space_path(arm: str, spaces: str) -> Path:
    name = (SPACE_FILE_C if spaces in COMMON_SETS else SPACE_FILE)[arm]
    root = {"start": V11_SPACES, "start8": COMMON_SPACES}.get(spaces, SPACES_OUT / spaces)
    return (root / name).with_suffix(".yaml")


SPACES_JSON = MAN / "search_spaces.json"      # the frozen space files of the campaigns, keyed by space set and file name


def space(arm: str, spaces: str) -> dict:
    name = space_path(arm, spaces).stem
    doc = json.loads(SPACES_JSON.read_text())
    if spaces not in doc or name not in doc[spaces]:
        raise SystemExit(f"space {spaces}/{name} is not in {SPACES_JSON}")
    return doc[spaces][name]


def trials(arm: str, cell: str, n: int, spaces: str, count: int = T) -> list[dict]:
    from csfrbench.pipeline import _sample_space
    from csfrbench.utils.seeds import derive
    task, preset, _ = CELLS[cell]
    seed = derive(hpo_seed(), "m17", spaces, TRIAL_KEY.get(arm, arm), task, preset, int(n))
    return _sample_space(space(arm, spaces), count, seed, f"m17.{arm}")


# ------------------------------------------------------------------ plan / build

def unit_name(cell, arm, fs, n, trial) -> str:
    return f"{cell}__{arm}__fs{fs}__n{n}__t{trial:03d}"


def parse_unit(name: str) -> dict | None:
    parts = name.split("__")
    if len(parts) != 5:
        return None
    return dict(cell=parts[0], arm=parts[1], fs=int(parts[2][2:]), n=int(parts[3][1:]),
                trial=int(parts[4][1:]))


def cohort_of(panel: str, stage: str | None) -> list[int]:
    if stage == "a" and panel in STAGE_A:
        return list(STAGE_A[panel])
    return list(PANELS[panel]["cohort"])


def plan(panel: str, stage: str | None = None, cells: list[str] | None = None) -> list[dict]:
    p = PANELS[panel]
    if p.get("winners"):
        return plan_winners(panel, cells)
    out = []
    for cell, arms in p["cells"].items():
        if cells and cell not in cells:
            continue
        pairs = (p.get("pairs") or {}).get(cell)        # amendment 13: a panel may run chosen (instance, size) cells of a workload only
        for arm in arms:
            for n in p["ns"]:
                draws = trials(arm, cell, n, p["spaces"])
                for fs in (p.get("cell_cohort") or {}).get(cell) or cohort_of(panel, stage):
                    if pairs is not None and (int(fs), int(n)) not in pairs:
                        continue
                    for t, hp in enumerate(draws):
                        out.append(dict(cell=cell, arm=arm, fs=int(fs), n=int(n), trial=t, hp=hp))
    return out


def plan_winners(panel: str, cells: list[str] | None = None) -> list[dict]:
    """One unit per condition: the source panel's validation-selected winner, re-run
    under its own spec (same trial index, same hyperparameters, same seeds). The unit
    names match the source panel's, so the record pairs the re-run with the sealed row."""
    import pandas as pd
    p = PANELS[panel]
    src = STUDY / "ledger" / f"winners_{p['winners']}.csv"
    if not src.exists():
        raise SystemExit(f"{src} missing: run m17_analysis.py {p['winners']} first")
    w = pd.read_csv(src)
    cohort = set(p["cohort"])
    out = []
    for cell, arms in p["cells"].items():
        if cells and cell not in cells:
            continue
        sub = w[(w["cell"] == cell) & w["arm"].isin(arms) & w["fs"].isin(cohort) & w["n"].isin(p["ns"])]
        for _, r in sub.sort_values(["arm", "n", "fs"]).iterrows():
            out.append(dict(cell=cell, arm=r["arm"], fs=int(r["fs"]), n=int(r["n"]),
                            trial=int(r["trial"]), hp=json.loads(r["hp_json"])))
    return out


def _raw(panel, cell, arm, fs, n, hp, trial) -> dict:
    task, preset, d = CELLS[cell]
    prof = PROFILES[PANELS[panel]["profile"]]
    return {
        "task": task, "task_preset": preset,
        "arch": {"width": prof["width"], "hidden": prof["hidden"], "act": "relu"},
        "method": PANELS[panel].get("arm_method", {}).get(arm) or ARM_METHOD.get(arm, arm), "d": int(d),     # amendment 11: a panel may run an arm key as another registry id
        "function_seed": int(fs), "dataset_seed": int(fs), "n": int(n),
        "hp": {**ARM_HP.get(arm, {}), **hp},
        "sigma_sq": 0.01,
        "sweep_id": f"m17.{panel}", "hpo_trial": int(trial),
        "test_evaluation": "final",
        # Retained only where a panel's winners feed the post-hoc probes (amendment 4):
        # init.pt is the linear probes' baseline, best.pt the model every readout loads.
        "retain": {"init_pt": bool(PANELS[panel].get("retain")),
                   "best_pt": bool(PANELS[panel].get("retain"))},
        **ARM_SPEC.get(arm, {}),
    }


# ------------------------------------------------------------------ 18-main-d8 intake/analysis.md (decision 2026-09-18):
# the mechanism ablations of the final LeWi at d = 8 (src/abl_methods.py) and the sensitivity of its position
# dimension r and radius multiplier, on the twin's cells, instances, grid, budget and LeWi's frozen space (p8b;
# the same trial draws as LeWi's, so every comparison is paired per instance and matched in selection).
ABL_ARMS = ["lewi-abl-soft", "lewi-abl-fixed", "lewi-abl-random", "lewi-abl-ste"]
SENS_R_ARMS = {"lewi-r1": 1, "lewi-r2": 2, "lewi-r4": 4, "lewi-r8": 8}                 # r = 3 is the record
SENS_RAD_ARMS = {"lewi-rad05": 0.5, "lewi-rad075": 0.75, "lewi-rad15": 1.5, "lewi-rad2": 2.0}   # x 1 is the record
PAPER.update({"lewi-abl-soft": "LeWi, soft-only (no mask)", "lewi-abl-fixed": "LeWi, fixed embeddings",
              "lewi-abl-random": "random sparse (LeWi's initial density)", "lewi-abl-ste": "LeWi, straight-through mask",
              **{a: f"LeWi, r = {r}" for a, r in SENS_R_ARMS.items()}, **{a: f"LeWi, R² × {m}" for a, m in SENS_RAD_ARMS.items()}})
COLOR.update({"lewi-abl-soft": "#ff9896", "lewi-abl-fixed": "#c49c94", "lewi-abl-random": "#7f7f7f", "lewi-abl-ste": "#e377c2",
              "lewi-r1": "#fdd0a2", "lewi-r2": "#fdae6b", "lewi-r4": "#e6550d", "lewi-r8": "#a63603",
              "lewi-rad05": "#c6dbef", "lewi-rad075": "#6baed6", "lewi-rad15": "#2171b5", "lewi-rad2": "#08306b"})
for _a in ABL_ARMS:
    SPACE_FILE[_a] = SPACE_FILE_C[_a] = "lewi-v1"
    TRIAL_KEY[_a] = "lewi-v1"
for _a, _r in SENS_R_ARMS.items():
    ARM_METHOD[_a] = LEWI; ARM_HP[_a] = {"code_dim": _r}; SPACE_FILE[_a] = SPACE_FILE_C[_a] = "lewi-v1"; TRIAL_KEY[_a] = "lewi-v1"
for _a, _m in SENS_RAD_ARMS.items():
    ARM_METHOD[_a] = LEWI; ARM_HP[_a] = {"radius": _m}; SPACE_FILE[_a] = SPACE_FILE_C[_a] = "lewi-v1"; TRIAL_KEY[_a] = "lewi-v1"
ABL_CELLS = ["tree8", "ftree8", "dag8", "flat8", "dense8", "rand8"]      # rand8 = the extension to the sampled graphs
ABL32_CELLS = ["tree", "dag"]                                              # the extension to the nuisance-input setting (d = 32)
SENS_CELLS = ["tree8", "ftree8"]
SENS_COHORT = list(range(6000, 6008))
SENS_NS = [2048, 8192, 32768]
PANELS["abl8"] = dict(cells={c: list(ABL_ARMS) for c in ABL_CELLS}, cohort=COHORT, ns=WINDOW8B, profile="c64k", spaces="p8b", retain=True,
                      note="18-main-d8 analysis: the four mechanism ablations of the final LeWi on the twin's five core workloads, "
                           "full grid 2^9..2^15, instances 6000-6015, T = 32, LeWi's p8b space; init.pt and best.pt retained")
PANELS["abl8r"] = dict(cells={c: list(SENS_R_ARMS) for c in SENS_CELLS}, cohort=SENS_COHORT, ns=SENS_NS, profile="c64k", spaces="p8b", retain=True,
                       note="18-main-d8 analysis: position dimension r in {1, 2, 4, 8} (3 = the record) on the tree and the Fourier tree, "
                            "reduced grid 2^11 / 2^13 / 2^15, instances 6000-6007, T = 32")
PANELS["abl8rad"] = dict(cells={c: list(SENS_RAD_ARMS) for c in SENS_CELLS}, cohort=SENS_COHORT, ns=SENS_NS, profile="c64k", spaces="p8b", retain=True,
                         note="18-main-d8 analysis: radius multiplier R^2 x {0.5, 0.75, 1.5, 2} (x 1 = the record) on the tree and the Fourier tree, "
                              "reduced grid 2^11 / 2^13 / 2^15, instances 6000-6007, T = 32")
PANELS["abl32"] = dict(cells={c: list(ABL_ARMS) for c in ABL32_CELLS}, cohort=COHORT8N, ns=WINDOW8B, profile="c64k", spaces="p8n", retain=True,
                       note="18-main-d8 analysis, extension: the four mechanism ablations in the nuisance-input setting (d = 32: 8 active and 24 inert "
                            "inputs) on the tree and the reuse DAG, full grid, instances 7000-7015, T = 32, LeWi's p8n space and trial draws")
SENS_INIT_ARMS = {"lewi-init05": 0.5, "lewi-init2": 2.0}           # positions and radius scaled together (src/abl_methods.py); x 1 is the record
PAPER.update({a: f"LeWi, init scale \u00d7 {c}" for a, c in SENS_INIT_ARMS.items()})
COLOR.update({"lewi-init05": "#a1d99b", "lewi-init2": "#238b45"})
for _a in SENS_INIT_ARMS:
    SPACE_FILE[_a] = SPACE_FILE_C[_a] = "lewi-v1"; TRIAL_KEY[_a] = "lewi-v1"
PANELS["abl8init"] = dict(cells={c: list(SENS_INIT_ARMS) for c in SENS_CELLS}, cohort=SENS_COHORT, ns=SENS_NS, profile="c64k", spaces="p8b", retain=True,
                          note="18-main-d8 analysis: initialization scale x {0.5, 2} (positions and radius together; the initial masks are the record's) "
                               "on the tree and the Fourier tree, reduced grid 2^11 / 2^13 / 2^15, instances 6000-6007, T = 32")
SENS_SEED_ARMS = {"lewi-seed1": 1, "lewi-seed2": 2, "lewi-seed3": 3, "lewi-seed4": 4}     # seed_replicate (0 = the record's own seeds)
PAPER.update({a: f"LeWi, seed replicate {k}" for a, k in SENS_SEED_ARMS.items()})
COLOR.update({a: "#fb6a4a" for a in SENS_SEED_ARMS})
for _a, _k in SENS_SEED_ARMS.items():
    ARM_METHOD[_a] = LEWI; ARM_SPEC[_a] = {"seed_replicate": _k}; SPACE_FILE[_a] = SPACE_FILE_C[_a] = "lewi-v1"; TRIAL_KEY[_a] = "lewi-v1"
PANELS["abl8seed"] = dict(cells={c: list(SENS_SEED_ARMS) for c in SENS_CELLS}, cohort=SENS_COHORT, ns=SENS_NS, profile="c64k", spaces="p8b", retain=True,
                          note="18-main-d8 analysis: seed variation - the record's LeWi with four fresh draws of every training seed (initialization, positions, "
                               "batch order; train.py seed_replicate 1..4) under the same 32 learning-rate trials, on the tree and the Fourier tree, reduced grid, "
                               "instances 6000-6007")
PANELS8ABL = ["abl8", "abl8r", "abl8rad", "abl32", "abl8init", "abl8seed"]


# ------------------------------------------------------------------ 18-main-d8 completeness check (decision 2026-09-18):
# neuron positions on the 8-input suite. Amendment 7 left the arm out of the twin's roster (ARMS8_OF), so the d = 8
# setting was the one setting of the paper without it. It runs here as it ran in the record's setting (main8n): the
# v1.1 common-profile space refrozen at this profile on 6016-6019 (pos8p, the rule of DESIGN 6.2), then the twin's
# cells, cohort, grid, budget and T (pos8b), read into main8b's record (src/m17_fold8b.py).
POS8_ARM = "erb_positions"
COMMON_SETS.add("pos8p")
PANELS["pos8p"] = dict(cells={c: [POS8_ARM] for c in SUITE8}, cohort=PANEL_COHORT, ns=PANEL_N8B, profile="c64k", spaces="start8",
                       note="DESIGN 6.2 / 18-main-d8 completeness check: refreeze panel of neuron positions at d = 8 - every "
                            "workload of the 8-input suite at 2^9 / 2^12 / 2^15 on 6016-6019, as p8b froze every other arm")
PANELS["pos8b"] = dict(cells={c: [POS8_ARM] for c in SUITE8}, cohort=COHORT, ns=WINDOW8B, profile="c64k", spaces="pos8p", retain=True,
                       note="18-main-d8 completeness check: neuron positions on the seven workloads of the 8-input suite under "
                            "main8b's cohort (6000-6015), grid (2^9..2^15), budget (32x4, 64,000 steps) and T = 32; init.pt and "
                            "best.pt retained; read into main8b's record")
PANELS8POS = ["pos8p", "pos8b"]


# ------------------------------------------------------------------ 18-main-d8 (decision 2026-09-20): two more campaigns of the 8-input suite.
# (a) The sampled graphs at d = 8 on 32 instances. The record's setting holds 32 sealed held-out graphs (7000-7031: 14-main,
#     16-main-final, main8n) where the twin held 16 (6000-6015). Sixteen graphs, 6020-6035 (the authors' choice,): the block after the refreeze
#     instances 6016-6019, never drawn by a panel, calibration, refreeze or smoke test of this study; the seed ledger
#     (versions/v1.1/configs/seeds.yaml) records 6020-6023 for the benchmark's horizon and boundary checks (other families)
#     and 6030-6037 for an earlier LeWi study (random-graph instances 6030-6033 at d = 32 among them), which
#     18-main-d8 discloses - run under main8b's grid, budget, T and spaces (p8b; neuron positions: pos8p), so every (arm, size) carries the
#     record's own trial draws, and are read into main8b's record (src/m17_fold8b.py). The 16 existing instances are reused.
# (b) The whole suite on the record's cohort: instances 7000-7015, 7000-7031 on the random graphs (main8n's), the twin's
#     grid, budget, T and spaces, checkpoints retained; main8r is the record of that setting, pos8r is read into it. The
#     setting pairs d = 8 with d = 32 per instance (the same target functions).
RAND8X_COHORT = list(range(6020, 6036))
ARMS8_FULL = {c: list(ARMS8_OF[c]) + (["go1"] if c == "fdag8" and "go1" not in ARMS8_OF[c] else []) for c in SUITE8}   # go8b's rung too
PANELS["rand8b"] = dict(cells={"rand8": list(ARMS8_OF["rand8"])}, cohort=RAND8X_COHORT, ns=WINDOW8B, profile="c64k", spaces="p8b", retain=True,
                        note="18-main-d8 (2026-09-20): the sampled graphs at d = 8 on 16 more held-out instances (6020-6035) under main8b's "
                             "grid (2^9..2^15), budget (32x4, 64,000 steps), T = 32 and spaces (p8b); init.pt and best.pt retained; "
                             "read into main8b's record")
PANELS["rand8bpos"] = dict(cells={"rand8": [POS8_ARM]}, cohort=RAND8X_COHORT, ns=WINDOW8B, profile="c64k", spaces="pos8p", retain=True,
                           note="18-main-d8 (2026-09-20): neuron positions on the same 16 instances of the sampled graphs (pos8p's space, "
                                "as pos8b ran); read into main8b's record")
PANELS["main8r"] = dict(cells={c: list(a) for c, a in ARMS8_FULL.items()}, cohort=COHORT8N, cell_cohort={"rand8": COHORT8N_RAND},
                        ns=WINDOW8B, profile="c64k", spaces="p8b", retain=True,
                        note="18-main-d8 (2026-09-20): the 8-input suite on the record's cohort - instances 7000-7015, 7000-7031 on the "
                             "random graphs - every arm of the twin's roster (GO(1) on the Fourier reuse DAG included), 32x4 at 64,000 "
                             "steps, 2^9..2^15, T = 32, the twin's spaces (p8b); init.pt and best.pt retained")
PANELS["pos8r"] = dict(cells={c: [POS8_ARM] for c in SUITE8}, cohort=COHORT8N, cell_cohort={"rand8": COHORT8N_RAND}, ns=WINDOW8B,
                       profile="c64k", spaces="pos8p", retain=True,
                       note="18-main-d8 (2026-09-20): neuron positions on the record's cohort at d = 8 (pos8p's space, as pos8b ran); "
                            "read into main8r's record")
PANELS8X = ["rand8b", "rand8bpos", "main8r", "pos8r"]

# ---- the nuisance-input setting on the twin's seeds (DESIGN amendment 11, decision 2026-09-20; 18-main-d8/intake) ----
# The d = 8 campaign repeated with 24 inert coordinates on the same instances, everything else the same: the seven workloads at
# d = 32 (the record's cells tree ... rand - the same families and presets as the twin's tree8 ... rand8 - with 24 inert
# coordinates) on the twin's cohort 6000-6015 (6000-6015 and 6020-6035 on the random graphs), the twin's profile (32x4, 64,000
# steps), grid 2^9..2^15, T = 32, the record's roster at d = 32 (ARMS8N_OF; GO(0) is the active-input oracle here, Dense at
# d = 8); and the twin's depth-5 tree with 24 inert coordinates: the depth-5 workload at d = 40 (cell tree40 = the twin's tree16
# plus 24), the record's depth-5 profile (64x8, 128,000 steps, 2^11..2^17), the record's 13 depth-5 arms. LeWi is the twin's
# adopted arm: lewi-v1-rgm under the input rule in5 (18-main-d8/input-cutoff/src/cutoff_methods.py: an input with fewer than
# five open first-transition connections at initialization has its position drawn again; the readout's own rejection rule and
# its gate and mask are the arm's since 2026-09-12, every transition masked and gated alike). The arm keeps its key, so unit
# names, ledgers and analyses read it as LeWi; the panel maps the key to the registry id (arm_method). A refreeze at each
# profile first, on 6016-6019 (p32s at d = 32; p40t at d = 40), as the twin did (p8b, p0).
CELLS["tree40"] = ("tree_functional_anova", "depth4", 40)
CELL_LABEL["tree40"] = "tree (depth 5), 16 leaves and 24 inert inputs"
CELL_LEVELS["tree40"] = 4
SUITE32S = list(SUITE8N)                                    # dispatch order: the reuse DAG first
RULE_METHOD = {LEWI: "lewi-v1-rgm-in5"}                     # the LeWi key -> the registry id of the adopted arm in these panels
COHORT_RAND_S = COHORT + RAND8X_COHORT                      # the twin's random graphs: 6000-6015 and 6020-6035
ARMS40 = list(ARMS_ALL) + [GO0]                             # the record's 13 depth-5 arms (GO(0) = the active-input oracle at d = 40)
CUTOFF_SRC = STUDY.parent / "18-main-d8" / "input-cutoff" / "src"
COMMON_SETS.add("p32s")
PANELS["p32s"] = dict(cells={c: [a for a in ARMS8N_OF[c] if a != "go2"] for c in SUITE32S}, cohort=PANEL_COHORT, ns=PANEL_N8B,
                      profile="c64k", spaces="start8", arm_method=RULE_METHOD,
                      note="DESIGN 6.2 / amendment 11: refreeze panel of the nuisance-input setting on the twin's seeds - every workload "
                           "at d = 32 and every arm of the record's roster (GO(2) shares GO(all)'s file; LeWi = lewi-v1-rgm under rule in5) "
                           "at 2^9 / 2^12 / 2^15 on 6016-6019")
PANELS["main32s"] = dict(cells={c: list(ARMS8N_OF[c]) for c in SUITE32S}, cohort=COHORT, cell_cohort={"rand": COHORT_RAND_S},
                         ns=WINDOW8B, profile="c64k", spaces="p32s", retain=True, arm_method=RULE_METHOD,
                         note="amendment 11: the twin's campaign with 24 inert coordinates - seven workloads at d = 32 on instances "
                              "6000-6015 (6000-6015 and 6020-6035 on the random graphs), 32x4 at 64,000 steps, 2^9..2^15, the record's "
                              "roster with every GO rung below the target's depth, LeWi under rule in5, T = 32; init.pt and best.pt "
                              "retained for the post-hoc readouts")
PANELS["p40t"] = dict(cells={"tree40": list(ARMS40)}, cohort=PANEL_COHORT, ns=PANEL_N, profile="d5",
                      spaces="start", arm_method=RULE_METHOD,
                      note="DESIGN 6.2 / amendment 11: refreeze panel of the depth-5 tree with 24 inert inputs (d = 40) at the record's "
                           "depth-5 profile, every depth-5 arm (GO(2) has its own file in the depth-4 start set, as in p0) at 2^11 / 2^14 / 2^17 on 6016-6019")
PANELS["main40t"] = dict(cells={"tree40": list(ARMS40)}, cohort=COHORT, ns=WINDOW, profile="d5", spaces="p40t", retain=True,
                         arm_method=RULE_METHOD,
                         note="amendment 11: the twin's depth-5 tree with 24 inert inputs - d = 40, 64x8 at 128,000 steps, 2^11..2^17, "
                              "the record's 13 depth-5 arms, LeWi under rule in5, instances 6000-6015, T = 32; checkpoints retained")
PANELS32S = ["p32s", "main32s", "p40t", "main40t"]

# Amendment 12 (decision 2026-09-21): the depth-5 tree of the nuisance-input setting belongs at d = 32 - 16 leaves and 16
# inert inputs, the same ambient dimension as the seven workloads - not at d = 40. Its own refreeze (p32t) and campaign (main32t),
# the record's 13 depth-5 arms and profile, LeWi under rule in5, instances 6000-6015. The d = 40 panels stay on record, superseded.
CELLS["tree5x32"] = ("tree_functional_anova", "depth4", 32)
CELL_LABEL["tree5x32"] = "tree (depth 5), 16 leaves and 16 inert inputs"
CELL_LEVELS["tree5x32"] = 4
PANELS["p32t"] = dict(cells={"tree5x32": list(ARMS40)}, cohort=PANEL_COHORT, ns=PANEL_N, profile="d5",
                      spaces="start", arm_method=RULE_METHOD,
                      note="DESIGN 6.2 / amendment 12: refreeze panel of the depth-5 tree with 16 inert inputs (d = 32) at the record's "
                           "depth-5 profile, every depth-5 arm at 2^11 / 2^14 / 2^17 on 6016-6019")
PANELS["main32t"] = dict(cells={"tree5x32": list(ARMS40)}, cohort=COHORT, ns=WINDOW, profile="d5", spaces="p32t", retain=True,
                         arm_method=RULE_METHOD,
                         note="amendment 12: the twin's depth-5 tree with 16 inert inputs - d = 32, 64x8 at 128,000 steps, 2^11..2^17, "
                              "the record's 13 depth-5 arms, LeWi under rule in5, instances 6000-6015, T = 32; checkpoints retained")
PANELS32S += ["p32t", "main32t"]

# 18-main-d8's adoption of the input-cutoff rule for the twin's LeWi arm (decision 2026-09-20; amendment 10): rule in5
# (an input with fewer than five open first-transition connections at initialization has its position drawn again,
# 18-main-d8/input-cutoff, tables/repair_in5_cells.csv). A winners panel for the post-hoc readouts only: its run root links
# the 144 repaired winners (input-cutoff/runs/repair/in5) and the instance-6000 winners of the embedding panels; nothing is
# planned or run under it, and the record's panel main8b is untouched.
PANELS["main8b-in5"] = dict(PANELS["main8b"], winners="main8b",
                            note="18-main-d8's adoption of rule in5 for the twin's LeWi arm (decision 2026-09-20): the repaired winners "
                                 "linked under the run root for the post-hoc readouts (probes, DAS, embeddings); no unit is planned here")

# Amendment 13 (decision 2026-09-22, the final paper's figure set): the ablation family on the adopted arm's initialization. Every
# mechanism variant, initialization-scale arm and sensitivity arm is built from lewi-v1-rgm-in5 (src/abl_in5_methods.py: the input
# rule fires in the parent build, the variant's change follows; the r / radius / seed arms map the LeWi key to the rule's id and keep
# their fixed hyperparameters), on the record's cells, grid, budget, LeWi's frozen space and trial draws - the sampled graphs on all
# 32 instances now - and, for the nuisance-input setting, on the twin's seeds under the p32s space (abl32 ran on 7000-7015 under p8n).
# The reference of every contrast is the adopted LeWi (18-main-d8 ledger/d8, ledger/d32s). The record's panels abl8 ... abl32 stay.
ABL_IN5 = {a: f"{a}-in5" for a in ABL_ARMS}
INIT_IN5 = {a: f"{a}-in5" for a in SENS_INIT_ARMS}
SENS_IN5 = {a: RULE_METHOD[LEWI] for a in (*SENS_R_ARMS, *SENS_RAD_ARMS, *SENS_SEED_ARMS)}


def _abl_in5_pairs() -> dict[str, set[tuple[int, int]]]:
    """The cells whose initialization the rule changes (18-main-d8/input-cutoff census: an input with fewer than five open
    first-transition connections at initialization) plus two control cells per workload, frozen in manifests/abl_in5_cells.csv.
    On every other cell the in5 build equals the record's unit bit for bit, and the read takes the record's abl8 / abl8init row."""
    import csv
    out: dict[str, set[tuple[int, int]]] = {}
    with open(STUDY / "manifests" / "abl_in5_cells.csv") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["cell"], set()).add((int(r["fs"]), int(r["n"])))
    return out


ABL_IN5_PAIRS = _abl_in5_pairs()
PANELS["abl8-in5"] = dict(cells={c: list(ABL_ARMS) for c in ABL_CELLS}, cohort=COHORT, cell_cohort={"rand8": COHORT_RAND_S}, ns=WINDOW8B,
                          profile="c64k", spaces="p8b", retain=True, arm_method=ABL_IN5, pairs=ABL_IN5_PAIRS,
                          note="amendment 13: the four mechanism ablations on the adopted arm's initialization (rule in5) on the cells the rule "
                               "changes (142 of the twin's five core workloads and the sampled graphs on 32 instances) plus two controls per workload, "
                               "T = 32, LeWi's p8b space and trial draws; the record's abl8 stands on every other cell; init.pt and best.pt retained")
PANELS["abl8r-in5"] = dict(PANELS["abl8r"], arm_method=SENS_IN5, note="amendment 13: abl8r on the adopted arm's initialization (rule in5); "
                           "in full, since another position dimension draws other positions")
PANELS["abl8rad-in5"] = dict(PANELS["abl8rad"], arm_method=SENS_IN5, note="amendment 13: abl8rad on the adopted arm's initialization (rule in5); "
                             "in full, since another radius opens other connections")
PANELS["abl8init-in5"] = dict(PANELS["abl8init"], arm_method=INIT_IN5, pairs=ABL_IN5_PAIRS,
                              note="amendment 13: abl8init on the adopted arm's initialization (rule in5) on the cells the rule changes plus the controls "
                                   "(the scaled initialization keeps the record's masks); the record's abl8init stands on every other cell")
PANELS["abl8seed-in5"] = dict(PANELS["abl8seed"], arm_method=SENS_IN5, note="amendment 13: abl8seed on the adopted arm's initialization (rule in5); "
                              "in full, since a seed replicate draws other positions")
PANELS["abl32s-in5"] = dict(cells={c: list(ABL_ARMS) for c in ABL32_CELLS}, cohort=COHORT, ns=WINDOW8B, profile="c64k", spaces="p32s", retain=True,
                            arm_method=ABL_IN5,
                            note="amendment 13: the four mechanism ablations on the adopted arm's initialization in the nuisance-input setting "
                                 "(d = 32) on the tree and the reuse DAG, full grid, instances 6000-6015, T = 32, LeWi's p32s space and trial draws")
PANELS_ABL_IN5 = ["abl8-in5", "abl8r-in5", "abl8rad-in5", "abl8init-in5", "abl8seed-in5", "abl32s-in5"]

# Amendment 14 (decision 2026-09-22; 18-main-d8 intake/lewi_mask_only_r_sensitivity_goal.md): two prespecified LeWi studies on
# the adopted arm across the whole primary all-active suite - the seven workloads at d = 8 (the sampled graphs on 32 instances) and
# the depth-5 tree at d = 16 (cell tree16, the record's depth-5 profile). Arms: lewi-mask (the mask-only forward, src/lms_methods.py,
# registry id lewi-mask-in5) and lewi-dim1 / lewi-dim2 / lewi-dim4 / lewi-dim8 (the adopted arm with code_dim r; R^2 = r/2 by the
# method's own rule). Their learning-rate spaces are frozen by the benchmark's rule on the development instances 6016-6019 from
# LeWi's own start ranges (space sets startv8 / startv5: copies of LeWi's start files under the arms' names; panels pv8 / pv5), and
# installed into the canonical sets p8b / p0 under the arms' names - the final panels lms8 / lms5 read those sets with LeWi's trial
# key, so every arm draws LeWi's Sobol quantiles (the numerical learning rates follow each arm's frozen range). The r = 3 reference
# is the adopted LeWi of 18-main-d8's ledger; nothing of it is re-run here.
LMS_ARMS = ["lewi-mask", "lewi-dim1", "lewi-dim2", "lewi-dim4", "lewi-dim8"]
ARM_METHOD["lewi-mask"] = "lewi-mask-in5"
for _a, _r in (("lewi-dim1", 1), ("lewi-dim2", 2), ("lewi-dim4", 4), ("lewi-dim8", 8)):
    ARM_METHOD[_a] = RULE_METHOD[LEWI]; ARM_HP[_a] = {"code_dim": _r}
for _a in LMS_ARMS:
    SPACE_FILE[_a] = SPACE_FILE_C[_a] = _a; TRIAL_KEY[_a] = "lewi-v1"
PAPER.update({"lewi-mask": "LeWi, mask-only forward", "lewi-dim1": "LeWi, r = 1", "lewi-dim2": "LeWi, r = 2", "lewi-dim4": "LeWi, r = 4", "lewi-dim8": "LeWi, r = 8"})
COLOR.update({"lewi-mask": "#8c564b", "lewi-dim1": "#fdd0a2", "lewi-dim2": "#fdae6b", "lewi-dim4": "#e6550d", "lewi-dim8": "#a63603"})
PANELS["pv8"] = dict(cells={c: list(LMS_ARMS) for c in SUITE8}, cohort=PANEL_COHORT, ns=PANEL_N8B, profile="c64k", spaces="startv8",
                     note="amendment 14: the development refreeze of the two LeWi studies' arms at the 64,000-step common profile - every d = 8 "
                          "workload at 2^9 / 2^12 / 2^15 on 6016-6019, T = 32, from LeWi's start range (the benchmark's freeze rule)")
PANELS["pv5"] = dict(cells={"tree16": list(LMS_ARMS)}, cohort=PANEL_COHORT, ns=PANEL_N, profile="d5", spaces="startv5",
                     note="amendment 14: the development refreeze of the arms on the depth-5 tree (d = 16, 64x8, 128,000 steps) at "
                          "2^11 / 2^14 / 2^17 on 6016-6019, T = 32, from LeWi's depth-5 start range")
PANELS["lms8"] = dict(cells={c: list(LMS_ARMS) for c in SUITE8}, cohort=COHORT, cell_cohort={"rand8": COHORT_RAND_S}, ns=WINDOW8B,
                      profile="c64k", spaces="p8b", retain=True,
                      note="amendment 14: the five arms on the seven d = 8 workloads (the sampled graphs on 32 instances), full grid "
                           "2^9..2^15, instances 6000-6015, T = 32, the arms' frozen ranges under the p8b set with LeWi's trial key; "
                           "init.pt and best.pt retained")
PANELS["lms5"] = dict(cells={"tree16": list(LMS_ARMS)}, cohort=COHORT, ns=WINDOW, profile="d5", spaces="p0", retain=True,
                      note="amendment 14: the five arms on the depth-5 tree at d = 16 (64x8, 128,000 steps), 2^11..2^17, instances "
                           "6000-6015, T = 32, the arms' frozen ranges under the p0 set with LeWi's trial key; checkpoints retained")
PANELS_LMS = ["pv8", "pv5", "lms8", "lms5"]


# Amendment 15 (decision 2026-09-22; 18-main-d8 intake/lewi_cached_geometry_efficiency_goal.md): amortizing LeWi's geometry.
# Arms lewi-k16 ... lewi-k2048 = the adopted arm with its geometry refreshed every K optimizer steps and the gate-mask multiplier
# cached in between (src/lk_methods.py, registry id lewi-cache-in5 with the fixed hyperparameter refresh_every = K); K = 1 is the
# adopted LeWi of 18-main-d8's ledger (the canonical implementation; nothing of it is re-run). The learning-rate spaces are frozen
# by the benchmark's rule on the development instances 6016-6019 from LeWi's own start ranges (space sets startk8 / startk5: copies
# of LeWi's start files under the arms' names; panels pk8 on the tree, the reuse DAG and the high-arity node at d = 8, pk5 on the
# depth-5 tree), then installed into the canonical sets p8b / p0 by the prespecified decision rule
# (18-main-d8/lewi-cached/src/freeze_decision.py) - the finals lk8 / lk5 read those sets with LeWi's trial key, so every arm draws
# LeWi's Sobol quantiles. One learning rate for the weights and the embeddings, as the record (no separate embedding multiplier:
# prespecified, lewi-cached/README.md).
LK_ARMS = ["lewi-k16", "lewi-k64", "lewi-k256", "lewi-k512", "lewi-k1024", "lewi-k2048"]
LK_OF = {a: int(a[len("lewi-k"):]) for a in LK_ARMS}
LK_CELLS8 = ["tree8", "dag8", "dense8"]
for _a, _k in LK_OF.items():
    ARM_METHOD[_a] = "lewi-cache-in5"; ARM_HP[_a] = {"refresh_every": _k}
    SPACE_FILE[_a] = SPACE_FILE_C[_a] = _a; TRIAL_KEY[_a] = "lewi-v1"
PAPER.update({a: f"LeWi, K = {k}" for a, k in LK_OF.items()})
COLOR.update({"lewi-k16": "#c6dbef", "lewi-k64": "#9ecae1", "lewi-k256": "#6baed6", "lewi-k512": "#4292c6", "lewi-k1024": "#2171b5", "lewi-k2048": "#08306b"})
PANELS["pk8"] = dict(cells={c: list(LK_ARMS) for c in LK_CELLS8}, cohort=PANEL_COHORT, ns=PANEL_N8B, profile="c64k", spaces="startk8",
                     note="amendment 15: the development refreeze of the six K arms at the 64,000-step common profile - the depth-4 tree, "
                          "the reuse DAG and the high-arity node at 2^9 / 2^12 / 2^15 on 6016-6019, T = 32, from LeWi's start range")
PANELS["pk5"] = dict(cells={"tree16": list(LK_ARMS)}, cohort=PANEL_COHORT, ns=PANEL_N, profile="d5", spaces="startk5",
                     note="amendment 15: the development refreeze of the six K arms on the depth-5 tree (d = 16, 64x8, 128,000 steps) at "
                          "2^11 / 2^14 / 2^17 on 6016-6019, T = 32, from LeWi's depth-5 start range")
PANELS["lk8"] = dict(cells={c: list(LK_ARMS) for c in SUITE8}, cohort=COHORT, cell_cohort={"rand8": COHORT_RAND_S}, ns=WINDOW8B,
                     profile="c64k", spaces="p8b", retain=True,
                     note="amendment 15: the six K arms on the seven d = 8 workloads (the sampled graphs on 32 instances), full grid "
                          "2^9..2^15, instances 6000-6015, T = 32, the arms' frozen ranges under the p8b set with LeWi's trial key; "
                          "init.pt and best.pt retained")
PANELS["lk5"] = dict(cells={"tree16": list(LK_ARMS)}, cohort=COHORT, ns=WINDOW, profile="d5", spaces="p0", retain=True,
                     note="amendment 15: the six K arms on the depth-5 tree at d = 16 (64x8, 128,000 steps), 2^11..2^17, instances "
                          "6000-6015, T = 32, the arms' frozen ranges under the p0 set with LeWi's trial key; checkpoints retained")
PANELS_LK = ["pk8", "pk5", "lk8", "lk5"]


# Amendment 17 (decision 2026-09-24; 18-main-d8 intake/cached_geometry_followup.md): the fine-grained refresh intervals K = 2, 4, 8
# of amendment 15's cached-geometry arm, between the canonical K = 1 and the smallest interval run so far (K = 16). The same method
# (src/lk_methods.py, registry id lewi-cache-in5, refresh_every = K; unchanged since amendment 15's freeze), the same pilot
# structure (the sets startkf8 / startkf5: copies of LeWi's start files under the new arms' names; panels pkf8 on the tree, the reuse DAG and
# the high-arity node at d = 8, pkf5 on the depth-5 tree, instances 6016-6019), the same prespecified decision rule
# (18-main-d8/lewi-cached-fine/src/freeze_decision.py: amendment 15's script with the new names) installing each arm's final range into
# p8b / p0, then the finals lkf8 / lkf5 with LeWi's trial key (96,768 trainings). K = 1 (18-main-d8's ledger) and K = 16 (lk8 / lk5) are read,
# never re-run.
LKF_ARMS = ["lewi-k2", "lewi-k4", "lewi-k8"]
LKF_OF = {a: int(a[len("lewi-k"):]) for a in LKF_ARMS}
for _a, _k in LKF_OF.items():
    ARM_METHOD[_a] = "lewi-cache-in5"; ARM_HP[_a] = {"refresh_every": _k}
    SPACE_FILE[_a] = SPACE_FILE_C[_a] = _a; TRIAL_KEY[_a] = "lewi-v1"
PAPER.update({a: f"LeWi, K = {k}" for a, k in LKF_OF.items()})
COLOR.update({"lewi-k2": "#e3eef8", "lewi-k4": "#d7e6f4", "lewi-k8": "#cee0f1"})
PANELS["pkf8"] = dict(cells={c: list(LKF_ARMS) for c in LK_CELLS8}, cohort=PANEL_COHORT, ns=PANEL_N8B, profile="c64k", spaces="startkf8",
                      note="amendment 17: the development refreeze of the three fine K arms at the 64,000-step common profile - the depth-4 "
                           "tree, the reuse DAG and the high-arity node at 2^9 / 2^12 / 2^15 on 6016-6019, T = 32, from LeWi's start range")
PANELS["pkf5"] = dict(cells={"tree16": list(LKF_ARMS)}, cohort=PANEL_COHORT, ns=PANEL_N, profile="d5", spaces="startkf5",
                      note="amendment 17: the development refreeze of the three fine K arms on the depth-5 tree (d = 16, 64x8, 128,000 "
                           "steps) at 2^11 / 2^14 / 2^17 on 6016-6019, T = 32, from LeWi's depth-5 start range")
PANELS["lkf8"] = dict(cells={c: list(LKF_ARMS) for c in SUITE8}, cohort=COHORT, cell_cohort={"rand8": COHORT_RAND_S}, ns=WINDOW8B,
                      profile="c64k", spaces="p8b", retain=True,
                      note="amendment 17: the three fine K arms on the seven d = 8 workloads (the sampled graphs on 32 instances), full grid "
                           "2^9..2^15, instances 6000-6015, T = 32, the arms' frozen ranges under the p8b set with LeWi's trial key; "
                           "init.pt and best.pt retained")
PANELS["lkf5"] = dict(cells={"tree16": list(LKF_ARMS)}, cohort=COHORT, ns=WINDOW, profile="d5", spaces="p0", retain=True,
                      note="amendment 17: the three fine K arms on the depth-5 tree at d = 16 (64x8, 128,000 steps), 2^11..2^17, instances "
                           "6000-6015, T = 32, the arms' frozen ranges under the p0 set with LeWi's trial key; checkpoints retained")
PANELS_LKF = ["pkf8", "pkf5", "lkf8", "lkf5"]


# Amendment 16 (decision 2026-09-23; 18-main-d8 intake/sensitivity-addendum.md): the parameter sensitivity on the canonical full
# grid. The radius-multiplier arms (lewi-rad05 ... lewi-rad2) and the initialization-scale arms (lewi-init05, lewi-init2) of amendment 13
# on the depth-4 tree and the depth-4 Fourier tree, instances 6000-6015, the full grid 2^9..2^15, T = 32, on the adopted arm's
# initialization (rule in5), paired with the adopted LeWi of 18-main-d8's ledger. The reduced-grid panels (abl8rad-in5, abl8init-in5 /
# abl8init: instances 6000-6007 at 2^11 / 2^13 / 2^15, the same contract) are reused on their cells (manifests/sens_full_cells.csv,
# kind = reused); the finals sens8rad / sens8init run the other 88 cells per workload (kind = new). The embedding dimension is complete
# on the full grid (amendment 14: lms8 / lms5). Development pilot pr8: the six arms on 6016-6019 at 2^9 / 2^12 / 2^15 from LeWi's start
# range (space set startr8: copies of LeWi's start file under the arms' names), T = 32; then the decision rule of amendment 14
# (18-main-d8/sensitivity-full/src/freeze_decision.py) installs each arm's range under its name in p8b. The arms' own p8b files were
# installed on 2026-09-23 as verbatim copies of LeWi's p8b range, so the reduced-grid panels' plans resolve unchanged; a decision for an
# arm's own range replaces its file and the reused cells of that arm are then re-run under it.
SENS_FULL_ARMS = [*SENS_RAD_ARMS, *SENS_INIT_ARMS]
for _a in SENS_FULL_ARMS:
    SPACE_FILE[_a] = SPACE_FILE_C[_a] = _a


def _sens_full_pairs(kind: str) -> dict[str, set[tuple[int, int]]]:
    import csv
    out: dict[str, set[tuple[int, int]]] = {}
    with open(STUDY / "manifests" / "sens_full_cells.csv") as f:
        for r in csv.DictReader(f):
            if r["kind"] == kind:
                out.setdefault(r["cell"], set()).add((int(r["fs"]), int(r["n"])))
    return out


SENS_FULL_NEW = _sens_full_pairs("new")
SENS_FULL_METHOD = {**{a: SENS_IN5[a] for a in SENS_RAD_ARMS}, **INIT_IN5}
PANELS["pr8"] = dict(cells={c: list(SENS_FULL_ARMS) for c in SENS_CELLS}, cohort=PANEL_COHORT, ns=PANEL_N8B, profile="c64k", spaces="startr8",
                     arm_method=SENS_FULL_METHOD,
                     note="amendment 16: the development pilot of the six sweep arms (radius multiplier x 0.5 / 0.75 / 1.5 / 2, initialization "
                          "scale x 0.5 / 2) on the tree and the Fourier tree at 2^9 / 2^12 / 2^15 on 6016-6019, T = 32, from LeWi's start range")
PANELS["sens8rad"] = dict(cells={c: list(SENS_RAD_ARMS) for c in SENS_CELLS}, cohort=COHORT, ns=WINDOW8B, profile="c64k", spaces="p8b", retain=True,
                          arm_method={a: SENS_IN5[a] for a in SENS_RAD_ARMS}, pairs=SENS_FULL_NEW,
                          note="amendment 16: the radius multiplier x {0.5, 0.75, 1.5, 2} on the tree and the Fourier tree, instances 6000-6015, "
                               "the full grid 2^9..2^15 on the 88 cells per workload the reduced grid did not run, T = 32, the arms' ranges under "
                               "the p8b set with LeWi's trial key; init.pt and best.pt retained")
PANELS["sens8init"] = dict(cells={c: list(SENS_INIT_ARMS) for c in SENS_CELLS}, cohort=COHORT, ns=WINDOW8B, profile="c64k", spaces="p8b", retain=True,
                           arm_method=INIT_IN5, pairs=SENS_FULL_NEW,
                           note="amendment 16: the initialization scale x {0.5, 2} on the tree and the Fourier tree, instances 6000-6015, the full "
                                "grid 2^9..2^15 on the 88 cells per workload the reduced grid did not run, T = 32, the arms' ranges under the p8b "
                                "set with LeWi's trial key; init.pt and best.pt retained")
# Amendment 16b (decision 2026-09-23, "do the rerun"): the 40 reduced-grid cells of the initialization-scale sweep on which the input rule
# does not fire were kept from the record's abl8init (built before the rule; bit for bit the in5 build there, the two controls verified). They are
# re-run under the in5 arm inside sens8init (manifests/sens_init_rerun_cells.csv = the reused cells minus the cells the rule changes; 80
# conditions, 2,560 units), so every row of the initialization sweep on the full grid comes from an in5 panel: sens8init (the 88 new cells per
# workload + these 40) and abl8init-in5 (the 8 cells the rule changes). The radius sweep's reused cells (abl8rad-in5, an in5 panel) stand.
def _sens_init_rerun_pairs() -> dict[str, set[tuple[int, int]]]:
    import csv
    out: dict[str, set[tuple[int, int]]] = {}
    with open(STUDY / "manifests" / "sens_init_rerun_cells.csv") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["cell"], set()).add((int(r["fs"]), int(r["n"])))
    return out


SENS_INIT_RERUN = _sens_init_rerun_pairs()
PANELS["sens8init"]["pairs"] = {c: SENS_FULL_NEW.get(c, set()) | SENS_INIT_RERUN.get(c, set()) for c in SENS_CELLS}
PANELS["sens8init"]["note"] += ("; amendment 16b: the 40 reduced-grid cells the input rule leaves alone (kept from the record's abl8init until "
                                "2026-09-23) re-run here under the in5 arm at the authors' word (80 conditions, 2,560 units)")
PANELS_SENS = ["pr8", "sens8rad", "sens8init"]


