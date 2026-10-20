"""Figure: the world-model transfer figure on every task; in the SWM-16 data folder the nineteen tasks of
16-main-final (the seventeen claim-bearing ones first, then Humanoid walk and OGBench cube marked not claim-bearing)
with all nine arms; historically sixteen tasks with the LeWi-v3 arm.

Caption - Rollout sample efficiency on sixteen world-modeling tasks. Sixteen-step
open-loop rollout error (feature-space NMSE on held-out test windows) against the
number of training transitions, for the dense MLP backbone (black), its
hard-concrete L0 variant (blue), LeWi (orange) and LeWi with learned frames and the
straight-through gate gradient (crimson; the l1-regularized variant is the thin
line). The first three panels are the tasks of the world-model transfer study
(PushT, OGBench cube, Humanoid walk; 256x4 backbone, 2,000 to 128,000
transitions); the remaining thirteen are the held-out bank of the LeWi
world-model study (128x4 backbone, 8,192 to 524,288 transitions), on which every
arm was selected and paired under that study's protocol. Curves are means over 16
paired training seeds; bands are 95% Student-t intervals of the arithmetic mean (decision 2026-09-23; the SWM-16 build). On the
thirteen-task bank the frame + straight-through arm improves the paired rollout
AULC (mean log10 rollout NMSE over the window; negative favors the arm) over
L0 on every task, by 0.12 (Walker) to 2.79 (Pendulum) log10 units, and over
LeWi on every task, by 0.04 (Walker) to 0.86 (Cartpole). On the three
transfer-study tasks (the arm's amendment-A6 extension, the study's own trapezoid
AULC over log2 n) it improves on Dense by 0.07 (Humanoid walk) to 0.20 (OGBench
cube) and on L0 by 0.09 to 0.23, each on 16 of 16 seeds, and on LeWi by 0.05 on
PushT (16 of 16 seeds), 0.02 on OGBench cube (13 of 16) and 0.005 on Humanoid walk
(12 of 16; interval [-0.010, -0.001]).

Built for inclusion at the full text width. Data: data/world_model_transfer_all/
cells.csv and paired.csv, harvested by src/harvest_world_model_all.py from the
sealed 07-external-swm summary (+ its amendment-A6 arm) and from the 09-swm-lewi
held-out bank beside an earlier campaign's finals; each bank keeps its own bootstrap and AULC
statistic (07: trapezoid over log2 n of log10 NMSE, normalised; 09: the mean).

Wave 2 (13-main, data-v2/world_model_transfer_all/, runner section 8; the
07/09 records and the LeWi-v3 arm are superseded): the thirteen tasks of
13-main's own world-model lane (8,192 ... 524,288 transitions, 128x4,
128,000 steps, 4 paired held-out seeds, every arm with the plain readout of
CSFRBench v1.0) for Dense, hard-concrete L0 and LeWi-v1 (heavy, banded) with
l1 thin - the arms of the main-text figure, whose six tasks are among the
thirteen; means over the 4 seeds with two-sided 95% Student-t bands, the y
ticks derived from the data; 4 x 4 grid, three cells empty. The paired
contrasts printed at build time are LeWi-v1 minus Dense, L0 and l1 per task
(the paired AULC over the 7 sizes of log10 rollout NMSE at h = 16, Student-t
interval over the seeds; 13-main/src/harvest_swm_v10.py, paired.csv).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import paperstyle as ps
from figstyle import CAMPAIGN, DATA, save

WAVE2 = CAMPAIGN.name != "08-main"
ARM = "lewi_v3_fst"
PANELS = [("07", "pusht", "PushT"), ("07", "ogbcube", "OGBench cube"), ("07", "humanoid", "Humanoid walk"),
          ("09", "acrobot", "Acrobot"), ("09", "ballincup", "Ball in cup"), ("09", "cartpole", "Cartpole"),
          ("09", "cheetah", "Cheetah"), ("09", "finger", "Finger"), ("09", "hopper", "Hopper"),
          ("09", "ogbmaze", "OGBench maze"), ("09", "pendulum", "Pendulum"), ("09", "pendulumctl", "Pendulum (Gymnasium)"),
          ("09", "pointmaze", "Point maze"), ("09", "reacher", "Reacher"), ("09", "tworoom", "Two-room"),
          ("09", "walker", "Walker")]
ARMS = ["l1", "l0hc", "dense", "lewi", ARM]
STYLE_OF = {"dense": "noreg", "l1": "l1", "l0hc": "l0hc", "lewi": ps.LEWI, ARM: "lewi-v3"}
WEIGHT = {"lewi": 1.6, ARM: 1.6, "dense": 1.3, "l0hc": 1.2}       # the four compared arms are heavy and banded
TICKS = {"07": [2000, 8000, 32000, 128000], "09": [8192, 32768, 131072, 524288]}
# wave 2: 13-main's thirteen tasks (bank "13"), the record arm ids (lewi-v1 = ps.LEWI)
TASKS_V2 = [("acrobot", "Acrobot"), ("ballincup", "Ball in cup"), ("cartpole", "Cartpole"), ("cheetah", "Cheetah"),
            ("finger", "Finger"), ("hopper", "Hopper"), ("ogbmaze", "OGBench maze"), ("pendulum", "Pendulum"),
            ("pendulumctl", "Pendulum (Gymnasium)"), ("pointmaze", "Point maze"), ("reacher", "Reacher"),
            ("tworoom", "Two-room"), ("walker", "Walker")]
BANK_V2 = CAMPAIGN.name.split("-")[0]          # the campaign's bank id in the harvest: "13" (13-main), "14" (14-main)
if WAVE2:
    PANELS = [(BANK_V2, task, title) for task, title in TASKS_V2]
    ARMS = ["l1", "l0hc", "dense", ps.LEWI]
    STYLE_OF = {"dense": "noreg", "l1": "l1", "l0hc": "l0hc", ps.LEWI: ps.LEWI}
    WEIGHT = {ps.LEWI: 1.6, "dense": 1.3, "l0hc": 1.2}
    TICKS[BANK_V2] = TICKS["09"]
# Trial variant (design request 2026-09-18; 18-main-d8/swm-rollout): with CSFR_SWM_TRIAL=1 the builder reads the rollout study's
# trial rows (cells_trial.csv / paired_trial.csv beside the record's files: the record with the cells repaired by the
# validation-only re-selection rule, the same for every arm) and writes the stem with the suffix _trial. The record's stem
# and files stay untouched; adoption is the authors' decision.
TRIAL = os.environ.get("CSFR_SWM_TRIAL") == "1"
# The input rule in5 is adopted (decision 2026-09-23): the default tables carry the 30 substituted seeds (harvest_swm16.py); with
# CSFR_SWM_SEALED=1 the builder reads cells_sealed.csv / paired_sealed.csv (the record without the substitution) and writes the stem
# with the suffix _sealed (provenance only).
SEALED = os.environ.get("CSFR_SWM_SEALED") == "1"
_SUFFIX = "_trial" if TRIAL else ("_sealed" if SEALED else "")
SWM16_META = DATA / "world_model_transfer_all" / "swm16.json"
SWM16 = SWM16_META.exists()          # the SWM-16 campaign (16-main-final/external): nineteen tasks, nine arms, one contract
FINAL = DATA.name == "data-final" and not SWM16     # the final arm (record id lewi) on the three transfer tasks and on 13-main's thirteen
M, LO, HI = ("rollout16_mean", "rollout16_lo", "rollout16_hi")   # every build: the arithmetic mean over the seeds (decision 2026-09-23)
if FINAL:                             # the sixteen panels of the 08-main build, the final arm on all of them
    ARMS = ["l1", "l0hc", "dense", "lewi"]
    STYLE_OF = {"dense": "noreg", "l1": "l1", "l0hc": "l0hc", "lewi": ps.LEWI}
    WEIGHT = {"lewi": 1.6, "dense": 1.3, "l0hc": 1.2}
    TICKS["13"] = TICKS["09"]
if SWM16:                             # the definitive campaign: every arm thin, LeWi / Dense / L0 heavy and banded
    _meta = json.loads(SWM16_META.read_text())
    _cb = list(_meta.get("claim_bearing") or _meta["tasks"])                  # the 17 counted tasks only: Humanoid walk and OGBench
    PANELS = [("16", t, _meta["titles"].get(t, t)) for t in _cb]              # cube dropped (decision 2026-09-23)
    ARMS = ["erb", "fiedler", "bimt", "rigl", "l2", "l1", "l0hc", "dense", "lewi-v1"]
    STYLE_OF = {"dense": "noreg", "l1": "l1", "l2": "l2", "l0hc": "l0hc", "rigl": "rigl", "bimt": "bimt", "fiedler": "fiedler",
                "erb": "erb_positions", "lewi-v1": ps.LEWI}
    WEIGHT = {"lewi-v1": 1.6, "dense": 1.3, "l0hc": 1.2}
    M, LO, HI = "rollout16_mean", "rollout16_lo", "rollout16_hi"     # decision 2026-09-23: the arithmetic mean over the seeds with its t band
    GRIDS = {t: list(g) for t, g in _meta["grids"].items()}
    TICKS["16"] = TICKS["09"]


def frame_of(sub: pd.DataFrame, m: str, lo: str, hi: str, arms) -> tuple[float, float, pd.DataFrame]:
    """The folder's off-scale rule for a log panel of arm means with bands: a
    mean more than ten times its arm's median over n (a seed that diverged
    inside the bootstrap) or a band beyond ten times the kept means does not
    set the frame; the frame is the hull of the kept means and bands, and the
    off-frame means are returned to be drawn at the frame edge."""
    kept, off = [], []
    for arm in arms:
        s = sub[(sub.arm == arm) & (sub[m] > 0)]
        if s.empty:
            continue
        med = float(s[m].median())
        inside = (s[m] <= 10 * med) & (s[m] >= med / 10)
        kept.append(s[inside])
        off.append(s[~inside])
    kept = pd.concat(kept) if kept else sub.iloc[0:0]
    off = pd.concat(off) if off else sub.iloc[0:0]
    if kept.empty:
        return float(sub[m].min()), float(sub[m].max()), off
    lo_m, hi_m = float(kept[m].min()), float(kept[m].max())
    lo_b = kept[lo][(kept[lo] > 0) & (kept[lo] >= lo_m / 10)]
    hi_b = kept[hi][kept[hi] <= hi_m * 10]
    return (float(lo_b.min()) if len(lo_b) else lo_m), (float(hi_b.max()) if len(hi_b) else hi_m), off


def draw_off_frame(ax, off: pd.DataFrame, m: str, lo: float, hi: float, title: str, notes: list[str]) -> None:
    for _, r in off.iterrows():
        above = float(r[m]) > hi
        if not above and float(r[m]) >= lo:          # inside the frame although it did not set it: drawn in place, no edge marker
            continue
        ax.plot([r.n], [hi if above else lo], marker="^" if above else "v", markersize=3.0, clip_on=False,
                color=ps.COLOR[STYLE_OF.get(r.arm, r.arm)], zorder=6, linestyle="none")
        notes.append(f"{title}: {r.arm} at n = {int(r.n):,} mean {float(r[m]):.3g} drawn at the frame edge")



def _fmt(n: int) -> str:
    if n < 1000:
        return f"{n}"
    return f"{n // 1000}k" if n < 1_000_000 else f"{n / 1e6:g}M"


def _yticks(lo: float, hi: float) -> list[float]:
    """Two to five round ticks inside [lo, hi]: 1-2-5 per decade, thinned to 1-3 then 1 when crowded."""
    decades = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    for pattern in ((1, 2, 5), (1, 3), (1,)):
        ticks = [m * 10.0 ** k for k in decades for m in pattern if lo <= m * 10.0 ** k <= hi]
        if len(ticks) <= 5:
            return ticks
    return ticks


def _ylabel(v: float, _pos=None) -> str:
    if v >= 1e-3:
        return f"{v:g}"
    k = int(np.floor(np.log10(v) + 1e-9)); m = v / 10.0 ** k
    return f"$10^{{{k}}}$" if abs(m - 1) < 1e-6 else f"${m:g}{{\\cdot}}10^{{{k}}}$"


def main() -> None:
    ps.apply_rc()
    path = DATA / f"world_model_transfer_all/cells{_SUFFIX}.csv"
    if WAVE2 and not path.exists():
        raise SystemExit("[world_model_transfer_all] no wave-2 data yet: run 13-main/src/harvest_swm_v10.py "
                         f"(missing {path.relative_to(DATA.parent)})")
    df = pd.read_csv(path)
    df["bank"] = df["bank"].astype(str).str.zfill(2)
    n_rows = max(4, (len(PANELS) + 3) // 4)            # four rows of four; SWM-16: seventeen panels in five rows, the last row panels in five
    fig, axes = plt.subplots(n_rows, 4, figsize=(5.5, 1.3 * n_rows))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.965, bottom=0.42 / n_rows, wspace=0.34, hspace=0.62)
    missing = []
    for ax in list(axes.flat)[len(PANELS):]:           # the spare panels of a partial last row
        ax.set_axis_off()
    off_notes: list[str] = []
    for ax, (bank, task, title) in zip(axes.flat, PANELS):
        sub = df[(df.bank == bank) & (df.task == task)]
        if sub.empty:
            missing.append((title, "every arm"))
            ax.set_axis_off()
            continue
        for arm in ARMS:
            s = sub[sub.arm == arm].sort_values("n")
            if s.empty:
                missing.append((title, arm))
                continue
            kw = ps.style(ax, STYLE_OF[arm])
            if arm in WEIGHT:
                kw["linewidth"] = WEIGHT[arm]
                kw["zorder"] = 3
                ax.fill_between(s.n, s[LO].clip(lower=s[M] / 50), s[HI], color=kw["color"], alpha=0.13, linewidth=0, zorder=1)   # a t bound at or below zero is clipped at 1/50 of the mean on the log axis
            ax.plot(s.n, s[M], marker="o", markersize=1.6, markeredgewidth=0, **kw)
        ns = sorted(sub.n.unique())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = [g for g in (GRIDS[task][0], GRIDS[task][2], GRIDS[task][4], GRIDS[task][6])] if SWM16 and task in GRIDS and len(GRIDS[task]) >= 7 else TICKS[bank]
        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt(n) for n in ticks])
        ax.set_xticks([], minor=True)
        ax.set_xlim(ns[0] / 1.5, ns[-1] * 1.5)
        ax.set_title(title, loc="left", pad=2)
        ps.tidy(ax)
        lo, hi, off = frame_of(sub[sub.arm.isin(ARMS)], M, LO, HI, ARMS)   # the folder's off-scale rule
        ax.set_ylim(lo / 1.3, hi * 1.3)
        ax.set_yticks(_yticks(lo, hi))
        draw_off_frame(ax, off, M, lo / 1.3, hi * 1.3, title, off_notes)
        ax.set_yticks([], minor=True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_ylabel))
        ax.tick_params(axis="both", labelsize=6.2, pad=1.5)
    # the x label under the last panel of every column (wave 2: the last row holds one panel)
    for c in range(4):                                 # under the lowest panel of every column
        ks = list(range(c, len(PANELS), 4))
        if ks:
            axes.flat[ks[-1]].set_xlabel("Training transitions $n$", labelpad=1.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Rollout NMSE ($h = 16$)", labelpad=1.5)
    ps.arm_legend(fig, [STYLE_OF[a] for a in ARMS], y=0.0)
    save(fig, "world_model_transfer_all" + _SUFFIX)
    for note in off_notes:
        print(f"[world_model_transfer_all] off scale: {note}")
    if missing:
        print(f"[world_model_transfer_all] panels without every arm: {sorted(set(m[0] for m in missing))} "
              f"(missing {sorted(set(m[1] for m in missing))})")
    # the caption's paired claims, printed at build time
    paired = pd.read_csv(DATA / f"world_model_transfer_all/paired{_SUFFIX}.csv")
    paired["bank"] = paired["bank"].astype(str).str.zfill(2)
    au = paired[paired.n.astype(str) == "aulc"]
    candidate, seeds, bank_id = ("lewi-v1", 16, "16") if SWM16 else (ps.LEWI, 4, BANK_V2) if WAVE2 else ("lewi", None, "09") if FINAL else (ARM, 16, "09")
    for bank, task, title in PANELS:
        rows = au[(au.bank == bank) & (au.task == task)]
        if rows.empty:
            continue
        print(f"[world_model_transfer_all] {title}: " + "; ".join(
            f"AULC({candidate}) - AULC({r.reference}) = {r.dlog10_mean:+.3f} [{r.dlog10_lo:+.3f}, {r.dlog10_hi:+.3f}] "
            f"better {int(r.better)}/{seeds if seeds else int(r.seeds)}" for _, r in rows.iterrows()))
    bank = au[au.bank == bank_id]
    for ref in (("dense", "l1", "l2", "l0hc", "rigl", "bimt", "fiedler", "erb") if SWM16 else ("dense", "l0hc", "l1") if (WAVE2 or FINAL) else ("l0hc", "lewi")):
        r = bank[bank.reference == ref]
        won = int((r.dlog10_hi < 0).sum())
        print(f"[world_model_transfer_all] bank {bank_id} vs {ref}: interval-backed wins on {won}/{len(r)} tasks; "
              f"range {r.dlog10_mean.min():+.3f} .. {r.dlog10_mean.max():+.3f}")


if __name__ == "__main__":
    main()
