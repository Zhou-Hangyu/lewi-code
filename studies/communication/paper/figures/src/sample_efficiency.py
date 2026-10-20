"""Figure: LeWi improves sample efficiency on CSFRBench and generalizes to
larger and unseen computational structures.

Caption - LeWi improves sample efficiency and generalizes to larger and
unseen computational structures. Sealed test NMSE against training-set
size n on six settings, one panel each, from left to right: the depth-4
tree (the primary compositional setting), the depth-5 tree (64x8 backbone,
128,000 steps, n = 2^11 ... 2^17), the depth-3 reuse DAG (a different
computational organization), 32 sampled depth-4 DAGs drawn from the
benchmark's graph distribution (common 32x4 profile), the flat modules
(four disjoint input pairs under an additive readout: modularity without
hierarchy) and the High-arity node (a single eight-input function, which
removes the low-arity compositional decomposition); 16 target instances
per setting except the 32 sampled depth-4 DAGs. Arithmetic means over
instances on a linear axis, each panel in its own frame (the frames are
matched between panels where the ranges allow; every panel carries its
own tick labels); shaded bands are 95% Student-t intervals for the
highlighted arms. LeWi
(orange) and Dense (black) are the learners the text compares; the dashed
green curves are the structure-informed references GraphOracle(0), which
receives the true active inputs, GraphOracle(1), which also receives the
first target level, GraphOracle(2), which receives the first two levels
(drawn where it differs from GraphOracle(all): the depth-5 tree and the
sampled depth-4 DAGs), and GraphOracle(all), which receives all target
connectivity; on the flat modules GraphOracle(1) already receives the
whole graph and is drawn as GraphOracle(all), and on the high-arity node every
rung coincides. On the depth-5 tree and the sampled depth-4 DAGs GraphOracle(1)
and GraphOracle(2) come from 12-extra wave 2 (the same sealed instances,
frozen spaces and trial lists as the 08-main GraphOracle(all) blocks), so
all four rungs of the ladder are drawn there. The generic structured and
sparse baselines (l0, RigL, BIMT, l1, l2, Fiedler, neuron positions; the
two rightmost settings ran l0, l1 and l2) are drawn in grey; the dense
node carries the complete roster, its RigL, BIMT, Fiedler and
neuron-positions curves coming from 12-extra wave 1 under the identical
protocol. The appendix figures learning_curves_all and
sample_efficiency_aulc identify each baseline and give the paired AULC
differences. On the depth-4 tree, the depth-3 reuse DAG and the flat
modules LeWi is ahead of every baseline on every instance (16/16) and
covers 51-81% of the Dense-to-GraphOracle(all) gap; on the high-arity node it
stays ahead of Dense on every instance (AULC difference -0.098 [-0.109,
-0.088]) but its margins over the baselines shrink to 0.011-0.036
(RigL -0.011 [-0.019, -0.004], 13/16; l0 -0.015 [-0.022, -0.009], 14/16),
consistent with LeWi being most useful when the target admits a low-arity
decomposition. At depth 5, where LeWi's definition and search space were
fixed beforehand, LeWi beats Dense, l0 and l1 on every instance (median
AULC 0.869 against 0.949, 0.913 and 0.925) while GraphOracle(all) (0.451)
marks how much of the deeper target no learned-wiring method reaches; the
Dense mean at n = 2^12 lies above the frame (1.4 x 10^7; a caret above the
frame's top edge marks it; every marked mean of a build is listed with its
value in <out>/marks/<stem>.csv) because one Dense winner collapsed to a
constant predictor whose retained weights blow up on rare test inputs, and
the protocol enters it unclipped. On the sampled depth-4 DAGs LeWi beats Dense,
l1 and l0 on all 32 instances (mean AULC 0.257 against 0.501, 0.389 and
0.346), is level with GraphOracle(0) (0.258) and covers 62% of the
Dense-to-GraphOracle(all) gap.

Data: data/sample_efficiency/curves.csv - one sealed test read per
selected winner (T = 32 Sobol trials per method, instance and n; winner =
lowest validation NMSE), harvested from the 08-main ledger by
src/harvest_08main.py and verified against the campaign's contract
analysis and the paper-facts tables (out/verification_08main.md);
data/sample_efficiency/dense_node_12extra.csv - the high-arity node's RigL,
BIMT, Fiedler and neuron-positions arms, which 08-main did not run, from
panel E of the 12-extra wave-1 campaign under the identical protocol
(src/harvest_12extra.py dense_node; sealed 2026-09-04);
data/sample_efficiency/oracle_rungs_12extra.csv - GraphOracle(1) and
GraphOracle(2) on the depth-5 tree and the random graphs from 12-extra
wave 2 (src/harvest_12extra.py oracle_rungs). The paired AULC forests this
figure carried until 2026-09-04 are the appendix stem
sample_efficiency_aulc; the two rightmost panels were the generalization
figure until 2026-09-04 (kept as a companion stem); the third panel was
the depth-4 Fourier tree until 2026-09-14, when the authors asked for the
flat modules in its place - the Fourier tree keeps its place in
learning_curves_all, in the AULC table and in the mechanism figure. One
row of six panels, built for the full text width (5.5 x 1.9 in) at the
authors' request of 2026-09-04, so the type renders at its set size when
included at \\linewidth.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

import paperstyle as ps
from figstyle import DATA, save

MAIN = [["tree_functional_anova", "tree_functional_anova:depth4", "dag_functional_anova"],
        ["rand_functional_anova", "flat_functional_anova", "dense_fourier"]]
FLAT_MAIN = [wl for row in MAIN for wl in row]
EXCLUDE = {"lewi_reg-v05"}          # the gate-penalty variant is an ablation arm
XLIM = (-0.72, 0.62)        # the forests' frame: every paired difference except the degenerate depth-5 Dense instance fits
# The curve standard (decision, 2026-09-18): arithmetic means on a linear NMSE axis, a frame per panel, the frames
# matched where possible. A panel takes the lowest of FRAME_TOPS that holds every mean and every band it draws, so panels
# of similar range get the same scale and heights compare between them; every panel carries its own tick labels. A mean
# above the largest frame (a degenerate winner) is marked by a caret at the frame's top edge, the line and the band skip
# that point, and the build log gives its value.
FRAME_TOPS = (0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.05)
FRAME_STEP = {0.01: 0.005, 0.02: 0.01, 0.03: 0.01, 0.05: 0.02, 0.1: 0.05, 0.15: 0.05, 0.2: 0.1, 0.3: 0.1, 0.4: 0.2,
              0.5: 0.25, 0.6: 0.2, 0.8: 0.25, 1.05: 0.5}
YMAX = FRAME_TOPS[-1]
FRAME_HEADROOM = 1.02       # a panel's highest drawn value stays 2% under its frame's top, so no marker sits on the edge
TITLE_PAD = 5.5             # points between a panel and its title: room for the carets, which stand above the frame
MARKS: list[dict] = []      # every mean marked above a frame in this build; figstyle.save writes them beside the figure
# main-text weights of the five highlighted arms; the grey family is thin
MAIN_WEIGHT = {ps.LEWI: 1.7, "noreg": 1.4, "graph_oracle(levels=all)": 1.3,
               "graph_oracle(levels=2)": 1.3, "graph_oracle(levels=1)": 1.3, "graph_oracle(levels=0)": 1.3}
REFERENCES = ("graph_oracle(levels=0)", "graph_oracle(levels=1)", "graph_oracle(levels=2)",
              "graph_oracle(levels=all)")
GENERIC_LINE = {"linewidth": 0.7, "alpha": 0.6}
TICK = 6.4
TITLE_SIZE = 7.0
# The main-text figure of the d = 8 build leaves neuron positions to the appendix grid learning_curves_all (decision,
# 2026-09-18): the arm lies far above every other curve there and would set the panels' frames.
MAIN_TEXT_EXCLUDE = {"erb_positions"} if DATA.name.endswith(("-d8", "-7000")) else set()     # the d = 8 builds (decision 2026-09-18)
REPAIRS = DATA / "sample_efficiency" / "input_cutoff_repairs.csv"
TITLE_WRAP = 13                 # characters per title line in the one-row layout


def load() -> pd.DataFrame:
    """The sealed 08-main curves plus the 12-extra additions under the same
    protocol (same columns, marked by `source`): the dense node's missing
    roster arms (wave 1) and the GraphOracle middle rungs on the depth-5 tree
    and the random graphs (wave 2)."""
    df = pd.read_csv(DATA / "sample_efficiency/curves.csv")
    for name in ("dense_node_12extra.csv", "oracle_rungs_12extra.csv"):
        extra = DATA / "sample_efficiency" / name
        if extra.exists():
            df = pd.concat([df, pd.read_csv(extra)], ignore_index=True)
    df = df[(df.sigma_sq == 0.01) & ~df.arm.isin(EXCLUDE)]
    return df[df.workload.isin(ps.WORKLOAD)]


def with_repairs(df: pd.DataFrame, tag: str, verbose: bool = False) -> pd.DataFrame:
    """The rows with the input-cutoff repairs (decision, 2026-09-18): where the data folder holds
    sample_efficiency/input_cutoff_repairs.csv (18-main-d8/input-cutoff, rule lewi-v1-rgm-in5: an input with fewer than
    five open first-transition connections at initialization has its position drawn again), the LeWi rows of the cells
    it names take the test NMSE of the validation-selected winner of the full 32-trial re-run under the rule. Every
    cell in which the rule changes the initialization is replaced, better or worse; every other cell is the record bit
    for bit by the rule's construction. Without the file the rows are the record's. Since 2026-09-20 the twin's ledger
    carries the rule itself (18-main-d8/ledger/d8/repairs.csv keeps the record's rows) and the file is absent: every
    stem draws the adopted arm from curves.csv, and this overlay does nothing."""
    if not REPAIRS.exists():
        return df
    fix = pd.read_csv(REPAIRS)
    key = ["workload", "arm", "function_seed", "n"]
    merged = df.merge(fix[key + ["test_nmse", "rule"]].rename(columns={"test_nmse": "repaired"}), on=key, how="left")
    hit = merged.repaired.notna()
    if int(hit.sum()) != len(fix):
        raise SystemExit(f"[{tag}] {len(fix) - int(hit.sum())} repaired cells have no row in the figure data")
    if verbose:
        for r in merged[hit].itertuples():
            print(f"[{tag}] repaired cell: {ps.TITLE[r.workload]} instance {r.function_seed} n = {r.n}: {r.test_nmse:.4g} -> {r.repaired:.4g} ({r.rule})")
    per = merged[hit].groupby("workload").size()
    print(f"[{tag}] input-cutoff repairs applied: {int(hit.sum())} LeWi cells (" + ", ".join(f"{ps.TITLE[w]} {k}" for w, k in per.items()) + ")")
    merged.loc[hit, "test_nmse"] = merged.loc[hit, "repaired"]
    return merged.drop(columns=["repaired", "rule"])


def main_text(df: pd.DataFrame) -> pd.DataFrame:
    """The arms of the main-text figures (MAIN_TEXT_EXCLUDE)."""
    return df[~df.arm.isin(MAIN_TEXT_EXCLUDE)]


def per_instance_aulc(df: pd.DataFrame, window=None) -> pd.DataFrame:
    """AULC per (workload, arm, instance) over the workload's own window
    (or a given one)."""
    rows = []
    for (wl, arm, seed), g in df.groupby(["workload", "arm", "function_seed"]):
        win = window if window is not None else ps.WINDOW_OF[wl]
        g = g.set_index("n").reindex(win)
        if g.test_nmse.isna().any():
            continue
        rows.append({"workload": wl, "arm": arm, "function_seed": seed,
                     "aulc": ps.aulc(win, g.test_nmse.to_numpy())})
    return pd.DataFrame(rows)


def generic_label(arms=None) -> str:
    names = [ps.NAME[a] for a in ps.GENERIC if arms is None or a in arms]
    return "Baselines: " + ", ".join(names)


def line_style(arm: str, palette: str) -> tuple[dict, bool]:
    """Line keywords and whether the arm gets a band. `main` = the five
    highlighted arms in their colours and the baselines as one light
    grey family; `full` = every arm in its own colour (the appendix)."""
    if palette == "main" and arm in ps.GENERIC:
        return {"color": ps.GRAY[arm], "linestyle": "-", "solid_capstyle": "round",
                "zorder": 2, **GENERIC_LINE}, False
    kw = ps.style(None, arm)
    if palette == "main":
        kw["linewidth"] = MAIN_WEIGHT[arm]
        kw["zorder"] = 3
        return kw, True
    return kw, arm in ps.BANDED


def curve_arms(df: pd.DataFrame, palette: str) -> list[str]:
    present = set(df.arm)
    if palette == "main":
        order = [a for a in ps.CURVE_ORDER if a in ps.GENERIC] + list(REFERENCES) + ["noreg", ps.LEWI]
    else:
        order = list(ps.CURVE_ORDER)
    return [a for a in order if a in present]


def panel_frame(df: pd.DataFrame, window, palette: str = "main") -> tuple[float, float]:
    """The panel's frame: 0 to the lowest of FRAME_TOPS that holds every mean and every band the panel draws (a mean
    above the largest frame is marked, not framed)."""
    top = 0.0
    for arm in curve_arms(df, palette):
        st = ps.curve_stats(df[df.arm == arm])
        st = st[st.n.isin(window) & (st["mean"] <= YMAX)]
        if st.empty:
            continue
        top = max(top, float(st.hi.max()) if line_style(arm, palette)[1] else float(st["mean"].max()))
    # a band whose end lies above the largest frame is cut by that frame (its mean is drawn, so no caret marks it)
    return (0.0, next((t for t in FRAME_TOPS if t >= FRAME_HEADROOM * top), YMAX))


def frame_axis(ax, ylim: tuple[float, float]) -> None:
    """The frame's ticks: fixed per frame, so two panels with the same frame read the same."""
    ax.set_ylim(*ylim)
    step = FRAME_STEP.get(ylim[1])
    if step:
        ax.set_yticks(np.arange(0.0, ylim[1] + 1e-9, step))


def mark_above(ax, n: float, color: str, ylim: tuple[float, float], slot: int = 0, of: int = 1) -> None:
    """A caret above the frame's top edge over the n of a mean that lies above the frame. It stands outside the frame, so
    it hides no data at the top of the panel; carets that share an n stand side by side (slot of `of`)."""
    shift = (slot - (of - 1) / 2) * 4.2 / 72
    ax.plot([n], [ylim[1]], marker="^", markersize=3.4, color=color, markeredgewidth=0, zorder=6, clip_on=False,
            transform=ax.transData + ScaledTranslation(shift, 2.4 / 72, ax.figure.dpi_scale_trans))


def draw_curves(ax, df: pd.DataFrame, window, *, ylabel: bool, palette: str = "main",
                xlabel: bool = False, frame: tuple[float, float] | None = None) -> list[str]:
    """One workload's curves in the panel's own frame (panel_frame, or the given one). A mean above the frame is
    marked by a caret at the frame's top edge, never silently clipped, and the line and the band skip that point.
    Returns the marked cells for the build log."""
    off, marks = [], []
    ylim = frame or panel_frame(df, window, palette)
    panel = ps.TITLE.get(df.workload.iloc[0], str(df.workload.iloc[0])) if "workload" in df and df.workload.nunique() == 1 else ""
    for arm in curve_arms(df, palette):
        rows = df[df.arm == arm]
        st = ps.curve_stats(rows)
        st = st[st.n.isin(window)].copy()
        kw, band = line_style(arm, palette)
        high = st["mean"] > ylim[1]
        for _, row in st[high].iterrows():
            values = rows[rows.n == row.n].test_nmse.to_numpy()
            above = np.sort(values[values > ylim[1]])[::-1]
            off.append(f"{panel + ': ' if panel else ''}{ps.NAME[arm]} at n = {int(row.n)}: mean {row['mean']:.4g} above the frame (top {ylim[1]:g}), "
                       f"{len(above)} of {len(values)} instances above it, the highest at {above[0]:.4g}")
            marks.append((float(row.n), kw["color"]))
            MARKS.append({"panel": panel, "arm": ps.NAME[arm], "n": int(row.n), "mean": float(row["mean"]), "frame_top": float(ylim[1]),
                          "instances_above": int(len(above)), "instances": int(len(values)), "highest": float(above[0])})
        st.loc[high, ["mean", "lo", "hi"]] = np.nan
        if band:
            ax.fill_between(st.n, st.lo.clip(lower=0.0), st.hi, color=kw["color"], alpha=0.13,
                            linewidth=0, zorder=1)
        heavy = kw["linewidth"] >= 1.3
        ax.plot(st.n, st["mean"], marker="o" if (heavy or palette == "full") else None,
                markersize=1.9 if heavy else 1.4, markeredgewidth=0, **kw)
    for n in sorted({m[0] for m in marks}):                    # carets that share an n stand side by side
        same = [c for x, c in marks if x == n]
        for slot, color in enumerate(same):
            mark_above(ax, n, color, ylim, slot, len(same))
    ps.n_axis(ax, window, label=xlabel)
    frame_axis(ax, ylim)
    ps.tidy(ax)
    ax.tick_params(labelsize=TICK)
    if ylabel:
        ax.set_ylabel("Test NMSE", labelpad=2)
    return off


def legend_entries(palette: str, arms=None) -> tuple[list, list]:
    """Handles and canonical names for one shared legend."""
    handles, labels = [], []
    if palette == "main":
        for arm in (ps.LEWI, "noreg", *REFERENCES):
            if arms is not None and arm not in arms:
                continue
            kw, _ = line_style(arm, palette)
            handles.append(Line2D([], [], **kw))
            labels.append(ps.NAME[arm])
        handles.append(Line2D([], [], color=ps.GRAY["l1"], linewidth=0.9))
        labels.append(generic_label(arms))
        return handles, labels
    order = [ps.LEWI, "noreg", "graph_oracle(levels=0)", "graph_oracle(levels=1)",
             "graph_oracle(levels=2)", "graph_oracle(levels=all)", "l0hc", "rigl", "bimt",
             "l1", "l2", "fiedler", "erb_positions"]
    for arm in order:
        if arms is not None and arm not in arms:
            continue
        kw, _ = line_style(arm, palette)
        handles.append(Line2D([], [], **kw))
        labels.append(ps.NAME[arm])
    return handles, labels


def shared_legend(fig, palette: str, arms=None, *, rows=(0.045, 0.0), handlelength=2.0,
                  columnspacing=1.3) -> None:
    """The main palette's legend: one centred row for the highlighted arms,
    one for the grey family (each row must stay inside the frame, or the
    tight bounding box widens the PDF)."""
    handles, labels = legend_entries(palette, arms)
    k = len(handles) - 1
    common = dict(loc="lower center", frameon=False, fontsize=TICK, handlelength=handlelength,
                  handletextpad=0.5, columnspacing=columnspacing, borderaxespad=0.0)
    fig.legend(handles[:k], labels[:k], bbox_to_anchor=(0.5, rows[0]), ncol=k, **common)
    fig.legend(handles[k:], labels[k:], bbox_to_anchor=(0.5, rows[1]), ncol=1, **common)


def draw_forest(ax, aulc: pd.DataFrame, *, labels: bool, order: list[str] | None = None) -> list[str]:
    """Paired AULC(LeWi) - AULC(arm) per instance. An instance outside the
    frame is drawn as an arrow at the frame edge; a mean whose interval leaves
    the frame is replaced by a hollow diamond at the median (the 08-main
    convention for the degenerate depth-5 Dense winner). Returns the off-scale
    notes for the build log."""
    off = []
    lewi = aulc[aulc.arm == ps.LEWI].set_index("function_seed").aulc
    order = list(order or ps.FOREST_ORDER)          # the figure's row grid: the arms that exist in any of its panels
    rows = [a for a in order if a in set(aulc.arm)]
    ys = {a: len(order) - 1 - order.index(a) for a in rows}
    rng = np.random.default_rng(0)
    for arm in rows:
        other = aulc[aulc.arm == arm].set_index("function_seed").aulc
        d = (lewi - other.reindex(lewi.index)).to_numpy()
        m, lo, hi = ps.t_interval(d)
        med = float(np.median(d))
        y = ys[arm]
        color = ps.COLOR[arm]
        inside = (d >= XLIM[0]) & (d <= XLIM[1])
        ax.scatter(d[inside], y + rng.uniform(-0.22, 0.22, int(inside.sum())), s=4, color=color,
                   alpha=0.35, linewidths=0, zorder=2)
        for value in d[~inside]:
            left = value < XLIM[0]
            ax.plot([XLIM[0] + 0.012 if left else XLIM[1] - 0.012], [y], marker="<" if left else ">",
                    markersize=3.4, color=color, alpha=0.7, linewidth=0, zorder=2)
            off.append(f"{ps.NAME[arm]}: one instance at {value:+.3g} drawn at the frame edge")
        if XLIM[0] <= lo and hi <= XLIM[1]:
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.1, zorder=3, solid_capstyle="butt")
            ax.plot([m], [y], marker="D", markersize=3.2, color=color,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=4)
        else:
            ax.plot([med], [y], marker="D", markersize=3.4, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=0.8, linestyle="none", zorder=4)
            off.append(f"{ps.NAME[arm]}: mean {m:+.3g} [{lo:+.3g}, {hi:+.3g}] off scale; "
                       f"median {med:+.3f} drawn as a hollow diamond")
    ax.axvline(0, color="#444444", linewidth=0.6, zorder=1)
    ax.set_xlim(*XLIM)
    ax.set_xticks([-0.6, -0.3, 0, 0.3, 0.6])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_yticks([len(order) - 1 - i for i in range(len(order))])
    if labels:
        ax.set_yticklabels([ps.NAME[a] for a in order])
        for tick, arm in zip(ax.get_yticklabels(), order):
            tick.set_color(ps.COLOR[arm])
    else:
        ax.set_yticklabels([])
    ps.tidy(ax, grid="x")
    ax.tick_params(labelsize=TICK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    return off


def report(aulc: pd.DataFrame, workloads, tag: str) -> None:
    """The captions' claims, checked at build time: instances favoring LeWi
    over each baseline, the paired gaps (mean, interval, median), the share of
    the Dense-to-reference gap covered, the arms' median AULCs; every arm that
    ran on the workload is listed."""
    for wl in workloads:
        a = aulc[aulc.workload == wl]
        if a.empty or ps.LEWI not in set(a.arm):
            print(f"[{tag}] {ps.TITLE[wl]}: no rows in this data folder (the setting has no such workload)")
            continue
        lewi = a[a.arm == ps.LEWI].set_index("function_seed").aulc
        gaps, medians = {}, {}
        for arm in [ps.LEWI] + ps.FOREST_ORDER:
            if arm not in set(a.arm):
                continue
            mine = a[a.arm == arm].set_index("function_seed").aulc.reindex(lewi.index)
            medians[arm] = float(mine.median())
            if arm == ps.LEWI:
                continue
            d = (lewi - mine).to_numpy()
            gaps[arm] = (ps.t_interval(d), float(np.median(d)), int((d < 0).sum()))
        generic = {arm: v for arm, v in gaps.items() if arm in ps.GENERIC or arm == "noreg"}
        worst_arm = min(generic, key=lambda k: generic[k][2])
        ref = ("graph_oracle(levels=all)" if "graph_oracle(levels=all)" in gaps
               else "graph_oracle(levels=0)")
        dense = gaps["noreg"][0][0]
        share = ("no oracle arm on this workload (the all-active dense node)" if ref not in gaps
                 else f"{dense / (dense - gaps[ref][0][0]):.0%} of the Dense-to-{ps.NAME[ref]} gap covered (means)"
                 if abs(dense) < 2 else "gap share undefined (Dense mean off scale)")
        n = len(lewi)
        pairs = "; ".join(f"{ps.NAME[arm]} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (median {med:+.3f}; {k}/{n})"
                          for arm, ((m, lo, hi), med, k) in gaps.items())
        meds = ", ".join(f"{ps.NAME[arm]} {v:.3f}" for arm, v in medians.items())
        print(f"[{tag}] {ps.TITLE[wl]}: {len(gaps) + 1} arms, {n} instances; fewest instances favoring "
              f"LeWi over a baseline {generic[worst_arm][2]}/{n} ({ps.NAME[worst_arm]}); {share}; "
              f"median AULC {meds}; LeWi - arm: {pairs}")


def figure(df: pd.DataFrame, stem: str) -> None:
    """One row of six panels at the full text width, each in its own frame with its own tick labels; long titles wrap
    onto two lines above a common baseline, and the legend is a two-row block below the shared x label (highlighted
    arms, then the grey family)."""
    df = main_text(df)
    panels = present(df, FLAT_MAIN)         # a setting without a workload (the 7000-series set has no depth-5 tree) drops its panel
    fig = plt.figure(figsize=(5.5, 1.9))
    gs = GridSpec(1, len(panels), figure=fig, left=0.078, right=0.992, top=0.8, bottom=0.34, wspace=0.5)
    for c, wl in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        off = draw_curves(ax, df[df.workload == wl], ps.WINDOW_OF[wl], ylabel=c == 0, palette="main")
        ax.set_title(textwrap.fill(ps.TITLE[wl], TITLE_WRAP), fontsize=TITLE_SIZE, pad=TITLE_PAD,
                     linespacing=1.0)
        for cell in off:
            print(f"[{stem}] mean above the frame marked, {cell}")
    fig.text((gs.left + gs.right) / 2, 0.19, "Training samples $n$", ha="center",
             va="center", fontsize=plt.rcParams["axes.labelsize"])
    shared_legend(fig, "main", arms=set(df[df.workload.isin(panels)].arm), rows=(0.075, 0.0),
                  handlelength=1.6, columnspacing=1.0)
    save(fig, stem)


def present(df: pd.DataFrame, workloads: list[str]) -> list[str]:
    """The workloads of a panel list that the data folder carries (in the list's order)."""
    have = set(df.workload)
    return [wl for wl in workloads if wl in have]


def main() -> None:
    ps.apply_rc()
    df = load()
    figure(df, "sample_efficiency")
    report(per_instance_aulc(df), present(df, FLAT_MAIN), "sample_efficiency")


if __name__ == "__main__":
    main()
