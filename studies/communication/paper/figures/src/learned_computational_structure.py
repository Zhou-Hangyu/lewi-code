"""Figure: LeWi's target-aligned intermediate computations are both linearly
accessible and causally effective - the combined probe figure that
replaces the separate probe-heatmap and causal-use figures of the
2026-09-03 draft.

Caption - LeWi develops target-aligned intermediate computations that are
both linearly accessible and causally effective. Linear probes (left) on
the depth-4 tree at n = 2^15. Top: held-out ridge-probe R^2 for each
target intermediate (the same seven rows in the same order in both
blocks, grouped by target depth and named once, on the far left of the
figure: four level-1 intermediates at the bottom,
two level-2 above them, the root alone at the top; thin rules separate the
groups in both blocks, and the appendix figures give each intermediate its
number from the target graph of Figure 1) and hidden
layer (columns in both blocks) for Dense, LeWi and GraphOracle(all). Bottom: the same probe R^2
averaged over the intermediates at each target depth within each instance:
earlier target computations are read out earlier in the learner and deeper
ones later (LeWi's level-1, level-2 and level-3 curves peak at layers 2, 3
and 4 with R^2 0.81, 0.67 and 0.72, GraphOracle(all)'s at the same layers
with 0.87, 0.84 and 0.86, whereas Dense's peak at layers 1, 2 and 4 with
0.58, 0.39 and 0.37). Causal interventions (right) under joint
multi-variable DAS (Appendix B.8): the held-out counterfactual R^2 of the
patched network, the same quantity the probe block reports, so both blocks
are read as R^2; the two curve axes carry different ranges, because the
intervention values run below zero. Top: the patched R^2 for each target
intermediate and hidden layer, on a diverging colour scale centred at 0,
so a red cell marks a patched prediction farther from the teacher's
counterfactual target than that target's own mean - what happens when a
deep intermediate is patched at an early layer. Bottom: the patched R^2
averaged over the intermediates at each target depth within each instance,
a summary of where jointly addressable causal alignment is strongest for
each depth (LeWi: level 1 0.50 at layer 1, level 2 0.34 at layer 3, level
3 0.55 at layer 4; GraphOracle(all) 0.62, 0.49 and 0.72 at the same
layers; Dense 0.20, 0.15 and 0.29). The patched R^2 is not corrected for
the network's own unmodified prediction: that no-patch reference
R^2_no-patch, one constant per network and intermediate, and the
difference Delta R^2_cf = R^2_patch - R^2_no-patch are the appendix
figures causal_probe_detail and causal_use; the reference is negative for
the deeper intermediates, whose interventions move the target far from the
base output. Every value is the mean over 16 target instances, each
contributed by the validation-selected winner run of its block; bands are
95% Student-t intervals. Because the joint causal probe partitions one
shared representation among all target intermediates, individual causal
cells and the curves summarizing them are not independent estimates of
causal accessibility. Both blocks read the same validation-selected winner
checkpoints. The probe R^2 at initialization (R^{2,init}) and the
training-induced change Delta R^2 = R^{2,trained} - R^{2,init} are the
appendix figure linear_probe_detail; the raw patched and no-patch
counterfactual R^2 per intermediate are the appendix figure
causal_probe_detail.

Data: data/learned_structure/probe_values.csv (the campaign's per-instance
ridge-probe R^2 of every block's validation-selected winner run;
src/harvest_08main.py) and data/causal_nopatch/cells.csv (per-instance,
per-layer patched and no-patch counterfactual R^2 of the joint fits after
the frozen restart selection, with their difference delta_cf; the causal
block draws the patched column test_r2;
src/harvest_12extra.py causal_nopatch, from 12-extra wave 3b, checked
against data/causal_probe_detail/das_cells.csv and the campaign's
das_star.csv and das_scores.csv). Every plotted aggregate is written to
out/learned_computational_structure.csv at build time. The dimensions the
joint probe allocates, the raw patched and no-patch values and the
independent probe are the appendix figure causal_probe_detail; the causal
score against training-set size is the appendix figure causal_use. Built
for the full text width. The causal block switched from the raw patched
R^2 to Delta R^2_cf on 2026-09-05 (intake/wave-3.md of 12-extra) and back
to the patched R^2 on 2026-09-14 at the authors' request; neither the
header nor the curve axis repeats the metric, since both blocks report the
same R^2. The heatmap rows lost their intermediate numbers the same day
and are named by target depth instead; the names are plain black text
since 2026-09-22 (design decision), the level colours draw the curves only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import paperstyle as ps
from figstyle import DATA, OUT, save
from learned_structure import ARMS, N_SHOWN, load_embeddings, numbering

DAS_LAYERS = [1, 2, 3, 4]
PROBE_LAYERS = ["h1", "h2", "h3", "h4"]     # the hidden layers; the raw-input probe is left out
LAYER_LABEL = [str(k) for k in DAS_LAYERS]
DEPTH_OF = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3}   # Figure-1 number -> target depth
LEVEL_LABEL = {1: "Level 1", 2: "Level 2", 3: "Level 3"}
# target depth: amber, muted blue, burgundy - light to dark with depth; the
# colours draw the curves and their legend (the heatmap rows are named in
# plain black text, decision 2026-09-22)
LEVEL_COLOR = {1: "#d9a441", 2: "#5a86b8", 3: "#8b2635"}
# probe R^2 on a sequential scale, counterfactual R^2 on a diverging scale
# centred exactly at 0 (white); text switches to white on dark cells
SEQUENTIAL = plt.get_cmap("Blues")
DIVERGING = plt.get_cmap("RdBu")
LINEAR_NORM = Normalize(0.0, 1.0)
CAUSAL_NORM = Normalize(-1.0, 1.0)         # raw counterfactual R^2: the causal block and the appendix detail
DELTA_NORM = Normalize(-1.65, 1.65)        # dR^2_cf: 0 = no patch; the root reaches 1.6 in GraphOracle(all)
                                           # (the appendix figure causal_probe_detail draws both)
SEPARATOR = "#8a8a8a"                      # thin lines after intermediates 4 and 6
# blocks side by side: linear accessibility, causal alignment
# the two blocks draw at the same width (0.391 of the figure each); the depth names, the
# y title and the curve labels live in the margins outside them
BLOCKS = {"linear": (0.112, 0.503), "causal": (0.594, 0.985)}
HEADER = {"linear": "Linear probes", "causal": "Causal interventions"}
# the two-row layout: heatmaps over depth curves. The compact layout (decision 2026-09-22; the n = 2^14 stem) draws
# the heatmap row alone at exactly this layout's heatmap size (0.689 in x 0.859 in), the blocks 0.28 in apart
# instead of 0.50 in, without the level legend (no curves to name)
TWO_ROW = dict(figsize=(5.5, 2.55), top=0.875, bottom=0.18, height_ratios=(1.0, 0.72), hspace=0.4)
WSPACE = 0.06
COMPACT = dict(gap=0.28, left=0.616, right=0.083, top=0.30, bottom=0.30)   # inches
HEADER_GAP = 0.166                                                            # inches, heatmap top to block header
VALUE_SIZE, TICK_SIZE, LABEL_SIZE = 5.8, 7.0, 7.4


def heatmap_inches() -> tuple[float, float]:
    """The heatmap's width and height in the two-row layout, by GridSpec's cell arithmetic."""
    w, h = TWO_ROW["figsize"]
    left, right = BLOCKS["linear"]
    cell = (TWO_ROW["top"] - TWO_ROW["bottom"]) / (2 + TWO_ROW["hspace"])
    return ((right - left) / (3 + 2 * WSPACE) * w,
            cell * 2 / sum(TWO_ROW["height_ratios"]) * TWO_ROW["height_ratios"][0] * h)


def compact_figure() -> tuple[plt.Figure, dict[str, tuple[float, float]]]:
    """The heatmap-only figure and its blocks' (left, right) as figure fractions."""
    hw, hh = heatmap_inches()
    block = (3 + 2 * WSPACE) * hw
    width = COMPACT["left"] + 2 * block + COMPACT["gap"] + COMPACT["right"]
    height = COMPACT["top"] + hh + COMPACT["bottom"]
    x0 = COMPACT["left"]
    x1 = x0 + block + COMPACT["gap"]
    return (plt.figure(figsize=(width, height)),
            {"linear": (x0 / width, (x0 + block) / width), "causal": (x1 / width, (x1 + block) / width)})


def draw_heatmap(ax, table: pd.DataFrame, cmap, norm, *, title: str, row_labels: bool,
                 ylabel: bool, layer_labels: list[str] = LAYER_LABEL,
                 group_rows: bool = False) -> None:
    """One heatmap. `group_rows` names the rows by target depth at the centre of
    each depth's block (the main-text figure); otherwise every row carries its
    intermediate's number from the target graph (the appendix details)."""
    m = table.to_numpy(dtype=float)
    ax.imshow(m, cmap=cmap, norm=norm, origin="lower", aspect="auto", interpolation="nearest")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            r, g, b, _ = cmap(norm(v))
            dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
            ax.text(j, i, f"{round(v, 2) + 0.0:.2f}".replace("0.", "."), ha="center", va="center",
                    fontsize=VALUE_SIZE, color="white" if dark else "black")
    for edge in (3.5, 5.5):                    # level 1 | level 2 | level 3
        ax.axhline(edge, color=SEPARATOR, linewidth=0.6, zorder=3)
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(layer_labels, fontsize=TICK_SIZE)
    if row_labels and group_rows:      # one name per target depth, centred on its rows
        ax.set_yticks([float(np.mean([i for i, k in enumerate(table.index) if DEPTH_OF[k] == d]))
                       for d in (1, 2, 3)])
        ax.set_yticklabels([LEVEL_LABEL[d] for d in (1, 2, 3)], fontsize=TICK_SIZE)
    elif row_labels:
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels([str(k) for k in table.index], fontsize=TICK_SIZE)
    else:
        ax.set_yticks([])
    if ylabel:
        ax.set_ylabel("Target intermediate", labelpad=2, fontsize=LABEL_SIZE)
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="center", pad=2, fontsize=6.6)


def cell_table(df: pd.DataFrame, value: str, layers: list) -> pd.DataFrame:
    """Mean over instances per (Figure-1 number, layer)."""
    return (df.groupby(["number", "layer"])[value].mean().unstack("layer")
            .reindex(index=range(1, 8), columns=layers))


def level_curves(df: pd.DataFrame, value: str, layers: list) -> dict[int, pd.DataFrame]:
    """Per target depth: the value averaged over the depth's intermediates
    within each instance, then mean and 95% t-interval over the instances."""
    per = df.groupby(["function_seed", "depth", "layer"])[value].mean().reset_index()
    out = {}
    for depth in (1, 2, 3):
        rows = []
        for k, layer in enumerate(layers):
            v = per[(per.depth == depth) & (per.layer == layer)][value].to_numpy()
            m, lo, hi = ps.t_interval(v)
            rows.append({"x": k, "layer": str(layer), "mean": m, "lo": lo, "hi": hi,
                         "instances": len(v)})
        out[depth] = pd.DataFrame(rows)
    return out


def peaks_of(curves: dict[int, pd.DataFrame]) -> dict[int, tuple[int, float]]:
    """Per target depth: the index of the layer with the highest mean, and that mean."""
    return {d: (int(st["mean"].idxmax()), float(st["mean"].max())) for d, st in curves.items()}


def draw_curves(ax, curves: dict[int, pd.DataFrame], *, ylim, yticks, ylabel: str | None,
                ticks: bool = True, layer_labels: list[str] = LAYER_LABEL,
                ) -> dict[int, tuple[int, float]]:
    ax.axhline(0, color="#c8c8c8", linewidth=0.5, zorder=0)
    for depth in (1, 2, 3):
        st = curves[depth]
        color = LEVEL_COLOR[depth]
        ax.fill_between(st.x, st.lo, st.hi, color=color, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(st.x, st["mean"], color=color, linewidth=1.15, marker="o", markersize=1.8,
                markeredgewidth=0, zorder=3)
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, fontsize=TICK_SIZE)
    ax.set_xlim(-0.35, len(layer_labels) - 0.65)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ps.tidy(ax)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=2, fontsize=LABEL_SIZE)
    if not ticks:              # an interior panel keeps the grid, not the marks
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    return peaks_of(curves)


def load(n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The per-instance probe values and joint causal cells on the tree at training-set size n, numbered as in Figure 1."""
    runs, _order = load_embeddings()
    number = {(s, v): k for s, d in runs.items() for v, k in numbering(d).items()}
    pv = pd.read_csv(DATA / "learned_structure/probe_values.csv")
    pv = pv[(pv.task == "tree_functional_anova") & (pv.n == n)].copy()
    pv["number"] = [number[(s, int(v))] for s, v in
                    zip(pv.function_seed, pv.node.str.extract(r"z(\d+)")[0])]
    cells = pd.read_csv(DATA / "causal_nopatch/cells.csv")
    cells = cells[(cells.fit_list == "joint7") & (cells.task == "tree_functional_anova")
                  & (cells.n == n)].copy()
    cells["number"] = [number[(s, int(v[1:]))] for s, v in zip(cells.function_seed, cells["var"])]
    for df in (pv, cells):
        df["depth"] = df.number.map(DEPTH_OF)
    if not (pv.depth == pv.node_depth).all():
        raise SystemExit("[learned_computational_structure] depth numbering mismatch")
    return pv, cells


def main(n: int = N_SHOWN, stem: str = "learned_computational_structure", curves: bool = True) -> None:
    """The figure at training-set size n (default the paper's 2^15) under the stem name; every size of the records is
    available. `curves=False` draws the compact layout: the heatmap row alone (see TWO_ROW / COMPACT)."""
    ps.apply_rc()
    pv, cells = load(n)
    # both blocks: the heatmap value is also the curve value (R^2 throughout);
    # the causal block carries no y label, its header names the quantity
    blocks = {"linear": (pv, "r2_trained", PROBE_LAYERS, SEQUENTIAL, LINEAR_NORM,
                         (-0.03, 0.98), [0, 0.4, 0.8], r"Mean $R^2$"),
              "causal": (cells, "test_r2", DAS_LAYERS, DIVERGING, CAUSAL_NORM,
                         (-0.5, 0.85), [-0.4, 0.0, 0.4, 0.8], None)}
    records = []
    if curves:
        fig, positions = plt.figure(figsize=TWO_ROW["figsize"]), BLOCKS
    else:
        fig, positions = compact_figure()
    for block, (df, value, layers, cmap, norm, ylim, yticks, ylabel) in blocks.items():
        left, right = positions[block]
        if curves:
            gs = GridSpec(2, 3, figure=fig, height_ratios=list(TWO_ROW["height_ratios"]), left=left, right=right,
                          top=TWO_ROW["top"], bottom=TWO_ROW["bottom"], wspace=WSPACE, hspace=TWO_ROW["hspace"])
            curve = [fig.add_subplot(gs[1, k]) for k in range(3)]
        else:
            gs = GridSpec(1, 3, figure=fig, left=left, right=right, wspace=WSPACE,
                          top=1 - COMPACT["top"] / fig.get_figheight(), bottom=COMPACT["bottom"] / fig.get_figheight())
        heat = [fig.add_subplot(gs[0, k]) for k in range(3)]
        for k, (method, title) in enumerate(ARMS):
            sub = df[df.method == method]
            table = cell_table(sub, value, layers)
            first = block == "linear" and k == 0      # the depth names sit once, on the far left
            draw_heatmap(heat[k], table, cmap, norm, title=title, row_labels=first,
                         ylabel=first, group_rows=True)
            for num, row in table.iterrows():
                for layer, v in row.items():
                    records.append((block, "cells", method, f"intermediate {num}", str(layer),
                                    v, np.nan, np.nan))
            level = level_curves(sub, value, layers)
            peaks = (draw_curves(curve[k], level, ylim=ylim, yticks=yticks,
                                 ylabel=ylabel if k == 0 else None, ticks=k == 0)
                     if curves else peaks_of(level))
            for depth, st in level.items():
                for _, r in st.iterrows():
                    records.append((block, "by_depth", method, LEVEL_LABEL[depth], r.layer,
                                    r["mean"], r.lo, r.hi))
            print(f"[learned_computational_structure] {block} {title} peaks: " + ", ".join(
                f"{LEVEL_LABEL[d]} at layer {LAYER_LABEL[k_]} ({v:.2f})"
                for d, (k_, v) in peaks.items()))
        # the group header, centred over the block's three panels
        x0, x1 = heat[0].get_position().x0, heat[-1].get_position().x1
        header_gap = 0.065 if curves else HEADER_GAP / fig.get_figheight()
        fig.text((x0 + x1) / 2, heat[0].get_position().y1 + header_gap, HEADER[block],
                 ha="center", va="bottom", fontsize=7.6, color="#3a3a3a")
    # one x label for both blocks, and (with the curves) one legend for the target depths
    fig.text((positions["linear"][0] + positions["causal"][1]) / 2, 0.012, "Hidden layer",
             ha="center", va="bottom", fontsize=LABEL_SIZE)
    if curves:
        handles = [Line2D([], [], color=LEVEL_COLOR[d], linewidth=1.3, marker="o",
                          markersize=1.8, markeredgewidth=0) for d in (1, 2, 3)]
        fig.legend(handles, [LEVEL_LABEL[d] for d in (1, 2, 3)], loc="lower right",
                   bbox_to_anchor=(BLOCKS["causal"][1], 0.0), ncol=3, frameon=False,
                   fontsize=6.6, handlelength=1.4, handletextpad=0.5, columnspacing=1.2,
                   borderaxespad=0.0)
    save(fig, stem)
    pd.DataFrame(records, columns=["block", "panel", "method", "series", "layer", "value",
                                   "lo", "hi"]).to_csv(
        OUT / f"{stem}.csv", index=False)


if __name__ == "__main__":
    main()
