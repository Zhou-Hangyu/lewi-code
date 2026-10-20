"""Appendix companion of Figure 6(b): structure alignment against the training-set size.

Caption - Edge F1 of the module graph read from LeWi's learned geometry against the target
computation graph, (a) over all target edges and (b) over the edges originating at target
intermediate nodes, per structured workload (d = 8, every input active, 64,000 steps), over
the full grid n = 2^9 ... 2^15, for the trained network and three controls: degree-preserving
rewiring, the exact pre-training initialization and the modules shuffled at fixed per-layer
sizes. Lines are means over the 16 target instances (32 on the sampled depth-4 DAGs); bands are 95%
Student-t intervals (trained network and initialization). The procedure is the one of
Figure 6(b) (Appendix, structure discovery). No curve is required to be monotone in n.

Decision 2026-09-23: the main text's "edges originating at target intermediate nodes" in place of
"higher-level edges", and larger type: built at the include width (\\linewidth, 5.5 in), so the
sizes below are the printed sizes; the legend names are Figure 6(b)'s.

Data: data/structure/alignment_vs_n.csv (18-main-d8/structure, src/sd_read.py).
"""
from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

import paperstyle as ps
from figstyle import DATA, save
from fig6 import CONTROLS
from structure_f1_panel import COLOR, CONDITIONS, FAMILIES, FILLED, MARKER

TITLE = {"tree": "Depth-4 tree", "ftree": "Fourier tree", "dag": "Reuse DAG", "rand": "Sampled depth-4 DAGs"}   # Figure 6's target names (decision 2026-09-23)
ROWS = (("edge_f1", "(a) All target edges"), ("higher_f1", "(b) Edges originating at target intermediate nodes"))
NAME = {cond: label for cond, label, *_ in CONTROLS}
BANDS = ("trained", "init")
HEADING_PT, TITLE_PT, LABEL_PT, TICK_PT, KEY_PT = 9.0, 8.5, 8.0, 7.5, 8.0
WIDTH, LEFT, RIGHT, GAP = 5.5, 0.44, 0.05, 0.12      # inches
PANEL, HEAD, BETWEEN, FOOT = 1.0, 0.33, 0.10, 0.54   # panel height; row heading and panel titles; row gap; ticks, x label, legend


def main() -> None:
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "path",
                         "axes.spines.top": False, "axes.spines.right": False})   # figstyle's type, in case an earlier builder of the same process changed it
    ps.apply_rc()
    df = pd.read_csv(DATA / "structure/alignment_vs_n.csv")
    width = (WIDTH - LEFT - RIGHT - (len(FAMILIES) - 1) * GAP) / len(FAMILIES)
    height = 2 * (HEAD + PANEL) + BETWEEN + FOOT
    fig = plt.figure(figsize=(WIDTH, height))
    BASE = ScaledTranslation(-TICK_PT / 4 / 72, 0, fig.dpi_scale_trans)   # half a STIX digit (0.5 em wide) to the left
    for i, (metric, heading) in enumerate(ROWS):
        top = HEAD + i * (PANEL + BETWEEN + HEAD)
        fig.text(LEFT / WIDTH, 1 - (top - HEAD + 0.02) / height, heading, ha="left", va="top", fontsize=HEADING_PT, fontweight="bold")
        for j, fam in enumerate(FAMILIES):
            ax = fig.add_axes([(LEFT + j * (width + GAP)) / WIDTH, 1 - (top + PANEL) / height, width / WIDTH, PANEL / height])
            for cond in CONDITIONS[::-1]:
                r = df[(df.family == fam) & (df.condition == cond) & (df.metric == metric)].sort_values("n")
                if r.empty:
                    continue
                c = COLOR[cond]
                if cond in BANDS:
                    ax.fill_between(r.n, r.lo, r.hi, color=c, alpha=0.15, linewidth=0)
                ax.plot(r.n, r["mean"], marker=MARKER[cond], color=c, mfc=c if FILLED[cond] else "white", ms=3.2, mew=0.7,
                        lw=1.0 if cond == "trained" else 0.8, zorder=4 if cond == "trained" else 3)
            ps.n_axis(ax, ps.WINDOW, ticks=ps.WINDOW[::2], label=False)
            ax.set_ylim(-0.02, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ps.tidy(ax)
            ax.tick_params(labelsize=TICK_PT)
            ax.set_title(TITLE[fam], loc="left", pad=3, fontsize=TITLE_PT)
            if i == 0:
                ax.set_xticklabels([])                    # the column's x axis is labelled once, in row (b)
            else:
                for lab in ax.get_xticklabels():          # decision 2026-09-23: the base "2" sits under the tick, never an exponent's "1"
                    lab.set_horizontalalignment("left")
                    lab.set_transform(lab.get_transform() + BASE)
            if j == 0:
                ax.set_ylabel("Edge $F_1$", fontsize=LABEL_PT, labelpad=2)
            else:
                ax.set_yticklabels([])
    fig.text((LEFT + (WIDTH - LEFT - RIGHT) / 2) / WIDTH, (FOOT - 0.22) / height, "Training samples $n$", ha="center", va="top", fontsize=LABEL_PT)
    handles = [Line2D([], [], marker=MARKER[c], color=COLOR[c], mfc=COLOR[c] if FILLED[c] else "white", ms=3.6, mew=0.7, lw=1.0, label=NAME[c])
               for c in CONDITIONS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=KEY_PT, handlelength=1.8,
               handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0)
    save(fig, "structure_alignment_vs_n")


if __name__ == "__main__":
    main()
