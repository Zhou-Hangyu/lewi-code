"""Figure 6(b) candidate: structure discovery on the clean-input cohort, points with 95% intervals.

Caption - Edge F1 (left) and higher-level edge F1 (right; edges from an intermediate
node to a node or to the output) of the module graph read from LeWi's learned geometry
against the target computation graph, per structured workload (d = 8, every input
active, 64,000 steps, n = 2^15), for the trained network and three controls: the
exact pre-training initialization, the modules shuffled at fixed per-layer sizes, and
the wiring rewired at fixed degrees. Points are means over the 16 target instances
with 95% Student-t intervals. Modules: within-layer average-linkage clusters of the
neuron positions cut at the model's radius; edges: majority coverage through the
executed mask; names: one-to-one held-out ridge probes against shape-matched chance
sets (Appendix, structure discovery).

Data: data/structure/fig6b.csv (18-main-d8/structure, src/sd_read.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import paperstyle as ps
from figstyle import DATA, save

FAMILIES = ("tree", "dag", "ftree", "rand")          # the workload order of Figure 6 panel (a)
TICK = {"tree": "Depth-4\ntree", "ftree": "Fourier\ntree", "dag": "Reuse\nDAG", "rand": "Sampled\ndepth-4\nDAGs"}   # Figure 6's target names (decision 2026-09-23)
CONDITIONS = ("trained", "rewired", "init", "shuffled")   # the order of Figure 6(b) (decision 2026-09-22)
LABEL = {"trained": "trained LeWi", "init": "initialization", "shuffled": "modules shuffled", "rewired": "wiring rewired"}
COLOR = {"trained": ps.COLOR["lewi-v1-rgm"], "init": "#9e9e9e", "shuffled": "#8c6d31", "rewired": "#4d4d4d"}   # as Figure 6(b) (decision 2026-09-23)
MARKER = {"trained": "o", "init": "o", "shuffled": "D", "rewired": "^"}
FILLED = {"trained": True, "init": False, "shuffled": True, "rewired": True}
METRICS = (("edge_f1", "Edge $F_1$"), ("higher_f1", "Higher-level edge $F_1$"))


def main() -> None:
    ps.apply_rc()
    df = pd.read_csv(DATA / "structure/fig6b.csv")
    fig, axes = plt.subplots(1, 2, figsize=(2.95, 1.65))          # room between the family names and for the three-line one
    fig.subplots_adjust(left=0.115, right=0.995, top=0.887, bottom=0.399, wspace=0.12)   # the axes of the 1.55 in build, 0.1 in more below
    offs = np.linspace(-0.27, 0.27, len(CONDITIONS))
    for ax, (metric, title) in zip(axes, METRICS):
        for k, cond in enumerate(CONDITIONS):
            for i, fam in enumerate(FAMILIES):
                r = df[(df.family == fam) & (df.condition == cond) & (df.metric == metric)]
                if r.empty:
                    continue
                r = r.iloc[0]
                c = COLOR[cond]
                ax.errorbar(i + offs[k], r["mean"], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt=MARKER[cond], color=c,
                            mfc=c if FILLED[cond] else "white", ms=2.9, mew=0.7, elinewidth=0.6, capsize=0, zorder=3)
        ax.set_xticks(range(len(FAMILIES))); ax.set_xticklabels([TICK[f] for f in FAMILIES], fontsize=5.8)
        ax.set_xlim(-0.6, len(FAMILIES) - 0.4); ax.set_ylim(-0.02, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, loc="left", pad=3, fontsize=6.8)
        ps.tidy(ax)
    axes[0].set_ylabel("$F_1$ against the target graph", labelpad=2, fontsize=6.4)
    axes[1].set_yticklabels([])
    handles = [Line2D([], [], marker=MARKER[c], color=COLOR[c], mfc=COLOR[c] if FILLED[c] else "white", ms=3.2, mew=0.7, lw=0, label=LABEL[c])
               for c in CONDITIONS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=5.8, handletextpad=0.4,
               columnspacing=1.2, borderaxespad=0.0)
    save(fig, "structure_f1_panel")


if __name__ == "__main__":
    main()
