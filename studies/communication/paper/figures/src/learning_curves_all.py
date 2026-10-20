"""Figure (appendix): learning curves on every released workload, every arm.

Caption - Learning curves on the six released CSFRBench workloads, the
depth-5 tree and the 32 sampled depth-4 DAGs: sealed test NMSE against
training-set size n for every arm that ran on the setting (mean over the
target instances; 95% Student-t bands for LeWi, Dense and GraphOracle(all)),
each arm in its own colour; the GraphOracle references are dashed. The
Fourier reuse DAG ran the Dense, l1, l2, l0, LeWi and GraphOracle arms only;
on the High-arity node the RigL, BIMT, Fiedler and neuron-positions arms
come from the 12-extra wave-1 campaign under the identical protocol, and
GraphOracle(all) coincides with GraphOracle(0), so a single reference is
shown; the depth-5 tree (64x8 backbone, 128,000 steps, n = 2^11 ... 2^17) ran
Dense, l1, l0, LeWi, GraphOracle(0) and GraphOracle(all), and the sampled depth-4
DAGs add l2; on both of these settings GraphOracle(1) and GraphOracle(2)
come from 12-extra wave 2 (the same sealed instances, frozen spaces and trial
lists as the GraphOracle(all) blocks), completing the oracle ladder. On the flat-modules workload GraphOracle(0), which receives
only the true active inputs, is already ahead of LeWi: on flat structure the
input support alone carries most of the gain. On the dense-node control
LeWi's gain over Dense shrinks and RigL and l0 nearly match it: the low-arity
decomposition is what LeWi exploits. The depth-5 Dense mean at n = 2^12 lies
above the frame (1.4 x 10^7; a caret at the frame's top edge marks it): one
Dense winner collapsed to a constant predictor whose retained weights blow up
on rare test inputs, entered unclipped by the protocol.

Data: data/sample_efficiency/curves.csv (src/harvest_08main.py),
data/sample_efficiency/dense_node_12extra.csv and
data/sample_efficiency/oracle_rungs_12extra.csv (src/harvest_12extra.py
dense_node, oracle_rungs).
"""
from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import paperstyle as ps
from figstyle import save
from sample_efficiency import TICK, TITLE_SIZE, draw_curves, legend_entries, load, per_instance_aulc, report, with_repairs

ORDER = ["tree_functional_anova", "tree_functional_anova:depth4", "dag_functional_anova",
         "rand_functional_anova", "flat_functional_anova", "dense_fourier",
         "tree_fourier", "dag_fourier_v2"]


def main() -> None:
    ps.apply_rc()
    df = with_repairs(load(), "learning_curves_all")      # the d = 8 build: LeWi with the input-cutoff repairs, as the main-text figure
    fig = plt.figure(figsize=(5.5, 5.3))
    gs = GridSpec(3, 3, figure=fig, left=0.085, right=0.975, top=0.962,
                  bottom=0.07, wspace=0.3, hspace=0.42)
    first_column = []
    order = [wl for wl in ORDER if wl in set(df.workload)]      # a set without the depth-5 tree (the 7000-series set) leaves its cell blank
    for k, wl in enumerate(order):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        if k % 3 == 0:
            first_column.append(ax)
        off = draw_curves(ax, df[df.workload == wl], ps.WINDOW_OF[wl], ylabel=k % 3 == 0, palette="full")
        ax.set_title(ps.TITLE[wl], fontsize=TITLE_SIZE, pad=5.5)
        for cell in off:
            print(f"[learning_curves_all] {ps.TITLE[wl]}: mean above the frame marked, {cell}")
    fig.align_ylabels(first_column)          # the rows' tick labels differ in width (a frame per panel)
    fig.text((gs.left + gs.right) / 2, 0.018, "Training samples $n$", ha="center",
             va="center", fontsize=plt.rcParams["axes.labelsize"])
    # the ninth cell carries the legend: every arm in its own colour, one
    # column (two columns would be wider than the cell)
    key = fig.add_subplot(gs[2, 2])
    key.axis("off")
    handles, labels = legend_entries("full", arms=set(df.arm))
    key.legend(handles, labels, loc="center", ncol=1, frameon=False, fontsize=6.0,
               handlelength=1.8, handletextpad=0.5, labelspacing=0.22, borderaxespad=0.0)
    save(fig, "learning_curves_all")
    report(per_instance_aulc(df), order, "learning_curves_all")


if __name__ == "__main__":
    main()
