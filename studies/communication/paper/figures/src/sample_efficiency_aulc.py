"""Figure (appendix): the paired AULC differences behind Figure 3.

Caption - Paired AULC differences behind Figure 3. AULC(LeWi) - AULC(arm)
on the same target instances (dots), with the mean and its paired 95%
Student-t interval (diamond and bar), on the six settings of Figure 3;
AULC is the average of the sealed test NMSE over log n for n = 2^9 ...
2^15 (2^11 ... 2^17 on the depth-5 tree), so values left of zero favor
LeWi. Row labels are coloured as the arms in the appendix learning-curve
figure. On the depth-4 tree, the depth-3 reuse DAG and the flat modules
LeWi is ahead of every baseline on every instance (16/16); it is level
with GraphOracle(0), which receives the true active inputs, on the tree
and the reuse DAG, and behind it by 0.015 [0.005, 0.024] on the flat
modules, where its margins over the baselines run from 0.021 (l0)
to 0.353 (neuron positions). GraphOracle(1) and GraphOracle(all), which
receive target connectivity, stay ahead of LeWi on all three. On the dense
node GraphOracle(1) and GraphOracle(all) coincide with GraphOracle(0),
which leads LeWi by 0.061 [0.050, 0.072]; LeWi keeps its lead over Dense
(-0.098 [-0.109, -0.088], 16/16), Fiedler (-0.084) and neuron positions
(-0.325), and its margins over the remaining baselines shrink to
0.011-0.036 (RigL -0.011 [-0.019, -0.004], 13/16; l0 -0.015 [-0.022,
-0.009], 14/16). On the depth-5 tree LeWi is ahead of Dense, l0 and l1 on
16/16 instances; the Dense pair's mean is undefined on the frame's scale
because one Dense winner is degenerate (test NMSE 2.2 x 10^8, drawn as an
arrow at the frame edge), so that row shows the median difference as a
hollow diamond. On the 32 sampled depth-4 DAGs LeWi is ahead of Dense, l1
and l0 on 32/32 instances and level with GraphOracle(0). On these two
settings GraphOracle(1) and GraphOracle(2) come from 12-extra wave 2 (the
same sealed instances, frozen spaces and trial lists as the 08-main
GraphOracle(all) blocks); on the four other settings GraphOracle(2)
coincides with GraphOracle(all) and its row is left empty, as is the
GraphOracle(1) row on the flat modules and the high-arity node, where that rung
receives the whole graph as well. The ladder is monotone on both: on the
depth-5 tree LeWi is ahead of GraphOracle(0) on 16/16 instances (-0.046)
and behind GraphOracle(1), GraphOracle(2) and GraphOracle(all) by 0.159,
0.336 and 0.402 (0/16); on the sampled depth-4 DAGs LeWi is level with
GraphOracle(0) (-0.001, 17/32) and behind the three connectivity rungs by
0.068, 0.121 and 0.152 (0/32).

Data: as Figure 3 (data/sample_efficiency/curves.csv from the 08-main
ledger, data/sample_efficiency/dense_node_12extra.csv for the high-arity node's
RigL, BIMT, Fiedler and neuron-positions arms from the 12-extra wave-1
campaign, data/sample_efficiency/oracle_rungs_12extra.csv for
GraphOracle(1) and GraphOracle(2) on the depth-5 tree and the random
graphs from wave 2). This figure is the bottom row of the
sample_efficiency figure as it stood until 2026-09-04, extended to the two
generalization settings; it follows Figure 3's panel set, so the third
forest became the flat modules on 2026-09-14.
"""
from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import paperstyle as ps
from figstyle import save
from sample_efficiency import FLAT_MAIN, MAIN, TITLE_SIZE, draw_forest, load, per_instance_aulc, report


def main() -> None:
    ps.apply_rc()
    aulc = per_instance_aulc(load())
    order = [a for a in ps.FOREST_ORDER if a in set(aulc[aulc.workload.isin(FLAT_MAIN)].arm)]      # no empty row for an arm the setting does not have (GO(0) at d = 8)
    fig = plt.figure(figsize=(5.5, 4.7))
    gs = GridSpec(2, 3, figure=fig, left=0.185, right=0.995, top=0.955, bottom=0.075,
                  wspace=0.16, hspace=0.3)
    have = set(aulc.workload)
    for r, row in enumerate(MAIN):
        for c, wl in enumerate(row):
            if wl not in have:                   # a setting without the workload (the 7000-series set has no depth-5 tree) leaves the cell blank
                continue
            ax = fig.add_subplot(gs[r, c])
            off = draw_forest(ax, aulc[aulc.workload == wl], labels=c == 0, order=order)
            ax.set_title(ps.TITLE[wl], fontsize=TITLE_SIZE, pad=3)
            for note in off:
                print(f"[sample_efficiency_aulc] {ps.TITLE[wl]}: {note}")
    fig.text((gs.left + gs.right) / 2, 0.015, "AULC(LeWi) − AULC(arm)", ha="center",
             va="center", fontsize=plt.rcParams["axes.labelsize"])
    save(fig, "sample_efficiency_aulc")
    report(aulc, [wl for wl in FLAT_MAIN if wl in have], "sample_efficiency_aulc")


if __name__ == "__main__":
    main()
