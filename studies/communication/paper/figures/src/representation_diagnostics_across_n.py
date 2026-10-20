"""Figure (appendix): the main text's representation diagnostics across training-set size, on three target families.

Caption - Linear decodability and counterfactual agreement across
training-set size on the same three target families, all inputs active
(d = 8). (a) Linear decodability: for each target intermediate, the maximum
held-out ridge-probe R^2 over hidden layers 1-4, averaged over the
intermediates. (b) Counterfactual agreement: held-out counterfactual R^2
under joint multi-variable Boundless DAS (the target intermediates aligned
jointly, one learned rotation per hidden layer, each intermediate read at its
validation-selected layer), averaged over the intermediates. Columns: the
depth-4 tree, the depth-3 reuse DAG and the depth-4 Fourier tree. Curves are
means over 16 target instances with 95% Student-t intervals, for Dense, LeWi
and GraphOracle(all).

Replaces probe_curves and causal_use (decision 2026-09-23); a 2 x 3 grid so both
diagnostics read the same families (design decision, the same night). Row (b) is
causal_use's solid patched curves (S). The maximum of row (a) is taken over
the hidden layers only (the harness's probe_levels.csv also admits the input
layer h0). The plotted means and intervals go to <stem>.csv beside the figure;
the PDF and the CSV are copied to out-paper/appendix (the paper's
figures/appendix); src/verify_representation_diagnostics.py recomputes every
value from the evaluations' own records.

Data: data/learned_structure/probe_values.csv and data/causal_nopatch/scores.csv
(joint7 rows), the tables of probe_curves and causal_use.
"""
from __future__ import annotations

import shutil

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import paperstyle as ps
from figstyle import DATA, OUT, output, save

STEM = "representation_diagnostics_across_n"
ARM_OF = {"noreg": "noreg", "graph_oracle": "graph_oracle(levels=all)",   # method id in the tables -> arm key,
          ps.LEWI: ps.LEWI}                                               # in drawing order (LeWi on top)
LEGEND = ["noreg", ps.LEWI, "graph_oracle"]
TASKS = ["tree_functional_anova", "dag_functional_anova", "tree_fourier"]
HIDDEN = ["h1", "h2", "h3", "h4"]
PARTS = {"a": ("(a) Linear decodability", "Maximum probe $R^2$"),
         "b": ("(b) Counterfactual agreement", "Counterfactual $R^2$")}

# inches: three equal columns at the text width (the type renders at its set size); rows share each column's x axis
WIDTH, LEFT, RIGHT, GAP = 5.5, 0.42, 0.04, 0.16
PANEL, HEAD, BETWEEN, FOOT = 1.0, 0.34, 0.12, 0.48     # panel height; row and column titles; row gap; ticks, x label, legend


def key_handle(arm: str) -> Line2D:
    """The arm's line for the legend, a dash pattern drawn from its start so the key shows whole dashes."""
    kw = ps.style(None, arm)
    if isinstance(kw["linestyle"], tuple):
        kw["linestyle"] = (0, kw["linestyle"][1])
    return Line2D([], [], **kw)


def curves() -> pd.DataFrame:
    """Mean and 95% t interval over instances at each n, one row per (part, task, method, n)."""
    pv = pd.read_csv(DATA / "learned_structure/probe_values.csv")
    best = (pv[pv.layer.isin(HIDDEN)].groupby(["task", "method", "n", "function_seed", "node"]).r2_trained.max()
            .groupby(["task", "method", "n", "function_seed"]).mean().rename("r2").reset_index())
    cf = pd.read_csv(DATA / "causal_nopatch/scores.csv")
    cf = cf[cf.fit_list == "joint7"]
    return pd.concat([ps.curve_stats(sub[(sub.task == t) & (sub.method == m)], value=value).assign(part=part, task=t, method=m)
                      for part, sub, value in (("a", best, "r2"), ("b", cf, "S")) for t in TASKS for m in ARM_OF],
                     ignore_index=True)


def build():
    """The figure and its plotted values."""
    ps.apply_rc()
    st = curves()
    width = (WIDTH - LEFT - RIGHT - 2 * GAP) / 3
    height = 2 * (HEAD + PANEL) + BETWEEN + FOOT
    fig = plt.figure(figsize=(WIDTH, height))
    ylim = (min(-0.05, st.lo.min() - 0.03), max(1.0, st.hi.max() + 0.03))     # one frame for all six panels; nothing leaves it
    for row, (part, (title, ylabel)) in enumerate(PARTS.items()):
        top = HEAD + row * (PANEL + BETWEEN + HEAD)
        fig.text(LEFT / WIDTH, 1 - (top - HEAD + 0.02) / height, title, ha="left", va="top", fontsize=8.2, fontweight="bold")
        for col, task in enumerate(TASKS):
            ax = fig.add_axes([(LEFT + col * (width + GAP)) / WIDTH, 1 - (top + PANEL) / height, width / WIDTH, PANEL / height])
            for m, arm in ARM_OF.items():
                c = st[(st.part == part) & (st.task == task) & (st.method == m)]
                kw = ps.style(ax, arm)
                ax.fill_between(c.n, c.lo, c.hi, color=kw["color"], alpha=0.13, linewidth=0, zorder=1)
                ax.plot(c.n, c["mean"], marker="o", markersize=2.0, markeredgewidth=0, **kw)
            ps.n_axis(ax, ps.WINDOW, label=(part, col) == ("b", 1))       # one x label, under the middle column
            if part == "a":
                ax.set_xticklabels([])                                     # the column's x axis is labelled once, in row (b)
            ax.set_ylim(*ylim)
            ax.set_yticks([0, 0.5, 1.0])
            ps.tidy(ax)
            ax.set_title(ps.TITLE[task], loc="left", pad=3)
            if col == 0:
                ax.set_ylabel(ylabel, labelpad=2)
            else:
                ax.set_yticklabels([])
    fig.legend([key_handle(ARM_OF[m]) for m in LEGEND], [ps.NAME[ARM_OF[m]] for m in LEGEND], loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=7.2, handlelength=2.3, columnspacing=1.8,
               borderaxespad=0.0)
    return fig, st


def main() -> None:
    if DATA.name != "data-paper-d8":
        print(f"[{STEM}] the all-inputs-active paper set only (data-paper-d8); skipped for {DATA.name}")
        return
    fig, st = build()
    save(fig, STEM)
    table = OUT / f"{STEM}.csv"
    (st.assign(panel=st.task.map(ps.TITLE), method=st.method.map(lambda m: ps.NAME[ARM_OF[m]]))
       .rename(columns={"lo": "ci95_lo", "hi": "ci95_hi"})
       [["part", "panel", "method", "n", "instances", "mean", "ci95_lo", "ci95_hi"]]
       .to_csv(table, index=False))
    appendix = OUT.parent / "appendix"
    appendix.mkdir(exist_ok=True)
    for path in (output(STEM, "pdf"), table):
        shutil.copy2(path, appendix / path.name)
    print(f"[{STEM}] {table.name} ({len(st)} rows); pdf and csv copied to {appendix.relative_to(OUT.parents[1])}")


if __name__ == "__main__":
    main()
