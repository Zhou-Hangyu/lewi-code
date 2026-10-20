"""Figure 6(a), "Nuisance inputs": LeWi's paired AULC advantage over Dense in the clean setting (d = 8, every input
active; 18-main-d8's all-active twin) and in the nuisance-input setting on the same instances (d = 32: 8 active inputs and
24 inert ones, 18-main-d8 setting d32s; decision 2026-09-22 - before that the record's instances), grouped by workload. Within
each setting the comparison is paired per instance (16 instances; 32 sampled depth-4 DAGs); the two settings sit side by side,
not paired with each other. A hollow marker is the median where the mean is off scale (a Dense winner with a test NMSE
above 10 at d = 32: one on the depth-4 tree, two on the reuse DAG); the caption explains it, the legend does not (decision
2026-09-23). Data: data-final-clean/fig6/feature_selection.csv (18-main-d8 analysis/d32s/contract and
analysis/d8/contract, contrasts.csv). Appendix companion: the same panel with L0 and GO(all) as the reference (fig6a_refs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import paperstyle as ps
from figstyle import DATA, save

ORDER = ["tree_functional_anova", "dag_functional_anova", "tree_fourier", "flat_functional_anova", "dense_fourier", "dag_fourier_v2", "rand_functional_anova"]
SHORT = {"tree_functional_anova": "Depth-4 tree", "dag_functional_anova": "Reuse DAG", "tree_fourier": "Fourier tree", "flat_functional_anova": "Flat modules",
         "dense_fourier": "High-arity node", "dag_fourier_v2": "Fourier reuse DAG", "rand_functional_anova": "Sampled depth-4 DAGs"}   # Figure 6's target names (decision 2026-09-23)
SETTINGS = [("clean (d = 8)", "clean inputs (d = 8)", ps.COLOR[ps.LEWI], "o"), ("nuisance (d = 32)", "24 nuisance inputs (d = 32)", "#555555", "s")]
REF_NAME = {"noreg": "Dense", "l0hc": r"$L_0$", "graph_oracle": "GraphOracle(all)"}


def handles() -> list:
    """Legend entries by hand: the first d = 32 row is an off-scale median, so no drawn series can carry the label."""
    from matplotlib.lines import Line2D
    return [Line2D([], [], marker=m, color=c, linestyle="none", markersize=4, label=l) for _, l, c, m in SETTINGS]


def panel(ax, df: pd.DataFrame, ref: str, title: str | None = None, wins: bool = True) -> None:
    sub = df[df.ref == ref]
    ys = np.arange(len(ORDER))[::-1]
    for k, (key, label, color, marker) in enumerate(SETTINGS):
        off = 0.18 if k == 0 else -0.18
        for wl, y in zip(ORDER, ys):
            r = sub[(sub.setting == key) & (sub.workload == wl)]
            if r.empty:
                continue
            r = r.iloc[0]
            if r.off_scale:
                ax.plot([r["median"]], [y + off], marker=marker, markerfacecolor="white", markeredgecolor=color, markersize=5, linestyle="none")
            else:
                ax.errorbar([r["mean"]], [y + off], xerr=[[r["mean"] - r.lo], [r.hi - r["mean"]]], fmt=marker, color=color, markersize=4, capsize=2, linewidth=0.9)
            if wins:
                ax.text(1.01, y + off, f"{int(r.wins)}/{int(r.n_pairs)}", transform=ax.get_yaxis_transform(), fontsize=6.5, va="center", color=color)
    ax.axvline(0, color="#888", linewidth=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([ps.wrap_name(SHORT[w]) for w in ORDER])
    ax.set_xlabel(f"AULC(LeWi) $-$ AULC({REF_NAME[ref]}), paired per instance")
    if title:
        ax.set_title(title, fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    ax.tick_params(labelsize=8)


def main() -> None:
    ps.apply_rc()
    df = pd.read_csv(DATA / "fig6" / "feature_selection.csv")
    fig, ax = plt.subplots(figsize=(3.2, 3.15))
    panel(ax, df, "noreg")
    ax.legend(handles=handles(), loc="upper center", bbox_to_anchor=(0.42, -0.20), ncol=1, fontsize=7, frameon=False, borderaxespad=0.0)
    fig.subplots_adjust(left=0.30, right=0.89, top=0.98, bottom=0.36)
    save(fig, "fig6a_feature_selection")
    fig2, axes = plt.subplots(1, 3, figsize=(7.0, 2.95), sharey=True)
    for ax, ref in zip(axes, ("noreg", "l0hc", "graph_oracle")):
        panel(ax, df, ref, title=f"LeWi vs {REF_NAME[ref]}")
        ax.set_xlabel(f"AULC(LeWi) $-$ AULC({REF_NAME[ref]})")      # the long label overlapped its neighbour; "paired" is in the caption
    fig2.legend(handles=handles(), loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig2.subplots_adjust(left=0.13, right=0.955, top=0.91, bottom=0.25, wspace=0.34)
    save(fig2, "fig6a_refs")
    for ref in ("noreg", "l0hc", "graph_oracle"):
        for key, *_ in SETTINGS:
            sub = df[(df.ref == ref) & (df.setting == key)]
            print(f"[fig6a] {REF_NAME[ref]:16s} {key:18s} " + "; ".join(f"{SHORT[r.workload]} {r['mean']:+.3f} ({int(r.wins)}/{int(r.n_pairs)})" + (" median" if r.off_scale else "") for _, r in sub.iterrows()))


if __name__ == "__main__":
    main()
