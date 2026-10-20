"""Figure 6: (a) nuisance inputs, (b) wiring alignment, (c) mechanism ablations, one row at the text width.

Caption - (a) Nuisance inputs: the paired AULC difference between LeWi and Dense per target family with every input
active (d = 8, orange) and with 24 nuisance inputs added (d = 32, grey); negative values favour LeWi. A hollow square
marks the median where single diverged Dense winners put the mean off the scale (d = 32: on the depth-4 tree one winner
with a test NMSE of 2.9e8 at n = 2^11, mean -3.0e6; on the reuse DAG two winners with test NMSE near 13 at n = 2^9 and
2^10, mean -0.51). (b) Wiring alignment: edge F1 of the module graph read from the learned geometry against the target
graph (n = 2^15), over all target edges (filled) and over the edges originating at target intermediate nodes (hollow), for trained LeWi and three
controls: degree-preserving rewiring, the initialization and modules shuffled. (c) Mechanism ablations: AULC(LeWi) -
AULC(variant) for soft-only (no mask), the mask-only forward, fixed embeddings and a random sparse mask; negative values
favour LeWi. Points are means over 16 target instances (32 on the sampled depth-4 DAGs) with 95% Student-t intervals; in (a)
and (c) the vertical line at zero marks no difference.

Every panel is its own stem too (fig6a_feature_selection, structure_f1_panel, fig6c_ablations); this file composes them
from the same data, and fig6_with_modules draws the same panels() above the module graphs. Decision 2026-09-23: fonts about
12% larger, the panels 0.90 : 1.05 : 0.95 wide, orange for LeWi and its default condition alone (greys for the
initialization and the random controls, brown for the module shuffle, teal soft-only, blue mask-only, purple fixed
embeddings), the marker keys in the caption.
"""
from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt

import paperstyle as ps
from figstyle import DATA, save
import fig6_feature_selection as A
import fig6_ablations as C

WL_ORDER = ["Depth-4 tree", "Reuse DAG", "Fourier tree", "Sampled depth-4 DAGs"]          # the workload order of panel (a)
FAMILY = {"tree": "Depth-4 tree", "ftree": "Fourier tree", "dag": "Reuse DAG", "rand": "Sampled depth-4 DAGs"}   # fig6b.csv keys its rows by family
CONTROLS = [("trained", "LeWi (trained)", ps.COLOR[ps.LEWI], "o"), ("rewired", "degree-preserving rewiring", "#4d4d4d", "^"),
            ("init", "initialization", "#9e9e9e", "s"), ("shuffled", "modules shuffled", "#8c6d31", "D")]   # decision 2026-09-22: trained first, rewiring second
TITLE, TICK, LABEL, KEY = 10.0, 9.0, 8.1, 7.1       # decision 2026-09-23: about 12% above the earlier 9 / 8 / 7.2 / 6.3 pt
WIDTHS = [0.90, 1.05, 0.95]                         # (b) carries the most: four conditions x two markers


def panel_b(ax, df: pd.DataFrame) -> None:
    """Points with 95% t intervals: edge F1 over all target edges (filled) and over the edges originating at target
    intermediate nodes (hollow), per workload and condition."""
    if "workload" not in df.columns:
        df = df.assign(workload=df.family.map(FAMILY).fillna(df.family))
    wls = [w for w in WL_ORDER if w in set(df.workload)] or sorted(set(df.workload))
    import numpy as np
    ys = np.arange(len(wls))[::-1]
    for j, (cond, label, color, marker) in enumerate(CONTROLS):
        off = 0.27 - j * 0.18
        for wl, y in zip(wls, ys):
            for metric, fill in (("edge_f1", color), ("higher_f1", "white")):
                r = df[(df.workload == wl) & (df.condition == cond) & (df.metric == metric)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar([r["mean"]], [y + off], xerr=[[r["mean"] - r.lo], [r.hi - r["mean"]]], fmt=marker, color=color, markerfacecolor=fill, markersize=4, capsize=2, linewidth=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([ps.wrap_name(w) for w in wls])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Edge $F_1$")                     # decision 2026-09-23: the filled / hollow key is in the caption
    ax.grid(True, axis="x", alpha=0.25)
    ax.tick_params(labelsize=8)


def handles_b() -> list:
    from matplotlib.lines import Line2D
    return [Line2D([], [], marker=m, color=c, linestyle="none", markersize=4, label=l) for _, l, c, m in CONTROLS]


def panels(axes, fa: pd.DataFrame, fb: pd.DataFrame, fc: pd.DataFrame) -> None:
    """The three panels on three axes, each titled, with its legend below it."""
    kw = dict(loc="upper center", fontsize=KEY, frameon=False, borderaxespad=0.0, columnspacing=0.7, handlelength=1.0, handletextpad=0.3)
    under = (0.4, -0.23)                                  # a legend's centre in its panel's axes coordinates
    A.panel(axes[0], fa, "noreg", wins=False)           # the win counts are in the standalone panel and the tables
    axes[0].set_xlabel("AULC(LeWi) $-$ AULC(Dense)")
    axes[0].legend(handles=A.handles(), ncol=1, bbox_to_anchor=under, **kw)
    if not fb.empty:
        panel_b(axes[1], fb)
        axes[1].legend(handles=handles_b(), ncol=2, bbox_to_anchor=under, **kw)
    else:
        axes[1].text(0.5, 0.5, "structure discovery\n(pending)", ha="center", va="center", transform=axes[1].transAxes)
    if not fc.empty:
        C.draw(axes[2], fc, [c for c in C.CELLS if c in set(fc.cell)], legend=False, wins=False)
        axes[2].set_xlabel("AULC(LeWi) $-$ AULC(variant)")
        axes[2].legend(handles=C.handles(), ncol=2, bbox_to_anchor=(0.3, under[1]), **kw)    # ends inside the text width
    else:
        axes[2].text(0.5, 0.5, "mechanism ablations\n(pending)", ha="center", va="center", transform=axes[2].transAxes)
    for ax, title in zip(axes, ("(a) Nuisance inputs", "(b) Wiring alignment", "(c) Mechanism ablations")):
        ax.set_title(title, fontsize=TITLE, loc="left")
        ax.tick_params(labelsize=TICK)
        ax.xaxis.label.set_size(LABEL)


def main() -> None:
    ps.apply_rc()
    fa = pd.read_csv(DATA / "fig6" / "feature_selection.csv")
    fb_path = DATA / "structure" / "fig6b.csv"
    fb = pd.read_csv(fb_path) if fb_path.exists() else pd.DataFrame()
    fc_path = DATA / "fig6" / "ablations.csv"
    fc = pd.read_csv(fc_path) if fc_path.exists() else pd.DataFrame()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": WIDTHS})
    panels(axes, fa, fb, fc)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.92, bottom=0.37, wspace=0.74)
    save(fig, "fig6")


if __name__ == "__main__":
    main()
