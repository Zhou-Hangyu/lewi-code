"""Figure 6(c): mechanism ablations - paired AULC differences between the canonical LeWi (lewi-v1-rgm; the all-active
twin at d = 8, 64,000 steps) and its four variants (soft-only, fixed embeddings, random sparse mask, straight-through mask;
17-main-clean panel abl8: the same instances 6000-6015, grid 2^9 ... 2^15, budget, space and trial draws), grouped by
target family with a common zero reference. The sign is the study's: AULC(LeWi) - AULC(variant), negative = LeWi better (the
direction of panel (a) and of every table). Data:
data-final-clean/fig6/ablations.csv (18-main-d8 analysis/ablations/contrasts.csv). The appendix companions read
sensitivity_r.csv, sensitivity_rad.csv and density.csv (fig6_sensitivity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import paperstyle as ps
from figstyle import DATA, save

CELLS = ["tree8", "dag8", "ftree8", "flat8", "dense8"]          # the workload order of panel (a)
NAME = {"tree8": "Depth-4 tree", "dag8": "Reuse DAG", "ftree8": "Fourier tree", "flat8": "Flat modules", "dense8": "High-arity node",
        "rand8": "Sampled depth-4 DAGs", "tree": "Depth-4 tree", "dag": "Reuse DAG"}     # Figure 6's target names (decision 2026-09-23); tree, dag: d = 32
# the two mask variants first, side by side (decision 2026-09-22), then the two frozen-geometry variants
# panel (c): the mask-only forward (17-main-clean amendment 14, the two LeWi studies) in the straight-through arm's place (decision
# 2026-09-22); the appendix extension keeps the straight-through arm beside it. Colours (decision 2026-09-23): orange is LeWi's
# alone, so soft-only is teal; mask-only blue, fixed embeddings purple, the random mask grey, the straight-through arm magenta
VARIANTS = [("lewi-abl-soft", "soft-only (no mask)", "#2a9d8f", "o"), ("lewi-mask", "mask-only forward", "#3182bd", "^"),
            ("lewi-abl-fixed", "fixed embeddings", "#756bb1", "s"), ("lewi-abl-random", "random sparse mask", "#636363", "D")]
VARIANTS_EXT = [VARIANTS[0], VARIANTS[1], ("lewi-abl-ste", "straight-through mask", "#e7298a", "v"), VARIANTS[2], VARIANTS[3]]


def handles(variants=None) -> list:
    from matplotlib.lines import Line2D
    return [Line2D([], [], marker=m, color=c, linestyle="none", markersize=4, label=l) for _, l, c, m in (variants or VARIANTS)]


def draw(ax, con: pd.DataFrame, cells: list[str], legend: bool = True, wins: bool = True, variants=None) -> None:
    VARIANTS = variants or globals()["VARIANTS"]
    ys = np.arange(len(cells))[::-1]
    k = len(VARIANTS)
    for j, (v, label, color, marker) in enumerate(VARIANTS):
        off = (k - 1) / 2 * 0.2 - j * 0.2
        for cell, y in zip(cells, ys):
            r = con[(con.cell == cell) & (con.variant == v)]
            if r.empty:
                continue
            r = r.iloc[0]
            x, lo, hi = r["mean"], r.ci95_lo, r.ci95_hi       # AULC(LeWi) - AULC(variant): negative = LeWi better, the direction of panel (a) and of every table
            if r.off_scale:
                ax.plot([r["median"]], [y + off], marker=marker, markerfacecolor="white", markeredgecolor=color, markersize=5, linestyle="none")
            else:
                ax.errorbar([x], [y + off], xerr=[[x - lo], [hi - x]], fmt=marker, color=color, markersize=4, capsize=2, linewidth=0.9)
            if wins:
                ax.text(1.01, y + off, f"{int(r.wins)}/{int(r.n_pairs)}", transform=ax.get_yaxis_transform(), fontsize=5.5, va="center", color=color)
    ax.axvline(0, color="#888", linewidth=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([ps.wrap_name(NAME[c]) for c in cells])
    ax.set_xlabel("AULC(LeWi) $-$ AULC(variant), paired per instance")
    ax.grid(True, axis="x", alpha=0.25)
    ax.tick_params(labelsize=8)
    if legend:
        ax.legend(handles=handles(), loc="upper center", bbox_to_anchor=(0.42, -0.17), ncol=2, fontsize=6.5, frameon=False, borderaxespad=0.0, columnspacing=0.8, handletextpad=0.3)


def main() -> None:
    ps.apply_rc()
    path = DATA / "fig6" / "ablations.csv"
    con = pd.read_csv(path) if path.exists() and path.stat().st_size > 5 else pd.DataFrame()
    if con.empty:
        print("[fig6c] no ablation contrasts yet (18-main-d8 src/ablations.py): nothing drawn")
        return
    cells = [c for c in CELLS if c in set(con.cell)]
    fig, ax = plt.subplots(figsize=(3.2, 3.5))
    draw(ax, con, cells)
    fig.subplots_adjust(left=0.30, right=0.89, top=0.98, bottom=0.29)
    save(fig, "fig6c_ablations")
    for _, r in con.iterrows():
        print(f"[fig6c] {r.workload:14s} {r.label:24s} LeWi - variant {r['mean']:+.3f} [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}] wins {int(r.wins)}/{int(r.n_pairs)}")
    # the extension (appendix): every clean workload incl. the sampled graphs beside the nuisance-input setting (tree, reuse DAG)
    p32 = DATA / "fig6" / "ablations_d32.csv"
    con32 = pd.read_csv(p32) if p32.exists() and p32.stat().st_size > 5 else pd.DataFrame()
    all8 = [c for c in CELLS + ["rand8"] if c in set(con.cell)]
    if len(con32) or "rand8" in all8:
        fig2, axes = plt.subplots(1, 2, figsize=(6.6, 4.0), gridspec_kw={"width_ratios": [1, 1]})
        draw(axes[0], con, all8, legend=False, variants=VARIANTS_EXT)
        axes[0].set_title("clean inputs (d = 8)", fontsize=9)
        if len(con32):
            draw(axes[1], con32, [c for c in ("tree", "dag") if c in set(con32.cell)], legend=False)
        else:
            axes[1].text(0.5, 0.5, "pending", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("24 nuisance inputs (d = 32)", fontsize=9)
        fig2.legend(handles=handles(VARIANTS_EXT), loc="lower center", ncol=5, fontsize=7, frameon=False)
        fig2.subplots_adjust(left=0.14, right=0.95, top=0.93, bottom=0.21, wspace=0.62)
        save(fig2, "fig6c_ablations_ext")


if __name__ == "__main__":
    main()
