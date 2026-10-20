"""Appendix: LeWi's overall connection density at the end of training against the training-set size n on the final runs of the
two main settings (decision 2026-09-24).

Source: 18-main-d8/density/tables (density/src/density.py). by_size.csv: per setting, workload and n, the mean over target
instances of the overall active-edge fraction (open edges / all edges over every transition) of the final selected LeWi model
(lewi-v1-rgm; the selected checkpoint best.pt) with its 95% Student-t interval, N = the instances with a run on record at that
size; per_run.csv: the initialization density of every run (init.pt). Panel (a): d = 8, the seven workloads (32x4, 64,000
steps, n = 2^9 .. 2^15) and the depth-5 tree at d = 16 (64x8, 128,000 steps, 2^11 .. 2^17; dashed); panel (b): d = 32 (24
nuisance inputs), the same workloads and the depth-5 tree at d = 32. A band is drawn where every instance of the workload has a
run on record at that size; a hollow marker is a size with runs from fewer instances (its mean only). The dotted line is the
initialization density, the mean over every run of the panel. The rows drawn: connection_density.csv beside the figure.
"""
from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import paperstyle as ps
from connection_density_table import NAME, SETTINGS, TABLES, family
from figstyle import OUT, save

STEM = "connection_density"
INIT_COLOR = "#6b6b6b"
STYLE = {"tree": ("#1a1a1a", "o", "-"), "dag": ("#3d7fd6", "s", "-"), "ftree": ("#2a9d8f", "^", "-"), "flat": ("#8f67c2", "D", "-"),
         "dense": ("#b3173a", "v", "-"), "fdag": ("#b8860b", "P", "-"), "rand": ("#8d8d8d", "X", "-"), "tree5": ("#1f7a45", "o", (0, (4, 2)))}
TICKS = [2 ** k for k in (9, 11, 13, 15, 17)]
TITLE_PT, LABEL_PT, TICK_PT, KEY_PT = 8.5, 8.0, 7.5, 8.0     # print sizes at the include width (structure_alignment_vs_n's scale)
WIDTH, LEFT, RIGHT, GAP = 5.5, 0.46, 0.04, 0.16               # inches; the paper's text width
TOP, PANEL, FOOT = 0.26, 1.95, 0.76                         # titles; panel height (the y label's length); ticks, x label, legend


def panel(ax, by_size: pd.DataFrame, runs: pd.DataFrame, setting: str, cells: list[str], title: str) -> list[dict]:
    rows = []
    for cell in cells:
        color, marker, dash = STYLE[family(cell)]
        b = by_size[(by_size.setting == setting) & (by_size.cell == cell)].sort_values("n")
        instances = runs[(runs.setting == setting) & (runs.cell == cell)].function_seed.nunique()
        full = b.N == instances
        ax.plot(b.n, b.density_end_overall_mean, color=color, linestyle=dash, linewidth=0.9, zorder=3)
        for sel, face in ((full, color), (~full, "white")):
            ax.plot(b.n[sel], b.density_end_overall_mean[sel], marker=marker, linestyle="none", color=color, markerfacecolor=face,
                    markeredgewidth=0.7, zorder=4)
        ax.fill_between(b.n[full], b.density_end_overall_lo[full], b.density_end_overall_hi[full], color=color, alpha=0.13, linewidth=0, zorder=2)
        for r, f in zip(b.itertuples(), full):
            rows.append(dict(setting=setting, cell=cell, workload=NAME[family(cell)], n=int(r.n), N=int(r.N), full_coverage=bool(f),
                             density_end_mean=r.density_end_overall_mean, density_end_lo=r.density_end_overall_lo, density_end_hi=r.density_end_overall_hi,
                             density_init_mean=r.density_init_overall_mean))
    init = float(runs[runs.setting == setting].density_init_overall.mean())
    ax.axhline(init, color=INIT_COLOR, linewidth=0.8, linestyle=(0, (1.2, 1.6)), zorder=1)
    ps.n_axis(ax, TICKS, ticks=TICKS)
    ax.xaxis.label.set_fontsize(LABEL_PT)
    ax.tick_params(labelsize=TICK_PT)
    ax.set_ylim(0, 0.75)
    ax.set_title(title, fontsize=TITLE_PT, pad=3)
    ps.tidy(ax)
    return rows


def main() -> None:
    ps.apply_rc()
    by_size = pd.read_csv(TABLES / "by_size.csv")
    runs = pd.read_csv(TABLES / "per_run.csv")
    height = TOP + PANEL + FOOT
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, height), sharey=True)
    rows = []
    for ax, letter, (setting, title, cells) in zip(axes, "ab", SETTINGS):
        rows += panel(ax, by_size, runs, setting, cells, title)
        ps.panel_label(ax, letter, x=-0.04, fontsize=TITLE_PT)
    axes[0].set_ylabel("Active-edge fraction at selected checkpoint", fontsize=LABEL_PT, labelpad=2)
    handles = [Line2D([], [], color=c, marker=m, linestyle=d, linewidth=0.9, markersize=2.6, label=NAME[k]) for k, (c, m, d) in STYLE.items()]
    handles.append(Line2D([], [], color=INIT_COLOR, linewidth=0.8, linestyle=(0, (1.2, 1.6)), label="initialization"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=KEY_PT, frameon=False, handlelength=1.5, columnspacing=0.85, handletextpad=0.5)   # the key within the 5.5 in width
    width = (WIDTH - LEFT - RIGHT - GAP) / 2
    fig.subplots_adjust(left=LEFT / WIDTH, right=1 - RIGHT / WIDTH, top=1 - TOP / height, bottom=FOOT / height, wspace=GAP / width)
    save(fig, STEM)
    pd.DataFrame(rows).to_csv(OUT / f"{STEM}.csv", index=False)
    print(f"[{STEM}] {len(rows)} rows drawn from {TABLES.relative_to(TABLES.parents[4])}")


if __name__ == "__main__":
    main()
