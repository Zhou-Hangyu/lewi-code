"""Shared vocabulary of the result figures: arm names, one colour per arm,
workload names, the sample-size axis, and the v0.5 reporting interval.

Every result figure draws the same arm in the same colour and line style, so
a reader who has learned one panel can read the rest without a legend. The
palette keeps the campaign report's family logic (penalties blue, LeWi warm,
GraphOracle green, other wiring methods muted) so the paper and the report
never disagree about which curve is which.

The LeWi arm key of the build is the switch CSFR_FIGURES_LEWI (LEWI below):
``lewi-v05`` for the 08-main build (CSFRBench v0.5), ``lewi-v1`` for wave 2
(13-main, CSFRBench v1.0). Every result builder keys its LeWi rows on
``ps.LEWI``; the two arms share one label and one colour.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from figstyle import DATA

LEWI = os.environ.get("CSFR_FIGURES_LEWI", "lewi-v05")

# ---- arms --------------------------------------------------------------
LABEL = {
    "noreg": "Dense",
    "l1": r"$\ell_1$",
    "l2": r"$\ell_2$",
    "l0hc": r"$L_0$",                   # hard-concrete L0 (design decision, intake/analysis-5.md 2026-09-04; ell_1 / ell_2 keep ell)
    "rigl": "RigL",
    "bimt": "BIMT",
    "fiedler": "Fiedler",
    "erb_positions": "neuron positions",
    "lewi-v05": "LeWi",
    "lewi-v1": "LeWi",                  # CSFRBench v1.0's LeWi (13-main): the floorless ball, plain readout
    "lewi-v1-rgm": "LeWi",              # the final arm (12-extra wave 5): LeWi-v1 with the readout gate and mask, rejection sampling of a disconnected initialization
    "lewi_reg-v05": "LeWi + gate penalty",
    "lewi-v3": "LeWi + learned frames + straight-through",   # the LeWi-v3 arm lv3_f_st (world_model_transfer_all)
    "graph_oracle(levels=0)": "GraphOracle(0)",
    "graph_oracle(levels=1)": "GraphOracle(1)",
    "graph_oracle(levels=2)": "GraphOracle(2)",     # 12-extra wave 2: depth-5 tree + random graphs
    "graph_oracle(levels=all)": "GraphOracle(all)",
    # the 12-extra wave-1 control arms (intake/analysis-4.md)
    "input_l0": r"Input $L_0$",                     # one hard-concrete gate per input coordinate before a Dense MLP
    "lewi_untied": "Untied low-rank scores",        # rank-3 scores with free sender / receiver embeddings per transition
    "lewi_edge": "Independent edge scores",         # one free score per connection
    # the 12-extra wave-3b matched independent-edge controls (intake/wave-3.md)
    "lewi_edge_offset": "Independent edge scores + shared offset",   # per-edge scores through LeWi's learned layer scale and offset
    "lewi_edge_st": "Independent edge scores + straight-through",    # per-edge scores whose closed edges keep a score gradient
}

COLOR = {
    "noreg": "#1a1a1a",
    "l1": "#3d7fd6",
    "l2": "#8cb8e8",
    "l0hc": "#24489a",
    "rigl": "#b0a23a",
    "bimt": "#8f67c2",
    "fiedler": "#8d8d8d",
    "erb_positions": "#9d6a49",
    "lewi-v05": "#f26b1d",              # bright industrial orange (decision, 2026-09-04; was red #d1352b)
    "lewi-v1": "#f26b1d",               # the same orange: one LeWi per build (LEWI)
    "lewi-v1-rgm": "#f26b1d",
    "lewi_reg-v05": "#ec8a3c",
    "lewi-v3": "#b3173a",               # crimson: the LeWi-v3 arm beside the orange of LeWi
    "graph_oracle(levels=0)": "#95c7a4",   # the light rung of the green ladder
    "graph_oracle(levels=1)": "#63ad7d",
    "graph_oracle(levels=2)": "#3f9360",            # the rung between GraphOracle(1) and (all)
    "graph_oracle(levels=all)": "#1f7a45",
    "input_l0": "#b5266e",              # raspberry: a gate family of its own, apart from the blue penalties
    "lewi_untied": "#b8860b",           # dark goldenrod: a LeWi variant, one step from the orange
    "lewi_edge": "#5b6b8f",             # slate: the far rung of the score ladder
    "lewi_edge_offset": "#7d5ba6",      # violet: per-edge with LeWi's shared offset (wave 3b)
    "lewi_edge_st": "#2a9d8f",          # teal: per-edge with the straight-through score gradient (wave 3b)
}

# curves: the three arms the paper compares are heavy, the field is light
WEIGHT = {"lewi-v05": 1.7, "lewi-v1": 1.7, "lewi-v1-rgm": 1.7, "lewi-v3": 1.7, "noreg": 1.4, "graph_oracle(levels=all)": 1.3,
          "input_l0": 1.3, "lewi_untied": 1.3, "lewi_edge": 1.3,
          "lewi_edge_offset": 1.3, "lewi_edge_st": 1.3}     # the controls are heavy in their own figures
DASHED = {"graph_oracle(levels=0)", "graph_oracle(levels=1)", "graph_oracle(levels=2)",
          "graph_oracle(levels=all)"}
BANDED = ("lewi-v05", "lewi-v1", "lewi-v1-rgm", "lewi-v3", "noreg", "graph_oracle(levels=all)", "input_l0", "lewi_untied", "lewi_edge",
          "lewi_edge_offset", "lewi_edge_st")

# comparator order in every forest: references first, then the field
FOREST_ORDER = ["graph_oracle(levels=all)", "graph_oracle(levels=2)", "graph_oracle(levels=1)",
                "graph_oracle(levels=0)", "noreg", "l1", "l2", "l0hc",
                "rigl", "bimt", "fiedler", "erb_positions"]
CURVE_ORDER = ["erb_positions", "fiedler", "bimt", "rigl", "l2", "l1", "l0hc",
               "graph_oracle(levels=0)", "graph_oracle(levels=1)", "graph_oracle(levels=2)",
               "graph_oracle(levels=all)", "lewi_edge", "lewi_edge_offset", "lewi_edge_st", "lewi_untied", "input_l0",
               "noreg", "lewi-v05", "lewi-v1", "lewi-v1-rgm"]
if LEWI not in LABEL:
    raise SystemExit(f"[paperstyle] CSFR_FIGURES_LEWI must name a LeWi arm of LABEL, not {LEWI!r}")

# ---- canonical paper notation (design note, 2026-09-04) -------------------
NAME = {**LABEL, "erb_positions": "Neuron positions"}

# main-text learning curves (sample_efficiency): four highlighted arms, the
# generic structured / sparse baselines in one muted grey family
HIGHLIGHT = ("lewi-v05", "lewi-v1", "lewi-v1-rgm", "noreg", "graph_oracle(levels=0)", "graph_oracle(levels=1)",
             "graph_oracle(levels=2)", "graph_oracle(levels=all)")
GENERIC = ["l0hc", "rigl", "bimt", "l1", "l2", "fiedler", "erb_positions"]
GRAY = {arm: "#9c9c9c" for arm in GENERIC}      # one light grey, thin and translucent (design decision)

# panel titles of the result figures (chosen wording, 2026-09-04)
TITLE = {
    "tree_functional_anova": "Depth-4 tree",
    "dag_functional_anova": "Depth-3 reuse DAG",
    "tree_fourier": "Depth-4 Fourier tree",
    "flat_functional_anova": "Flat modules",
    "dag_fourier_v2": "Depth-3 Fourier reuse DAG",
    "dense_fourier": "High-arity node",
    "rand_functional_anova": "Sampled depth-4 DAGs",
    "tree_functional_anova:depth4": "Depth-5 tree",
}


def wrap_name(name: str, width: int = 17) -> str:
    """A tick label or panel title longer than `width` characters breaks at its first space ("Sampled" over "depth-4
    DAGs"), so a long name neither widens the figure nor reaches the next panel."""
    return name.replace(" ", "\n", 1) if len(name) > width else name

# ---- workloads (public graph-theoretic depth: the readout counts) -------
WORKLOAD = {
    "tree_functional_anova": "tree (depth 4)",
    "dag_functional_anova": "reuse DAG (depth 3)",
    "tree_fourier": "Fourier tree (depth 4)",
    "flat_functional_anova": "flat modules",
    "dag_fourier_v2": "Fourier reuse DAG (depth 3)",
    "dense_fourier": "Fourier dense node",
    "rand_functional_anova": "held-out graphs (depth 4)",
    "tree_functional_anova:depth4": "tree (depth 5)",
}

WINDOW = [2 ** k for k in range(9, 16)]
# the depth-5 tree's grid: 2^11 ... 2^17 in 08-main (v0.5), 2^12 ... 2^18 in 13-main (v1.0) and 14-main (v1.1)
WINDOW_DEPTH5 = [2 ** k for k in (range(12, 19) if DATA.name in ("data-v2", "data-main-14") else range(11, 18))]
# the reporting window of each workload (the depth-5 tree has its own)
WINDOW_OF = {wl: (WINDOW_DEPTH5 if wl == "tree_functional_anova:depth4" else WINDOW)
             for wl in WORKLOAD}


def style(ax, k: int, *, band: bool = False) -> dict:
    """Line keyword arguments for arm k."""
    return {"color": COLOR[k], "linewidth": WEIGHT.get(k, 0.85),
            "linestyle": (2.6, (3.2, 1.6)) if k in DASHED else "-",
            "solid_capstyle": "round", "zorder": 3 if k in WEIGHT else 2}


def arm_legend(fig, keys, *, y: float = 0.0, ncol: int | None = None, labels=None, extra=(), fontsize: float = 6.5,
               handlelength: float = 2.0) -> None:
    """One centred legend row under the panels naming the arms drawn (decision, 2026-09-12: every result figure
    carries a legend). `extra` adds (handle, label) pairs after the arms."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], **style(None, k)) for k in keys] + [h for h, _ in extra]
    names = list(labels or [NAME[k] for k in keys]) + [l for _, l in extra]
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol or len(handles), frameon=False,
               fontsize=fontsize, handlelength=handlelength, handletextpad=0.5, columnspacing=1.3, borderaxespad=0.0)


def t_interval(values) -> tuple[float, float, float]:
    """Mean and two-sided 95% Student-t interval over target instances - the
    v0.5 reporting interval, computed by the benchmark package itself."""
    from csfrbench.metrics import instance_t_interval
    iv = instance_t_interval(np.asarray(values, dtype=float))
    return iv["mean"], iv["ci95"][0], iv["ci95"][1]


def aulc(ns, values) -> float:
    from csfrbench.metrics import aulc as _aulc
    return _aulc(ns, values)


def curve_stats(df: pd.DataFrame, value: str = "test_nmse") -> pd.DataFrame:
    """Mean and 95% t-interval of `value` over instances at each n."""
    rows = []
    for n, g in df.groupby("n"):
        m, lo, hi = t_interval(g[value].to_numpy())
        rows.append({"n": int(n), "mean": m, "lo": lo, "hi": hi,
                     "instances": len(g)})
    return pd.DataFrame(rows).sort_values("n")


def n_axis(ax, ns, *, ticks=None, label: bool = True) -> None:
    ax.set_xscale("log", base=2)
    ticks = list(ticks) if ticks is not None else [ns[0], ns[len(ns) // 2], ns[-1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$2^{{{int(np.log2(t))}}}$" for t in ticks])
    ax.set_xticks([], minor=True)
    ax.set_xlim(ns[0] / 1.25, ns[-1] * 1.25)
    if label:
        ax.set_xlabel("Training samples $n$", labelpad=1.5)


def panel_label(ax, text: str, *, x: float = -0.02, y: float = 1.02, fontsize: float = 7.6) -> None:
    """Panel letter in the journal convention: bold "(a)" at the title size,
    set flush with the panel's top-left corner."""
    ax.text(x, y, f"({text})", transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", ha="left", va="bottom")


def tidy(ax, *, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=2.5, pad=1.5)
    if grid:
        ax.grid(axis=grid, color="#e6e6e6", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)


def apply_rc() -> None:
    plt.rcParams.update({
        "font.size": 7.2,
        "axes.titlesize": 7.6,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.6,
        "axes.linewidth": 0.6,
        "lines.markersize": 2.6,
        # the type (STIXGeneral text, STIX math) is figstyle's, for every builder
    })
