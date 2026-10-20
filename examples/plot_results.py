"""Figures of the paper's CSFRBench results, drawn from the records in the package (no training).

    python examples/plot_results.py curves                       # learning curves, primary setting (Figures 4 and 11)
    python examples/plot_results.py curves --setting nuisance    # with inert inputs (Figure 15)
    python examples/plot_results.py contrasts                    # paired AULC differences LeWi - method (Figures 12, 16)
    python examples/plot_results.py inputs                       # LeWi - Dense with and without inert inputs (Figure 6a)
    python examples/plot_results.py curves --runs runs/mine      # your runs (csfrbench run) beside the records

Means are arithmetic over the target instances, bands and bars two-sided 95% Student-t intervals. Needs matplotlib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from csfrbench import analysis, record, runner, suite

COLOUR = {"lewi": "#e66101", "dense": "#333333", "go1": "#9e9ac8", "go2": "#756bb1", "go_all": "#54278f"}
OTHER = "#b5b5b5"


def _winners(setting: str, runs: str | None) -> list[dict]:
    winners = list(record.winners(setting))
    if runs:
        mine = [r for r in runner.read_runs(runs) if r["setting"] == setting and not r.get("failed")]
        winners += analysis.select_winners(mine)
    return winners


def curves(args, plt) -> None:
    rows = analysis.learning_curves(_winners(args.setting, args.runs))
    names = list(suite.SETTINGS[args.setting])
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), constrained_layout=True)
    extra = set()
    for ax, name in zip(axes.flat, names):
        w = suite.workload(args.setting, name)
        arms = sorted({r["arm"] for r in rows if r["workload"] == name},
                      key=lambda a: (a in COLOUR or a not in suite.ARMS, a))
        main = [r["mean"] for r in rows if r["workload"] == name and (r["arm"] in COLOUR or r["arm"] not in suite.ARMS)
                and r["mean"] <= 1.5]                                # a mean above 1.5 comes from a divergent run
        top = min(1.05, 1.1 * max(main)) if main else 1.05           # a highlighted mean beyond it is marked with a caret
        for arm in arms:
            pts = sorted((r for r in rows if r["workload"] == name and r["arm"] == arm), key=lambda r: r["n"])
            x = np.log2([r["n"] for r in pts])
            mean = np.array([r["mean"] for r in pts])
            known = arm in suite.ARMS
            if not known:
                extra.add(arm)
            bold = arm in COLOUR or not known
            colour = COLOUR.get(arm, OTHER if known else "#1b9e77")
            ax.plot(x, np.minimum(mean, top), color=colour, lw=1.8 if bold else 0.8, zorder=3 if bold else 1)
            over = mean > top
            if bold and over.any():
                ax.plot(x[over], np.full(int(over.sum()), top), "^", color=colour, ms=5, clip_on=False, zorder=4)
            if bold:
                ax.fill_between(x, np.minimum([r["lo"] for r in pts], top), np.minimum([r["hi"] for r in pts], top),
                                color=colour, alpha=0.15, lw=0)
        ax.set_title(w.paper_name, fontsize=9)
        ax.set_xticks(np.log2(w.profile.sizes[::3]))
        ax.set_xticklabels([f"$2^{{{int(np.log2(n))}}}$" for n in w.profile.sizes[::3]])
        ax.set_ylim(0, top)
        ax.grid(alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("test NMSE")
    handles = [plt.Line2D([], [], color=COLOUR[a], lw=1.8) for a in ("lewi", "dense", "go1", "go2", "go_all")]
    labels = [suite.ARMS[a].paper_name for a in ("lewi", "dense", "go1", "go2", "go_all")]
    handles += [plt.Line2D([], [], color="#1b9e77", lw=1.8) for _ in sorted(extra)] + [plt.Line2D([], [], color=OTHER, lw=0.8)]
    labels += sorted(extra) + ["L0, RigL, BIMT, L1, L2, Fiedler, neuron positions"]
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), fontsize=8, frameon=False)
    fig.supxlabel("training samples n (a caret marks a mean above the axis)", fontsize=9)
    _save(fig, args, f"curves_{args.setting}")


def contrasts(args, plt) -> None:
    rows = record.contrasts(args.setting)
    names = list(suite.SETTINGS[args.setting])
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True, sharex=False)
    for ax, name in zip(axes.flat, names):
        sub = [r for r in rows if r["workload"] == name]
        sub.sort(key=lambda r: suite.ARM_ORDER.index(r["arm"]))
        y = np.arange(len(sub))[::-1]
        for yi, r in zip(y, sub):
            colour = COLOUR.get(r["arm"], "#555555")
            ax.errorbar(r["mean"], yi, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt="o", ms=3.5,
                        color=colour, capsize=2, lw=1)
        ax.axvline(0, color="#999999", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([suite.ARMS[r["arm"]].paper_name for r in sub], fontsize=7)
        ax.set_title(suite.workload(args.setting, name).paper_name, fontsize=9)
        ax.grid(axis="x", alpha=0.2)
    fig.supxlabel("AULC(LeWi) - AULC(method), mean and 95% interval over target instances (negative favours LeWi)",
                  fontsize=9)
    _save(fig, args, f"contrasts_{args.setting}")


def inputs(args, plt) -> None:
    names = [n for n in suite.SETTINGS["primary"] if n != "depth5_tree"] + ["depth5_tree"]
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    for k, (setting, colour, dy) in enumerate((("primary", "#e66101", -0.12), ("nuisance", "#7a3d00", 0.12))):
        rows = {r["workload"]: r for r in record.contrasts(setting) if r["arm"] == "dense"}
        for i, name in enumerate(names):
            r = rows[name]
            ax.errorbar(r["mean"], i + dy, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt="s" if k else "o",
                        ms=4, color=colour, capsize=2, lw=1,
                        label=("all inputs relevant" if setting == "primary" else "24 inert inputs added") if i == 0 else None)
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([suite.workload("primary", n).paper_name for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AULC(LeWi) - AULC(Dense), mean and 95% interval (negative favours LeWi)", fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.2)
    _save(fig, args, "inputs")


def _save(fig, args, stem: str) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    print(f"written {out / stem}.png and .pdf")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("figure", choices=["curves", "contrasts", "inputs"])
    p.add_argument("--setting", default="primary", choices=["primary", "nuisance"])
    p.add_argument("--runs", help="a folder of csfrbench run: its methods are drawn beside the records")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    {"curves": curves, "contrasts": contrasts, "inputs": inputs}[args.figure](args, plt)


if __name__ == "__main__":
    main()
