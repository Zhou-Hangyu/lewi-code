"""The computation graphs of CSFRBench (Figures 3 and 8) and the GraphOracle connectivity masks (Figure 10).

    python examples/plot_targets.py graphs           # one target of every workload (Figure 3)
    python examples/plot_targets.py sampled          # the 32 sampled depth-4 DAGs (Figure 8)
    python examples/plot_targets.py masks            # the masks of GraphOracle(1), (2), (all) on the depth-5 tree (Figure 10)

Local functions are numbered left to right, bottom to top; an input dot has the colour of the function that reads it;
the black disc is the output. Needs matplotlib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from csfrbench import suite, viz
from csfrbench.methods import graph_oracle


def graphs(plt, out: Path) -> None:
    names = ["depth4_tree", "depth5_tree", "reuse_dag", "sampled_dags", "flat_modules", "high_arity_node"]
    fig, axes = plt.subplots(1, len(names), figsize=(2.1 * len(names), 2.2), constrained_layout=True)
    for ax, name in zip(axes, names):
        w = suite.workload("primary", name)
        viz.draw_target(ax, w.target(w.instances[0]))
        ax.set_title(w.paper_name, fontsize=8)
    _save(fig, out, "graphs")


def sampled(plt, out: Path) -> None:
    w = suite.workload("primary", "sampled_dags")
    fig, axes = plt.subplots(4, 8, figsize=(13, 6.6), constrained_layout=True)
    for ax, instance in zip(axes.flat, w.instances):
        viz.draw_target(ax, w.target(instance), numbers=False)
        ax.set_title(str(instance), fontsize=7)
    _save(fig, out, "sampled_dags")


def masks(plt, out: Path) -> None:
    w = suite.workload("primary", "depth5_tree")
    target = w.target(w.instances[0])
    levels = [1, 2, "all"]
    fig, axes = plt.subplots(len(levels), w.profile.hidden + 1, figsize=(1.3 * (w.profile.hidden + 1), 1.4 * len(levels)),
                             constrained_layout=True)
    for row, level in zip(axes, levels):
        ms = graph_oracle.graph_masks(target, w.profile.width, w.profile.hidden, levels=level)
        for ax, (k, m) in zip(row, enumerate(ms)):
            ax.imshow(m.numpy(), cmap="Greys", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if level == levels[0]:
                ax.set_title(f"layer {k + 1}" if k < w.profile.hidden else "readout", fontsize=7)
        row[0].set_ylabel(f"GraphOracle({level})", fontsize=7)
    _save(fig, out, "graph_oracle_masks")


def _save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    print(f"written {out / stem}.png and .pdf")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("figure", choices=["graphs", "sampled", "masks"])
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    {"graphs": graphs, "sampled": sampled, "masks": masks}[args.figure](plt, Path(args.out))


if __name__ == "__main__":
    main()
