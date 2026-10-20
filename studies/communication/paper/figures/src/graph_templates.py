"""Figure: the fixed graph templates.

Caption - The six settings of Figure 3, as graphs: qualitatively different
structural regimes, not a progression of matched interventions. Light nodes
are input leaves, blue discs are local functions, green is the readout; the
panels follow Figure 3's order, so a reader can carry a learning curve back
to its target. The depth-3 reuse DAG matches the depth-4 tree in
active-input count (8) and local-function count (7); its two shared
intermediates are each read by two level-2 nodes, and shared ancestry merges
only at the additive readout. The flat modules are four disjoint input pairs
under one additive readout, the high-arity node is a single eight-input function,
and the depth-5 tree doubles the active inputs (16) and adds a level. The
last panel is one draw from the held-out graph distribution (development
seed 6000, drawn by the benchmark's own sampler; the sealed instances stay
unshown), which fixes seven local functions at depth 3 while strand shape,
readout arity and reuse vary.

Built for the full text width (5.5 x 1.02 in) and read at a glance: node
numbers, input indices and the readout's y label are dropped (decision
2026-09-14 - the paper never refers to them), so the discs carry no type and
the panel is half its former height. One horizontal scale over all six
panels (the column widths are proportional to each graph's leaf span), so a
disc means the same size everywhere; the vertical scale fills each panel, so
levels are read within a panel, not across. Titles carry Figure 3's wording at
7.8 pt (decision 2026-09-14: about 9 % below the first six-panel build), with
one exception: the last panel is named "Sampled depth-4 DAG" (decision
2026-09-14; "DAG" in place of "graph" since 2026-09-25), because it shows the single draw it draws, where Figure 3's
panel pools the 32 sealed graphs of the same distribution. Every graph is drawn by csfrbench.viz.draw_graph (one grammar for
the paper, the spec page and the slides); the builder then lightens the
panel - smaller discs, thinner edges - because a benchmark-definition figure
should not carry the weight of a result.
"""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from csfrbench.tasks.graph import GraphSpec, balanced_refs
from csfrbench.tasks.topologies import dag8_graph, sample_random_graph7
from csfrbench.utils.seeds import derive
from csfrbench.viz import LEVEL_DY, draw_graph, layout_graph

from figstyle import save

RAND_FAMILY = "rand_functional_anova"
RAND_SEED = 6000         # a development draw; the final bank stays sealed


def balanced_tree(profile) -> GraphSpec:
    """The benchmark's balanced binary tree at this layer profile."""
    parents = [np.array([[2 * j, 2 * j + 1] for j in range(width)])
               for width in profile[1:]]
    return balanced_refs(profile, 2, parents)


def random_graph(function_seed: int) -> GraphSpec:
    """The graph the benchmark itself draws for this held-out function seed."""
    return sample_random_graph7(
        np.random.default_rng(derive(RAND_FAMILY, int(function_seed), "structure")))


TEMPLATES = (
    ("Depth-4 tree", balanced_tree((4, 2, 1))),
    ("Depth-5 tree", balanced_tree((8, 4, 2, 1))),
    ("Depth-3 reuse DAG", dag8_graph()),
    ("Sampled depth-4 DAG", random_graph(RAND_SEED)),
    ("Flat modules", GraphSpec(8, [(0, 1), (2, 3), (4, 5), (6, 7)],
                               (0, 1, 2, 3), (1.0,) * 4)),
    ("High-arity node", GraphSpec(8, [tuple(range(8))], (0,), (1.0,))),
)
WIDTH, HEIGHT = 5.5, 1.02
X_PAD = 0.6              # leaf units of margin on each side of a panel
Y_PAD = 0.09             # top and bottom margin as a fraction of the panel's drawn height
LEAF_SIZE = 24           # disc areas in points^2: leaves, local functions and readout
NODE_SIZE = 62
EDGE_WIDTH = 0.55
DISC_EDGE = 0.7
TITLE_SIZE = 7.8
TITLE_WRAP = 13          # characters per title line, as in the result figures


def extent(graph) -> float:
    """Width of one panel in leaf units: the drawn span plus both margins."""
    xs = [x for x, _y in layout_graph(graph)[0].values()]
    return max(xs) - min(xs) + 2 * X_PAD


def panel(ax, title: str, graph, scale: float = 1.0) -> None:
    """One template in the shared grammar, stripped of its labels and scaled
    to the row's one horizontal scale; `scale` shrinks the discs and edges
    with a panel drawn smaller than this figure's (the title keeps its size)."""
    draw_graph(ax, graph, title=textwrap.fill(title, TITLE_WRAP))
    pos, depths = layout_graph(graph)
    xs = [x for x, _y in pos.values()]
    ax.set_xlim(min(xs) - X_PAD, max(xs) + X_PAD)
    top = LEVEL_DY * (max(depths) + 1)
    ax.set_ylim(-Y_PAD * top, (1 + Y_PAD) * top)    # margins in units of the panel's own
                                                    # height, so leaves and readouts line up across the row
    for text in tuple(ax.texts):          # input indices, node numbers, the readout's y
        text.remove()
    for line in ax.lines:
        line.set_linewidth(EDGE_WIDTH * scale)
    sizes = [LEAF_SIZE * scale**2] * graph.A + [NODE_SIZE * scale**2] * (graph.S + 1)
    for coll, size in zip(ax.collections, sizes):
        coll.set_sizes([size])
        coll.set_linewidths([DISC_EDGE * scale])
    ax.title.set_fontsize(TITLE_SIZE)


def main() -> None:
    fig, axes = plt.subplots(1, len(TEMPLATES), figsize=(WIDTH, HEIGHT),
                             gridspec_kw={"width_ratios": [extent(g) for _t, g in TEMPLATES]})
    fig.subplots_adjust(left=0.022, right=0.978, top=0.785, bottom=0.025, wspace=0.14)   # the end titles overhang their panels: the margins keep the tight bounding box at the text width
    for ax, (title, graph) in zip(axes, TEMPLATES):
        panel(ax, title, graph)
    save(fig, "graph_templates")


if __name__ == "__main__":
    main()
