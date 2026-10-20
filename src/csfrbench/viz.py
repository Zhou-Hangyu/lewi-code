"""The one graph-drawing grammar for every CSFRBench figure.

Every computation-graph panel in the paper and the specification page renders
through ``draw_graph`` so the visual language is identical everywhere: light
leaves numbered 1..A left to right just under the node, blue discs for local
functions numbered left to right then bottom to top, a green disc marked y for
the readout, thin warm-gray edges. Fixed axis limits keep node sizes and level
heights comparable across panels.

Layout and ordering are pure functions of a ``GraphSpec`` (no matplotlib
import at module level); ``draw_graph`` only needs the axes it is given.
"""
from __future__ import annotations

import numpy as np

FUNC_FILL = "#2b5fac"
READOUT_GREEN = "#79a884"
LEAF_FILL = "#e4e2db"
LEAF_EDGE = "#ffffff"
EDGE_COLOR = "#52514e"
LEAF_NUM_COLOR = "#948f85"

LEVEL_DY = 1.25
MIN_DX = 1.4           # minimum horizontal gap between function discs
XLIM = (-4.45, 4.45)


def node_depths(graph) -> list[int]:
    """Depth of every function node: 1 + max parent depth (leaves are 0)."""
    depths: list[int] = []
    for refs in graph.node_parents:
        depths.append(1 + max((0 if r < graph.A else depths[r - graph.A])
                              for r in refs))
    return depths


def layout_graph(graph):
    """Positions for one graph: leaves at y = 0 in first-use order (so the
    left-to-right numbering is 1..A by construction), function nodes above
    their parents with a minimal-spacing collision sweep."""
    order: list[int] = []

    def visit(ref: int) -> None:
        if ref < graph.A:
            if ref not in order:
                order.append(ref)
            return
        for parent in graph.node_parents[ref - graph.A]:
            visit(parent)

    for i in graph.readout_indices:
        visit(graph.A + i)
    for leaf in range(graph.A):
        if leaf not in order:
            order.append(leaf)

    pos = {leaf: (rank - (graph.A - 1) / 2, 0.0)
           for rank, leaf in enumerate(order)}
    depths = node_depths(graph)
    for level in range(1, max(depths) + 1):
        row = [v for v in range(graph.S) if depths[v] == level]
        xs = [float(np.mean([pos[r][0] for r in graph.node_parents[v]]))
              for v in row]
        ordered = sorted(zip(row, xs), key=lambda pair: pair[1])
        placed: list[float] = []
        for _v, x in ordered:
            placed.append(x if not placed else max(x, placed[-1] + MIN_DX))
        shift = float(np.mean(xs)) - float(np.mean(placed))
        for (v, _x), x_new in zip(ordered, placed):
            pos[graph.A + v] = (x_new + shift, LEVEL_DY * level)
    return pos, depths


def function_order(graph, pos, depths) -> list[int]:
    """Function nodes numbered left to right, bottom to top: sorted by
    (depth, x). Monotonic increasing along both axes by construction."""
    return sorted(range(graph.S),
                  key=lambda v: (depths[v], pos[graph.A + v][0]))


def draw_graph(ax, graph, *, title=None, note=None):
    """Render one computation graph in the canonical grammar."""
    pos, depths = layout_graph(graph)

    for v, refs in enumerate(graph.node_parents):
        for r in refs:
            (x0, y0), (x1, y1) = pos[r], pos[graph.A + v]
            ax.plot([x0, x1], [y0, y1], color=EDGE_COLOR, lw=0.85,
                    alpha=0.9, zorder=1)
    yx = float(np.mean([pos[graph.A + i][0] for i in graph.readout_indices]))
    yy = LEVEL_DY * (max(depths) + 1)
    for i in graph.readout_indices:
        (x0, y0) = pos[graph.A + i]
        ax.plot([x0, yx], [y0, yy], color=EDGE_COLOR, lw=0.85,
                alpha=0.9, zorder=1)

    leaves_lr = sorted(range(graph.A), key=lambda leaf: pos[leaf][0])
    for rank, leaf in enumerate(leaves_lr, start=1):
        x, y = pos[leaf]
        ax.scatter([x], [y], s=55, color=LEAF_FILL, zorder=2,
                   edgecolor=LEAF_EDGE, lw=0.9)
        ax.annotate(str(rank), (x, y), xytext=(0, -6.6),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=5.8, color=LEAF_NUM_COLOR)

    for number, v in enumerate(function_order(graph, pos, depths), start=1):
        x, y = pos[graph.A + v]
        ax.scatter([x], [y], s=150, color=FUNC_FILL, zorder=2,
                   edgecolor="#ffffff", lw=1.0)
        ax.text(x, y, str(number), ha="center", va="center",
                fontsize=6.1, color="#ffffff", zorder=3)

    ax.scatter([yx], [yy], s=150, color=READOUT_GREEN, zorder=2,
               edgecolor="#ffffff", lw=1.0)
    ax.text(yx, yy, "y", ha="center", va="center", fontsize=6.5,
            style="italic", color="#ffffff", zorder=3)

    if title:
        ax.set_title(title, fontsize=9.6)
    if note:
        ax.text(0.5, -0.13, note, ha="center", fontsize=7.6, color="#555",
                transform=ax.transAxes)
    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.85, yy + 0.55)
    ax.axis("off")
