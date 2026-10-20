"""Figures of targets and learned embeddings (needs matplotlib: ``pip install matplotlib``).

``draw_target(ax, target)``        the computation graph of a target: inputs, local functions, the output
``plot_embeddings(model, target)`` the learned neuron embeddings of a LeWi network, one 3-D panel per layer transition,
                                   units coloured by the target intermediate they follow (:mod:`csfrbench.structure`:
                                   correlation alone, with the permutation threshold) and the open connections drawn

Local functions are numbered left to right, bottom to top, and every figure uses one colour per number, so a unit
and the local function it follows have the same colour. Hidden units without an association are left out.
"""
from __future__ import annotations

import numpy as np

from . import structure
from .targets import Target

#: one colour per local function, in the numbering order (Okabe-Ito based; the paper's palette), extended for larger graphs
PALETTE = ["#d55e00", "#e69f00", "#0072b2", "#009e73", "#cc79a7", "#8c564b", "#66c200",
           "#56b4e9", "#f0e442", "#7f3c8d", "#11a579", "#e73f74", "#3969ac", "#f2b701", "#80ba5a", "#008695"]
OUTPUT = "#000000"
EDGE = "#8f8f8f"
SHAPES = ["o", "s", "^", "D", "v", "P", "X", "h", "p", "<", ">"]
VIEW = (22.0, -55.0)
LEVEL_DY = 1.25
MIN_DX = 1.4


# ------------------------------------------------------------------ the graph's layout and numbering


def node_depths(graph) -> list[int]:
    """The level of every local function: 1 + the largest level of its parents (inputs are at level 0)."""
    depths: list[int] = []
    for refs in graph.node_parents:
        depths.append(1 + max((0 if r < graph.A else depths[r - graph.A]) for r in refs))
    return depths


def layout_graph(graph) -> tuple[dict[int, tuple[float, float]], list[int]]:
    """Positions of the inputs (``0 .. A-1``) and the local functions (``A + v``): inputs in the order of first use,
    every function above its parents, with a minimal horizontal spacing per level."""
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
    pos = {leaf: (rank - (graph.A - 1) / 2, 0.0) for rank, leaf in enumerate(order)}
    depths = node_depths(graph)
    for level in range(1, max(depths) + 1):
        row = [v for v in range(graph.S) if depths[v] == level]
        xs = [float(np.mean([pos[r][0] for r in graph.node_parents[v]])) for v in row]
        ordered = sorted(zip(row, xs), key=lambda pair: pair[1])
        placed: list[float] = []
        for _v, x in ordered:
            placed.append(x if not placed else max(x, placed[-1] + MIN_DX))
        shift = float(np.mean(xs)) - float(np.mean(placed))
        for (v, _x), x_new in zip(ordered, placed):
            pos[graph.A + v] = (x_new + shift, LEVEL_DY * level)
    return pos, depths


def numbering(graph) -> dict[int, int]:
    """Local function index -> its number (1, 2, ...): left to right, bottom to top."""
    pos, depths = layout_graph(graph)
    order = sorted(range(graph.S), key=lambda v: (depths[v], pos[graph.A + v][0]))
    return {v: k + 1 for k, v in enumerate(order)}


def colour(number: int) -> str:
    return PALETTE[(number - 1) % len(PALETTE)]


def node_colours(graph) -> dict[int, str]:
    """Local function index -> its colour."""
    return {v: colour(k) for v, k in numbering(graph).items()}


# ------------------------------------------------------------------ the target


def draw_target(ax, target: Target, *, numbers: bool = True) -> None:
    """The computation graph of a target: small dots for the inputs (in the colour of the function that reads them),
    discs for the local functions (numbered), a black disc for the output."""
    g = target.graph
    pos, depths = layout_graph(g)
    colours = node_colours(g)
    num = numbering(g)
    top = LEVEL_DY * (max(depths) + 1)
    out_x = float(np.mean([pos[g.A + i][0] for i in g.readout_indices]))
    for v, refs in enumerate(g.node_parents):
        for r in refs:
            (x0, y0), (x1, y1) = pos[r], pos[g.A + v]
            ax.plot([x0, x1], [y0, y1], color=EDGE, lw=0.8, zorder=1)
    for i in g.readout_indices:
        x0, y0 = pos[g.A + i]
        ax.plot([x0, out_x], [y0, top], color=EDGE, lw=0.8, zorder=1)
    for v, refs in enumerate(g.node_parents):
        for r in refs:
            if r < g.A:
                ax.scatter(*pos[r], s=14, color=colours[v], zorder=2, edgecolors="none")
    for v in range(g.S):
        x, y = pos[g.A + v]
        ax.scatter([x], [y], s=110, color=colours[v], zorder=3, edgecolors="white", linewidths=0.8)
        if numbers:
            ax.text(x, y, str(num[v]), ha="center", va="center", fontsize=5.5, color="white", zorder=4)
    ax.scatter([out_x], [top], s=110, color=OUTPUT, zorder=3, edgecolors="white", linewidths=0.8)
    ax.text(out_x, top, "y", ha="center", va="center", fontsize=6, color="white", style="italic", zorder=4)
    xs = [p[0] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(-0.6, top + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------------------------------------------------------ the embeddings


def _layers_and_embeddings(model) -> tuple[list, list[np.ndarray]]:
    """The affine layers and the embeddings of every layer boundary of a LeWi network (CSFRBench's or lewi.LeWiMLP)."""
    if hasattr(model, "gate_codes"):
        layers, codes = list(model.linear_layers()), list(model.gate_codes)
    elif hasattr(model, "embeddings") and hasattr(model, "layers"):
        layers, codes = list(model.layers), list(model.embeddings)
    else:
        raise TypeError("plot_embeddings needs a LeWi network (one with neuron embeddings)")
    codes = [c.detach().cpu().numpy().astype(np.float64) for c in codes]
    r = codes[0].shape[1]
    if r != 3:                                   # project onto the three leading principal axes of all embeddings
        allc = np.concatenate(codes)
        mean = allc.mean(0)
        if r > 3:
            _, _, vt = np.linalg.svd(allc - mean, full_matrices=False)
            codes = [(c - mean) @ vt[:3].T for c in codes]
        else:
            codes = [np.pad(c - mean, ((0, 0), (0, 3 - r))) for c in codes]
    return layers, codes


def plot_embeddings(model, target: Target, X=None, *, assoc: list[np.ndarray] | None = None, view=VIEW,
                    panel: float = 1.55, show_unassociated: bool = False, title: str | None = None):
    """A figure of the learned embeddings of a LeWi network: the target graph, then one 3-D panel per transition
    (inputs -> layer 1, ..., last layer -> output) with the units of its two layers at their embedding positions and the
    open connections between them in the sending unit's colour. Units are coloured by the intermediate they follow
    (correlation alone, :func:`csfrbench.structure.associations`); hidden units without an association are left out
    (``show_unassociated`` draws them in light grey). Returns the matplotlib figure."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    layers, codes = _layers_and_embeddings(model)
    assoc = structure.associations(model, target, X) if assoc is None else assoc
    colours = node_colours(target.graph)
    levels = len(codes)
    shown = [assoc[k] >= 0 for k in range(levels)]
    shown[-1] = np.ones_like(shown[-1], dtype=bool)
    pts = np.concatenate([codes[k][shown[k]] for k in range(levels)])
    centre = (pts.max(0) + pts.min(0)) / 2
    half = float(np.abs(pts - centre).max()) * 1.05
    names = ["Input"] + [f"Layer {k}" for k in range(1, levels - 1)] + ["Output"]
    n_panels = levels - 1
    band = 0.2 if title else 0.0                                     # the figure's height share above the panels
    fig = plt.figure(figsize=(panel * (n_panels + 0.8), panel * (1.1 + band)))
    top = 1.0 - band / (1.1 + band)
    width0 = 0.8 / (n_panels + 0.8)
    ax0 = fig.add_axes([0.0, 0.06 * top, width0, 0.74 * top])
    draw_target(ax0, target)
    fig.text(width0 / 2, 0.9 * top, "Target", ha="center", va="bottom", fontsize=8)

    def unit_colour(k, i):
        if k == levels - 1:
            return OUTPUT
        v = assoc[k][i]
        return colours[v] if v >= 0 else "#d0d0d0"

    for t in range(n_panels):
        a, b = t, t + 1
        left = (0.8 + t) / (n_panels + 0.8)
        ax = fig.add_axes([left, 0.0, 1.0 / (n_panels + 0.8), 0.88 * top], projection="3d")
        ax.set_proj_type("ortho")
        ax.view_init(elev=view[0], azim=view[1])
        mask = layers[t].event_mask.detach().cpu().numpy() > 0                # [units of b, units of a]
        keep_a = shown[a] | show_unassociated
        keep_b = shown[b] | show_unassociated
        rows, cols = np.nonzero(mask & keep_b[:, None] & keep_a[None, :])
        if len(rows):
            seg = np.stack([codes[a][cols], codes[b][rows]], axis=1)
            ax.add_collection(Line3DCollection(seg, colors=[unit_colour(a, c) for c in cols], linewidths=0.45,
                                               alpha=0.55))
        for k, keep in ((a, keep_a), (b, keep_b)):
            idx = np.nonzero(keep)[0]
            if not len(idx):
                continue
            size = 70 if k == levels - 1 else 14
            ax.scatter(*codes[k][idx].T, s=size, c=[unit_colour(k, i) for i in idx],
                       marker="*" if k == levels - 1 else SHAPES[k % len(SHAPES)], depthshade=False,
                       edgecolors="white", linewidths=0.3)
        for axis, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
            axis(c - half, c + half)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half + centre
        cube = [(corners[i], corners[j]) for i in range(8) for j in range(i + 1, 8)
                if np.sum(corners[i] != corners[j]) == 1]
        ax.add_collection(Line3DCollection(cube, colors="#dddddd", linewidths=0.5))
        fig.text(left + 0.5 / (n_panels + 0.8), 0.9 * top, f"{names[a]} → {names[b]}", ha="center", va="bottom",
                 fontsize=8)
    if title:
        fig.text(0.5, 0.98, title, ha="center", va="top", fontsize=9)
    return fig
