"""Figure 1 (teaser, under the abstract): LeWi's learned neuron embeddings
follow the target computation, one layer transition at a time.

Caption - LeWi learns neuron embeddings that follow the target computation.
Left: the target computation of one depth-4 Fourier-tree instance (active
inputs at the bottom, the seven local functions as numbered intermediates,
the output y on top). Right, in one shared frame, view and cube: the
embeddings in R^3 that a sealed LeWi network trained on this instance
(n = 2^15, test NMSE 0.003) learned for every input coordinate, hidden unit
and the output - their inner products score the connections. Each panel
highlights one layer transition: its two levels, coloured by the target
intermediate they are most associated with (inputs by the local function
that reads them, hidden units by the largest absolute correlation on the
diagnostic pool among the intermediates whose inputs the unit reads through
its kept connections, grey below the permutation floor), and the connections the
network keeps between them, each in the colour of its sending unit; every
other level is light grey, and farther points and connections are drawn
fainter. The output's embedding (black star) is shown in every panel, and
the last panel adds the connections into it. Marker shape gives the level.
The active inputs and hidden units
of one local function align and connect within their module; the units of
the two level-2 functions (5, 6) lie between the modules they combine, on
opposite sides of the space, and the deeper layers contract toward the
origin. Colours use the known target intermediates for descriptive
visualization only; this is not a graph-recovery measurement.

Data: data/teaser/fourier_f7011_n32768__lewi-v05.npz, written by
src/harvest_teaser.py from the sealed LeWi winner of Fourier-tree instance
7011 at n = 2^15 (the checkpoint reproduces its sealed test NMSE). The
network was chosen by eye among the sealed LeWi winners of both depth-4
trees ranked by the cosine silhouette of their hidden-1 units by module and
of their node-5 against node-6 units (this one: 0.76 and 0.77, with 13-19
units on every intermediate); the viewpoint is the 10-degree grid point
whose projection best separates the coloured groups (best_view, printed at
build time). Colours, marker shapes, the association rule and the embedding
frame are those of the retired learned_structure figure; teaser_comparison
reuses these panels' helpers on the tree instance the wiring-detail figure shows.
"""
from __future__ import annotations

import itertools

import numpy as np
from csfrbench.viz import LEVEL_DY, READOUT_GREEN, layout_graph
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import paperstyle as ps
from figstyle import DATA, save
from learned_structure import (CUBE_FRACTION, GREY, NODE_COLOR, OUTPUT_COLOR, SHAPES,
                               TRANSITIONS, VIEW, ZOOM, associations, colour_of, graph_of,
                               kept, level_colours, numbering)

NETWORK = f"fourier_f7011_n32768__{ps.LEWI}"    # the record data/teaser/<NETWORK>.npz (08-main's instance choice)
LEVEL = ["Input", "H1", "H2", "H3", "H4", "Output"]
HEADINGS = ["Target"] + [f"{LEVEL[a].capitalize()} → {LEVEL[b]}" for a, b in TRANSITIONS]
POINT = 9.0                  # embedding markers in a large panel
POINT_SMALL = 6.5            # in a row of small panels
POINT_BG = 3.0
BACKGROUND = "#d4d4d4"
IDLE = "#b3b3b3"             # faded units of a highlighted level (no association)
FADE_FAR = 0.45              # alpha at the far corner of the cube
EDGE = "#bdbdbd"
MUTED = "#6a6a6a"
TITLE = 7.6


def load(name: str) -> dict:
    return dict(np.load(DATA / f"teaser/{name}.npz"))


# ---- the target graph ----------------------------------------------------

def tree(d, margin: float = 1.2):
    """Layout of the target graph: node positions, the readout position and
    the axis limits every graph panel shares (`margin` leaf spacings of
    room beside the outermost inputs)."""
    g = graph_of(d)
    pos, depths = layout_graph(g)
    yx = float(np.mean([pos[g.A + i][0] for i in g.readout_indices]))
    yy = LEVEL_DY * (max(depths) + 1)
    xs = [p[0] for p in pos.values()]
    return g, pos, (yx, yy), ((min(xs) - margin, max(xs) + margin), (-1.0, yy + 1.1))


def draw_edges(ax, g, pos, readout, colour) -> None:
    for v, refs in enumerate(g.node_parents):
        x1, y1 = pos[g.A + v]
        for r in refs:
            ax.plot([pos[r][0], x1], [pos[r][1], y1], color=colour, linewidth=0.7, zorder=1)
    for i in g.readout_indices:
        ax.plot([pos[g.A + i][0], readout[0]], [pos[g.A + i][1], readout[1]],
                color=colour, linewidth=0.7, zorder=1)


def draw_target(ax, d, *, margin: float = 1.2, label: str = "y") -> None:
    """The target graph as in learned_structure's key: numbered, coloured
    intermediates, active inputs as dots of the function that reads them,
    the readout in the grammar's green; it fills its axes box."""
    g, pos, readout, (xlim, ylim) = tree(d, margin)
    num = numbering(d)
    owner = {r: v for v, refs in enumerate(g.node_parents) for r in refs if r < g.A}
    draw_edges(ax, g, pos, readout, EDGE)
    for leaf in range(g.A):
        ax.scatter(*pos[leaf], s=16, color=NODE_COLOR[num[owner[leaf]] - 1], zorder=3,
                   linewidths=0)
    for v in range(g.S):
        x, y = pos[g.A + v]
        ax.scatter([x], [y], s=105, color=NODE_COLOR[num[v] - 1], zorder=4,
                   edgecolor="#ffffff", lw=0.8)
        ax.text(x, y, str(num[v]), ha="center", va="center", fontsize=5.6,
                color="#ffffff", zorder=5)
    ax.scatter(*readout, s=105, color=OUTPUT_COLOR, zorder=4, edgecolor="#ffffff", lw=0.8)
    ax.text(*readout, label, ha="center", va="center", fontsize=6.0, style="italic",
            color="#ffffff", zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()


# ---- the embedding space -------------------------------------------------

def frame(records: list[dict]) -> tuple[np.ndarray, float]:
    """One frame for every geometry panel: the principal directions of the
    last record's embeddings through the origin, and a cube edge set by the
    largest embedding norm over all records."""
    codes = [np.asarray(records[-1][f"codes{i}"]) for i in range(6)]
    _, _, vt = np.linalg.svd(np.concatenate(codes), full_matrices=False)
    lim = max(float(np.linalg.norm(np.asarray(r[f"codes{i}"]), axis=1).max())
              for r in records for i in range(6)) * CUBE_FRACTION
    return vt.T, lim


def project(X: np.ndarray, elev: float, azim: float) -> np.ndarray:
    """Screen coordinates of points under a view (matplotlib's elev/azim)."""
    e, a = np.radians(elev), np.radians(azim)
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    return np.stack([X @ right, X @ up], axis=1)


def silhouette(P: np.ndarray, labels: np.ndarray) -> float:
    """Mean Euclidean silhouette of labelled points."""
    dist = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    values = []
    for i in range(len(P)):
        same = labels == labels[i]
        same[i] = False
        others = [l for l in set(labels) if l != labels[i]]
        if not same.any() or not others:
            continue
        a = dist[i, same].mean()
        b = min(dist[i, labels == l].mean() for l in others)
        values.append((b - a) / max(a, b))
    return float(np.mean(values)) if values else float("nan")


def best_view(d, rotation) -> dict:
    """The viewpoint, on a 10-degree grid, whose projection best separates
    the coloured groups: the mean of the silhouette of every associated
    input and hidden unit by intermediate and the silhouette of the node-5
    against the node-6 units."""
    codes = [np.asarray(d[f"codes{i}"]) @ rotation for i in range(5)]
    assoc = associations(d)
    X = np.concatenate(codes)
    labels = np.concatenate([np.asarray(a) for a in assoc[:5]])
    X, labels = X[labels > 0], labels[labels > 0]
    level2 = (labels == 5) | (labels == 6)
    best = None
    for elev in range(-60, 61, 10):
        for azim in range(0, 360, 10):
            P = project(X, elev, azim)
            score = 0.5 * silhouette(P, labels) + 0.5 * silhouette(P[level2], labels[level2])
            if best is None or score > best[0]:
                best = (score, elev, azim)
    print(f"[teaser] viewpoint elev {best[1]}, azim {best[2]} (projected separation {best[0]:.2f})")
    return {"elev": best[1], "azim": best[2]}


def cube(ax, lim: float, view: dict = VIEW):
    """The cube, limits, view and axes-off of one geometry panel; returns the
    projection and the depth span of the cube for the depth fading."""
    ax.computed_zorder = False
    for a, b in itertools.combinations(itertools.product((-lim, lim), repeat=3), 2):
        if sum(x != y for x, y in zip(a, b)) == 1:
            ax.plot(*zip(a, b), color="#e2e2e2", linewidth=0.45, zorder=0)
    for k in range(3):
        p = np.zeros((2, 3))
        p[0, k], p[1, k] = -lim, lim
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color="#c4c4c4", linewidth=0.5, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1), zoom=ZOOM)
    ax.view_init(**view)
    ax.set_axis_off()
    corners = np.array(list(itertools.product((-lim, lim), repeat=3)), dtype=float)
    M = ax.get_proj()
    z = proj3d.proj_transform(corners[:, 0], corners[:, 1], corners[:, 2], M)[2]
    return M, (float(z.min()), float(z.max()))


def fade(depth, xyz: np.ndarray) -> np.ndarray:
    """Alpha factor per point from its projected depth: 1 at the near corner
    of the cube, FADE_FAR at the far corner."""
    M, (near, far) = depth
    z = proj3d.proj_transform(xyz[:, 0], xyz[:, 1], xyz[:, 2], M)[2]
    return 1.0 - (1.0 - FADE_FAR) * np.clip((z - near) / (far - near), 0.0, 1.0)


def points(ax, depth, xyz, level: int, colours, size: float, *, zorder: int = 3,
           alpha: float = 1.0) -> None:
    if len(xyz) == 0:
        return
    f = fade(depth, xyz) * alpha
    rgba = np.array([to_rgba(c) for c in colours])
    rgba[:, 3] = f
    edge = np.ones_like(rgba)
    edge[:, 3] = f
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=size * (0.6 + 0.4 * f), c=rgba,
               marker=SHAPES[level], linewidths=0.3, edgecolors=edge, depthshade=False,
               zorder=zorder)


def connections(ax, depth, send, recv, weight, numbers, *, alpha: float, width: float,
                faint: float | None = None, recv_numbers=None) -> None:
    """Kept connections from `send` to `recv` in the sending unit's colour;
    with `faint`, a connection touching a unit without an association
    (number 0, at either end) keeps only that share of its alpha."""
    rows, cols = np.nonzero(weight > 0)
    if len(rows) == 0:
        return
    segments = np.stack([send[cols], recv[rows]], axis=1)
    rgba = np.array([to_rgba(colour_of(numbers[c])) for c in cols])
    rgba[:, 3] = alpha * weight[rows, cols] * fade(depth, segments.mean(1))
    if faint is not None:
        idle = np.array([numbers[c] == 0 or recv_numbers[r] == 0 for r, c in zip(rows, cols)])
        rgba[idle, 3] *= faint
    ax.add_collection(Line3DCollection(segments, colors=rgba, linewidths=width, zorder=2))


def draw_geometry(ax, d, rotation, lim, *, transition=None, pairs=None, assoc=None,
                  point=POINT, edge_alpha=None, view: dict = VIEW,
                  readout: bool | None = None, weights=kept,
                  output_star: bool = True, star_scale: float = 1.0,
                  faint: float | None = None, faint_edges: float | None = None,
                  background: float = 1.0) -> None:
    """One panel of the embedding space in a given frame, cube and view:
    every level with the kept connections of `pairs` (default: every
    transition), or the two levels of one transition and its connections
    over the other levels in light grey. Units are coloured by `assoc`
    (default: the record's own associations). Farther points and
    connections are fainter and smaller. `readout` adds the connections
    into the output (default: in the panel of the last hidden transition);
    `weights(d, boundary)` gives the connection strengths (default: the
    kept-connection rule of learned_structure). The output is a larger star
    on top in every panel, or with `output_star` off a level like any
    other, its marker area scaled by `star_scale` (a five-point star fills
    about a third of the area a square of the same marker size does). With
    `faint`, the units of a highlighted level that have no association (the
    inputs no local function reads, the hidden units that correlate with no
    intermediate) are drawn as small light-grey markers at that alpha, a
    step above the other levels' background markers, whose alpha is
    `background`; the connections touching them keep the share
    `faint_edges` (default `faint`) of theirs."""
    codes = [np.asarray(d[f"codes{i}"]) @ rotation for i in range(6)]
    assoc = associations(d) if assoc is None else assoc
    depth = cube(ax, lim, view)
    focus = list(range(6)) if transition is None else list(transition)
    for k in range(5 if output_star else 6):
        if k not in focus:
            points(ax, depth, codes[k], k, [BACKGROUND] * len(codes[k]),
                   POINT_BG * (star_scale if k == 5 else 1.0), zorder=1, alpha=background)
    if transition is not None:
        pairs = [transition]
    elif pairs is None:
        pairs = TRANSITIONS
    if edge_alpha is None:
        edge_alpha = 0.3 if transition is None else 0.8
    width = 0.3 if transition is None else 0.4
    if faint_edges is None:
        faint_edges = faint
    for a, b in pairs:
        connections(ax, depth, codes[a], codes[b], weights(d, a), assoc[a], alpha=edge_alpha,
                    width=width, faint=faint_edges, recv_numbers=assoc[b])
    # the last transition also shows the readout: the connections into the
    # output's embedding, which anchors the geometry and is shown in every
    # panel as a larger star
    if readout is None:
        readout = transition == TRANSITIONS[-1]
    if readout:
        connections(ax, depth, codes[4], codes[5], weights(d, 4), assoc[4], alpha=edge_alpha,
                    width=width, faint=faint_edges, recv_numbers=assoc[5])
    for k in focus:
        if k != 5 or not output_star:
            colours = level_colours(assoc, k)
            size = point * (star_scale if k == 5 else 1.0)
            if faint is None:
                points(ax, depth, codes[k], k, colours, size)
                continue
            idle = np.array([n == 0 for n in assoc[k]])
            points(ax, depth, codes[k][~idle], k, [c for c, i in zip(colours, idle) if not i],
                   size)
            points(ax, depth, codes[k][idle], k, [IDLE] * int(idle.sum()), POINT_BG * 1.6,
                   zorder=1, alpha=faint)
    if output_star:
        points(ax, depth, codes[5], 5, level_colours(assoc, 5), point * 2.4, zorder=4)


def marker_handles(star_scale: float = 1.0) -> list[Line2D]:
    """Legend handles per level; the star's marker area scaled by `star_scale`."""
    return [Line2D([], [], marker=m, linestyle="none",
                   markersize=4.0 * (star_scale ** 0.5 if m == "*" else 1.0),
                   markerfacecolor=MUTED, markeredgecolor="white", markeredgewidth=0.3)
            for m in SHAPES]


def fit_cubes(fig, panels, zoom: float = ZOOM) -> float:
    """Shrink the panels' zoom until every rendered cube edge lies inside
    its axes box, so no corner is clipped whatever viewpoint was chosen."""
    while True:
        for ax in panels:
            ax.set_box_aspect((1, 1, 1), zoom=zoom)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inside = all(ax.get_window_extent(renderer).contains(x, y)
                     for ax in panels for line in ax.lines
                     for b in [line.get_window_extent(renderer)]
                     for x, y in ((b.x0, b.y0), (b.x1, b.y1)))
        if inside or zoom < 0.5:
            print(f"[teaser] cube zoom {zoom:.2f}")
            return zoom
        zoom *= 0.96


def cube_extent(fig, panels) -> tuple[float, float]:
    """Vertical extent of the drawn cubes, in figure coordinates, read from
    the rendered cube edges so other panels can align with them."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [line.get_window_extent(renderer) for ax in panels for line in ax.lines]
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    (_, f0), (_, f1) = fig.transFigure.inverted().transform([(0.0, y0), (0.0, y1)])
    return float(f0), float(f1)


def main() -> None:
    ps.apply_rc()
    lewi = load(NETWORK)
    rotation, lim = frame([lewi])
    view = best_view(lewi, rotation)
    fig = plt.figure(figsize=(5.5, 1.75))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[0.95, 1, 1, 1, 1],
                  left=0.005, right=0.995, top=0.86, bottom=0.17, wspace=0.02)
    target = fig.add_subplot(gs[0, 0])
    draw_target(target, lewi)
    panels = []
    for j, transition in enumerate(TRANSITIONS):
        ax = fig.add_subplot(gs[0, 1 + j], projection="3d")
        draw_geometry(ax, lewi, rotation, lim, transition=transition, point=POINT_SMALL,
                      view=view)
        panels.append(ax)
    # the cubes fit their boxes uncut, the target graph spans exactly their
    # height, and every heading sits on one line just above them
    fit_cubes(fig, panels)
    low, high = cube_extent(fig, panels)
    box = target.get_position()
    target.set_position([box.x0, low, box.width, high - low])
    for ax, heading in zip([target] + panels, HEADINGS):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, high + 0.035, heading, ha="center", va="bottom",
                 fontsize=TITLE, color="#2a2a2a")
    # reading cue under the geometry panels, marker key under the target
    x0 = panels[0].get_position().x0 + 0.03
    x1 = panels[-1].get_position().x1 - 0.03
    y = low - 0.07
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=5, color="#9a9a9a",
                                   linewidth=0.6, shrinkA=0, shrinkB=0))
    fig.text((x0 + x1) / 2, y - 0.025, "Network depth", ha="center", va="top",
             fontsize=6.4, color=MUTED)
    box = target.get_position()
    order = [0, 3, 1, 4, 2, 5]           # legends fill columns first; this reads by rows
    fig.legend([marker_handles()[i] for i in order], [LEVEL[i] for i in order],
               loc="upper center", ncol=3, fontsize=5.8, frameon=False, handlelength=0.9,
               handletextpad=0.2, columnspacing=0.6, labelspacing=0.2, borderaxespad=0.0,
               bbox_to_anchor=((box.x0 + box.x1) / 2, low - 0.03))
    save(fig, "teaser")


if __name__ == "__main__":
    main()
