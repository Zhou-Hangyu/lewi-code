"""Retired stem (2026-09-04) and shared helper module: the probe heatmaps
and the depth-progression panel now live in learned_computational_structure
together with the causal probes, and this file is no longer registered in
build.py. It stays because the teaser, learned_geometry,
learned_structure_v2, learned_wiring_detail, causal_probe_detail and
learned_computational_structure import its helpers (NODE_COLOR, SHAPES,
GREY, VIEW, CUBE_FRACTION, ZOOM, TRANSITIONS, MASKED, graph_of, colour_of,
kept, numbering, associations, load_embeddings, module_silhouette,
level_progression, unit_orders, draw_matrix, draw_masks, draw_key,
draw_embeddings) - never rename them. `python src/learned_structure.py`
still builds the retired two-row figure for reference.

Caption (retired) - LeWi develops stronger and more depth-aligned intermediate
representations. (a) Held-out ridge-probe R^2 of the seven target
intermediates (rows, level 1 at the bottom, root at the top, numbered and
coloured as the target graph of Figure 1) from each learner layer (columns)
for Dense, LeWi and GraphOracle(all) on the depth-4 tree at n = 2^15,
averaged over 16 target instances (Appendix B.7); "in" is the raw-input
probe, identical for all arms. (b) Training-induced accessibility Delta R^2
(trained minus initialization probe R^2) averaged within each target depth
- the four level-1 intermediates, the two level-2 intermediates, the root -
per instance, then mean and 95% Student-t band over the 16 instances
(colours light to dark with target depth). Earlier target computations
emerge earlier in the learner, while deeper computations become accessible
later: LeWi's level-1 curve peaks at layer 2 (0.71), level 2 at layer 3
(0.65) and the root at layer 4 (0.71), the GraphOracle(all) profile peaks
at the same layers (0.31, 0.67, 0.80), whereas Dense's level-1 and level-2
curves both peak at layer 2 (0.42, 0.36) and its root reaches 0.36 at
layer 4. The learned wiring of one of these networks is the appendix
figure learned_wiring_detail.

Data: data/learned_structure/probe_values.csv (the campaign's per-instance
probe values: trained and initialization R^2 of every intermediate from
every layer) and data/learned_structure/embeddings/*.npz (codes, masks,
gates and unit-node correlations of the 16 sealed tree LeWi winners at
n = 2^15, read from their retained best.pt; used here for the Figure-1
numbering of the intermediates and by the appendix figures), both written
by src/harvest_08main.py. The embedding geometry itself is the appendix
figure learned_geometry.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from csfrbench.tasks.graph import GraphSpec
from csfrbench.viz import LEVEL_DY, READOUT_GREEN, function_order, layout_graph
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import paperstyle as ps
from figstyle import DATA, save

# intermediates 1..7 in Figure-1 order, from the Okabe-Ito colour-blind-safe
# set: vermillion, orange, blue, bluish green, reddish purple, then brown in
# place of sky blue (which reads like faded blue beside node 3; a violet
# reads like node 5's pink), and lime for the root (the colour farthest, in
# CIELAB, from every other node colour and its faded version); the output y
# is black
NODE_COLOR = ["#d55e00", "#e69f00", "#0072b2", "#009e73", "#cc79a7", "#8c564b",
              "#66c200"]
OUTPUT_COLOR = "#000000"
GREY = "#9a9a9a"                              # unassociated, in a highlighted level
BACKGROUND = "#d4d4d4"                        # every other level
CHARCOAL = "#1a1a1a"
SHAPES = ["o", "s", "^", "D", "v", "*"]       # per level, as in the demo
LEVEL_NAME = ["Inputs", "Hidden 1", "Hidden 2", "Hidden 3", "Hidden 4", "Output"]
TRANSITIONS = [(0, 1), (1, 2), (2, 3), (3, 4)]
# the transitions the build's LeWi arm masks: LeWi-v05 the first two, LeWi-v1
# (CSFRBench v1.0) all four hidden transitions (the readout stays plain)
MASKED = {"lewi-v05": (0, 1), "lewi-v1": (0, 1, 2, 3), "lewi-v1-rgm": (0, 1, 2, 3, 4)}[ps.LEWI]   # the final arm masks the readout too
ARMS = [("noreg", "Dense"), (ps.LEWI, "LeWi"),          # method id in the tables, panel title
        ("graph_oracle", "GraphOracle(all)")]
LAYERS = ["h0", "h1", "h2", "h3", "h4"]
LAYER_LABEL = ["in", "1", "2", "3", "4"]
N_SHOWN = 32768
# panel (b): one curve per target depth, light to dark with depth
LEVEL_COLOR = {1: "#b3a5d9", 2: "#7a5fb8", 3: "#3a1a7e"}
LEVEL_LABEL = {1: "Level 1", 2: "Level 2", 3: "Level 3"}
LABELLED_ARM = "graph_oracle"          # the panel that carries the curve labels
VIEW = {"elev": 22, "azim": -55}
POINT = 8.0
POINT_BG = 3.0
CUBE_FRACTION = 0.8      # cube half-edge as a fraction of the largest norm
ZOOM = 1.05


def graph_of(d) -> GraphSpec:
    return GraphSpec(int(d["leaves"]), [tuple(p) for p in d["node_parents"]],
                     tuple(int(i) for i in d["readout_indices"]),
                     (1.0,) * len(d["readout_indices"]))


def numbering(d) -> dict[int, int]:
    """csfrbench node index -> Figure-1 number (left to right, bottom to top)."""
    g = graph_of(d)
    pos, depths = layout_graph(g)
    return {v: k + 1 for k, v in enumerate(function_order(g, pos, depths))}


def load_embeddings() -> tuple[dict, list]:
    """The 16 sealed tree LeWi winners at n = 2^15, ordered by the clarity of
    their hidden-1 module structure (see module_silhouette)."""
    files = sorted((DATA / "learned_structure/embeddings").glob(f"tree_f*_n{N_SHOWN}.npz"))
    if not files:
        raise SystemExit(f"[learned_structure] no embedding record under {DATA.name}/learned_structure/embeddings/ "
                         "(harvest_08main.py writes them)")
    runs = {int(np.load(f)["function_seed"]): np.load(f) for f in files}
    order = sorted(runs, key=lambda s: -module_silhouette(runs[s]))
    return runs, order


def module_silhouette(d) -> float:
    """How clearly the hidden-1 embeddings separate by the level-1 local
    function their units are associated with: the mean cosine silhouette of
    those units (units associated with a level-2 or root intermediate, or
    with none, are left out). Connections score by inner products, so the
    cosine geometry is the one the wiring reads."""
    labels = np.asarray(associations(d)[1])
    keep = (labels >= 1) & (labels <= 4)
    X = np.asarray(d["codes1"])[keep]
    labels = labels[keep]
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    dist = 1.0 - X @ X.T
    values = []
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        others = [l for l in set(labels) if l != labels[i]]
        if not same.any() or not others:
            continue
        a = dist[i, same].mean()
        b = min(dist[i, labels == l].mean() for l in others)
        values.append((b - a) / max(a, b))
    return float(np.mean(values)) if values else float("-inf")


def input_support(d) -> list[set[int]]:
    """Per intermediate (csfrbench node index), the raw input coordinates it
    depends on: its own active inputs and those of the intermediates it
    reads."""
    A = int(d["leaves"])
    raw = [np.atleast_1d(r) for r in np.asarray(d["active_raw"])]
    support = []
    for v, refs in enumerate(d["node_parents"]):
        # a record whose graph is not a balanced tree stores one row per node, padded with -1 where the node
        # reads fewer inputs than the widest one; the padding is not an input coordinate
        s = {int(c) for c in raw[v] if int(c) >= 0} if v < len(raw) else set()
        for r in refs:
            if r >= A:
                s |= support[int(r) - A]
        support.append(s)
    return support


def receptive_fields(d) -> list[list[set[int]]] | None:
    """Per level, the raw input coordinates each unit reads through the
    kept connections (the binary mask at the masked boundaries, a gate above
    one half elsewhere); None for a record without the wiring."""
    if not all((f"mask{b}" if b in MASKED else f"gate{b}") in d for b in range(4)):
        return None
    fields = [[{u} for u in range(np.asarray(d["codes0"]).shape[0])]]
    for b in range(4):
        w = kept(d, b)
        fields.append([set().union(*(fields[b][s] for s in np.nonzero(w[r] > 0)[0]))
                       for r in range(w.shape[0])])
    return fields


def associations(d) -> list[list[int]]:
    """Per level, the Figure-1 number of the associated intermediate per
    unit (0 = none): inputs by the local function that reads them, hidden
    units by the largest absolute correlation above the permutation floor
    among the intermediates whose inputs the unit reads through its kept
    connections (every intermediate when the record carries no wiring), the
    output by the root. A unit correlates with every ancestor of the
    intermediate it computes, so without the restriction a first-layer unit
    can be attributed to the root."""
    num = numbering(d)
    floor = float(d["corr_floor"])
    owner = {}
    for v, row in enumerate(d["active_raw"]):
        for c in row:
            owner[int(c)] = num[v]
    out = [[owner.get(u, 0) for u in range(d["codes0"].shape[0])]]
    support = input_support(d)
    fields = receptive_fields(d)
    for lv in range(1, 5):
        corr = np.abs(np.asarray(d[f"corr{lv}"]))
        level = []
        for u in range(corr.shape[0]):
            allowed = [v for v in range(corr.shape[1])
                       if fields is None or support[v] <= fields[lv][u]]
            best = max(allowed, key=lambda v: corr[u, v]) if allowed else None
            level.append(num[best] if best is not None and corr[u, best] >= floor else 0)
        out.append(level)
    # the readout: one entry per OUTPUT UNIT, not per readout node. On a tree the two coincide; a workload whose
    # readout sums several target nodes (a reuse DAG, the flat modules, a random graph) names several while the
    # network still has one output. The number only marks the unit as associated - the output is drawn black.
    roots = [num[int(r)] for r in d["readout_indices"]]
    n_out = int(np.asarray(d["codes5"]).shape[0]) if "codes5" in d else len(roots)
    out.append(roots if len(roots) == n_out else [(roots[0] if roots else 0)] * n_out)
    return out


def colour_of(number: int, *, muted: bool = False) -> str:
    if muted:
        return BACKGROUND
    return NODE_COLOR[number - 1] if number else GREY


def level_colours(assoc, level: int) -> list[str]:
    """Marker colours of one level: the output is black, every other unit
    takes the colour of its associated intermediate."""
    if level == 5:
        return [OUTPUT_COLOR] * len(assoc[level])
    return [colour_of(n) for n in assoc[level]]


def draw_key(ax, d) -> None:
    """The target graph as the colour key: the package's layout, discs
    coloured and numbered per intermediate, active inputs as small dots of
    the function that reads them, the readout as in the grammar; no input
    numbers."""
    g = graph_of(d)
    pos, depths = layout_graph(g)
    num = numbering(d)
    owner = {}
    for v, refs in enumerate(g.node_parents):
        for r in refs:
            if r < g.A:
                owner[r] = v
    for v, refs in enumerate(g.node_parents):
        x1, y1 = pos[g.A + v]
        for r in refs:
            x0, y0 = pos[r]
            ax.plot([x0, x1], [y0, y1], color="#bdbdbd", linewidth=0.7, zorder=1)
    for leaf in range(g.A):
        x, y = pos[leaf]
        ax.scatter([x], [y], s=16, color=NODE_COLOR[num[owner[leaf]] - 1],
                   zorder=3, linewidths=0)
    for v in range(g.S):
        x, y = pos[g.A + v]
        ax.scatter([x], [y], s=105, color=NODE_COLOR[num[v] - 1], zorder=4,
                   edgecolor="#ffffff", lw=0.8)
        ax.text(x, y, str(num[v]), ha="center", va="center", fontsize=5.6,
                color="#ffffff", zorder=5)
    yx = float(np.mean([pos[g.A + i][0] for i in g.readout_indices]))
    yy = LEVEL_DY * (max(depths) + 1)
    for i in g.readout_indices:
        x0, y0 = pos[g.A + i]
        ax.plot([x0, yx], [y0, yy], color="#bdbdbd", linewidth=0.7, zorder=1)
    ax.scatter([yx], [yy], s=105, color=OUTPUT_COLOR, zorder=4,
               edgecolor="#ffffff", lw=0.8)
    ax.text(yx, yy, "y", ha="center", va="center", fontsize=6.0, style="italic",
            color="#ffffff", zorder=5)
    xs = [p[0] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(-3.1, yy + 0.9)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("target", loc="center", pad=2)
    # marker key: one glyph per level of the learner
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=m, linestyle="none", markersize=4.2,
                      markerfacecolor="#6a6a6a", markeredgecolor="white",
                      markeredgewidth=0.3)
               for m in SHAPES]
    ax.legend(handles, LEVEL_NAME, loc="lower center", ncol=3, fontsize=6.0,
              frameon=False, handlelength=1.0, handletextpad=0.25,
              columnspacing=0.9, labelspacing=0.25, borderaxespad=0.0,
              bbox_to_anchor=(0.5, -0.01))


def _cube(ax, lim: float) -> None:
    for a, b in itertools.combinations(itertools.product((-lim, lim), repeat=3), 2):
        if sum(x != y for x, y in zip(a, b)) == 1:
            ax.plot(*zip(a, b), color="#e2e2e2", linewidth=0.45, zorder=0)
    for k in range(3):
        p = np.zeros((2, 3))
        p[0, k], p[1, k] = -lim, lim
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color="#c4c4c4", linewidth=0.5, zorder=0)


def kept(d, boundary: int) -> np.ndarray:
    """(receivers x senders) kept-connection weights in [0, 1]: the binary
    mask at a masked transition, the decided part of the soft gate elsewhere
    (2g - 1 above one half, else 0)."""
    if boundary in MASKED:
        return np.asarray(d[f"mask{boundary}"], dtype=float)
    g = np.asarray(d[f"gate{boundary}"], dtype=float)
    return np.clip(2.0 * g - 1.0, 0.0, 1.0)


def _scatter(ax, xyz, level: int, colours, size: float, *, zorder: int) -> None:
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=size, c=colours,
               marker=SHAPES[level], linewidths=0.3, edgecolors="white",
               depthshade=False, zorder=zorder)


def _edges(ax, send_xyz, recv_xyz, weight: np.ndarray, colours, *,
           alpha: float, width: float, zorder: int) -> None:
    rows, cols = np.nonzero(weight > 0)
    if len(rows) == 0:
        return
    segments = np.stack([send_xyz[cols], recv_xyz[rows]], axis=1)
    rgba = np.array([plt.matplotlib.colors.to_rgba(colours[c]) for c in cols])
    rgba[:, 3] = alpha * weight[rows, cols]
    ax.add_collection(Line3DCollection(segments, colors=rgba, linewidths=width,
                                       zorder=zorder))


def draw_embeddings(axes, d) -> None:
    codes = [np.asarray(d[f"codes{i}"]) for i in range(6)]
    assoc = associations(d)
    # one frame for every panel: the principal directions of all embeddings
    # through the origin (a rigid rotation), one view, one cube - the cube
    # edge is a fixed fraction of the largest embedding norm, so the panels
    # share a scale and the few far-out inputs sit just outside the cube
    _, _, vt = np.linalg.svd(np.concatenate(codes), full_matrices=False)
    rotated = [c @ vt.T for c in codes]
    lim = float(max(np.linalg.norm(c, axis=1).max() for c in codes)) * CUBE_FRACTION
    panels = [(t, "pair") for t in TRANSITIONS] + [(None, "all")]
    for ax, (transition, kind) in zip(axes, panels):
        focus = list(transition) if kind == "pair" else list(range(6))
        ax.computed_zorder = False
        _cube(ax, lim)
        for k in range(6):
            if k not in focus:
                _scatter(ax, rotated[k], k, BACKGROUND, POINT_BG, zorder=1)
        if kind == "pair":
            a, b = transition
            _edges(ax, rotated[a], rotated[b], kept(d, a),
                   [colour_of(n) for n in assoc[a]], alpha=0.6, width=0.4, zorder=2)
            for k in (a, b):
                _scatter(ax, rotated[k], k, level_colours(assoc, k),
                         POINT, zorder=3)
            title = f"{LEVEL_NAME[a]} → {LEVEL_NAME[b]}"
        else:
            for a, b in TRANSITIONS:
                _edges(ax, rotated[a], rotated[b], kept(d, a),
                       [colour_of(n) for n in assoc[a]], alpha=0.3, width=0.3,
                       zorder=2)
            for k in range(6):
                _scatter(ax, rotated[k], k, level_colours(assoc, k),
                         POINT, zorder=3)
            title = "all layers"
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1), zoom=ZOOM)
        ax.view_init(**VIEW)
        ax.set_axis_off()
        # the panel name sits above the axes box, clear of the cube
        ax.text2D(0.04, 1.0, title, transform=ax.transAxes, fontsize=6.6,
                  color="#4a4a4a", ha="left", va="bottom")


def draw_heatmaps(axes, runs, *, value_size: float = 5.4,
                  xlabel: bool = True) -> None:
    pv = pd.read_csv(DATA / "learned_structure/probe_values.csv")
    pv = pv[(pv.task == "tree_functional_anova") & (pv.n == N_SHOWN)]
    pv["v"] = pv.node.str.extract(r"z(\d+)").astype(int)
    pv["number"] = [numbering(runs[s])[v] for s, v in zip(pv.function_seed, pv.v)]
    cmap = plt.get_cmap("viridis")
    for ax, (method, title) in zip(axes, ARMS):
        sub = pv[pv.method == method]
        table = (sub.groupby(["number", "layer"]).r2_trained.mean()
                 .unstack("layer").reindex(index=range(1, 8), columns=LAYERS))
        m = table.to_numpy()
        ax.imshow(m, cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto",
                  interpolation="nearest")
        for i in range(7):
            for j in range(5):
                v = m[i, j]
                ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center",
                        fontsize=value_size, color="white" if v < 0.55 else "black")
        ax.set_xticks(range(5))
        ax.set_xticklabels(LAYER_LABEL)
        ax.set_yticks(range(7))
        if ax is axes[0]:
            ax.set_yticklabels([str(k) for k in range(1, 8)])
            for tick, k in zip(ax.get_yticklabels(), range(1, 8)):
                tick.set_color(NODE_COLOR[k - 1])
                tick.set_fontweight("bold")
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0, pad=1.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc="left", pad=2)
        if xlabel:
            ax.set_xlabel("Hidden layer", labelpad=1.5)
    axes[0].set_ylabel("Target intermediate", labelpad=2)


def level_progression(pv: pd.DataFrame, method: str) -> dict[int, pd.DataFrame]:
    """Per target depth, the mean and 95% t-interval over instances of
    Delta R^2 at each learner layer, after averaging the depth's
    intermediates within each instance."""
    sub = pv[pv.method == method]
    per = sub.groupby(["function_seed", "node_depth", "layer"]).delta_r2.mean().reset_index()
    out = {}
    for depth, g in per.groupby("node_depth"):
        rows = []
        for k, layer in enumerate(LAYERS):
            v = g[g.layer == layer].delta_r2.to_numpy()
            m, lo, hi = ps.t_interval(v)
            rows.append({"x": k, "mean": m, "lo": lo, "hi": hi, "instances": len(v)})
        out[int(depth)] = pd.DataFrame(rows)
    return out


def draw_progression(axes, pv: pd.DataFrame) -> dict[tuple[str, int], tuple[int, float]]:
    """Panel (b): one small plot per arm, one curve per target depth; returns
    the peak layer and value of every curve for the build-time print."""
    peaks = {}
    for ax, (method, _title) in zip(axes, ARMS):
        prog = level_progression(pv, method)
        for depth in (1, 2, 3):
            st = prog[depth]
            color = LEVEL_COLOR[depth]
            ax.fill_between(st.x, st.lo, st.hi, color=color, alpha=0.16, linewidth=0, zorder=1)
            ax.plot(st.x, st["mean"], color=color, linewidth=1.2, marker="o",
                    markersize=1.9, markeredgewidth=0, zorder=3)
            k = int(st["mean"].idxmax())
            peaks[(method, depth)] = (k, float(st["mean"].iloc[k]))
            if method == LABELLED_ARM:          # curve names at the right end
                ax.annotate(LEVEL_LABEL[depth], (len(LAYERS) - 1, float(st["mean"].iloc[-1])),
                            xytext=(3, 0), textcoords="offset points", ha="left",
                            va="center", fontsize=6.0, color=color)
        ax.set_xticks(range(len(LAYERS)))
        ax.set_xticklabels(LAYER_LABEL)
        ax.set_xlim(-0.35, len(LAYERS) + 0.35)     # room for the curve names
        ax.set_ylim(-0.03, 0.95)
        ax.set_yticks([0, 0.4, 0.8])
        ps.tidy(ax)
        if ax is axes[0]:
            ax.set_ylabel("Mean intermediate\n" + r"$\Delta R^2$", labelpad=2)
        else:
            ax.set_yticklabels([])
    return peaks


def unit_orders(d) -> dict[str, tuple[list[int], list[int]]]:
    """Unit order and group (Figure-1 number, 0 = none) per level for the
    matrix views: inputs by the local function that reads them, inactive
    last; hidden units by associated intermediate, strongest first."""
    assoc = associations(d)
    corr_strength = [None] + [np.abs(np.asarray(d[f"corr{lv}"])).max(1) for lv in range(1, 5)]

    def order_units(level):
        a, s = assoc[level], corr_strength[level]
        return sorted(range(len(a)), key=lambda u: (a[u] == 0, a[u], -float(s[u])))

    owner = dict(enumerate(assoc[0]))
    in_order = sorted(range(d["codes0"].shape[0]),
                      key=lambda c: (owner[c] == 0, owner[c], c))
    out = {"in": (in_order, [owner[c] for c in in_order])}
    for level, name in ((1, "h1"), (2, "h2")):
        order = order_units(level)
        out[name] = (order, [assoc[level][u] for u in order])
    return out


def draw_matrix(ax, m: np.ndarray, rows: list[int], cols: list[int], title: str, *,
                cmap=None, vmax: float | None = None, bar: float = 1.6) -> None:
    """A (receivers x senders) matrix as cells, receivers top to bottom,
    with the unit-group colour bars on the left and on top: a binary matrix
    is drawn in charcoal, a magnitude matrix through `cmap` up to `vmax`."""
    n_r, n_c = m.shape
    if cmap is None:
        rects = [Rectangle((j, n_r - 1 - i), 1, 1) for i in range(n_r)
                 for j in range(n_c) if m[i, j]]
        ax.add_collection(PatchCollection(rects, facecolor=CHARCOAL, edgecolor="none"))
    else:
        ax.imshow(np.asarray(m, dtype=float), cmap=cmap, vmin=0, vmax=vmax,
                  extent=(0, n_c, 0, n_r), origin="upper", interpolation="nearest",
                  aspect="auto", zorder=1)
    for i, g in enumerate(rows):
        ax.add_patch(Rectangle((-bar - 0.4, n_r - 1 - i), bar, 1,
                               facecolor=colour_of(g, muted=not g), edgecolor="none"))
    for j, g in enumerate(cols):
        ax.add_patch(Rectangle((j, n_r + 0.4), 1, bar,
                               facecolor=colour_of(g, muted=not g), edgecolor="none"))
    ax.add_patch(Rectangle((0, 0), n_c, n_r, facecolor="none", edgecolor="black",
                           linewidth=0.5, zorder=2))
    ax.set_xlim(-bar - 0.6, n_c + 0.2)
    ax.set_ylim(-0.2, n_r + bar + 0.6)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=2)


def draw_masks(axes, d) -> None:
    """The binary masks of the two masked transitions (the companion figure
    learned_structure_v2 and, with the effective weights, the appendix figure
    learned_wiring_detail)."""
    orders = unit_orders(d)
    (in_order, in_groups), (h1_order, h1_groups), (h2_order, h2_groups) = (
        orders["in"], orders["h1"], orders["h2"])
    panels = [("in → h1", np.asarray(d["mask0"])[np.ix_(h1_order, in_order)],
               h1_groups, in_groups),
              ("h1 → h2", np.asarray(d["mask1"])[np.ix_(h2_order, h1_order)],
               h2_groups, h1_groups)]
    for ax, (title, m, rows, cols) in zip(axes, panels):
        draw_matrix(ax, m, rows, cols, title)


def main() -> None:
    ps.apply_rc()
    runs, order = load_embeddings()
    pv = pd.read_csv(DATA / "learned_structure/probe_values.csv")
    pv = pv[(pv.task == "tree_functional_anova") & (pv.n == N_SHOWN)]
    fig = plt.figure(figsize=(5.5, 3.0))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.6, 1.0],
                  left=0.075, right=0.995, top=0.925, bottom=0.095,
                  wspace=0.10, hspace=0.36)
    heat = [fig.add_subplot(gs[0, k]) for k in range(3)]
    draw_heatmaps(heat, runs, value_size=6.6, xlabel=False)
    prog = [fig.add_subplot(gs[1, k]) for k in range(3)]
    peaks = draw_progression(prog, pv)
    # one shared x label under the bottom row; the columns are the same arms
    x0 = prog[0].get_position().x0
    x1 = prog[-1].get_position().x1
    fig.text((x0 + x1) / 2, 0.012, "Hidden layer", ha="center", va="bottom")
    # panel letters sit at the figure's left edge, on each row's top line
    for ax, text in ((heat[0], "a"), (prog[0], "b")):
        box = ax.get_position()
        fig.text(0.008, box.y1 + 0.006, f"({text})", fontsize=7.6,
                 fontweight="bold", ha="left", va="bottom")
    save(fig, "learned_structure")
    # the caption's peak layers, printed at build time
    for method, title in ARMS:
        print(f"[learned_structure] {title} Delta R^2 peaks: " + ", ".join(
            f"{LEVEL_LABEL[depth]} at layer {LAYER_LABEL[k]} ({v:.2f})"
            for depth in (1, 2, 3) for k, v in [peaks[(method, depth)]]))
    shown = runs[order[0]]
    print(f"[learned_structure] wiring-detail instance (highest hidden-1 module "
          f"silhouette): seed {int(shown['function_seed'])}, test NMSE "
          f"{float(shown['test_nmse']):.3f}; all 16 at n = {N_SHOWN}: "
          + ", ".join(f"{s}:{module_silhouette(runs[s]):.2f}" for s in order))


if __name__ == "__main__":
    main()
