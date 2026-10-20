"""The paper's per-workload embedding strips (decision 2026-09-22): one strip per workload of the d = 8 setting in the
delivered teaser look (design/teaser/settings-starburst-cube-side.json: the cube frame and axes, glyph titles, the
target as discs and edges, no legend or depth key, the background units hidden), drawn from the workload's own LeWi
winner at n = 2^15 (harvest.py, choose.py).

    python build.py rank [workload ...]    # candidates.csv: metrics + the searched view of every record
    python build.py figure [workload ...]  # settings-<workload>.json -> out-paper/d8/embedding_<workload>.{pdf,svg,png}
    python build.py contact                # out-paper/d8/embedding_all.png: the strips stacked

The rendering is design/teaser-clean-d8/build.py's (the delivered strip's rules: Styled, the cube frame, the glyph
titles, the packing) with its tree-only parts generalized to any target graph: the view criterion and the screen
orientation read the target's depths, wings and readout (label_sets, wings, frame_for), the target panel keeps one
vertical scale for every graph (draw_target); the halos, the learned dendrogram, the ball and the workbench are left out.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("CSFR_FIGURES_OUT", "out-paper/d8")
os.environ.setdefault("CSFR_FIGURES_DATA", "data-paper-d8")
os.environ.setdefault("CSFR_FIGURES_LEWI", "lewi-v1-rgm")

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parents[1]
sys.path.insert(0, str(FIGURES / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402
from scipy.cluster.hierarchy import linkage  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

import paperstyle as ps  # noqa: E402
import teaser as T  # noqa: E402
import teaser_v2 as T2  # noqa: E402
from csfrbench.viz import LEVEL_DY, layout_graph, node_depths  # noqa: E402
from learned_structure import SHAPES, associations, graph_of, input_support, kept, numbering  # noqa: E402

DATA = HERE / "data"
OUT = HERE / "out"
CANDIDATES = HERE / "candidates.csv"
ARM = "lewi-v1-rgm"
N = 32768
# workload key (the record's stem prefix, harvest.py) -> the paper's name, the first title of its strip
NAMES = {"tree": "Depth-4 tree", "ftree": "Fourier tree", "dag": "Reuse DAG", "fdag": "Fourier reuse DAG", "flat": "Flat modules",
         "dense": "High-arity node", "rand": "Sampled depth-4 DAG"}
FIT = 0.02                      # a candidate must reach this test NMSE ...
MIN_UNITS = 5                   # ... and carry at least this many hidden units on every intermediate
COVERAGE = 1.0                  # the cube holds every unit that carries an association
STAR = 2.5                      # the output star's marker area, relative to the other levels
CAMERA = {"elev": 22, "azim": -55}   # the paper's embedding camera: the cube stands in it (cube_frame)
LATTICE = 2000                  # viewing directions tried on the sphere (about 4.5 degrees apart) ...
FINE = (400, 8.0)               # ... then this many within this angular radius of the best
EDGE_ALPHA = 0.6
TREE = 0.56                     # width of the target panel, in cubes
GAP = 0.03                      # inches between neighbouring cubes
GAP_TREE = 0.04                 # inches between the target panel and the first cube
HEIGHT = 1.85                   # the strip's canvas height, inches; the saved file is cropped tight
DISC, LEAF = 62.0, 10.0         # the target graph's node disc and leaf dot areas (pt^2)
TARGET_ROWS = 4                 # the target panel's vertical scale: the depth-4 tree's rows (three levels and the readout)


# ---------------------------------------------------------------- records and metrics

def load(stem: str) -> dict:
    return dict(np.load(DATA / f"{stem}.npz"))


def workload_of(stem: str) -> str:
    return stem.split("_f")[0]


def units(d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions, association (Figure-1 number, 0 = none) and level of every input and hidden unit."""
    assoc = associations(d)
    X = np.concatenate([np.asarray(d[f"codes{i}"], dtype=float) for i in range(5)])
    lab = np.concatenate([np.asarray(a) for a in assoc[:5]])
    lev = np.concatenate([np.full(len(assoc[i]), i) for i in range(5)])
    return X, lab, lev


def depths_of(d) -> dict[int, int]:
    """Figure number of each intermediate -> its depth in the target (1 = reads inputs only)."""
    num = numbering(d)
    return {num[v]: int(dep) for v, dep in enumerate(node_depths(graph_of(d)))}


def wings(d) -> dict[int, int]:
    """Figure number of each intermediate -> the wing it belongs to: the readout nodes when the readout sums several
    (the flat modules, a reuse DAG's three top nodes, a sampled graph's strands), else the intermediates the single
    readout node reads (the tree's two level-2 functions). An intermediate below two wings (a reused node) and the
    single readout node itself belong to none."""
    g = graph_of(d)
    num = numbering(d)
    readout = list(g.readout_indices)
    roots = readout if len(readout) >= 2 else [r - g.A for r in g.node_parents[readout[0]] if r >= g.A]

    def below(v: int) -> set[int]:
        s = {v}
        for r in g.node_parents[v]:
            if r >= g.A:
                s |= below(r - g.A)
        return s

    member: dict[int, set[int]] = {}
    for root in roots:
        for v in below(root):
            member.setdefault(v, set()).add(root)
    return {num[v]: num[next(iter(rs))] for v, rs in member.items() if len(rs) == 1}


def silhouette(P: np.ndarray, labels) -> float:
    """Mean Euclidean silhouette; points of a singleton group are skipped."""
    labels = np.asarray(labels)
    groups = np.unique(labels)
    if len(groups) < 2 or len(P) < 3:
        return float("nan")
    D = cdist(P, P)
    sums = np.stack([D[:, labels == g].sum(1) for g in groups], axis=1)
    counts = np.array([(labels == g).sum() for g in groups])
    own = np.searchsorted(groups, labels)
    rows = np.arange(len(P))
    a = sums[rows, own] / np.maximum(counts[own] - 1, 1)
    other = sums / counts
    other[rows, own] = np.inf
    b = other.min(1)
    keep = counts[own] > 1
    s = (b - a) / np.maximum(a, b)
    return float(s[keep].mean()) if keep.any() else float("nan")


def label_sets(X, lab, d) -> list[tuple[np.ndarray, np.ndarray]]:
    """The groupings the clarity of the picture is judged by: per target depth with two or more intermediates, the
    units of those intermediates by intermediate (the tree's four modules; its two level-2 functions); the units by
    wing (wings); every associated unit by intermediate. A grouping with fewer than two groups is left out, so a
    target with one intermediate has none."""
    depth = depths_of(d)
    sets = []
    for lv in sorted(set(depth.values())):
        members = [k for k, v in depth.items() if v == lv]
        if len(members) >= 2:
            m = np.isin(lab, members)
            sets.append((X[m], lab[m]))
    w = wings(d)
    if len(set(w.values())) >= 2:
        m = np.isin(lab, list(w))
        sets.append((X[m], np.array([w[l] for l in lab[m]])))
    m = lab >= 1
    sets.append((X[m], lab[m]))
    return [(P, l) for P, l in sets if len(np.unique(l)) >= 2]


def clade_recall(d) -> float:
    """The recovery study's graph-level measure: the average-linkage (UPGMA) dendrogram of every input and hidden
    position, restricted after the fact to the active inputs, holds this share of the target's non-trivial input
    supports (a tree's clades; a DAG's; NaN for a target with none, the dense node)."""
    X, _, _ = units(d)
    Z = linkage(X, method="average")
    active = {int(c) for c in np.asarray(d["active_raw"]).ravel() if int(c) >= 0}   # -1 pads a node that reads fewer
    n = len(X)
    sets = {i: frozenset([i]) for i in range(n)}
    found = set()
    for k, (a, b, _, _) in enumerate(Z):
        s = sets[int(a)] | sets[int(b)]
        sets[n + k] = s
        found.add(frozenset(s & active))
    targets = {frozenset(s) for s in input_support(d) if 1 < len(s) < len(active)}
    if not targets:
        return float("nan")
    return len(targets & found) / len(targets)


def metrics(d) -> dict:
    X, lab, lev = units(d)
    S = len(numbering(d))
    counts = [int(((lab == k) & (lev > 0)).sum()) for k in range(1, S + 1)]
    values = [s for s in (silhouette(P, l) for P, l in label_sets(X, lab, d)) if not np.isnan(s)]
    return {"recall": clade_recall(d), "sil3d": float(np.mean(values)) if values else float("nan"),
            "units": " ".join(str(c) for c in counts), "min_units": min(counts), "intermediates": S,
            "dead": int(sum(np.asarray(d[f"dead{lv}"]).sum() for lv in range(1, 5))),
            "open_readout": int(np.asarray(d["mask4"]).sum())}


# ---------------------------------------------------------------- the view: a direction, then a frame

def camera_basis(elev: float, azim: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Screen right, screen up and the direction toward the viewer of a matplotlib camera (as teaser.project)."""
    e, a = np.radians(elev), np.radians(azim)
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    return right, up, np.cross(right, up)


def plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A right-handed screen basis (right, up) of the plane seen along n."""
    n = n / np.linalg.norm(n)
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(helper, n)
    right /= np.linalg.norm(right)
    return right, np.cross(n, right)


def lattice(k: int, centre: np.ndarray | None = None, cap: float | None = None) -> np.ndarray:
    """k directions spread evenly over the sphere (a Fibonacci lattice), or over the cap of angular radius `cap`
    degrees around `centre`."""
    i = np.arange(k) + 0.5
    theta = np.pi * (1 + 5 ** 0.5) * i
    if centre is None:
        z = 1 - 2 * i / k
    else:
        z = 1 - (1 - np.cos(np.radians(cap))) * i / k
    rad = np.sqrt(np.clip(1 - z * z, 0, None))
    pts = np.c_[rad * np.cos(theta), rad * np.sin(theta), z]
    if centre is not None:
        r, u = plane_basis(centre)
        pts = pts @ np.stack([r, u, centre / np.linalg.norm(centre)], axis=0)
    return pts


def view_sets(d) -> tuple[list, np.ndarray]:
    """The groupings of the view criterion and the centred positions of the associated units (its fallback)."""
    X, lab, _ = units(d)
    P = X[lab >= 1]
    return label_sets(X, lab, d), P - P.mean(0)


def separation(sets, spread: np.ndarray, n: np.ndarray) -> float:
    """The view criterion: the mean projected silhouette of the groupings when the positions are seen along n; for a
    target with nothing to separate (one intermediate) the share of the associated units' variance the screen plane
    keeps, so the picture is the widest."""
    r, u = plane_basis(n)
    Q = np.stack([r, u], axis=1)
    values = [s for s in (silhouette(P @ Q, l) for P, l in sets) if not np.isnan(s)]
    if values:
        return float(np.mean(values))
    return float(((spread @ Q) ** 2).sum() / max((spread ** 2).sum(), 1e-12))


def view_search(d, *, k: int = LATTICE, fine=FINE) -> tuple[float, np.ndarray]:
    """The viewing direction (in the record's frame) that separates the target's groupings best: a lattice over the
    sphere, then a finer one around the winner."""
    sets, spread = view_sets(d)
    best = max(((separation(sets, spread, n), n) for n in lattice(k)), key=lambda t: t[0])
    if fine:
        best = max(((separation(sets, spread, n), n) for n in lattice(fine[0], best[1], fine[1])), key=lambda t: t[0])
    return best


def as_angles(n: np.ndarray) -> tuple[float, float]:
    """Elevation and azimuth (degrees) of a viewing direction."""
    n = n / np.linalg.norm(n)
    return float(np.degrees(np.arcsin(n[2]))), float(np.degrees(np.arctan2(n[1], n[0])))


def from_angles(elev: float, azim: float) -> np.ndarray:
    return camera_basis(elev, azim)[2]


def angles_of(rotation: np.ndarray) -> tuple[float, float, float]:
    """The (elevation, azimuth, roll) of the camera that shows the record's positions as the fixed CAMERA shows the
    rotated ones (the workbench's roll convention: plane_basis, then r = cos(roll) r0 + sin(roll) u0)."""
    r_c, u_c, n_c = camera_basis(**CAMERA)
    frame = rotation @ np.stack([r_c, u_c, n_c], axis=1)        # columns r, u, m in the record's frame
    r, u, m = frame[:, 0], frame[:, 1], frame[:, 2]
    r0, u0 = plane_basis(m)
    elev, azim = as_angles(m)
    return elev, azim, float(np.degrees(np.arctan2(r @ u0, r @ r0)))


IDENTITY = np.eye(3)


def mpl_view(view: dict) -> dict:
    """matplotlib's camera for a workbench view (elev, azim, roll): its roll turns the other way."""
    return {"elev": float(view["elev"]), "azim": float(view["azim"]), "roll": -float(view.get("roll", 0.0))}


def screen_basis(view: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The workbench's screen right, up and toward-the-viewer vectors of a view."""
    n = from_angles(float(view["elev"]), float(view["azim"]))
    r0, u0 = plane_basis(n)
    c, s_ = np.cos(np.radians(float(view.get("roll", 0.0)))), np.sin(np.radians(float(view.get("roll", 0.0))))
    return c * r0 + s_ * u0, -s_ * r0 + c * u0, n


def cube_view(cube: dict) -> dict:
    """The camera a cube setting stands its cube in (the paper's 22 / -55 / 0 by default)."""
    return {"elev": float(cube.get("elev", CAMERA["elev"])), "azim": float(cube.get("azim", CAMERA["azim"])),
            "roll": float(cube.get("roll", 0.0))}


def cube_frame(view: dict, cube: dict) -> np.ndarray:
    """The cube's three axes in the record's frame (columns), for a cube that the data's camera `view` shows exactly
    as the camera `cube` shows an axis-aligned cube: R = B B_c^T with B and B_c the two screen bases. LeWi's geometry
    is invariant under rigid motions (its scores are distances), so the frame the cube and the axes mark may stand any
    way the picture needs while the units' picture stays exactly as chosen."""
    B = np.stack(screen_basis(view), axis=1)
    Bc = np.stack(screen_basis(cube_view(cube)), axis=1)
    return B @ Bc.T


def frame_check(view: dict, cube: dict, R: np.ndarray) -> float:
    """matplotlib's own answer to cube_frame: two orthographic cameras over the same limits differ, on the linear part
    of their projections, by inv(M_view) M_cube. Returns the largest deviation from R."""
    fig = plt.figure(figsize=(1, 1))
    linear = []
    for v in (view, cube_view(cube)):
        ax = fig.add_subplot(projection="3d")
        ax.set_proj_type("ortho")
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-1.0, 1.0)
        ax.set_box_aspect((1, 1, 1), zoom=T.ZOOM)
        ax.view_init(**mpl_view(v))
        linear.append(np.asarray(ax.get_proj())[:3, :3])
        ax.remove()
    plt.close(fig)
    return float(np.abs(np.linalg.solve(linear[0], linear[1]) - R).max())


def tight_cube(P: np.ndarray, R: np.ndarray, pad: float = 1.04) -> tuple[np.ndarray, float]:
    """The smallest cube with axes R around the units P: its centre in the record's frame and its half-edge, padded
    by 4% as half_edge pads."""
    Y = P @ R
    lo, hi = Y.min(0), Y.max(0)
    return R @ ((lo + hi) / 2.0), float((hi - lo).max() / 2.0 * pad)


def view_of(d, n: np.ndarray) -> dict:
    """The camera (elev, azim, roll in the record's frame) that shows the units as frame_for(d, n) does."""
    elev, azim, roll = angles_of(frame_for(d, n))
    return {"elev": elev, "azim": azim, "roll": roll}


def frame_for(d, n: np.ndarray) -> np.ndarray:
    """The rotation of the positions under which the fixed CAMERA looks along n, turned in the screen plane so that
    the target's depth-1 intermediates (its modules) run left to right as in the target drawing - the axis from the
    leftmost module's units to the rightmost's is horizontal; with one module, the units' principal axis is - and seen
    from the side on which the readout nodes' units and the output lie above the modules' units."""
    X, lab, lev = units(d)
    g = graph_of(d)
    num = numbering(d)
    pos, depths = layout_graph(g)
    order = sorted(range(g.S), key=lambda v: pos[g.A + v][0])
    modules = [num[v] for v in order if depths[v] == 1 and (lab == num[v]).any()]
    tops = [num[i] for i in g.readout_indices]
    out = np.asarray(d["codes5"], dtype=float)
    r_c, u_c, n_c = camera_basis(**CAMERA)
    B_c = np.stack([r_c, u_c, n_c], axis=1)
    chosen = None
    for side in (1.0, -1.0):
        m = side * n / np.linalg.norm(n)
        r, u = plane_basis(m)
        P = X @ np.stack([r, u], axis=1)
        if len(modules) >= 2:
            left, right = P[lab == modules[0]].mean(0), P[lab == modules[-1]].mean(0)
            phi = float(np.arctan2(*(right - left)[::-1]))
        else:
            C = P[lab >= 1] - P[lab >= 1].mean(0)
            axis = np.linalg.svd(C, full_matrices=False)[2][0]
            phi = float(np.arctan2(axis[1], axis[0]))
        r2, u2 = np.cos(phi) * r + np.sin(phi) * u, -np.sin(phi) * r + np.cos(phi) * u
        Q = np.stack([r2, u2], axis=1)
        P = X @ Q
        high = float(np.concatenate([P[np.isin(lab, tops)], out @ Q])[:, 1].mean())
        low = float(P[np.isin(lab, modules)][:, 1].mean()) if modules else float(P[lab >= 1][:, 1].mean())
        frame = np.stack([r2, u2, m], axis=1) @ B_c.T
        if chosen is None or high - low > chosen[0]:
            chosen = (high - low, frame)
    return chosen[1]


# ---------------------------------------------------------------- drawing

def filtered(d, hide: dict) -> tuple[dict, dict[int, np.ndarray]]:
    """The record without the unassociated units of the levels `hide` names (inert inputs, dead or unassociated
    hidden units): every per-unit array and connection matrix filtered consistently, the active-input table
    re-indexed; and, per level, the indices of the kept units, so the caller can carry the full record's
    associations over (re-derived on the filtered record, a kept unit's receptive field can lose the dropped
    units' paths and its association with them)."""
    levels = {int(k) for k, v in hide.items() if v}
    assoc = associations(d)
    keep = {lv: (np.nonzero(np.asarray(assoc[lv]) > 0)[0] if lv in levels else np.arange(len(assoc[lv]))) for lv in range(5)}
    if not levels:
        return d, keep
    d = dict(d)
    if 0 in levels:
        position = {int(raw): i for i, raw in enumerate(keep[0])}
        # -1 pads a node that reads fewer inputs than the widest one; it is not a coordinate to re-index
        d["active_raw"] = np.array([[position[int(c)] if int(c) >= 0 else -1 for c in np.atleast_1d(row)]
                                    for row in np.asarray(d["active_raw"])])
    for lv, idx in keep.items():
        for key in (f"codes{lv}", f"corr{lv}", f"dead{lv}"):
            d[key] = np.asarray(d[key])[idx]
    for b in range(5):
        for key in (f"gate{b}", f"mask{b}"):
            m = np.asarray(d[key])
            if b + 1 in keep:
                m = m[keep[b + 1]]
            d[key] = m[:, keep[b]]
    return d, keep


STYLE = dict(point=6.5, focus=1.0, background=0.7, faint=0.7, edge_alpha=EDGE_ALPHA, edge_width=0.4,
             fade_far=T.FADE_FAR,
             axis_alpha=1.0, axis_width=0.0,   # 0 keeps whatever teaser drew; below 1 fades the three axes
             star=STAR, readout_boost=1.0,     # the last panel is naturally sparse: its star and its edges can be lifted
             readout_width=1.0,                # ... its edges widened by this factor ...
             star_near=0.0,                    # ... and its star drawn at the near face's alpha and size whatever its depth
             axis_reach=1.0,                   # without a cube the axes reach this quantile of |coordinate|
             cube_alpha=1.0, cube_width=0.0)   # the cube's twelve edges: alpha below 1 fades them, width 0 keeps teaser's


class Styled:
    """Applies a style to teaser's module-level rules for the duration of a render: the connection width, the depth
    fade's far value, the depth span of the fade (the drawn units' nearest and farthest depths at the view), the cube
    and its axes moved as one rigid body onto the frame's axes, the last panel's star drawn at its own depth."""

    def __init__(self, style: dict | None = None, codes: np.ndarray | None = None, show_cube: bool = True, show_axes: bool = True,
                 frame: tuple | None = None):
        self.style = {**STYLE, **(style or {})}
        self.codes = codes
        self.show_cube, self.show_axes = show_cube, show_axes
        self.frame = frame                     # (R, centre): the cube's axes (columns) and centre in the record's frame

    def __enter__(self):
        self.saved = (T.connections, T.FADE_FAR, T.cube)
        self.saved_points = T.points
        width_ = float(self.style["edge_width"])
        original = T.connections
        original_cube = T.cube
        codes = self.codes

        state = self.state = {"width": 1.0, "star": None}      # the panel loop sets these for the last panel

        def connections(ax, depth, send, recv, weight, numbers, *, alpha, width=None, faint=None, recv_numbers=None):
            return original(ax, depth, send, recv, weight, numbers, alpha=alpha, width=width_ * state["width"], faint=faint,
                            recv_numbers=recv_numbers)

        original_points = T.points
        star_near = float(self.style.get("star_near", 0.0)) > 0

        def points(ax, depth, xyz, level, colours, size, *, zorder=3, alpha=1.0):
            star = state["star"]
            if star is not None and level == 4 and alpha >= 1.0 and len(xyz):
                # the last panel: the H4 units nearer the camera than the star are held back and drawn after it, so
                # the star sits in the cloud at its own depth instead of on top of everything
                M = depth[0]
                z = proj3d.proj_transform(xyz[:, 0], xyz[:, 1], xyz[:, 2], M)[2]
                z_star = proj3d.proj_transform(star[:, 0], star[:, 1], star[:, 2], M)[2][0]
                front = z < z_star
                original_points(ax, depth, xyz[~front], level, [c for c, f in zip(colours, front) if not f], size,
                                zorder=zorder, alpha=alpha)
                state["in_front"] = (xyz[front], [c for c, f in zip(colours, front) if f], size)
                return
            if level == 5 and alpha >= 1.0:
                if star_near:                     # the star exempt from the depth fade that would grey and shrink it
                    saved = T.FADE_FAR
                    T.FADE_FAR = 1.0
                    try:
                        original_points(ax, depth, xyz, level, colours, size, zorder=zorder, alpha=alpha)
                    finally:
                        T.FADE_FAR = saved
                else:
                    original_points(ax, depth, xyz, level, colours, size, zorder=zorder, alpha=alpha)
                held = state.pop("in_front", None)
                if held is not None:
                    original_points(ax, depth, held[0], 4, held[1], held[2], zorder=zorder, alpha=alpha)
                return
            return original_points(ax, depth, xyz, level, colours, size, zorder=zorder, alpha=alpha)

        show_cube, show_axes = self.show_cube, self.show_axes
        axis_alpha = float(self.style.get("axis_alpha", 1.0))
        axis_width = float(self.style.get("axis_width", 0.0))
        axis_reach = float(self.style.get("axis_reach", 1.0))
        cube_alpha = float(self.style.get("cube_alpha", 1.0))
        cube_width = float(self.style.get("cube_width", 0.0))
        frame = self.frame

        def cube(ax, lim, view=None):
            M, span = original_cube(ax, lim, view if view is not None else T.VIEW)
            # the cube's twelve edges and three axes are the first lines of the panel
            if frame is not None:
                R, centre = frame
                for line in ax.lines[:15]:
                    q = np.stack(line.get_data_3d(), axis=1)
                    p = centre[None, :] + q @ R.T
                    line.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
            for line in ax.lines[:12]:
                line.set_visible(show_cube)
                if cube_alpha < 1.0:
                    line.set_alpha(cube_alpha)
                if cube_width > 0:
                    line.set_linewidth(cube_width)
            for line in ax.lines[12:15]:
                line.set_visible(show_axes)
                if axis_alpha < 1.0:
                    line.set_alpha(axis_alpha)
                if axis_width > 0:
                    line.set_linewidth(axis_width)
            if codes is not None and len(codes):
                z = proj3d.proj_transform(codes[:, 0], codes[:, 1], codes[:, 2], M)[2]
                span = (float(z.min()), float(z.max()))
                if not show_cube:                     # no cube: the axes span the drawn units along each axis
                    ext = np.quantile(np.abs(codes), axis_reach, axis=0) if axis_reach < 1.0 else np.abs(codes).max(0)
                    for k, line in enumerate(ax.lines[12:15]):
                        p = np.zeros((2, 3))
                        p[0, k], p[1, k] = -ext[k], ext[k]
                        line.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
            return M, span
        T.connections = connections
        T.cube = cube
        T.points = points
        T.FADE_FAR = float(self.style["fade_far"])
        return self

    def __exit__(self, *exc):
        T.connections, T.FADE_FAR, T.cube = self.saved
        T.points = self.saved_points


@contextmanager
def palette(colours: dict | None):
    """Node colours overridden for one render, {number: hex}; the paper's palette (learned_structure.NODE_COLOR) is
    shared by every figure, so the list is edited in place and restored."""
    import learned_structure as LS
    saved = list(LS.NODE_COLOR)
    for k, v in (colours or {}).items():
        LS.NODE_COLOR[int(k) - 1] = v
    try:
        yield
    finally:
        LS.NODE_COLOR[:] = saved


def all_codes(d) -> np.ndarray:
    return np.concatenate([np.asarray(d[f"codes{i}"], dtype=float) for i in range(6)])


def half_edge(d, coverage: float = COVERAGE) -> float:
    """The cube half-edge in the record's own frame: the largest |coordinate| over the units that carry an
    association and the output, padded by 4%."""
    X, lab, lev = units(d)
    P = np.concatenate([X[lab >= 1], np.asarray(d["codes5"], dtype=float)])
    return float(np.quantile(np.abs(P).max(1), coverage)) * 1.04


def save_paper(fig, stem: str, bbox=None) -> None:
    """figstyle.save's three files (out-paper/d8/<stem>.pdf, svg/, png/) with an explicit crop box, and a PNG copy in
    out/ beside this pipeline."""
    from figstyle import FOLDER, PNG_DPI
    options = {"pdf": {"metadata": {"CreationDate": None}}, "svg": {"metadata": {"Date": None}}, "png": {"dpi": PNG_DPI}}
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", bbox_inches=bbox if bbox is not None else "tight", dpi=PNG_DPI)
    for ext, folder in FOLDER.items():
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{stem}.{ext}", bbox_inches=bbox if bbox is not None else "tight", **options[ext])
    plt.close(fig)
    print(f"[fig] {stem}  " + " ".join(f"{ext} {(folder / f'{stem}.{ext}').stat().st_size / 1e3:.0f}K" for ext, folder in FOLDER.items()))


def drawn_bbox(fig, keep, pad: float = 0.1) -> Bbox:
    """The crop box, in inches, around the artists `keep` (the tight crop's padding), for strips whose empty panel
    boxes must not count."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    parts = []
    for a in keep:
        try:
            bb = a.get_tightbbox(r)
        except TypeError:
            bb = a.get_window_extent(r)
        if bb is not None and bb.width > 0:
            parts.append(bb)
    bb = Bbox.union(parts).transformed(fig.dpi_scale_trans.inverted())
    return Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)


LAST: dict = {}                          # what the last strip rendered: page scale (in per unit), cube width, half-edge


def page_scale(fig, ax, view: dict) -> float:
    """Inches per unit of the embedding on the page: the rendered length of the view's screen-right unit vector."""
    r, _, _ = screen_basis(view)
    P = np.array([[0.0, 0.0, 0.0], r])
    x, y, _ = proj3d.proj_transform(P[:, 0], P[:, 1], P[:, 2], ax.get_proj())
    pts = ax.transData.transform(np.c_[x, y])
    return float(np.hypot(*(pts[1] - pts[0])) / fig.dpi)


def draw_target(ax, d, *, margin: float = T2.TARGET_MARGIN) -> None:
    """The target graph as discs and edges, no numbers and no y (the colours name the intermediates): the active
    inputs as dots of the function that reads them, the intermediates as discs of their colour, the output black, in
    the grammar's layout (teaser.tree over csfrbench.viz.layout_graph, so a reuse DAG, the flat modules, the dense
    node and a sampled graph are drawn as the benchmark draws them). One vertical scale for every graph - the depth-4
    tree's four rows - with the graph centred on the panel, so a level is the same height in every strip."""
    from learned_structure import NODE_COLOR, OUTPUT_COLOR
    g, pos, readout, (xlim, _) = T.tree(d, margin)
    num = numbering(d)
    owner = {r: v for v, refs in enumerate(g.node_parents) for r in refs if r < g.A}
    T.draw_edges(ax, g, pos, readout, T.EDGE)
    for leaf in range(g.A):
        ax.scatter(*pos[leaf], s=LEAF, color=NODE_COLOR[num[owner[leaf]] - 1], zorder=3, linewidths=0)
    for v in range(g.S):
        x, y = pos[g.A + v]
        ax.scatter([x], [y], s=DISC, color=NODE_COLOR[num[v] - 1], zorder=4, edgecolor="#ffffff", lw=0.8)
    ax.scatter(*readout, s=DISC, color=OUTPUT_COLOR, zorder=4, edgecolor="#ffffff", lw=0.8)
    span = LEVEL_DY * TARGET_ROWS + 2.1         # the tree's own limits, (-1.0, yy + 1.1) with yy = LEVEL_DY * 4
    mid = readout[1] / 2.0
    ax.set_xlim(*xlim)
    ax.set_ylim(mid - span / 2, mid + span / 2)
    ax.set_axis_off()


# ---------------------------------------------------------------- titles that carry their own key

LEVEL_NAME = ["Input", "Layer 1", "Layer 2", "Layer 3", "Layer 4", "Output"]


def title_runs(headings: list[str], glyphs: bool) -> list[list[tuple]]:
    """Each title as a run of pieces: ("t", text) and ("m", level). With `glyphs`, a transition title carries the two
    markers it is about - "Input * -> Layer 1 *" with the panels' own shapes."""
    first = [("t", line) for line in headings[0].split("\n")]          # a strip title given on two lines ("\n") keeps them
    if not glyphs:
        return [first] + [[("t", h)] for h in headings[1:]]
    runs = [first]
    for (a, b) in T2.PANELS:
        runs.append([("t", LEVEL_NAME[a]), ("m", a), ("t", "→"), ("t", LEVEL_NAME[b]), ("m", b)])
    return runs


# at one nominal size a circle fills 79% of its box, a triangle 50%, a square all of it; these lift the round and the
# pointed glyphs to one visual size
GLYPH_SCALE = {"o": 1.13, "^": 1.10, "v": 1.10}
GLYPH = 0.56          # a title glyph's nominal size as a fraction of the title size
RUN_GAP = 0.14        # the space between a title's pieces, as a fraction of the title size


def _marker_pt(level: int, size: float) -> float:
    return size * GLYPH * GLYPH_SCALE.get(SHAPES[level], 1.0) * (STAR ** 0.5 if level == 5 else 1.0)


def run_width(fig, renderer, run: list[tuple], size: float) -> float:
    """The run's width in inches at `size`, markers and word spaces included."""
    gap = size * RUN_GAP / 72.0
    total = 0.0
    for kind, value in run:
        if kind == "t":
            probe = fig.text(0, 0, value, fontsize=size)
            total += probe.get_window_extent(renderer).width / fig.dpi
            probe.remove()
        else:
            total += _marker_pt(value, size) / 72.0
        total += gap
    return max(total - gap, 1e-6)


def draw_run(fig, renderer, run: list[tuple], centre: float, y: float, size: float, colour: str) -> list:
    """Lay the run out left to right about `centre` (figure fractions) and return its artists."""
    from matplotlib.lines import Line2D
    width = run_width(fig, renderer, run, size)
    gap = size * RUN_GAP / 72.0
    x = centre - width / 2 / T2.WIDTH
    out = []
    for kind, value in run:
        if kind == "t":
            t = fig.text(x, y, value, ha="left", va="baseline", fontsize=size, color=colour)
            fig.canvas.draw()
            w = t.get_window_extent(renderer).width / fig.dpi
            out.append(t)
        else:
            ms = _marker_pt(value, size)
            w = ms / 72.0
            # the marker sits on the cap-line rather than the baseline, so it reads as part of the word
            line = Line2D([x + w / 2 / T2.WIDTH], [y + size * 0.30 / 72.0 / fig.get_figheight()],
                          marker=SHAPES[value], linestyle="none", markersize=ms,
                          markerfacecolor=colour, markeredgecolor="white", markeredgewidth=0.3,
                          transform=fig.transFigure, clip_on=False)
            fig.add_artist(line)
            out.append(line)
        x += (w + gap) / T2.WIDTH
    return out


def strip(d, stem: str, view: dict, *, colours: dict | None = None, **kw) -> None:
    """The strip under an optional per-render palette (see _strip)."""
    with palette(colours):
        _strip(d, stem, view, **kw)


def _strip(d, stem: str, view: dict, *, title: str, lim: float | None = None, cube_fill: float = 1.0,
           style: dict | None = None, hide: dict | None = None, show_background: bool = True, show_readout: bool = True,
           gap: float = GAP, gap_tree: float = GAP_TREE, height: float = HEIGHT, tree: float = TREE,
           show_cube: bool = True, show_axes: bool = True, show_legend: bool = False, show_depth: bool = False,
           glyphs: bool = True, title_lines: int = 1, pad: float = 0.03, cube: dict | None = None) -> None:
    """The strip: the target graph, then the five transition panels in one frame, cube and view (the delivered strip's
    layout and rules; the connections are the executed masks, the readout's included). `cube` names the camera the
    cube stands in and the offset of its centre (screen right, up; units) from the smallest cube of that stance around
    the associated units and the output; `lim` is the half-edge (None: that smallest cube times `cube_fill`)."""
    ps.apply_rc()
    frame = None
    if show_cube and cube:
        R = cube_frame(view, cube)
        drift = frame_check(view, cube, R)
        if drift > 1e-6:
            raise RuntimeError(f"cube frame: the workbench's camera and matplotlib's disagree by {drift:.2e}")
        X, lab, _ = units(d)
        centre, tight = tight_cube(np.concatenate([X[lab >= 1], np.asarray(d["codes5"], dtype=float)]), R)
        r, u, _ = screen_basis(view)
        ox, oy = (list(cube.get("offset") or ()) + [0.0, 0.0])[:2]
        frame = (R, centre + float(ox) * r + float(oy) * u)
        if lim is None:
            lim = tight * float(cube_fill)
        print(f"[{stem}] cube frame: camera {cube_view(cube)}, half-edge {lim:.3f} (tight {tight:.3f}), offset ({ox}, {oy}), "
              f"matplotlib drift {drift:.1e}")
    if lim is None:
        lim = half_edge(d, COVERAGE)
    style = {**STYLE, **(style or {})}
    full = d
    assoc_full = associations(full)
    d, keep_units = filtered(d, hide or {})
    assoc = [list(np.asarray(assoc_full[lv])[keep_units[lv]]) for lv in range(5)] + [list(assoc_full[5])]
    saved_height = T2.HEIGHT
    T2.HEIGHT = height                                  # pack() places the row from the module's height
    fig = plt.figure(figsize=(T2.WIDTH, height), label=stem)
    target = fig.add_axes([T2.MARGIN / T2.WIDTH, 0.17, 0.1, 0.69])
    draw_target(target, full)
    panels = []
    with Styled(style, all_codes(d), show_cube, show_axes, frame) as styled:
        for transition in T2.PANELS:
            ax = fig.add_axes([0.2, 0.17, 0.15, 0.69], projection="3d")
            ax.set_facecolor("none")
            ax.set_proj_type("ortho")
            weights = kept if show_readout or transition[0] < 4 else (lambda rec, b: np.zeros_like(np.asarray(rec[f"mask{b}"]), dtype=float))
            last = transition[0] == 4                      # Layer 4 -> Output: a handful of edges against four dense panels
            boost = float(style.get("readout_boost", 1.0)) if last else 1.0
            styled.state["width"] = float(style.get("readout_width", 1.0)) if last else 1.0
            styled.state["star"] = np.asarray(d["codes5"], dtype=float) if last else None
            T.draw_geometry(ax, d, IDENTITY, lim, transition=transition, assoc=assoc, point=float(style["point"]), view=mpl_view(view),
                            readout=False, weights=weights, output_star=False,
                            star_scale=float(style.get("star", STAR)), faint=float(style["faint"]),
                            faint_edges=float(style["faint"]),
                            background=float(style["background"]) if show_background else 0.0,
                            edge_alpha=min(1.0, float(style["edge_alpha"]) * boost))
            panels.append(ax)
    cubes, block = T2.pack(fig, panels, target=tree, gap=gap, gap_target=gap_tree)
    low, high = T.cube_extent(fig, panels)
    keep = [line for ax in panels for line in ax.lines] + [target]
    target.set_position([T2.MARGIN / T2.WIDTH, low, block / T2.WIDTH, high - low])
    centres = [(T2.MARGIN + block / 2) / T2.WIDTH] + [(c[0] + c[2]) / 2 / T2.WIDTH for c in cubes]
    headings = [title] + [f"{LEVEL_NAME[a]} $\\rightarrow$ {LEVEL_NAME[b]}" for a, b in T2.PANELS]
    runs = title_runs(headings, glyphs)
    fig.text(0.5, high + 0.035, "", fontsize=T.TITLE)                  # the canvas must exist before a text is measured
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [run_width(fig, renderer, run, T.TITLE) for run in runs]
    pitch = min(np.diff(sorted(centres))) * T2.WIDTH if len(centres) > 1 else T2.WIDTH
    # a title never reaches its neighbour: title_lines 1 = one row at the size at which no title reaches its neighbour;
    # 2 = a transition title breaks after its arrow onto two lines when the row would have to shrink; "stagger" = the
    # odd panels' titles on an upper row and the even panels' on a lower one
    mode = str(title_lines)
    stagger = glyphs and mode == "stagger"
    two_lines = glyphs and mode == "2" and max(widths) > 0.94 * pitch
    lines = [(run[:3], run[3:]) if len(run) == 5 else (run, []) for run in runs] if two_lines else [(run, []) for run in runs]
    lines = [(run[:1], run[1:]) if len(run) == 2 else line for run, line in zip(runs, lines)]
    if two_lines or any(len(run) == 2 for run in runs):
        widths = [max(run_width(fig, renderer, a, T.TITLE), run_width(fig, renderer, b, T.TITLE) if b else 0.0) for a, b in lines]
    title_size = max(5.0, min(T.TITLE, T.TITLE * 0.94 * (2 * pitch if stagger else pitch) / max(widths)))
    line_step = title_size * 1.3 / 72.0 / fig.get_figheight()
    base = high + 0.035
    upper = [i == 0 or (i - 1) % 2 == 0 for i in range(len(runs))] if stagger else [bool(b) for _, b in lines]
    print(f"[{fig.get_label()}] titles at {title_size:.2f} pt, {'staggered on two rows' if stagger else ('two lines' if two_lines else 'one row')} "
          f"(pitch {pitch:.2f} in, the widest title {max(widths):.2f} in at {T.TITLE} pt)")
    # the titles sit on one optical line: place them, measure where each one's capitals land, shift each by the difference
    written = []
    for x, (a, b), up in zip(centres, lines, upper):
        group = draw_run(fig, renderer, a, x, base + (line_step if up else 0.0), title_size, "#2a2a2a")
        if b:
            group += draw_run(fig, renderer, b, x, base, title_size, "#2a2a2a")
        written.append(group)
    fig.canvas.draw()
    tops = [max(t.get_window_extent(renderer).y1 for t in group if hasattr(t, "get_window_extent")) for group in written]
    for group, top, up in zip(written, tops, upper):
        row_top = max(t for t, u in zip(tops, upper) if u == up)
        shift = (row_top - top) / fig.dpi / fig.get_figheight()
        for t in group:
            t.set_y(t.get_position()[1] + shift) if hasattr(t, "set_y") else t.set_ydata([t.get_ydata()[0] + shift])
    order = [0, 3, 1, 4, 2, 5]
    legend = None
    legend_right = T2.MARGIN / T2.WIDTH
    if show_legend:
        legend = fig.legend([T.marker_handles(STAR)[i] for i in order], [LEVEL_NAME[i] for i in order], loc="upper left", ncol=3,
                            fontsize=5.8, frameon=False, handlelength=0.9, handletextpad=0.2, columnspacing=0.6, labelspacing=0.2,
                            borderaxespad=0.0, bbox_to_anchor=(T2.MARGIN / T2.WIDTH, low - 0.03))
        fig.canvas.draw()
        legend_right = legend.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi / T2.WIDTH
    arrow = None
    if show_depth:
        x0 = max(cubes[0][0] / T2.WIDTH + 0.03, legend_right + 0.08 / T2.WIDTH)
        x1 = cubes[-1][2] / T2.WIDTH - 0.03
        y = low - 0.07
        inset = 0.06 * (x1 - x0)
        arrow = fig.add_artist(FancyArrowPatch((x0 + inset, y), (x1 - inset, y), transform=fig.transFigure, arrowstyle="-|>",
                                               mutation_scale=4, color="#bcbcbc", linewidth=0.42, shrinkA=0, shrinkB=0))
        fig.text((x0 + x1) / 2, y - 0.022, "Network depth", ha="center", va="top", fontsize=5.8, color="#9b9b9b")
    bbox = drawn_bbox(fig, keep + list(fig.texts) + list(fig.artists) + [a for a in (arrow, legend) if a is not None], pad=pad)
    LAST.clear()
    LAST.update(title_size=float(title_size), scale=page_scale(fig, panels[0], view), lim=float(lim),
                cube_width=float(cubes[0][2] - cubes[0][0]), page=[float(bbox.width), float(bbox.height)],
                frame=None if frame is None else {"centre": [float(v) for v in frame[1]], "axes": np.round(frame[0], 6).tolist()})
    save_paper(fig, stem, bbox)
    T2.HEIGHT = saved_height


# ---------------------------------------------------------------- verbs

LAYOUT_KEYS = ("gap", "gap_tree", "height", "tree", "pad")


def settings_kwargs(settings: dict) -> dict:
    """The strip's keyword arguments a settings file asks for."""
    return dict(lim=settings.get("lim"), cube_fill=settings.get("cube_fill", 1.0), style=settings.get("style"),
                hide=settings.get("hide"), show_background=settings.get("show_background", True),
                show_readout=settings.get("show_readout", True), show_cube=settings.get("show_cube", True),
                show_axes=settings.get("show_axes", True), show_legend=settings.get("show_legend", False),
                show_depth=settings.get("show_depth", False), glyphs=settings.get("glyphs", True),
                colours=settings.get("colours"), cube=settings.get("cube"), title_lines=settings.get("title_lines", 1),
                **{k: settings["layout"][k] for k in LAYOUT_KEYS if k in settings.get("layout", {})})


def settings_view(settings: dict) -> dict:
    return {"elev": float(settings["elev"]), "azim": float(settings["azim"]), "roll": float(settings.get("roll", 0.0))}


def rank(keys: list[str] | None = None) -> pd.DataFrame:
    """candidates.csv: every record's metrics and searched view; with `keys`, only those workloads are re-ranked and
    the other rows are kept."""
    rows = []
    for path in sorted(DATA.glob(f"*_n{N}__{ARM}.npz")):
        if keys and workload_of(path.stem) not in keys:
            continue
        d = load(path.stem)
        score, n = view_search(d)
        elev, azim = as_angles(n)
        m = metrics(d)
        rows.append({"stem": path.stem, "workload": workload_of(path.stem), "seed": int(d["function_seed"]), "n": int(d["n"]),
                     "test": float(d["test_nmse"]), **m, "elev": round(elev, 1), "azim": round(azim, 1), "view": score})
        print(f"[rank] {path.stem:34s} test {m and rows[-1]['test']:.4f} recall {m['recall']:.2f} sil {m['sil3d']:.2f} "
              f"units {m['units']} view {score:.2f} along ({elev:.0f}, {azim:.0f})", flush=True)
    df = pd.DataFrame(rows)
    if keys and CANDIDATES.exists():
        old = pd.read_csv(CANDIDATES)
        df = pd.concat([old[~old.workload.isin(keys)], df], ignore_index=True)
    # eligible: a network that fits its instance (test NMSE at most FIT) and shows every intermediate (at least
    # MIN_UNITS hidden units associated with each); ranked by the input supports the dendrogram recovers, then by
    # the projected separation at the searched view
    df["eligible"] = (df.test <= FIT) & (df.min_units >= MIN_UNITS)
    df = df.sort_values(["workload", "eligible", "recall", "view"], ascending=[True, False, False, False],
                        na_position="last").reset_index(drop=True)
    df.to_csv(CANDIDATES, index=False)
    pd.set_option("display.width", 250)
    print(df.to_string())
    return df


def figure(keys: list[str] | None = None) -> None:
    """settings-<workload>.json rendered as out-paper/d8/embedding_<workload>.{pdf,svg,png} (+ out/<stem>.png), the
    render recorded in figure.json."""
    record = json.loads((HERE / "figure.json").read_text()) if (HERE / "figure.json").exists() else {}
    for key in keys or list(NAMES):
        path = HERE / f"settings-{key}.json"
        if not path.exists():
            print(f"[figure] {key}: no {path.name} yet (choose.py)")
            continue
        settings = json.loads(path.read_text())
        d = load(settings["stem"])
        strip(d, f"embedding_{key}", settings_view(settings), title=settings.get("title", NAMES[key]), **settings_kwargs(settings))
        record[key] = {"stem": settings["stem"], "view": settings_view(settings), "render": dict(LAST)}
    (HERE / "figure.json").write_text(json.dumps(record, indent=1) + "\n")


def contact() -> None:
    """out-paper/d8/embedding_all.png: the strips stacked, each under its workload and record."""
    from figstyle import FOLDER, OUT as PAPER
    ps.apply_rc()
    choices = json.loads((HERE / "choices.json").read_text()) if (HERE / "choices.json").exists() else {}
    keys = [k for k in NAMES if (FOLDER["png"] / f"embedding_{k}.png").exists()]
    images = [plt.imread(FOLDER["png"] / f"embedding_{k}.png") for k in keys]
    width = 7.0
    heights = [width * im.shape[0] / im.shape[1] + 0.22 for im in images]
    fig, axes = plt.subplots(len(keys), 1, figsize=(width, sum(heights)), gridspec_kw={"height_ratios": heights})
    for ax, key, im in zip(np.atleast_1d(axes), keys, images):
        ax.imshow(im)
        ax.set_axis_off()
        c = choices.get(key, {})
        label = f"{NAMES[key]} - {c.get('stem', '')}, test NMSE {c.get('test_nmse', float('nan')):.4f}, {c.get('rule', '').split(':')[0]}"
        ax.set_title(label, loc="left", fontsize=7, pad=2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.12)
    fig.savefig(PAPER / "embedding_all.png", dpi=250)
    plt.close(fig)
    print(f"[contact] {len(keys)} strips -> {PAPER / 'embedding_all.png'}")


def main(argv: list[str]) -> None:
    verb, keys = (argv[0] if argv else "figure"), argv[1:]
    if verb == "rank":
        rank(keys or None)
    elif verb == "figure":
        figure(keys or None)
    elif verb == "contact":
        contact()
    else:
        raise SystemExit(f"unknown verb {verb!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
