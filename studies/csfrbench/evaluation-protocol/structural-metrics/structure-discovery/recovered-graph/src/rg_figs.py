"""Figures for the recovered-computation-graph report: inline SVG, no raster."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

HERE = Path(__file__).resolve().parent
RG = HERE.parent
FIG = RG / "figures"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RG.parent / "component-dag" / "src"))
import cd_score as cs  # noqa: E402

CELLS = ("tree", "ftree", "dag", "fdag", "flat", "rand", "dense")
STRUCT = CELLS[:-1]
SHORT = {"tree": "tree", "ftree": "Fourier tree", "dag": "reuse DAG", "fdag": "Fourier reuse DAG",
         "flat": "flat modules", "rand": "random graph", "dense": "dense node"}
STACK = {"tree": "tree", "ftree": "Fourier\ntree", "dag": "reuse\nDAG", "fdag": "Fourier\nreuse DAG",
         "flat": "flat\nmodules", "rand": "random\ngraph", "dense": "dense\nnode"}
SETTINGS = ("d32", "d8")
SET_SHORT = {"d32": "d = 32: 8 active + 24 nuisance inputs", "d8": "d = 8: every input active"}
COL = {"d32": "#7b2d5e", "d8": "#0f6f6c", "init": "#9a9a9a", "shuffle_all": "#c0392b",
       "shuffle_higher": "#e08214", "random_match": "#3b7dd8", "leaf_permutation": "#2a9d8f",
       "first": "#3b7dd8", "higher": "#b8342a", "direct": "#5b5b8a", "hit": "#2b2b2b",
       "extra": "#c0392b", "miss": "#c0392b"}
CTL_NAME = {"init": "initialization", "shuffle_all": "wiring rewired",
            "shuffle_higher": "higher wiring rewired", "random_match": "random correspondence",
            "leaf_permutation": "leaf labels permuted"}
# the paper's teacher palette (next-gen tracking.py), the readout root held out in grey
TEACHER = ("#c44e52", "#4c72b0", "#55a868", "#8172b3", "#dd8452", "#12a3a3", "#8c564b", "#e377c2")
READOUT_GREY, DIM_GREY, NO_NODE = "#555555", "#c4c4c4", "#9aa5b1"
LEAF, YCOL = "#eef3fb", "#fdf3e3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def _svg(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"{name}.svg"
    fig.savefig(p, format="svg", bbox_inches="tight")
    plt.close(fig)
    return p.read_text().split("<svg", 1)[1].join(["<svg", ""])


def _bar(ax, x, c, color, width=0.5, off=0.0, hatch=None, alpha=1.0):
    if not c or c["mean"] is None:
        return
    ax.bar(x + off, c["mean"], width, color=color, zorder=2, hatch=hatch, alpha=alpha,
           edgecolor="none" if hatch is None else color)
    if c["n"] > 1:
        ax.plot([x + off, x + off], [c["lo"], c["hi"]], color="#333", lw=.9, zorder=3)


def _xt(ax, cells, size=7.2):
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([STACK[c] for c in cells], fontsize=size)


def node_colours(rec):
    """The teacher palette over the target's canonical nodes: a single readout
    root is held out in grey, as in the embedding figures."""
    nodes = sorted(int(k) for k in rec["target_parents"])
    root = rec["target_root"]
    internal = [v for v in nodes if v != root]
    col = {v: TEACHER[k % len(TEACHER)] for k, v in enumerate(internal)}
    if root is not None:
        col[root] = READOUT_GREY
    return col


# ----------------------------------------------------------------- graphs --

def _layout(nodes, edges):
    depth = {n: 0 for n in nodes}
    for _ in range(len(nodes)):
        for a, b in edges:
            if a in depth and b in depth and depth[b] <= depth[a]:
                depth[b] = depth[a] + 1
    by = {}
    for n, d in depth.items():
        by.setdefault(d, []).append(n)
    pos, width = {}, 0
    for d, ns in by.items():
        ns = sorted(ns, key=lambda n: (n[0] == "extra", str(n[0]), -n[1]))
        width = max(width, len(ns))
        for i, n in enumerate(ns):
            pos[n] = (d, i - (len(ns) - 1) / 2)
    return pos, max(depth.values(), default=0), width


def _draw(ax, nodes, edges, labels, colours, title, span, edge_style, hollow=(), title_size=7.8):
    pos, D, _ = _layout(nodes, edges)
    for e in edges:
        a, b = e
        if a not in pos or b not in pos:
            continue
        col, ls = edge_style.get(e, (COL["hit"], "-"))
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>", mutation_scale=6,
                                     color=col, lw=.8 if ls == "-" else .9, linestyle=ls,
                                     shrinkA=7, shrinkB=7, zorder=1, connectionstyle="arc3,rad=0.05"))
    for n in nodes:
        if n not in pos:
            continue
        x, y = pos[n]
        if n in hollow:
            ax.scatter([x], [y], s=150, facecolor="white", edgecolor=colours.get(n, "#444"),
                       linewidths=1.2, linestyle="--", zorder=2)
        else:
            ax.scatter([x], [y], s=150, facecolor=colours.get(n, "#fff"), edgecolor="#444",
                       linewidths=.8, zorder=2)
        dark = colours.get(n, "#fff") in TEACHER or colours.get(n) == READOUT_GREY
        ax.text(x, y, labels.get(n, str(n)), ha="center", va="center", fontsize=5.6,
                color="white" if dark and n not in hollow else "#111", zorder=3)
    ax.set_title(title, fontsize=title_size)
    ax.axis("off")
    ax.set_xlim(-0.55, D + 0.55)
    ax.set_ylim(-span / 2 - 0.9, span / 2 + 0.9)


def target_and_recovered(rec):
    """(nodes, edges, labels, colours) of the target and of the recovered graph
    of one record, on the target's keys; edges classified hit / miss / extra."""
    raw = rec["active_raw"]
    t = cs.canonical_target({"A": rec["A"], "node_parents": rec["raw_node_parents"],
                             "readout_indices": rec["raw_readout_indices"]})
    tedges = sorted(cs.target_edges(t))
    tnodes = sorted({n for e in tedges for n in e} | {("leaf", a) for a in range(rec["A"])}, key=str)
    ncol = node_colours(rec)
    lab, col = {}, {}
    for n in tnodes:
        if n[0] == "leaf":
            lab[n], col[n] = f"x{raw[n[1]]}", LEAF
        elif n[0] == "y":
            lab[n], col[n] = "y", (ncol.get(rec["target_root"], YCOL) if rec["target_root"] is not None else YCOL)
        else:
            lab[n], col[n] = f"z{n[1]}", ncol[n[1]]
    g = rec["graph"]
    node_of = {}
    for k, v in g["map"].items():
        a, b = k.split(":")
        node_of[(a, int(b))] = tuple(v)
    E = {(tuple(a), tuple(b)) for a, b in g["edges"]}
    A = {(node_of[a], node_of[b]) for a, b in cs.reachability(E, set(node_of))}
    R = sorted(cs.transitive_reduction(A))
    dnodes = sorted({n for e in R for n in e} | set(node_of.values()), key=str)
    for n in dnodes:
        if n not in lab:
            if n[0] == "extra":
                lab[n], col[n] = f"x{n[1]}", "#f2f2f2"
            elif n[0] == "leaf":
                lab[n], col[n] = f"x{raw[n[1]]}", LEAF
    hits = set(R) & set(tedges)
    missing = [n for n in tnodes if n not in dnodes]
    return {"t": t, "tnodes": tnodes, "tedges": tedges, "dnodes": dnodes + missing, "redges": R,
            "labels": lab, "colours": col, "hits": hits, "missing": missing}


def graph_pair(axes, rec, setting, short=False):
    G = target_and_recovered(rec)
    sc = rec["wirings"]["learned"]
    span = max(_layout(G["tnodes"], G["tedges"])[2], _layout(G["dnodes"], G["redges"])[2])
    tstyle = {e: ("#b0b0b0", "-") if e in G["hits"] else (COL["miss"], "-") for e in G["tedges"]}
    rstyle = {e: (COL["hit"], "-") if e in G["hits"] else (COL["extra"], "--") for e in G["redges"]}
    matched = (f"{sc['n_matched']} of {rec['n_nodes']} nodes matched" if rec["n_nodes"]
               else "no intermediate node to match")
    if short:
        t1 = f"target: {SHORT[rec['cell']]} ({rec['function_seed']})"
        t2 = f"recovered: $F_1$ {sc['edge']['f1']:.2f}, {len(G['hits'])}/{len(G['tedges'])} edges"
    else:
        t1 = f"target: {SHORT[rec['cell']]}, instance {rec['function_seed']}, {SET_SHORT[setting].split(':')[0]}"
        t2 = (f"recovered from the wiring, n = {rec['n']:,}: edge $F_1$ {sc['edge']['f1']:.2f}, "
              f"{len(G['hits'])} of {len(G['tedges'])} target edges, {matched}")
    ts = 6.6 if short else 7.8
    _draw(axes[0], G["tnodes"], G["tedges"], G["labels"], G["colours"], t1, span, tstyle, title_size=ts)
    _draw(axes[1], G["dnodes"], G["redges"], G["labels"], G["colours"], t2, span, rstyle,
          hollow=set(G["missing"]), title_size=ts)
    return span


def graph_legend(fig, y=-0.02):
    handles = [Line2D([0], [0], color=COL["hit"], lw=1.2, label="recovered edge that is a target edge"),
               Line2D([0], [0], color=COL["extra"], lw=1.2, ls="--", label="recovered edge with no target edge"),
               Line2D([0], [0], color=COL["miss"], lw=1.2, label="target edge not recovered (on the target)"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#444",
                      markersize=7, label="target node with no matched component (hollow)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.8, frameon=False,
               bbox_to_anchor=(0.5, y))


def examples(recs, setting, per_figure=4):
    """Target -> recovered for one representative unit per family, in figures
    of at most ``per_figure`` families so each fits a printed page."""
    cells = [c for c in CELLS if c in recs]
    out = []
    for k in range(0, len(cells), per_figure):
        chunk = cells[k:k + per_figure]
        spans = []
        for c in chunk:
            G = target_and_recovered(recs[c])
            spans.append(max(_layout(G["tnodes"], G["tedges"])[2], _layout(G["dnodes"], G["redges"])[2]))
        heights = [max(2.2, 0.24 * s + 0.6) for s in spans]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(9.4, sum(heights)),
                                 gridspec_kw={"height_ratios": heights})
        axes = np.atleast_2d(axes)
        for i, c in enumerate(chunk):
            graph_pair(axes[i], recs[c], setting)
        graph_legend(fig, y=-0.012 - 0.02 * (4 - len(chunk)))
        fig.tight_layout()
        out.append(_svg(fig, f"examples_{setting}_{k // per_figure + 1}"))
    return out


def paper(read, recs32):
    """The paper figure: target -> recovered for a tree and a held-out random
    graph, then edge F1 with the controls in both settings."""
    fig = plt.figure(figsize=(9.6, 6.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0], hspace=0.5, wspace=0.25)
    top = [fig.add_subplot(gs[0, k]) for k in range(4)]
    for k, cell in enumerate(("tree", "rand")):
        if cell in recs32:
            graph_pair(top[2 * k:2 * k + 2], recs32[cell], "d32", short=True)
        else:
            for ax in top[2 * k:2 * k + 2]:
                ax.axis("off")
    for k, s in enumerate(SETTINGS):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        sr = read["settings"].get(s)
        if not sr:
            ax.axis("off"); continue
        n = str(sr["n_max"])
        cells = [c for c in CELLS if c in sr["cells"]]
        x = np.arange(len(cells))
        for i, c in enumerate(cells):
            _bar(ax, x[i], sr["table"][c][n]["edge"], COL[s], 0.34, -0.19)
            _bar(ax, x[i], sr["controls"][c][n]["init"]["edge"], COL["init"], 0.34, 0.19)
            for ctl, mk in (("shuffle_all", "v"), ("shuffle_higher", "^")):
                v = sr["controls"][c][n][ctl]["edge"]
                if v["mean"] is not None:
                    ax.plot([x[i] - 0.19], [v["mean"]], marker=mk, color=COL[ctl], ms=5.5, zorder=4,
                            markeredgecolor="white", markeredgewidth=.5)
        _xt(ax, cells, 6.8)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{SET_SHORT[s]}   (n = {sr['n_max']:,})", fontsize=8)
        if k == 0:
            ax.set_ylabel("edge $F_1$ against the target graph")
    handles = [Patch(color=COL["d32"], label="trained, d = 32"), Patch(color=COL["d8"], label="trained, d = 8"),
               Patch(color=COL["init"], label="initialization"),
               Line2D([0], [0], marker="v", color="w", markerfacecolor=COL["shuffle_all"], ms=7, label="wiring rewired (degrees kept)"),
               Line2D([0], [0], marker="^", color="w", markerfacecolor=COL["shuffle_higher"], ms=7, label="first transition kept, rest rewired")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    graph_legend(fig, y=0.455)
    return _svg(fig, "paper")


# ------------------------------------------------------------- method --

def method():
    """The three steps on a toy network: ancestry sets, the quotient graph,
    the correspondence and the comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [1.25, 0.9, 1.15]})
    # --- A: the wiring with the ancestry set of every unit
    ax = axes[0]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    X = {f"x{i}": (0.06, 0.86 - 0.2 * i) for i in range(4)}
    H1 = {"u0": (0.42, 0.9), "u1": (0.42, 0.72), "u2": (0.42, 0.54), "u3": (0.42, 0.36), "u4": (0.42, 0.14)}
    H2 = {"v0": (0.72, 0.78), "v1": (0.72, 0.5), "v2": (0.72, 0.26)}
    Y = {"y": (0.95, 0.52)}
    E = [("x0", "u0"), ("x1", "u0"), ("x0", "u1"), ("x1", "u1"), ("x2", "u2"), ("x3", "u2"), ("x2", "u3"), ("x3", "u3"),
         ("x1", "u4"), ("u0", "v0"), ("u1", "v0"), ("u1", "v1"), ("u2", "v1"), ("u3", "v1"), ("u2", "v2"), ("u3", "v2"),
         ("u4", "v2"), ("v0", "y"), ("v1", "y"), ("v2", "y")]
    P = {**X, **H1, **H2, **Y}
    anc = {"u0": "{x0,x1}", "u1": "{x0,x1}", "u2": "{x2,x3}", "u3": "{x2,x3}", "u4": "{x1}",
           "v0": "{x0,x1}", "v1": "{x0..x3}", "v2": "{x1,x2,x3}"}
    comp = {"u0": TEACHER[0], "u1": TEACHER[0], "u2": TEACHER[1], "u3": TEACHER[1], "u4": NO_NODE,
            "v0": NO_NODE, "v1": TEACHER[2], "v2": NO_NODE}
    for a, b in E:
        ax.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]], color="#b9b9b9", lw=.7, zorder=1)
    for k, (x, y) in P.items():
        ax.scatter([x], [y], s=95, facecolor=comp.get(k, "#fff" if k != "y" else YCOL), edgecolor="#444",
                   lw=.7, zorder=3)
        ax.text(x, y, k, ha="center", va="center", fontsize=5.4, zorder=4,
                color="white" if k in comp and comp[k] != NO_NODE else "#111")
        if k in anc:
            ax.text(x, y - 0.075, anc[k], ha="center", va="top", fontsize=5.2, color="#555")
    for lab, x in (("inputs", 0.06), ("hidden 1", 0.42), ("hidden 2", 0.72), ("y", 0.95)):
        ax.text(x, 1.0, lab, ha="center", fontsize=6.8, color="#333")
    ax.text(0.5, -0.05, "1  the input ancestry of every live unit,\nread from the open mask only; units of one\n"
            "stage with the same set form a component", ha="center", va="top", fontsize=6.6,
            color=COL["d32"], transform=ax.transAxes)
    # --- B: the quotient graph
    ax = axes[1]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    Q = {"x0": (0.08, 0.86), "x1": (0.08, 0.64), "x2": (0.08, 0.42), "x3": (0.08, 0.2),
         "C1": (0.42, 0.82), "C2": (0.42, 0.34), "C3": (0.42, 0.08),
         "C4": (0.7, 0.86), "C5": (0.7, 0.56), "C6": (0.7, 0.24), "y": (0.95, 0.56)}
    QE = [("x0", "C1"), ("x1", "C1"), ("x2", "C2"), ("x3", "C2"), ("x1", "C3"), ("C1", "C4"), ("C1", "C5"),
          ("C2", "C5"), ("C2", "C6"), ("C3", "C6"), ("C4", "y"), ("C5", "y"), ("C6", "y")]
    qc = {"C1": TEACHER[0], "C2": TEACHER[1], "C3": NO_NODE, "C4": NO_NODE, "C5": TEACHER[2], "C6": NO_NODE}
    for a, b in QE:
        ax.add_patch(FancyArrowPatch(Q[a], Q[b], arrowstyle="-|>", mutation_scale=6, color="#777", lw=.8,
                                     shrinkA=6, shrinkB=6, zorder=1))
    for k, (x, y) in Q.items():
        ax.scatter([x], [y], s=120 if k.startswith("C") else 80, facecolor=qc.get(k, LEAF if k != "y" else YCOL),
                   edgecolor="#444", lw=.7, zorder=3)
        ax.text(x, y, k, ha="center", va="center", fontsize=5.4, zorder=4,
                color="white" if k in qc and qc[k] != NO_NODE else "#111")
    ax.text(0.42, 0.95, "C1 = {u0,u1}   C2 = {u2,u3}   C3 = {u4}\nC4 = {v0}   C5 = {v1}   C6 = {v2}",
            ha="center", va="bottom", fontsize=5.6, color="#555")
    ax.text(0.5, -0.05, "2  the quotient graph: one node per\ncomponent, an arrow where an open\nconnection runs between two of them",
            ha="center", va="top", fontsize=6.6, color=COL["d32"], transform=ax.transAxes)
    # --- C: correspondence and comparison
    ax = axes[2]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    T = {"x0": (0.06, 0.86), "x1": (0.06, 0.64), "x2": (0.06, 0.42), "x3": (0.06, 0.2),
         "z0": (0.3, 0.75), "z1": (0.3, 0.31), "z2": (0.52, 0.53)}
    TE = [("x0", "z0"), ("x1", "z0"), ("x2", "z1"), ("x3", "z1"), ("z0", "z2"), ("z1", "z2")]
    for a, b in TE:
        ax.add_patch(FancyArrowPatch(T[a], T[b], arrowstyle="-|>", mutation_scale=6, color="#777", lw=.8,
                                     shrinkA=6, shrinkB=6, zorder=1))
    tc = {"z0": TEACHER[0], "z1": TEACHER[1], "z2": TEACHER[2]}
    for k, (x, y) in T.items():
        ax.scatter([x], [y], s=110 if k.startswith("z") else 80, facecolor=tc.get(k, LEAF), edgecolor="#444", lw=.7, zorder=3)
        ax.text(x, y, k, ha="center", va="center", fontsize=5.4, zorder=4, color="white" if k in tc else "#111")
    ax.text(0.3, 0.97, "target graph", ha="center", fontsize=6.8, color="#333")
    # the probe table
    x0, y0, w, h = 0.66, 0.9, 0.1, 0.1
    comps = ["C1", "C2", "C3", "C4", "C5", "C6"]
    vals = [[0.71, 0.02, 0.05], [0.01, 0.66, 0.08], [0.02, 0.0, 0.01], [0.62, 0.01, 0.12], [0.11, 0.09, 0.58], [0.03, 0.2, 0.31]]
    sig = [[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]]
    chosen = {(0, 0), (1, 1), (4, 2)}
    for j, z in enumerate(("z0", "z1", "z2")):
        ax.text(x0 + w * (j + 0.5), y0 + 0.02, z, ha="center", va="bottom", fontsize=5.8, color=tc[z])
    for i, cname in enumerate(comps):
        ax.text(x0 - 0.01, y0 - h * (i + 0.5), cname, ha="right", va="center", fontsize=5.8)
        for j in range(3):
            v = vals[i][j]
            ax.add_patch(plt.Rectangle((x0 + w * j, y0 - h * (i + 1)), w, h, facecolor=plt.cm.BuPu(0.15 + 0.85 * v),
                                       edgecolor="white", lw=.6, zorder=2))
            ax.text(x0 + w * (j + 0.5), y0 - h * (i + 0.5), f"{v:.2f}", ha="center", va="center", fontsize=4.8,
                    color="white" if v > 0.45 else "#222", zorder=4)
            if (i, j) in chosen:
                ax.add_patch(plt.Rectangle((x0 + w * j, y0 - h * (i + 1)), w, h, fill=False, edgecolor="#111",
                                           lw=1.4, zorder=5))
            elif sig[i][j]:
                ax.scatter([x0 + w * (j + 0.8)], [y0 - h * (i + 0.2)], s=4, color="#111", zorder=5)
    ax.text(x0 + 1.5 * w, y0 - h * 6.6, "held-out excess $R^2$; · beats every\nchance set; box = the one-to-one\nassignment (C1→z0, C2→z1, C5→z2)",
            ha="center", va="top", fontsize=5.4, color="#555")
    ax.text(0.5, -0.05, "3  held-out probes name the components one-to-one;\nthen the arrows between the named components\n"
            "are compared with the target's edges", ha="center", va="top", fontsize=6.6, color=COL["d32"],
            transform=ax.transAxes)
    fig.subplots_adjust(wspace=0.18, bottom=0.22)
    return _svg(fig, "method")


# ------------------------------------------------------------ analyses --

def controls(read):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), gridspec_kw={"height_ratios": [1.0, 0.9]})
    for k, s in enumerate(SETTINGS):
        sr = read["settings"].get(s)
        if not sr:
            for ax in axes[:, k]:
                ax.axis("off")
            continue
        n = str(sr["n_max"])
        cells = [c for c in CELLS if c in sr["cells"]]
        x = np.arange(len(cells))
        ax = axes[0, k]
        for i, c in enumerate(cells):
            _bar(ax, x[i], sr["table"][c][n]["edge"], COL[s], 0.5, 0)
            for ctl in ("init", "shuffle_all", "shuffle_higher", "random_match", "leaf_permutation"):
                v = sr["controls"][c][n][ctl]["edge"]
                if v["mean"] is not None:
                    ax.plot([x[i] - 0.25, x[i] + 0.25], [v["mean"]] * 2, color=COL[ctl], lw=1.4, zorder=4,
                            label=CTL_NAME[ctl] if i == 0 and k == 0 else None)
        _xt(ax, cells)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{SET_SHORT[s]}, n = {sr['n_max']:,}", fontsize=8)
        if k == 0:
            ax.set_ylabel("edge $F_1$: trained (bar) and the controls (lines)")
        ax = axes[1, k]
        w = 0.26
        for j, ctl in enumerate(("init", "shuffle_all", "shuffle_higher")):
            for i, c in enumerate(cells):
                v = sr["contrasts"][c][n][ctl]["edge"]
                if v["mean"] is None:
                    continue
                xx = x[i] + (j - 1) * w
                ax.plot([xx, xx], [v["lo"], v["hi"]], color=COL[ctl], lw=1.4)
                ax.plot([xx], [v["mean"]], "o", color=COL[ctl], ms=4, label=f"trained minus {CTL_NAME[ctl]}" if i == 0 and k == 0 else None)
        ax.axhline(0, color="#333", lw=.7)
        _xt(ax, cells)
        ax.set_ylim(-0.25, 1.0)
        if k == 0:
            ax.set_ylabel("paired difference in edge $F_1$")
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    h1, l1 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", ncol=5, fontsize=6.6, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.legend(h1, l1, loc="lower center", ncol=3, fontsize=6.6, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 0.96), h_pad=2.6)
    return _svg(fig, "controls")


def levels(read):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
    for k, s in enumerate(SETTINGS):
        ax = axes[k]
        sr = read["settings"].get(s)
        if not sr:
            ax.axis("off"); continue
        n = str(sr["n_max"])
        cells = [c for c in CELLS if c in sr["cells"]]
        x = np.arange(len(cells))
        w = 0.36
        for i, c in enumerate(cells):
            _bar(ax, x[i], sr["table"][c][n]["first"], COL["first"], w, -w / 2 - 0.01)
            _bar(ax, x[i], sr["table"][c][n]["higher"], COL["higher"], w, w / 2 + 0.01)
            v = sr["controls"][c][n]["shuffle_higher"]["higher"]
            if v["mean"] is not None:
                ax.plot([x[i] + w / 2 + 0.01], [v["mean"]], marker="^", color=COL["shuffle_higher"], ms=6,
                        markeredgecolor="white", markeredgewidth=.5, zorder=5,
                        label="higher level after the higher wiring is rewired" if i == 0 and k == 0 else None)
            v = sr["controls"][c][n]["shuffle_higher"]["first"]
            if v["mean"] is not None:
                ax.plot([x[i] - w / 2 - 0.01], [v["mean"]], marker="^", color="#7f9fd8", ms=6,
                        markeredgecolor="white", markeredgewidth=.5, zorder=5,
                        label="first level after the higher wiring is rewired" if i == 0 and k == 0 else None)
        _xt(ax, cells)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{SET_SHORT[s]}, n = {sr['n_max']:,}", fontsize=8)
        if k == 0:
            ax.set_ylabel("$F_1$ by level")
    h, l = axes[0].get_legend_handles_labels()
    h = [Patch(color=COL["first"], label="first level: leaf → node edges"),
         Patch(color=COL["higher"], label="higher level: node → node edges")] + h
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return _svg(fig, "levels")


def curves(read):
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.8), sharey=True)
    for i, cell in enumerate(CELLS):
        ax = axes.flat[i]
        for s in SETTINGS:
            sr = read["settings"].get(s)
            if not sr or cell not in sr["cells"]:
                continue
            ns = [int(n) for n in sr["ns"]]
            for key, ls, lab in (("edge", "-", f"trained, {s.replace('d', 'd = ')}"),):
                m = [sr["table"][cell][str(n)][key]["mean"] for n in ns]
                lo = [sr["table"][cell][str(n)][key]["lo"] for n in ns]
                hi = [sr["table"][cell][str(n)][key]["hi"] for n in ns]
                ok = [j for j, v in enumerate(m) if v is not None]
                if ok:
                    ax.plot([ns[j] for j in ok], [m[j] for j in ok], ls, marker="o", ms=2.6, color=COL[s], lw=1.3,
                            label=lab if i == 0 else None)
                    ax.fill_between([ns[j] for j in ok], [lo[j] for j in ok], [hi[j] for j in ok], color=COL[s], alpha=.14, lw=0)
            m = [sr["controls"][cell][str(n)]["init"]["edge"]["mean"] for n in ns]
            ok = [j for j, v in enumerate(m) if v is not None]
            if ok:
                ax.plot([ns[j] for j in ok], [m[j] for j in ok], "--", color=COL[s], lw=.9, alpha=.6,
                        label=f"initialization, {s.replace('d', 'd = ')}" if i == 0 else None)
        ax.set_xscale("log", base=2)
        ax.set_title(SHORT[cell], fontsize=8)
        ax.set_ylim(0, 1.02)
    ax = axes.flat[7]
    ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    ax.legend(h, l, loc="center", fontsize=7, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("training set size $n$")
    for ax in axes[:, 0]:
        ax.set_ylabel("edge $F_1$")
    fig.tight_layout()
    return _svg(fig, "curves")


def instances(read):
    fig, axes = plt.subplots(2, 7, figsize=(9.6, 4.4), sharey=True)
    rng = np.random.default_rng(0)
    for k, s in enumerate(SETTINGS):
        sr = read["settings"].get(s)
        for i, cell in enumerate(CELLS):
            ax = axes[k, i]
            if not sr or cell not in sr["cells"]:
                ax.axis("off"); continue
            rows = sr["per_instance"][cell][str(sr["n_max"])]
            for j, (key, col) in enumerate((("init", COL["init"]), ("shuffle_all", COL["shuffle_all"]),
                                            ("shuffle_higher", COL["shuffle_higher"]), ("edge", COL[s]))):
                ys = [r[key] for r in rows if r.get(key) is not None]
                xs = j + rng.uniform(-0.18, 0.18, len(ys))
                ax.scatter(xs, ys, s=9, color=col, alpha=.75, lw=0, zorder=3)
                if ys:
                    ax.plot([j - 0.3, j + 0.3], [np.mean(ys)] * 2, color="#222", lw=1.0, zorder=4)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["init", "rewired", "higher\nrewired", "trained"], fontsize=5.6)
            ax.set_ylim(-0.03, 1.03)
            if k == 0:
                ax.set_title(SHORT[cell], fontsize=7.6)
            if i == 0:
                ax.set_ylabel(f"edge $F_1$\n{s.replace('d', 'd = ')}", fontsize=7.5)
    fig.tight_layout()
    return _svg(fig, "instances")


def precision_recall(read):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for s, mk in (("d32", "o"), ("d8", "s")):
        sr = read["settings"].get(s)
        if not sr:
            continue
        n = str(sr["n_max"])
        for cell in CELLS:
            if cell not in sr["cells"]:
                continue
            p, r = sr["table"][cell][n]["precision"], sr["table"][cell][n]["recall"]
            if p["mean"] is None or r["mean"] is None:
                continue
            ax.plot([r["lo"], r["hi"]], [p["mean"]] * 2, color=COL[s], lw=.8, alpha=.6)
            ax.plot([r["mean"]] * 2, [p["lo"], p["hi"]], color=COL[s], lw=.8, alpha=.6)
            ax.plot([r["mean"]], [p["mean"]], marker=mk, color=COL[s], ms=6, markeredgecolor="white", markeredgewidth=.5)
            ax.annotate(SHORT[cell], (r["mean"], p["mean"]), textcoords="offset points", xytext=(5, 4), fontsize=6.2, color="#333")
    ax.plot([0, 1], [0, 1], color="#ccc", lw=.7, ls=":")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("edge recall (share of target edges recovered)")
    ax.set_ylabel("edge precision (share of recovered edges that are target edges)")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["d32"], ms=7, label="d = 32"),
                       Line2D([0], [0], marker="s", color="w", markerfacecolor=COL["d8"], ms=7, label="d = 8")],
              fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    return _svg(fig, "precision_recall")


def direct(read):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for k, s in enumerate(SETTINGS):
        ax = axes[k]
        sr = read["settings"].get(s)
        if not sr:
            ax.axis("off"); continue
        n = str(sr["n_max"])
        cells = [c for c in CELLS if c in sr["cells"]]
        x = np.arange(len(cells))
        w = 0.36
        for i, c in enumerate(cells):
            _bar(ax, x[i], sr["table"][c][n]["edge"], COL[s], w, -w / 2 - 0.01)
            _bar(ax, x[i], sr["table"][c][n]["direct_edge"], COL["direct"], w, w / 2 + 0.01)
            v = sr["controls"][c][n]["init"]["direct_edge"]
            if v["mean"] is not None:
                ax.plot([x[i] - 0.25, x[i] + 0.25], [v["mean"]] * 2, color=COL["init"], lw=1.3, zorder=4)
        _xt(ax, cells)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{SET_SHORT[s]}, n = {sr['n_max']:,}", fontsize=8)
        if k == 0:
            ax.set_ylabel("edge $F_1$")
    fig.legend(handles=[Patch(color=COL["d32"], label="connection through relays allowed (d = 32)"),
                        Patch(color=COL["d8"], label="connection through relays allowed (d = 8)"),
                        Patch(color=COL["direct"], label="direct open connection only"),
                        Line2D([0], [0], color=COL["init"], lw=1.3, label="direct, at initialization")],
               loc="lower center", ncol=4, fontsize=6.6, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return _svg(fig, "direct")


def settings_slope(read):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for k, key in enumerate(("edge", "first", "higher")):
        ax = axes[k]
        ends = []
        for i, cell in enumerate(CELLS):
            pts = []
            for j, s in enumerate(SETTINGS):
                sr = read["settings"].get(s)
                if not sr or cell not in sr["cells"]:
                    continue
                v = sr["table"][cell][str(sr["n_max"])][key]
                if v["mean"] is None:
                    continue
                pts.append((j, v))
                ax.plot([j, j], [v["lo"], v["hi"]], color=TEACHER[i % len(TEACHER)], lw=1.0, alpha=.8)
            if len(pts) == 2:
                ax.plot([0, 1], [pts[0][1]["mean"], pts[1][1]["mean"]], "-o", color=TEACHER[i % len(TEACHER)], ms=4,
                        lw=1.2, label=SHORT[cell])
                ends.append([pts[1][1]["mean"], SHORT[cell], TEACHER[i % len(TEACHER)]])
            elif len(pts) == 1:
                ax.plot([pts[0][0]], [pts[0][1]["mean"]], "o", color=TEACHER[i % len(TEACHER)], ms=4)
                ends.append([pts[0][1]["mean"], SHORT[cell], TEACHER[i % len(TEACHER)]]) if pts[0][0] == 1 else None
        ends.sort()
        ys = [e[0] for e in ends]
        for j in range(1, len(ys)):                      # push labels apart by at least 0.055
            ys[j] = max(ys[j], ys[j - 1] + 0.055)
        for (y0, lab, col), y in zip(ends, ys):
            ax.annotate(lab, (1, y0), xytext=(1.12, y), textcoords="data", fontsize=6, color=col, va="center",
                        arrowprops=dict(arrowstyle="-", color="#cccccc", lw=.5) if abs(y - y0) > 0.01 else None)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["d = 32\n(24 nuisance inputs)", "d = 8\n(all active)"], fontsize=7)
        ax.set_xlim(-0.3, 1.9); ax.set_ylim(0, 1.02)
        ax.set_title({"edge": "edge $F_1$", "first": "first level $F_1$", "higher": "higher level $F_1$"}[key], fontsize=8)
    fig.tight_layout()
    return _svg(fig, "settings")


def correspondence(rec, setting):
    """The probe table of one unit: every component against every target node."""
    sc = rec["wirings"]["learned"]
    rows = sc["probe_table"]
    nodes = rec["target_match_nodes"]
    ncol = node_colours(rec)
    order = sorted(range(len(rows)), key=lambda i: (min(rows[i]["stages"]), -rows[i]["size"]))
    M = np.array([rows[i]["excess"] for i in order]) if rows else np.zeros((0, len(nodes)))
    S = np.array([rows[i]["sig"] for i in order]) if rows else np.zeros((0, len(nodes)), bool)
    chosen = {}
    for key, (comp, stages, ex) in rec["graph"]["matched"].items():
        v = int(key)
        chosen[(order.index(comp), nodes.index(v))] = ex
    h = max(2.4, 0.19 * len(order) + 1.2)
    fig, ax = plt.subplots(figsize=(4.6 + 0.35 * len(nodes), h))
    vmax = max(0.3, float(np.nanmax(M))) if M.size else 1
    im = ax.imshow(M, cmap="BuPu", vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if (i, j) in chosen:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="#111", lw=1.6, zorder=5))
            elif S[i, j]:
                ax.scatter([j + .32], [i - .3], s=5, color="#111", zorder=5)
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.4,
                    color="white" if M[i, j] > 0.55 * vmax else "#222")
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels([f"z{v}" for v in nodes], fontsize=7)
    for lab, v in zip(ax.get_xticklabels(), nodes):
        lab.set_color(ncol.get(v, "#111")); lab.set_fontweight("bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"C{rows[i]['comp']}  (stage {','.join(str(s) for s in rows[i]['stages'])}, {rows[i]['size']} units)"
                        for i in order], fontsize=6)
    ax.set_title(f"{SHORT[rec['cell']]}, instance {rec['function_seed']}, {SET_SHORT[setting].split(':')[0]}, n = {rec['n']:,}: "
                 f"held-out excess $R^2$ of every component for every target node\n"
                 f"· = beats all 32 chance sets of the same stage composition; box = the one-to-one assignment",
                 fontsize=7.2)
    fig.colorbar(im, ax=ax, shrink=.7, label="excess $R^2$ (test pool)")
    fig.tight_layout()
    return _svg(fig, f"correspondence_{setting}_{rec['cell']}")


# ------------------------------------------------------------- embeddings --

def embed_static(embed, setting, recs, per_figure=4):
    """Print view of the inspection panels: the target graph, then the first two
    principal axes of the positions under each colouring, one row per family
    at the largest size, in figures of at most ``per_figure`` families."""
    E = embed.get(setting)
    if not E:
        return []
    n = max(E["ns"])
    cols = [a["arm"] for a in E["arms"]]
    return [_embed_rows(E, setting, recs, E["cells"][k:k + per_figure], n, cols, k // per_figure + 1)
            for k in range(0, len(E["cells"]), per_figure)]


def _embed_rows(E, setting, recs, cells, n, cols, part):
    fig, axes = plt.subplots(len(cells), 1 + len(cols), figsize=(2.4 * (1 + len(cols)), 2.35 * len(cells)), squeeze=False)
    for i, cell in enumerate(cells):
        rec = recs.get(cell)
        ax = axes[i][0]
        if rec:
            G = target_and_recovered(rec)
            span = _layout(G["tnodes"], G["tedges"])[2]
            _draw(ax, G["tnodes"], G["tedges"], G["labels"], G["colours"],
                  f"{SHORT[cell]}, instance {rec['function_seed']}: target", span, {})
        else:
            ax.axis("off")
        for j, arm in enumerate(cols, 1):
            ax = axes[i][j]
            p = next((q for q in E["panels"] if q["cell"] == cell and q["n"] == n and q["arm"] == arm), None)
            if p is None:
                ax.axis("off"); continue
            allpts = np.concatenate([np.array(l["pts"]) for l in p["layers"]])
            mean = allpts.mean(0)
            _u, _s, vt = np.linalg.svd(allpts - mean, full_matrices=False)
            P = vt[:2].T
            for l in p["layers"]:
                xy = (np.array(l["pts"]) - mean) @ P
                is_out = l["name"] == "readout"
                ax.scatter(xy[:, 0], xy[:, 1], s=(70 if is_out else 13), marker=l["marker"], c=l["colour"],
                           edgecolors="k" if is_out else "none", linewidths=0.6, alpha=0.95, zorder=4 if is_out else 3)
                if is_out:
                    ax.add_patch(plt.Circle((xy[0, 0], xy[0, 1]), p["radius"], fill=False, ls="--", lw=.8, color="#555"))
            ax.set_aspect("equal"); ax.tick_params(labelsize=5.5)
            title = {"track": "trained: tracked node (η²)", "component": "trained: recovered component",
                     "component_init": "initialization: components"}[arm]
            matched = (f"{p['n_matched']} of {p['n_nodes']} nodes matched" if p.get("n_nodes")
                       else "no intermediate node to match")
            ax.set_title(f"{title}\nn = {p['n']:,}, edge $F_1$ {p['edge_f1']:.2f}, {matched}"
                         if arm != "track" else f"{title}\nn = {p['n']:,}", fontsize=6.4)
    fig.tight_layout()
    return _svg(fig, f"embed_{setting}_{part}")


# ------------------------------------------------------------------ build --

def example_path(units_root, s, ex):
    """The representative record: in the panel, or in results/examples once the
    panel is archived."""
    name = f"{ex['cell']}__fs{ex['fs']}__n{ex['n']}__{ex['which']}.json"
    for root in (Path(units_root), RG / "results" / "examples"):
        if (root / s / name).exists():
            return root / s / name
    return None


def load_examples(read, units_root):
    out = {}
    for s, sr in read["settings"].items():
        out[s] = {}
        for cell, ex in sr["examples"].items():
            p = example_path(units_root, s, ex)
            if p is not None:
                out[s][cell] = json.loads(p.read_text())
    return out


def build(read, units_root, embed=None):
    recs = load_examples(read, units_root)
    out = {"method": method(), "paper": paper(read, recs.get("d32", {})), "controls": controls(read),
           "levels": levels(read), "curves": curves(read), "instances": instances(read),
           "precision_recall": precision_recall(read), "direct": direct(read), "settings": settings_slope(read)}
    for s in SETTINGS:
        if recs.get(s):
            out[f"examples_{s}"] = examples(recs[s], s)
            for cell in ("dag", "tree"):
                if cell in recs[s]:
                    out[f"correspondence_{s}_{cell}"] = correspondence(recs[s][cell], s)
            if embed:
                out[f"embed_{s}"] = embed_static(embed, s, recs[s])
    return out
