"""Figures of the modules study, inline SVG.  The graph drawing and the bar helpers are the
recovered-graph study's (rg_figs), imported; the records of this study are adapted to its layout
by ``view``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
MO = HERE.parent
SD = MO.parent
for p in (str(HERE), str(SD / "recovered-graph" / "src"), str(SD / "component-dag" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
import rg_figs as F     # noqa: E402
import cd_score as cs   # noqa: E402
import mo_score as ms   # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

F.FIG = MO / "figures"
FIG = F.FIG
CELLS, STRUCT, SHORT, STACK, COL, TEACHER = F.CELLS, F.STRUCT, F.SHORT, F.STACK, F.COL, F.TEACHER
LEAF, YCOL = F.LEAF, F.YCOL
UNNAMED, UNUSED, RELAY, HIT, SHORTCUT = "#c9ced4", "#ececec", "#c4c4c4", "#2b2b2b", "#8f8f8f"
COL = {**COL, "mod": "#1f5f8b", "anc": "#7b2d5e", "shuffle_modules": "#8e6c2f", "p05": "#1f5f8b",
       "significant": "#9a9a9a", "positive": "#b56576"}
CTL = ("init", "shuffle_all", "shuffle_higher", "shuffle_modules")
CTL_NAME = {"init": "initialization", "shuffle_all": "wiring rewired", "shuffle_higher": "higher wiring rewired",
            "shuffle_modules": "modules shuffled"}
CTL_MARK = {"shuffle_all": "v", "shuffle_higher": "^", "shuffle_modules": "D"}
STAGE_MARK = ["o", "s", "^", "D", "v", "*"]
GREYS = ("#9aa5b1", "#7f8c99", "#b8c0c8", "#6b7783", "#cfd5db", "#57636e")


def view(rec, key):
    """A record of this study, seen through one key, in the recovered-graph layout."""
    sc = rec["wirings"]["learned"][key]
    return {**rec, "graph": sc.get("graph"), "wirings": {"learned": sc}}


def primary_key(read):
    return "|".join(read["primary"])


# ------------------------------------------------------------ bars etc. --

def _ci_bar(ax, x, c, color, width, off, **kw):
    F._bar(ax, x, c, color, width, off, **kw)


def controls(read, split="confirm"):
    """Edge F1 by family: modules (trained) beside the four controls, at the largest size."""
    S = read["splits"][split]; k = primary_key(read); n = str(read["n_max"])
    cells = [c for c in CELLS if c in S["table"][k]]
    fig, ax = plt.subplots(figsize=(8.6, 3.1))
    x = np.arange(len(cells))
    for i, c in enumerate(cells):
        _ci_bar(ax, x[i], S["table"][k][c][n]["edge"], COL["mod"], 0.3, -0.17)
        _ci_bar(ax, x[i], S["controls"][k][c][n]["init"]["edge"], COL["init"], 0.3, 0.17)
        for ctl in ("shuffle_all", "shuffle_higher", "shuffle_modules"):
            v = S["controls"][k][c][n][ctl]["edge"]
            if v["mean"] is not None:
                ax.plot([x[i] - 0.17], [v["mean"]], marker=CTL_MARK[ctl], color=COL[ctl], ms=6, zorder=4,
                        markeredgecolor="white", markeredgewidth=.5)
    F._xt(ax, cells)
    ax.set_ylim(0, 1.02); ax.set_ylabel("edge $F_1$ against the target graph")
    handles = [Patch(color=COL["mod"], label="modules of the trained geometry"), Patch(color=COL["init"], label="initialization"),
               Line2D([0], [0], marker="v", color="w", markerfacecolor=COL["shuffle_all"], ms=8, label="wiring rewired (degrees kept)"),
               Line2D([0], [0], marker="^", color="w", markerfacecolor=COL["shuffle_higher"], ms=8, label="first transition kept, rest rewired"),
               Line2D([0], [0], marker="D", color="w", markerfacecolor=COL["shuffle_modules"], ms=7, label="modules shuffled (wiring kept)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=6.8, frameon=False)
    fig.tight_layout()
    return F._svg(fig, f"controls_{split}")


def levels(read, split="confirm"):
    """First-level, higher-level and readout F1 by family, with the higher-wiring-rewired control on
    the higher level and the initialization on every level."""
    S = read["splits"][split]; k = primary_key(read); n = str(read["n_max"])
    cells = [c for c in STRUCT if c in S["table"][k]]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
    for ax, lvl, title in zip(axes, ("first", "higher", "readout"), ("first level: input → node", "higher level: node → node", "readout: node → y")):
        x = np.arange(len(cells))
        for i, c in enumerate(cells):
            _ci_bar(ax, x[i], S["table"][k][c][n][lvl], COL["mod"], 0.3, -0.17)
            _ci_bar(ax, x[i], S["controls"][k][c][n]["init"][lvl], COL["init"], 0.3, 0.17)
            v = S["controls"][k][c][n]["shuffle_higher"][lvl]
            if v["mean"] is not None:
                ax.plot([x[i] - 0.17], [v["mean"]], marker="^", color=COL["shuffle_higher"], ms=6, zorder=4, markeredgecolor="white", markeredgewidth=.5)
            v = S["controls"][k][c][n]["shuffle_modules"][lvl]
            if v["mean"] is not None:
                ax.plot([x[i] - 0.17], [v["mean"]], marker="D", color=COL["shuffle_modules"], ms=5, zorder=4, markeredgecolor="white", markeredgewidth=.5)
        F._xt(ax, cells, 6.6); ax.set_title(title, fontsize=8); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("$F_1$")
    handles = [Patch(color=COL["mod"], label="trained"), Patch(color=COL["init"], label="initialization"),
               Line2D([0], [0], marker="^", color="w", markerfacecolor=COL["shuffle_higher"], ms=8, label="first transition kept, rest rewired"),
               Line2D([0], [0], marker="D", color="w", markerfacecolor=COL["shuffle_modules"], ms=7, label="modules shuffled")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    return F._svg(fig, f"levels_{split}")


def compare(read, split="confirm"):
    """Modules against the input-ancestry quotient: edge and higher-level F1 and the module count."""
    S = read["splits"][split]; n = str(read["n_max"])
    keys = [k for k in read["keys"] if k.endswith("|p05")]
    kmod = primary_key(read); kanc = next((k for k in keys if k.startswith("ancestry")), None)
    cells = [c for c in CELLS if c in S["table"][kmod]]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    for ax, lvl, title in zip(axes[:2], ("edge", "higher"), ("all edges", "higher level")):
        x = np.arange(len(cells))
        for i, c in enumerate(cells):
            _ci_bar(ax, x[i], S["table"][kmod][c][n][lvl], COL["mod"], 0.3, -0.17)
            if kanc:
                _ci_bar(ax, x[i], S["table"][kanc][c][n][lvl], COL["anc"], 0.3, 0.17)
        F._xt(ax, cells, 6.6); ax.set_title(f"{title}: $F_1$", fontsize=8); ax.set_ylim(0, 1.02)
    ax = axes[2]
    x = np.arange(len(cells))
    for i, c in enumerate(cells):
        _ci_bar(ax, x[i], S["table"][kmod][c][n]["diagnostics"]["n_components"], COL["mod"], 0.3, -0.17)
        if kanc:
            _ci_bar(ax, x[i], S["table"][kanc][c][n]["diagnostics"]["n_components"], COL["anc"], 0.3, 0.17)
    F._xt(ax, cells, 6.6); ax.set_title("modules per network", fontsize=8)
    handles = [Patch(color=COL["mod"], label="modules: clusters of the geometry at R, majority edges"),
               Patch(color=COL["anc"], label="input-ancestry quotient, any-connection edges (recovered-graph study)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    return F._svg(fig, f"compare_{split}")


def matchers(read, split="confirm"):
    """The three admissibility rules of the correspondence: edge and higher-level F1, trained and initialization."""
    S = read["splits"][split]; n = str(read["n_max"])
    design = read["primary"][0]
    ms = [m for m in ("significant", "p05", "positive") if f"{design}|{m}" in S["table"]]
    cells = [c for c in STRUCT if c in S["table"][f"{design}|{ms[0]}"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.9), sharey=True)
    w = 0.8 / len(ms)
    for ax, lvl in zip(axes, ("edge", "higher")):
        x = np.arange(len(cells))
        for j, m in enumerate(ms):
            k = f"{design}|{m}"
            for i, c in enumerate(cells):
                off = (j - (len(ms) - 1) / 2) * w
                _ci_bar(ax, x[i], S["table"][k][c][n][lvl], COL[m], w * 0.92, off)
                v = S["controls"][k][c][n]["init"][lvl]
                if v["mean"] is not None:
                    ax.plot([x[i] + off], [v["mean"]], marker="_", color="#222", ms=8, mew=1.4, zorder=4)
        F._xt(ax, cells, 6.6); ax.set_ylim(0, 1.02)
        ax.set_title({"edge": "all edges: $F_1$", "higher": "higher level: $F_1$"}[lvl], fontsize=8)
    name = {"significant": "beats all 32 chance draws (registered)", "p05": "p < 0.05 on 200 draws (resolved)", "positive": "any positive excess"}
    handles = [Patch(color=COL[m], label=name[m]) for m in ms] + [Line2D([0], [0], marker="_", color="#222", lw=0, ms=8, mew=1.4, label="initialization, same rule")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    return F._svg(fig, f"matchers_{split}")


def curves(read, split="confirm"):
    """Edge, first-level and higher-level F1 over the training-set size, trained and initialization."""
    S = read["splits"][split]; k = primary_key(read); ns = read["ns"]
    cells = [c for c in STRUCT if c in S["table"][k]]
    fig, axes = plt.subplots(1, len(cells), figsize=(1.55 * len(cells) + 0.6, 2.5), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), cells):
        for lvl, col, ls in (("edge", COL["mod"], "-"), ("first", "#4c9a6a", "--"), ("higher", "#c0392b", ":")):
            ys = [S["table"][k][c][str(n)][lvl] for n in ns]
            m = [y["mean"] if y["mean"] is not None else np.nan for y in ys]
            lo = [y["lo"] if y["lo"] is not None else np.nan for y in ys]
            hi = [y["hi"] if y["hi"] is not None else np.nan for y in ys]
            ax.plot(ns, m, color=col, ls=ls, lw=1.3, marker="o", ms=2.5)
            ax.fill_between(ns, lo, hi, color=col, alpha=0.12, lw=0)
        yi = [S["controls"][k][c][str(n)]["init"]["edge"] for n in ns]
        ax.plot(ns, [y["mean"] if y["mean"] is not None else np.nan for y in yi], color=COL["init"], lw=1.2, marker="o", ms=2.5)
        ax.set_xscale("log", base=2); ax.set_ylim(0, 1.02); ax.set_title(SHORT[c], fontsize=7.6)
        ax.set_xticks([ns[0], ns[len(ns) // 2], ns[-1]]); ax.tick_params(labelsize=6)
    np.atleast_1d(axes)[0].set_ylabel("$F_1$")
    handles = [Line2D([0], [0], color=COL["mod"], lw=1.3, label="all edges (trained)"),
               Line2D([0], [0], color="#4c9a6a", ls="--", lw=1.3, label="first level"),
               Line2D([0], [0], color="#c0392b", ls=":", lw=1.3, label="higher level"),
               Line2D([0], [0], color=COL["init"], lw=1.2, label="all edges, initialization")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.supxlabel("training samples n", fontsize=7.5, y=0.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return F._svg(fig, f"curves_{split}")


def purity(read, split="confirm"):
    """Purity and completeness of the modules against the tracked labels, per hidden layer, by family."""
    S = read["splits"][split]; k = primary_key(read); n = str(read["n_max"])
    cells = [c for c in STRUCT if c in S["purity"][k]]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.7), sharey=True)
    for ax, what in zip(axes, ("purity", "completeness")):
        for i, c in enumerate(cells):
            for st in ("1", "2", "3", "4"):
                v = S["purity"][k][c][n][st][what]
                if v["mean"] is None:
                    continue
                x = i + (int(st) - 2.5) * 0.19
                ax.bar(x, v["mean"], 0.17, color=plt.cm.Blues(0.35 + 0.16 * int(st)), zorder=2)
                if v["n"] > 1:
                    ax.plot([x, x], [v["lo"], v["hi"]], color="#333", lw=.8, zorder=3)
        F._xt(ax, cells, 6.6); ax.set_ylim(0, 1.02)
        ax.set_title({"purity": "purity: units of a module that track its majority node",
                      "completeness": "completeness: units of a node inside its largest module"}[what], fontsize=7.6)
    handles = [Patch(color=plt.cm.Blues(0.35 + 0.16 * s), label=f"hidden layer {s}") for s in range(1, 5)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    return F._svg(fig, f"purity_{split}")


def instances(read, split="confirm"):
    """Every instance at the largest size: edge F1 trained against the initialization, per family."""
    S = read["splits"][split]; k = primary_key(read)
    P = S["per_instance"][k]
    cells = [c for c in CELLS if c in P]
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    rng = np.random.default_rng(0)
    for i, c in enumerate(cells):
        for j, (key, col) in enumerate((("edge", COL["mod"]), ("init", COL["init"]), ("shuffle_modules", COL["shuffle_modules"]))):
            ys = [v for v in P[c][key] if v is not None]
            xs = i + (j - 1) * 0.25 + rng.uniform(-0.07, 0.07, len(ys))
            ax.scatter(xs, ys, s=9, color=col, alpha=0.75, zorder=3)
    F._xt(ax, cells); ax.set_ylim(-0.02, 1.04); ax.set_ylabel("edge $F_1$ per instance")
    handles = [Patch(color=COL["mod"], label="trained"), Patch(color=COL["init"], label="initialization"),
               Patch(color=COL["shuffle_modules"], label="modules shuffled")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=6.8, frameon=False)
    fig.tight_layout()
    return F._svg(fig, f"instances_{split}")



# ------------------------------------------------------------------ graphs --

def node_colours(rec):
    """One colour per target node, the root included; y keeps its own tint."""
    nodes = sorted(int(k) for k in rec["target_parents"])
    return {v: TEACHER[k % len(TEACHER)] for k, v in enumerate(nodes)}


def _crossings(pos, edges):
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    segs = [(pos[a], pos[b]) for a, b in edges if a in pos and b in pos]
    n = 0
    for i, (p1, p2) in enumerate(segs):
        for q1, q2 in segs[i + 1:]:
            if p1 in (q1, q2) or p2 in (q1, q2):
                continue
            if side(p1, p2, q1) * side(p1, p2, q2) < 0 and side(q1, q2, p1) * side(q1, q2, p2) < 0:
                n += 1
    return n


def layout(level, edges, seed):
    """Slots of a layered graph with few crossings.  ``level`` maps a node to its column and ``seed``
    to its starting rank in the column (top first); barycenter sweeps (down: the mean slot of a node's
    senders; up: of its readers) refine the ranks, and the order with the fewest edge crossings is kept."""
    cols = {}
    for n in sorted(level, key=lambda n: (seed.get(n, 1e9), str(n))):
        cols.setdefault(level[n], []).append(n)
    levels = sorted(cols)

    def positions(order):
        return {n: (l, (len(ns) - 1) / 2 - i) for l, ns in order.items() for i, n in enumerate(ns)}
    senders, readers = {}, {}
    for a, b in edges:
        readers.setdefault(a, []).append(b); senders.setdefault(b, []).append(a)
    order = {l: list(ns) for l, ns in cols.items()}
    best = positions(order); best_x = _crossings(best, edges)
    for sweep in range(8):
        down = sweep % 2 == 0
        pos = positions(order)
        for l in (levels[1:] if down else levels[-2::-1]):
            nb = senders if down else readers

            def rank(n):
                s = [pos[m][1] for m in nb.get(n, ()) if m in pos]
                return -float(np.mean(s)) if s else -pos[n][1]
            order[l] = sorted(order[l], key=rank)
            pos = positions(order)
        x = _crossings(pos, edges)
        if x < best_x:
            best, best_x = pos, x
    return best, max(len(ns) for ns in cols.values())


def _dfs_seed(t):
    """First-visit ranks of a depth-first walk from y through the parents, in their stored order:
    planar for a tree, a good start for a DAG."""
    par = {("node", v): [cs._lift(t, p) for p in ps] for v, ps in t["parents"].items()}
    par[("y", 0)] = [("node", v) for v in t["readout"]]
    rank = {}

    def visit(n):
        if n in rank:
            return
        rank[n] = len(rank)
        for p in par.get(n, ()):
            visit(p)
    visit(("y", 0))
    return rank


def graphs(rec, key):
    """The target graph and the module graph of one record, laid out, with every edge classified.
    Target edges: recovered by the module graph (reachability between named modules, reduced) or not.
    Module edges: a target edge between two named modules; a shortcut (ancestor, not an edge); no target
    relation; or through an unnamed module or an input the target does not use."""
    t = ms.target({"A": rec["A"], "node_parents": rec["raw_node_parents"], "readout_indices": rec["raw_readout_indices"]})
    raw = rec["active_raw"]; ncol = node_colours(rec)
    Et, At = cs.target_edges(t), cs.target_ancestors(t)
    level = {("leaf", a): 0 for a in range(rec["A"])}
    level.update({("node", v): t["depth"][v] for v in t["parents"]})
    level[("y", 0)] = max(level.values()) + 1
    tpos, tspan = layout(level, Et, _dfs_seed(t))
    lab, col = {("y", 0): "y"}, {("y", 0): YCOL}
    for a in range(rec["A"]):
        lab[("leaf", a)], col[("leaf", a)] = f"x{raw[a]}", LEAF
    for v in t["parents"]:
        lab[("node", v)], col[("node", v)] = f"z{v}", ncol[v]
    g = rec["wirings"]["learned"][key]["graph"]
    node_of = {}
    for k, v in g["map"].items():
        a, b = k.split(":")
        node_of[(a, int(b))] = tuple(v)
    named = {m: n for m, n in node_of.items() if n[0] != "extra"}
    E = {(tuple(a), tuple(b)) for a, b in g["edges"]}
    Ehat = cs.transitive_reduction({(named[a], named[b]) for a, b in cs.reachability(E, set(named))})
    hits = Ehat & Et
    L = len(rec["positions"]) - 1
    mlevel = {("in", i): 0 for i in g["inputs"]}
    for k, comp in enumerate(g["components"]):
        mlevel[("c", k)] = comp[0][0]
    mlevel[("out", 0)] = L
    seed = {m: -tpos[n][1] for m, n in node_of.items() if n in tpos}
    mpos, mspan = layout(mlevel, E, seed)
    mlab, mcol, small = {}, {}, set()
    for m in mlevel:
        n = node_of.get(m)
        if m[0] == "in" and (n is None or n[0] == "extra"):
            mlab[m], mcol[m] = "", UNUSED; small.add(m)
        elif n is None:
            mlab[m], mcol[m] = f"m{m[1]}", UNNAMED
        else:
            mlab[m], mcol[m] = lab[n], col[n]
    if small:                      # the unused live inputs: small dots at half spacing below the leaves
        used = [mpos[m][1] for m in mlevel if mlevel[m] == 0 and m not in small]
        base = min(used) if used else 0.5
        for j, m in enumerate(sorted(small, key=lambda m: -mpos[m][1])):
            mpos[m] = (0, base - 0.5 * (j + 1))
        ys = [mpos[m][1] for m in mlevel if mlevel[m] == 0]
        shift = (max(ys) + min(ys)) / 2
        for m in mlevel:
            if mlevel[m] == 0:
                mpos[m] = (0, mpos[m][1] - shift)
        mspan = max(max(ys) - min(ys) + 1, max(len([m for m in mlevel if mlevel[m] == l]) for l in range(1, L + 1)))
    tstyle = {e: (HIT, "-") if e in hits else (COL["miss"], "-") for e in Et}
    mstyle = {}
    for a, b in E:
        na, nb = named.get(a), named.get(b)
        if na is None or nb is None:
            mstyle[(a, b)] = (RELAY, "-")
        elif (na, nb) in Et:
            mstyle[(a, b)] = (HIT, "-")
        elif (na, nb) in At:
            mstyle[(a, b)] = (SHORTCUT, ":")
        else:
            mstyle[(a, b)] = (COL["extra"], "--")
    missing = [n for n in level if n[0] == "node" and n not in named.values()]
    return {"t": t, "tpos": tpos, "tspan": tspan, "tnodes": list(level), "tedges": sorted(Et), "tstyle": tstyle,
            "lab": lab, "col": col, "hits": hits, "missing": missing, "mpos": mpos, "mspan": mspan,
            "mnodes": list(mlevel), "medges": sorted(E), "mstyle": mstyle, "mlab": mlab, "mcol": mcol, "small": small}


def _draw(ax, pos, nodes, edges, style, lab, col, title, span, hollow=(), small=(), title_size=6.8):
    for e in edges:
        a, b = e
        if a not in pos or b not in pos:
            continue
        c, ls = style.get(e, (HIT, "-"))
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>", mutation_scale=4.5, color=c,
                                     lw=.6 if ls == "-" else .75, linestyle=ls, shrinkA=5.5, shrinkB=5.5, zorder=1))
    for n in nodes:
        x, y = pos[n]
        if n in small:
            ax.scatter([x], [y], s=14, facecolor=col[n], edgecolor="#aaaaaa", linewidths=.4, zorder=2)
            continue
        if n in hollow:
            ax.scatter([x], [y], s=95, facecolor="white", edgecolor=col[n], linewidths=1.0, linestyle="--", zorder=2)
        else:
            ax.scatter([x], [y], s=95, facecolor=col[n], edgecolor="#555555", linewidths=.5, zorder=2)
        dark = col[n] in TEACHER and n not in hollow
        ax.text(x, y, lab[n], ha="center", va="center", fontsize=5.0, color="white" if dark else "#222222", zorder=3)
    ax.set_title(title, fontsize=title_size); ax.axis("off")
    D = max(l for l, _ in pos.values())
    ax.set_xlim(-0.5, D + 0.5); ax.set_ylim(-span / 2 - 0.8, span / 2 + 0.8)


def graph_pair(axes, rec, key):
    """Left: the target graph (red = edge not recovered, hollow = node with no named module).  Right: the
    module graph, every module of every layer (colour = its named node, grey = unnamed) and every majority
    edge, the score in the title."""
    G = graphs(rec, key); sc = rec["wirings"]["learned"][key]
    span = max(G["tspan"], G["mspan"])
    t1 = f"target: {SHORT[rec['cell']]} ({rec['function_seed']}): {rec['n_nodes']} nodes, {len(G['tedges'])} edges"
    t2 = (f"module graph: $F_1$ {sc['edge']['f1']:.2f}, {len(G['hits'])}/{len(G['tedges'])} edges, "
          f"{sc['n_matched']}/{rec['n_nodes']} named")
    _draw(axes[0], G["tpos"], G["tnodes"], G["tedges"], G["tstyle"], G["lab"], G["col"], t1, span, hollow=set(G["missing"]))
    _draw(axes[1], G["mpos"], G["mnodes"], G["medges"], G["mstyle"], G["mlab"], G["mcol"], t2, span, small=G["small"])


def graph_legend(fig, y=0.0):
    handles = [Line2D([0], [0], color=HIT, lw=1.1, label="module edge = target edge (both ends named)"),
               Line2D([0], [0], color=SHORTCUT, lw=1.1, ls=":", label="shortcut: ancestor to descendant"),
               Line2D([0], [0], color=COL["extra"], lw=1.1, ls="--", label="no target relation"),
               Line2D([0], [0], color=RELAY, lw=1.1, label="through an unnamed module (grey) or an unused input (dot)"),
               Line2D([0], [0], color=COL["miss"], lw=1.1, label="target edge not recovered (left)"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#555", ms=6.5,
                      label="target node with no named module (hollow)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, y))

# ------------------------------------------------------- modules pictures --

def _project(rec):
    E = [np.array(e) for e in rec["positions"]]
    live = set()
    g = rec["wirings"]["learned"][primary_key_of(rec)]["graph"]
    for comp in g["components"]:
        for s, u in comp:
            live.add((s, u))
    pts = np.concatenate([E[0][g["inputs"]]] + [E[s][[u for (ss, u) in live if ss == s]] for s in range(1, len(E) - 1) if any(ss == s for ss, _ in live)] + [E[-1]])
    mu = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
    return E, mu, vt[:2].T, live, g


def primary_key_of(rec):
    return "|".join(rec["primary"])


def module_colours(rec, key):
    """Colour of every module: its matched node's colour, or a grey for an unmatched module."""
    g = rec["wirings"]["learned"][key]["graph"]
    ncol = node_colours(rec)
    matched = {}
    for k, (comp, stages, ex) in g["matched"].items():
        v = int(k)
        matched[comp] = v
    cols, labels = {}, {}
    grey = 0
    for k in range(len(g["components"])):
        if k in matched:
            cols[k] = ncol[matched[k]]; labels[k] = f"z{matched[k]}"
        else:
            cols[k] = GREYS[grey % len(GREYS)]; labels[k] = f"m{k}"; grey += 1
    return cols, labels, matched


def modules_picture(rec, key, axes, title_size=7.2):
    """One row: the layers of one network on the two leading axes of the trained positions, units
    coloured by module (matched node colour, greys for unmatched modules), the radius R for scale."""
    E = [np.array(e) for e in rec["positions"]]
    g = rec["wirings"]["learned"][key]["graph"]
    owner = {}
    for k, comp in enumerate(g["components"]):
        for s, u in comp:
            owner[(s, u)] = k
    cols, labels, matched = module_colours(rec, key)
    raw = rec["active_raw"]
    reader = {}
    for v, refs in enumerate(rec["raw_node_parents"]):
        for r in refs:
            if r < rec["A"]:
                reader.setdefault(int(raw[r]), v)
    ncol = node_colours(rec)
    pts = np.concatenate([E[0][g["inputs"]]] + [E[s][[u for (ss, u) in owner if ss == s]] for s in range(1, len(E) - 1)] + [E[-1]])
    mu = pts.mean(0); _, _, vt = np.linalg.svd(pts - mu, full_matrices=False); B = vt[:2].T
    lim = np.abs((pts - mu) @ B).max() * 1.15 + 0.05
    R = float(np.sqrt(rec["radius2"]))
    for s, ax in enumerate(axes):
        P = (E[s] - mu) @ B
        for u in range(len(P)):
            if s == 0:
                if u in reader:
                    ax.scatter(*P[u], s=26, c=ncol[reader[u]], marker="o", edgecolors="k", linewidths=.4, zorder=3)
                elif u in g["inputs"]:
                    ax.scatter(*P[u], s=22, c="#9aa5b1", marker="o", edgecolors="k", linewidths=.4, zorder=3)
                else:
                    ax.scatter(*P[u], s=7, c="#dddddd", marker="x", linewidths=.5, zorder=2)
            elif s == len(E) - 1:
                ax.scatter(*P[u], s=80, c="#555555", marker="*", zorder=3)
            else:
                k = owner.get((s, u))
                if k is None:
                    ax.scatter(*P[u], s=7, c="#dddddd", marker="x", linewidths=.5, zorder=2)
                else:
                    ax.scatter(*P[u], s=26, c=cols[k], marker="s", edgecolors="k", linewidths=.4, zorder=3)
        # module hulls: a light circle around every module of this layer
        for k, comp in enumerate(g["components"]):
            if comp[0][0] != s:
                continue
            Q = (E[s][[u for _, u in comp]] - mu) @ B
            c = Q.mean(0); r = max(np.sqrt(((Q - c) ** 2).sum(1)).max() * 1.15, 0.08)
            ax.add_patch(plt.Circle(c, r, fill=False, lw=.7, color=cols[k], alpha=.9, zorder=1))
            ax.text(c[0], c[1] + r + 0.03, labels[k], fontsize=5.2, ha="center", va="bottom", color=cols[k] if k in matched else "#666")
        ax.add_patch(plt.Circle((lim * 0.66, -lim * 0.66), R, fill=False, ls="--", lw=.7, color="#888"))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        name = "inputs" if s == 0 else ("readout" if s == len(E) - 1 else f"hidden {s}")
        ax.set_title(name, fontsize=title_size)


def module_examples(recs, key, per_figure=3):
    """Per family: the target, the layers with their modules, the recovered graph."""
    cells = [c for c in CELLS if c in recs]
    out = []
    for k0 in range(0, len(cells), per_figure):
        chunk = cells[k0:k0 + per_figure]
        n_lv = len(recs[chunk[0]]["positions"])
        fig = plt.figure(figsize=(11.6, 2.3 * len(chunk) + 0.5))
        gs = fig.add_gridspec(len(chunk), n_lv + 2, width_ratios=[1.25] + [1] * n_lv + [1.75], wspace=0.08, hspace=0.3,
                              top=0.93, bottom=0.12 if len(chunk) == 1 else 0.08)
        for i, c in enumerate(chunk):
            rec = recs[c]
            axl = fig.add_subplot(gs[i, 0]); axr = fig.add_subplot(gs[i, n_lv + 1])
            graph_pair([axl, axr], rec, key)
            axes = [fig.add_subplot(gs[i, 1 + s]) for s in range(n_lv)]
            modules_picture(rec, key, axes)
        graph_legend(fig, y=0.0)
        out.append(F._svg(fig, f"modules_{k0 // per_figure + 1}"))
    return out


def paper(read, recs):
    """The recommended paper figure: for a tree and a held-out random graph, the target graph, the
    trained positions with their modules (hidden layers 1 to 4) and the recovered graph; below, the
    edge F1 by family against the controls."""
    key = primary_key(read)
    fig = plt.figure(figsize=(10.8, 7.6))
    n_lv = 4
    gs = fig.add_gridspec(3, n_lv + 2, height_ratios=[1, 1, 1.15], width_ratios=[1.25] + [1] * n_lv + [1.7], hspace=0.55, wspace=0.08)
    for i, c in enumerate(("tree", "rand")):
        if c not in recs:
            continue
        rec = recs[c]
        axl = fig.add_subplot(gs[i, 0]); axr = fig.add_subplot(gs[i, n_lv + 1])
        graph_pair([axl, axr], rec, key)
        axes = [fig.add_subplot(gs[i, 1 + s]) for s in range(n_lv)]
        E = rec["positions"]
        # hidden layers only in the paper row
        allax = [None] + axes + [None]
        tmp_axes = [plt.Axes(fig, [0, 0, 0.01, 0.01]) if a is None else a for a in allax]
        modules_picture(rec, key, tmp_axes, title_size=6.8)
    ax = fig.add_subplot(gs[2, :])
    S = read["splits"]["confirm"]; n = str(read["n_max"])
    cells = [c for c in CELLS if c in S["table"][key]]
    x = np.arange(len(cells))
    for i, c in enumerate(cells):
        _ci_bar(ax, x[i], S["table"][key][c][n]["edge"], COL["mod"], 0.3, -0.17)
        _ci_bar(ax, x[i], S["controls"][key][c][n]["init"]["edge"], COL["init"], 0.3, 0.17)
        for ctl in ("shuffle_all", "shuffle_higher", "shuffle_modules"):
            vv = S["controls"][key][c][n][ctl]["edge"]
            if vv["mean"] is not None:
                ax.plot([x[i] - 0.17], [vv["mean"]], marker=CTL_MARK[ctl], color=COL[ctl], ms=6, zorder=4, markeredgecolor="white", markeredgewidth=.5)
    F._xt(ax, cells, 6.8); ax.set_ylim(0, 1.02); ax.set_ylabel("edge $F_1$ against the target graph")
    ax.set_title(f"held-out instances (odd seeds), n = {read['n_max']:,}", fontsize=8)
    handles = [Patch(color=COL["mod"], label="modules of the trained geometry"), Patch(color=COL["init"], label="initialization"),
               Line2D([0], [0], marker="v", color="w", markerfacecolor=COL["shuffle_all"], ms=8, label="wiring rewired"),
               Line2D([0], [0], marker="^", color="w", markerfacecolor=COL["shuffle_higher"], ms=8, label="first transition kept, rest rewired"),
               Line2D([0], [0], marker="D", color="w", markerfacecolor=COL["shuffle_modules"], ms=7, label="modules shuffled")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.6, frameon=False, bbox_to_anchor=(0.5, -0.01))
    graph_legend(fig, y=0.355)
    return F._svg(fig, "paper")


# ------------------------------------------------------------ exploration --

def geometry(x1_dir):
    """Exploration 1: silhouette of the tracked labels in position space per hidden layer (trained
    and initialization, with the permutation null), and the share of same-layer pairs within R."""
    recs = [json.loads(p.read_text()) for p in sorted(Path(x1_dir).glob("*.json"))]
    cells = [c for c in STRUCT if any(r["cell"] == c for r in recs)]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.8))
    for ax, what in zip(axes, ("silhouette", "pair_within_R")):
        for i, c in enumerate(cells):
            for st in range(1, 5):
                for which, col, off in (("best", COL["mod"], -0.09), ("init", COL["init"], 0.09)):
                    vals = [l.get(what) for r in recs if r["cell"] == c and r["which"] == which for l in r["layers"] if l["stage"] == st and l.get(what) is not None]
                    if not vals:
                        continue
                    x = i + (st - 2.5) * 0.2 + off
                    ax.bar(x, np.mean(vals), 0.08, color=col, alpha=0.4 + 0.15 * st, zorder=2)
                    if what == "silhouette" and which == "best":
                        nulls = [l.get("silhouette_null_mean") for r in recs if r["cell"] == c and r["which"] == which for l in r["layers"] if l["stage"] == st and l.get("silhouette_null_mean") is not None]
                        ax.plot([x], [np.mean(nulls)], marker="_", color="#222", ms=6, mew=1.2, zorder=4)
        F._xt(ax, cells, 6.6)
        ax.set_title({"silhouette": "silhouette of the tracked labels in position space (bar per layer 1-4; tick = permutation null)",
                      "pair_within_R": "share of same-layer unit pairs closer than R"}[what], fontsize=7)
        ax.axhline(0, color="#888", lw=.6)
    handles = [Patch(color=COL["mod"], label="trained (layers 1 to 4, light to dark)"), Patch(color=COL["init"], label="initialization")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    return F._svg(fig, "geometry")


def tracked_pictures(recs, cells=("tree", "dag", "rand")):
    """Exploration 2 from the panel records: the layers of one trained network per family, units
    coloured by the target node they track (the paper's colouring)."""
    cells = [c for c in cells if c in recs]
    n_lv = len(recs[cells[0]]["positions"])
    fig, axes = plt.subplots(len(cells), n_lv, figsize=(1.75 * n_lv, 1.9 * len(cells)), squeeze=False)
    for i, c in enumerate(cells):
        rec = recs[c]
        E = [np.array(e) for e in rec["positions"]]
        tr = {int(s): l for s, l in rec["tracked"].items()}
        ncol = node_colours(rec)
        raw = rec["active_raw"]
        reader = {}
        for v, refs in enumerate(rec["raw_node_parents"]):
            for r in refs:
                if r < rec["A"]:
                    reader.setdefault(int(raw[r]), v)
        key = primary_key_of(rec)
        g = rec["wirings"]["learned"][key]["graph"]
        live = {(s, u) for comp in g["components"] for s, u in comp}
        pts = np.concatenate([E[0][g["inputs"]]] + [E[s][[u for (ss, u) in live if ss == s]] for s in range(1, n_lv - 1)] + [E[-1]])
        mu = pts.mean(0); _, _, vt = np.linalg.svd(pts - mu, full_matrices=False); B = vt[:2].T
        lim = np.abs((pts - mu) @ B).max() * 1.15 + 0.05
        R = float(np.sqrt(rec["radius2"]))
        for s in range(n_lv):
            ax = axes[i][s]
            P = (E[s] - mu) @ B
            for u in range(len(P)):
                if s == 0:
                    if u in reader:
                        ax.scatter(*P[u], s=22, c=ncol[reader[u]], marker="o", edgecolors="k", linewidths=.4, zorder=3)
                    elif u in g["inputs"]:
                        ax.scatter(*P[u], s=18, c="#9aa5b1", marker="o", edgecolors="k", linewidths=.4, zorder=3)
                    else:
                        ax.scatter(*P[u], s=6, c="#dddddd", marker="x", linewidths=.5)
                elif s == n_lv - 1:
                    ax.scatter(*P[u], s=70, c="#555555", marker="*", zorder=3)
                elif (s, u) in live:
                    v = tr[s][u]
                    ax.scatter(*P[u], s=22, c=ncol[v] if v >= 0 else "#c4c4c4", marker="s", edgecolors="k", linewidths=.4, zorder=3)
                    if v >= 0:
                        ax.annotate(str(v), P[u], fontsize=4.2, ha="center", va="center", color="white", zorder=4)
                else:
                    ax.scatter(*P[u], s=6, c="#dddddd", marker="x", linewidths=.5)
            ax.add_patch(plt.Circle((lim * 0.66, -lim * 0.66), R, fill=False, ls="--", lw=.7, color="#888"))
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            name = "inputs" if s == 0 else ("readout" if s == n_lv - 1 else f"hidden {s}")
            ax.set_title(f"{SHORT[c]} ({rec['function_seed']}): {name}" if s == 0 else name, fontsize=6.6)
    fig.tight_layout()
    return F._svg(fig, "tracked")


def load_examples(read, units_root):
    out = {}
    for cell, ex in read["examples"].items():
        name = f"{cell}__fs{ex['fs']}__n{read['n_max']}__best.json"
        for root in (Path(units_root), MO / "results" / "examples"):
            if (root / name).exists():
                out[cell] = json.loads((root / name).read_text())
                break
    return out


def build(read, units_root, x1_dir=None):
    recs = load_examples(read, units_root)
    key = primary_key(read)
    out = {"controls": controls(read), "levels": levels(read), "compare": compare(read), "matchers": matchers(read),
           "curves": curves(read), "purity": purity(read), "instances": instances(read),
           "controls_dev": controls(read, "develop"), "levels_dev": levels(read, "develop")}
    if recs:
        out["modules"] = module_examples(recs, key)
        out["paper"] = paper(read, recs)
        out["tracked"] = tracked_pictures(recs)
    if x1_dir and Path(x1_dir).exists():
        out["geometry"] = geometry(x1_dir)
    return out
