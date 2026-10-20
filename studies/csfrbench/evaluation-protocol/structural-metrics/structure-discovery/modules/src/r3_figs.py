"""Round-3 figures (inline SVG): the checkpoint verification, the module sets, the four
correspondences with their multiplicity-matched null, persistence, superposition, the node statuses
and the missing-edge reasons, the edge rules, representative full module graphs, the sample-size
robustness and the confirmation cohort beside the development panel."""
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
sys.path.insert(0, str(HERE))
import mo_figs as MF                     # noqa: E402
import rg_figs as F                      # noqa: E402

CELLS, STRUCT, SHORT, COL = MF.CELLS, MF.STRUCT, MF.SHORT, MF.COL
SETCOL = {"avg@1": "#1f5f8b", "avg@0.75": "#6aa6d1", "avg@1.25": "#0b3a57", "complete@1": "#2a9d8f", "cc@1": "#8e6c2f",
          "xavg@0.75": "#b5179e", "xavg@1": "#7209b7"}
SETLAB = {"avg@1": "average linkage at R (reference)", "avg@0.75": "average at 0.75 R", "avg@1.25": "average at 1.25 R",
          "complete@1": "complete linkage at R", "cc@1": "single linkage at R", "xavg@0.75": "cross-layer average at 0.75 R",
          "xavg@1": "cross-layer average at R"}
NAMCOL = {"one_to_one": "#1f5f8b", "copies": "#e76f51", "one_to_many": "#e9c46a", "many_to_one": "#2a9d8f", "many_to_many": "#7b2d5e"}
NAMLAB = {"one_to_one": "one-to-one (reference)", "copies": "one-to-one plus copies (persistence)", "one_to_many": "one-to-many, module-centric",
          "many_to_one": "many-to-one (superposition)", "many_to_many": "many-to-many"}
TICK = {"tree": "tree", "ftree": "Fourier\ntree", "dag": "reuse\nDAG", "fdag": "Fourier\nreuse DAG", "flat": "flat\nmodules", "rand": "random\ngraphs", "dense": "dense\nnode"}


def _xt(ax, cells, size=6.2):
    ax.set_xticks(range(len(cells))); ax.set_xticklabels([TICK[c] for c in cells], fontsize=size)
    ax.tick_params(axis="y", labelsize=6.5)
RULECOL = {"coverage@0.5": "#1f5f8b", "coverage@0.25": "#6aa6d1", "coverage@0": "#9a9a9a", "weight_share@0.25": "#e9c46a", "contrib_share@0.25": "#e76f51"}
RULELAB = {"coverage@0.5": "majority coverage (reference)", "coverage@0.25": "coverage > 0.25", "coverage@0": "any open connection",
           "weight_share@0.25": "weight share > 0.25", "contrib_share@0.25": "contribution share > 0.25"}
MARK = ["o", "s", "D", "^", "v", "P", "X"]
STATUS_COL = {"named": "#1f5f8b", "whole layer": "#e9c46a", "module taken": "#e76f51", "below chance": "#9a9a9a", "not decodable": "#dddddd"}
REASON_COL = {"child whole layer": "#e9c46a", "parent whole layer": "#f4e3a1", "child not decodable": "#dddddd", "parent not decodable": "#eeeeee",
              "child module taken": "#e76f51", "parent module taken": "#f3a683", "child below chance": "#9a9a9a", "parent below chance": "#c0c0c0",
              "no connection": "#7b2d5e", "threshold": "#b5179e", "reduced": "#2a9d8f", "input dead": "#333333"}


def _series(ax, cells, series, ylabel, title, ylim=(0, 1.02), width=0.82, ms=4.2):
    """series: (label, colour, marker, {cell: ci}, filled)."""
    n = len(series)
    offs = np.linspace(-width / 2 + width / (2 * n), width / 2 - width / (2 * n), n) if n > 1 else [0.0]
    for k, (label, color, marker, vals, filled) in enumerate(series):
        for i, c in enumerate(cells):
            v = vals.get(c) if vals else None
            if not v or v.get("mean") is None:
                continue
            lo = v["mean"] - v["lo"] if v.get("lo") is not None else 0; hi = v["hi"] - v["mean"] if v.get("hi") is not None else 0
            ax.errorbar(i + offs[k], v["mean"], yerr=[[max(lo, 0)], [max(hi, 0)]], fmt=marker, color=color, mfc=color if filled else "white",
                        ms=ms, capsize=0, lw=.8, zorder=3, markeredgewidth=.8)
    _xt(ax, cells)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=7); ax.set_title(title, fontsize=8); ax.grid(axis="y", lw=.3, alpha=.4)


def _legend(fig, series, y=-0.02, ncol=4):
    handles = [Line2D([0], [0], marker=m, color=c, mfc=c if f else "white", ms=6, lw=0, markeredgewidth=.9, label=l) for l, c, m, _, f in series]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=6.6, frameon=False, bbox_to_anchor=(0.5, y))


def T(read, mset, rule, naming, wiring="learned"):
    return read["tables"][mset][f"{rule}|{naming}"][wiring]


# --------------------------------------------------------- verification --

def verify(ver):
    recs = [r for r in ver["records"] if "error" not in r]
    best = [r for r in recs if r["which"] == "best"]; init = [r for r in recs if r["which"] == "init"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.6))
    ax = axes[0]
    bins = np.arange(0, 34, 2)
    ax.hist([r["readout_open_edges"] for r in best], bins=bins, color=COL["mod"], alpha=.85, label="trained")
    ax.hist([r["readout_open_edges"] for r in init], bins=bins, color=COL["init"], alpha=.6, label="initialization")
    ax.set_xlabel("open readout edges (of 32 units)", fontsize=7); ax.set_ylabel("states", fontsize=7); ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("the readout mask", fontsize=8)
    ax = axes[1]
    ax.hist([r["readout_plain_gap"] for r in best], bins=30, color=COL["mod"], alpha=.85)
    ax.set_xlabel("output change without the readout gate and mask\n(max over 512 inputs, in output s.d.)", fontsize=7); ax.set_title("the readout gate and mask act", fontsize=8)
    ax = axes[2]
    ax.hist([r["readout_mask_gap"] for r in best], bins=30, color=COL["mod"], alpha=.85)
    ax.set_xlabel("output change without the readout mask alone", fontsize=7); ax.set_title("the mask beyond the gate", fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    return F._svg(fig, "r3_verify")


# --------------------------------------------------------- constructions --

def constructions(read, rule="coverage@0.5", naming="one_to_one"):
    sets = [s for s in read["module_sets"] if s != "xavg@1"]
    cells = list(CELLS)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))
    ser = [(SETLAB[s], SETCOL[s], MARK[k], {c: T(read, s, rule, naming)[c]["edge"] for c in cells}, True) for k, s in enumerate(sets)]
    _series(axes[0], cells, ser, "edge $F_1$", "edge $F_1$ by module set (trained)")
    ser2 = [(SETLAB[s], SETCOL[s], MARK[k], {c: T(read, s, rule, naming)[c]["node_coverage"] for c in cells}, True) for k, s in enumerate(sets)]
    _series(axes[1], cells, ser2, "named nodes / target nodes", "node coverage (one-to-one)")
    ser3 = [(SETLAB[s], SETCOL[s], MARK[k], {c: read["sizes"][s][c]["n_modules"] for c in cells}, True) for k, s in enumerate(sets)]
    _series(axes[2], cells, ser3, "modules per network", "size of the module graph", ylim=None)
    _legend(fig, ser, y=-0.06, ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return F._svg(fig, "r3_constructions")


def controls_table_series(read, mset, rule, naming):
    return {w: T(read, mset, rule, naming, w) for w in ("learned", "init", "shuffle_all", "shuffle_higher", "shuffle_modules", "random_names")}


# -------------------------------------------------------------- namings --

def namings(read, mset="avg@1", rule="coverage@0.5"):
    cells = list(CELLS); nm = read["namings"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))
    for ax, lvl, title in zip(axes, ("edge", "higher", "ancestor"), ("edge $F_1$", "higher level $F_1$ (from a node)", "ancestor $F_1$")):
        ser = []
        for k, n in enumerate(nm):
            ser.append((NAMLAB[n], NAMCOL[n], MARK[k], {c: T(read, mset, rule, n)[c][lvl] for c in cells}, True))
        for k, n in enumerate(nm):
            rn = T(read, mset, rule, n, "random_names")
            ser.append((NAMLAB[n] + ": names dealt at random", NAMCOL[n], MARK[k], {c: rn[c].get(lvl) for c in cells}, False))
        ser.append(("initialization (one-to-one)", COL["init"], "_", {c: T(read, mset, rule, "one_to_one", "init")[c][lvl] for c in cells}, True))
        _series(ax, cells, ser, "$F_1$", title, width=0.86, ms=3.6)
    handles = [Line2D([0], [0], marker=MARK[k], color=NAMCOL[n], ms=6, lw=0, label=NAMLAB[n]) for k, n in enumerate(nm)]
    handles += [Line2D([0], [0], marker="o", color="#555", mfc="white", ms=6, lw=0, label="hollow: the same names dealt at random among the modules of the same layers (16 draws)"),
                Line2D([0], [0], marker="_", color=COL["init"], ms=8, lw=0, label="initialization")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return F._svg(fig, "r3_namings")


# -------------------------------------------------- persistence, superposition --

def persistence(read, mset="avg@1", naming="copies"):
    cells = list(CELLS); P = read["persistence"][mset][naming]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.8))
    ser = [("first-level nodes", MF.COL["mod"], "o", {c: P[c]["by_class"]["first"] for c in cells}, True),
           ("higher nodes", "#e76f51", "s", {c: P[c]["by_class"]["higher"] for c in cells}, True),
           ("root", "#e9c46a", "D", {c: P[c]["by_class"]["root"] for c in cells}, True)]
    _series(axes[0], cells, ser, "modules named after the node", "modules per node (one-to-one plus copies)", ylim=(0, None))
    ser2 = [("two or more modules", MF.COL["mod"], "o", {c: P[c]["multi"] for c in cells}, True),
            ("modules in two or more layers", "#e76f51", "s", {c: P[c]["multi_stage"] for c in cells}, True),
            ("two or more modules at initialization", MF.COL["mod"], "o", {c: P[c].get("init_multi") for c in cells}, False)]
    _series(axes[1], cells, ser2, "share of target nodes", "persistent nodes (hollow: initialization)")
    ser3 = [("joined by a majority edge", MF.COL["mod"], "o", {c: P[c]["joined"] for c in cells}, True)]
    _series(axes[2], cells, ser3, "share of persistent nodes", "the copies are wired")
    _legend(fig, ser + ser2, y=-0.04, ncol=5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return F._svg(fig, "r3_persistence")


def superposition(read, mset="avg@1"):
    cells = list(CELLS); S = read["superposition"][mset]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.8))
    ser = [("modules with two or more names", MF.COL["mod"], "o", {c: S[c]["modules"] for c in cells}, True),
           ("the same at initialization", MF.COL["mod"], "o", {c: S[c]["init_modules"] for c in cells}, False),
           ("undivided layers with two or more names", "#e9c46a", "D", {c: S[c]["whole"] for c in cells}, True)]
    _series(axes[0], cells, ser, "per network", "multiply named modules (many-to-many)", ylim=(0, None))
    ser2 = [("serial: a node with its ancestor", "#2a9d8f", "s", {c: S[c]["serial"] for c in cells}, True),
            ("serial at initialization", "#2a9d8f", "s", {c: S[c]["init_serial"] for c in cells}, False),
            ("parallel: unrelated nodes", "#e76f51", "^", {c: S[c]["parallel"] for c in cells}, True),
            ("parallel at initialization", "#e76f51", "^", {c: S[c]["init_parallel"] for c in cells}, False)]
    _series(axes[1], cells, ser2, "name pairs per network", "kind of the pairs (hollow: initialization)", ylim=(0, None))
    ser3 = [("networks with a parallel pair", "#e76f51", "^", {c: S[c]["any_parallel"] for c in cells}, True),
            ("at initialization", "#e76f51", "^", {c: S[c]["init_any_parallel"] for c in cells}, False)]
    _series(axes[2], cells, ser3, "share of networks", "parallel superposition (hollow: initialization)")
    _legend(fig, ser + ser2, y=-0.06, ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return F._svg(fig, "r3_superposition")


# ------------------------------------------------- statuses, attribution --

def statuses(read, mset="avg@1"):
    cells = list(CELLS); S = read["status"][mset]; A = read["attribution"][mset]
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 5.6), gridspec_kw={"height_ratios": [1.25, 1]})
    ax = axes[0]
    classes = ("first", "higher", "root"); x = 0; ticks, labels = [], []
    for c in cells:
        for cl in classes:
            cnt = S[c].get(cl, {})
            tot = sum(cnt.values())
            if not tot:
                continue
            bottom = 0
            for st, colr in STATUS_COL.items():
                v = cnt.get(st, 0) / tot
                if v:
                    ax.bar(x, v, 0.8, bottom=bottom, color=colr, edgecolor="white", lw=.3); bottom += v
            ticks.append(x); labels.append(f"{SHORT[c]}\n{cl}"); x += 1
        x += 0.6
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=5.6, rotation=90); ax.set_ylim(0, 1); ax.set_ylabel("share of nodes", fontsize=7)
    ax.set_title("node status under the one-to-one naming, by family and node class", fontsize=8)
    ax.legend(handles=[Patch(color=v, label=k) for k, v in STATUS_COL.items()], fontsize=6.2, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax = axes[1]
    for i, c in enumerate(cells):
        reasons = A[c]["reasons"]; tot = sum(reasons.values())
        bottom = 0
        for rs, colr in REASON_COL.items():
            v = reasons.get(rs, 0) / tot if tot else 0
            if v:
                ax.bar(i, v, 0.7, bottom=bottom, color=colr, edgecolor="white", lw=.3); bottom += v
    _xt(ax, cells); ax.set_ylim(0, 1); ax.set_ylabel("share of the missing target edges", fontsize=7)
    ax.set_title("why a target edge is missing (reference)", fontsize=8)
    ax.legend(handles=[Patch(color=v, label=k) for k, v in REASON_COL.items()], fontsize=6.0, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=2)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    return F._svg(fig, "r3_status")


# ----------------------------------------------------------------- edges --

def edges(read, mset="avg@1", naming="one_to_one"):
    cells = list(CELLS); rules = read["edge_rules"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.9))
    ser = [(RULELAB[r], RULECOL[r], MARK[k], {c: T(read, mset, r, naming)[c]["edge"] for c in cells}, True) for k, r in enumerate(rules)]
    ser_c = [(RULELAB[r] + " (modules shuffled)", RULECOL[r], MARK[k], {c: T(read, mset, r, naming, "shuffle_modules")[c]["edge"] for c in cells}, False) for k, r in enumerate(rules)]
    _series(axes[0], cells, ser + ser_c, "edge $F_1$", "edge $F_1$ by edge rule (filled: trained; hollow: modules shuffled)", width=0.86, ms=3.6)
    ser2 = [(RULELAB[r], RULECOL[r], MARK[k], {c: T(read, mset, r, naming)[c]["edge_precision"] for c in cells}, True) for k, r in enumerate(rules)]
    _series(axes[1], cells, ser2, "precision", "edge precision")
    ser3 = [(RULELAB[r], RULECOL[r], MARK[k], {c: T(read, mset, r, naming)[c]["disconnected"] for c in cells}, True) for k, r in enumerate(rules)]
    _series(axes[2], cells, ser3, "share of hidden modules", "disconnected modules (no in- or no out-edge)")
    _legend(fig, ser, y=-0.04, ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return F._svg(fig, "r3_edges")


# -------------------------------------------------------------- examples --

def as_mo(rec, mset, edge_rule, naming):
    """A round-3 record seen through one variant, in the modules study's record layout (several modules
    may map to one node; a module with several names takes its first)."""
    L = rec["wirings"]["learned"][mset]; g = L["graph"]; sc = L["scores"][f"{edge_rule}|{naming}"]
    m, th = edge_rule.split("@"); idx = {"coverage": 2, "mask_share": 3, "weight_share": 4, "contrib_share": 5}[m]
    edges_ = [[row[0], row[1]] for row in g["measures"] if row[idx] > float(th)]
    active = {int(a): i for i, a in enumerate(rec["active_raw"])}
    mp = {f"in:{i}": (["leaf", active[i]] if i in active else ["extra", i]) for i in g["inputs"]}
    matched = {}
    for comp, ids in L["names"][naming].items():
        mp[f"c:{comp}"] = ["node", int(ids[0])]
        matched.setdefault(ids[0], [int(comp), L["component_stages"][int(comp)], 0.0])
    mp["out:0"] = ["y", 0]
    key = f"{mset}|{edge_rule}|{naming}"
    graph = {"components": g["components"], "inputs": g["inputs"], "edges": edges_, "map": mp, "matched": matched}
    return {**rec, "primary": [mset, edge_rule, naming],
            "wirings": {"learned": {key: {**sc, "graph": graph, "n_components": L["n_components"], "n_matched": sc["n_named_nodes"]}}}}, key


def example_graphs(recs, variants, cells=("tree", "ftree", "dag", "rand"), name="r3_examples"):
    cells = [c for c in cells if c in recs]
    fig = plt.figure(figsize=(2.6 * (len(variants) + 1), 2.5 * len(cells) + 0.6))
    gs = fig.add_gridspec(len(cells), len(variants) + 1, width_ratios=[1] + [1.3] * len(variants), wspace=0.1, hspace=0.35, top=0.94, bottom=0.08)
    for i, c in enumerate(cells):
        rec = recs[c]
        axt = fig.add_subplot(gs[i, 0])
        for j, (mset, edge_rule, naming, label) in enumerate(variants):
            v, key = as_mo(rec, mset, edge_rule, naming)
            axr = fig.add_subplot(gs[i, j + 1])
            if j == 0:
                MF.graph_pair([axt, axr], v, key)
                axr.set_title(f"{label}\n" + axr.get_title().replace("module graph: ", ""), fontsize=6.0)
                axt.set_title(axt.get_title().replace("): ", ")\n"), fontsize=6.0)
            else:
                G = MF.graphs(v, key); sc = v["wirings"]["learned"][key]
                mp = v["wirings"]["learned"][key]["graph"]["map"]
                for a, b in G["medges"]:                       # a copy joined to a module of the same name: a carry, not a relation
                    na, nb = mp.get(f"{a[0]}:{a[1]}"), mp.get(f"{b[0]}:{b[1]}")
                    if na is not None and na == nb and na[0] == "node":
                        G["mstyle"][(a, b)] = (MF.RELAY, "-")
                MF._draw(axr, G["mpos"], G["mnodes"], G["medges"], G["mstyle"], G["mlab"], G["mcol"],
                         f"{label}\n$F_1$ {sc['edge']['f1']:.2f}, {len(G['hits'])}/{len(G['tedges'])} edges, {sc['n_named_nodes']}/{rec['n_nodes']} named",
                         max(G["tspan"], G["mspan"]), small=G["small"], title_size=6.0)
    MF.graph_legend(fig, y=0.0)
    return F._svg(fig, name)


def paper_figure(recs, cells=("tree", "ftree", "dag", "rand"), name="r3_paper"):
    """The proposed main-text figure: row 1 the target graphs, row 2 the full module graphs of the same networks
    (the reference), one column per family."""
    cells = [c for c in cells if c in recs]
    fig = plt.figure(figsize=(2.9 * len(cells), 5.3))
    gs = fig.add_gridspec(2, len(cells), height_ratios=[1, 1.15], wspace=0.08, hspace=0.3, top=0.93, bottom=0.1)
    for j, c in enumerate(cells):
        v, key = as_mo(recs[c], "avg@1", "coverage@0.5", "one_to_one")
        axt, axr = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
        MF.graph_pair([axt, axr], v, key)
        axt.set_title(axt.get_title().replace("): ", ")\n"), fontsize=6.4); axr.set_title(axr.get_title().replace("module graph: ", "module graph\n"), fontsize=6.4)
    MF.graph_legend(fig, y=0.0)
    return F._svg(fig, name)


def pick_examples(read, units_dir, mset="avg@1", key="coverage@0.5|one_to_one", which="median"):
    """Per family the instance at the median reference edge F1 (or the worst, for the failure cases)."""
    out = {}
    for c in CELLS:
        vals = read["tables"][mset][key]["learned"][c]["values"]; seeds = read["instances"][c]
        if not vals or len(vals) != len(seeds):
            continue
        order = np.argsort(vals)
        i = order[len(order) // 2] if which == "median" else order[0]
        name = f"{c}__fs{seeds[i]}__n32768__best.json"
        for root in (Path(units_dir), MO / "results" / "r3_examples", MO / "results" / "r3_examples" / Path(units_dir).name):
            if (root / name).exists():
                out[c] = json.loads((root / name).read_text()); break
    return out


# ------------------------------------------------- sample size, confirmation --

def sample_size(read15, read13, mset="avg@1", key="coverage@0.5|one_to_one"):
    cells = list(CELLS)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8))
    for ax, lvl in zip(axes, ("edge", "higher")):
        ser = [("$n = 2^{15}$", MF.COL["mod"], "o", {c: read15["tables"][mset][key]["learned"][c][lvl] for c in cells}, True),
               ("$n = 2^{13}$", "#6aa6d1", "s", {c: read13["tables"][mset][key]["learned"][c][lvl] for c in cells}, True),
               ("initialization, $2^{15}$", COL["init"], "o", {c: read15["tables"][mset][key]["init"][c][lvl] for c in cells}, False),
               ("initialization, $2^{13}$", COL["init"], "s", {c: read13["tables"][mset][key]["init"][c][lvl] for c in cells}, False)]
        _series(ax, cells, ser, "$F_1$", f"{lvl} $F_1$ by sample size (reference)")
    _legend(fig, ser, y=-0.04, ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return F._svg(fig, "r3_sample_size")


def confirmation(dev, conf, variants, name="r3_confirm"):
    """Development panel beside the confirmation cohort: the reference and the adopted variants with
    the initialization and the modules-shuffled control."""
    cells = [c for c in CELLS if conf["tables"][variants[0][0]][f"{variants[0][1]}|{variants[0][2]}"]["learned"][c]["edge"]["mean"] is not None]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))
    for ax, lvl in zip(axes, ("edge", "higher", "ancestor")):
        ser = []
        for k, (mset, rule, naming, label) in enumerate(variants):
            key = f"{rule}|{naming}"
            ser.append((label + ", development", NAMCOL.get(naming, MF.COL["mod"]), MARK[k], {c: dev["tables"][mset][key]["learned"][c][lvl] for c in cells}, False))
            ser.append((label + ", confirmation", NAMCOL.get(naming, MF.COL["mod"]), MARK[k], {c: conf["tables"][mset][key]["learned"][c][lvl] for c in cells}, True))
        mset, rule, naming, _ = variants[0]; key = f"{rule}|{naming}"
        ser.append(("initialization, confirmation", COL["init"], "_", {c: conf["tables"][mset][key]["init"][c][lvl] for c in cells}, True))
        ser.append(("modules shuffled, confirmation", COL["shuffle_modules"], "D", {c: conf["tables"][mset][key]["shuffle_modules"][c][lvl] for c in cells}, True))
        _series(ax, cells, ser, "$F_1$", f"{lvl} $F_1$", width=0.86, ms=3.6)
    _legend(fig, ser, y=-0.06, ncol=3)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return F._svg(fig, name)


def build(read15, read13=None, verify_json=None, units_dir=None, conf=None, conf_variants=None, conf8=None, conf_units=None):
    figs = {}
    if verify_json:
        figs["verify"] = verify(verify_json)
    figs["constructions"] = constructions(read15)
    figs["namings"] = namings(read15)
    figs["persistence"] = persistence(read15)
    figs["superposition"] = superposition(read15)
    figs["status"] = statuses(read15)
    figs["edges"] = edges(read15)
    if units_dir:
        recs = pick_examples(read15, units_dir)
        variants = [("avg@1", "coverage@0.5", "one_to_one", "reference: one-to-one"), ("avg@1", "coverage@0.5", "copies", "one-to-one plus copies"),
                    ("complete@1", "coverage@0.5", "one_to_one", "complete linkage, one-to-one")]
        if recs:
            figs["examples"] = example_graphs(recs, variants)
        worst = pick_examples(read15, units_dir, which="worst")
        if worst:
            figs["failures"] = example_graphs(worst, variants[:2], cells=("tree", "ftree", "dag", "rand"), name="r3_failures")
    if read13:
        figs["sample_size"] = sample_size(read15, read13)
    if conf and conf_variants:
        figs["confirmation"] = confirmation(read15, conf, conf_variants)
    if conf8 and conf_variants:
        figs["confirmation8"] = confirmation(read15, conf8, conf_variants, name="r3_confirm8")
    src = conf if conf else conf8
    if src and conf_units:
        recs = pick_examples(src, conf_units)
        if recs:
            figs["paper"] = paper_figure(recs)
    return figs
