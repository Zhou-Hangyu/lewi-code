"""Figure (appendix): the extracted module graphs of the clean-input cohort.

Caption - Target graphs (top row) and the module graphs extracted from LeWi's learned
geometry (bottom row) for the depth-4 tree, the Fourier tree, the reuse DAG and one
sampled depth-4 DAG (d = 8, every input active, 64,000 steps, n = 2^15; the network at
the upper median edge F1 of its workload, the 9th of 16, the 17th of 32 on the sampled
depth-4 DAGs). Each module graph is compared with the target graph above it. Target
row: a matched target edge is black, a missed target edge red; a hollow node is a
target intermediate without a matched module. Module row: every module of every
layer, a circle in the colour of the target intermediate it is matched to or a grey
square m_i when unmatched (a dashed ring marks a module spanning the full layer, where
no within-layer chance test exists), the inputs, and every majority edge: black for a
matched target edge, dotted grey for a transitive shortcut (ancestor to descendant),
dashed red for a spurious edge (no target relation), light grey for a path through an
unmatched module. Above each module graph: the edge F1, the matched target edges and
the matched target intermediates. Modules: within-layer average-linkage clusters of
the neuron positions cut at the model's radius R; edges: a module reads another when
its units are connected, on average, to more than half of it through the executed
mask; matching: one-to-one assignment of held-out ridge probes against shape-matched
chance sets (Appendix, structure discovery).

Data: data/structure/examples.json + examples/<record>.json (18-main-d8/structure,
src/sd_read.py). The layout and the edge classification are the modules study's
(structure-discovery/modules/src/mo_figs.py, r3_figs.py), imported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
MO = REPO / "studies/csfrbench/evaluation-protocol/structural-metrics/structure-discovery/modules"
for p in (str(MO / "src"), str(MO.parent / "component-dag" / "src"), str(MO.parent / "recovered-graph" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
import mo_figs as MF      # noqa: E402  (sets its own rc; figstyle's is re-applied below)
import r3_figs as R3      # noqa: E402

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerPatch  # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

import paperstyle as ps   # noqa: E402
from figstyle import DATA, OUT, save  # noqa: E402

FAMILIES = ("tree", "ftree", "dag", "rand")
TITLE = {"tree": "Depth-4 tree", "ftree": "Fourier tree", "dag": "Reuse DAG", "rand": "Sampled depth-4 DAG"}   # the paper's setting names
ROWS = ("Target graph", "Extracted module graph")
TYPE = 7.0               # the appendix figure's titles, annotations and legend (pt)
REFERENCE = ("avg@1", "coverage@0.5", "one_to_one")
MISS = MF.COL["miss"]


def paper_rc() -> None:
    """figstyle's type after the modules study's modules set theirs at import."""
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "path",
                         "axes.spines.top": False, "axes.spines.right": False})
    ps.apply_rc()


def load_examples() -> dict:
    ex = json.loads((DATA / "structure/examples.json").read_text())
    return {fam: json.loads((DATA / "structure/examples" / ex[fam]["record"]).read_text()) for fam in FAMILIES if fam in ex}


def graph_of(rec):
    """The target and module graphs of one record laid out by the modules study, plus the whole-layer modules."""
    v, key = R3.as_mo(rec, *REFERENCE)
    G = MF.graphs(v, key)
    whole = {("c", int(w["comp"])) for w in rec["wirings"]["learned"][REFERENCE[0]]["whole_layers"]}
    sc = rec["wirings"]["learned"][REFERENCE[0]]["scores"][f"{REFERENCE[1]}|{REFERENCE[2]}"]
    return G, whole, sc


def draw(ax, pos, nodes, edges, style, lab, col, span, *, hollow=(), small=(), whole=(), size=34.0, font=3.9, font3=None, labels=True,
         arrow=3.4, shrink=3.6, lw=(0.45, 0.55), unmatched="o") -> None:
    """`font3` is the type size of a label of three or more characters (m10 ...), so that it stays inside its node; `lw` the
    width of a solid and of a dashed or dotted edge; `unmatched` the marker of a module no target intermediate is matched to."""
    for e in edges:
        a, b = e
        if a not in pos or b not in pos:
            continue
        c, ls = style.get(e, (MF.HIT, "-"))
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>", mutation_scale=arrow, color=c, lw=lw[0] if ls == "-" else lw[1],
                                     linestyle=ls, shrinkA=shrink, shrinkB=shrink, zorder=1))
    for n in nodes:
        x, y = pos[n]
        if n in small:
            ax.scatter([x], [y], s=size * 0.16, facecolor=col[n], edgecolor="#aaaaaa", linewidths=0.3, zorder=2)
            continue
        if n in hollow:
            ax.scatter([x], [y], s=size, facecolor="white", edgecolor=col[n], linewidths=0.7, linestyle="--", zorder=2)
        elif col[n] == MF.UNNAMED and unmatched != "o":          # a square of the circle's area
            ax.scatter([x], [y], s=size * 0.8, marker=unmatched, facecolor=col[n], edgecolor="#555555", linewidths=0.35, zorder=2)
        else:
            ax.scatter([x], [y], s=size, facecolor=col[n], edgecolor="#555555", linewidths=0.35, zorder=2)
        if n in whole:
            ax.scatter([x], [y], s=size * 2.6, facecolor="none", edgecolor="#333333", linewidths=0.5, linestyle=(0, (1.6, 1.2)), zorder=2)
        if labels:
            dark = col[n] in MF.TEACHER and n not in hollow
            ax.text(x, y, lab[n], ha="center", va="center", fontsize=font if len(lab[n]) < 3 else (font3 or font),
                    color="white" if dark else "#222222", zorder=3)
    ax.axis("off")
    depth = max(l for l, _ in pos.values())
    ax.set_xlim(-0.55, depth + 0.55); ax.set_ylim(-span / 2 - 0.7, span / 2 + 0.7)


def legend(fig, *, y: float, ncol: int, fontsize: float = 5.9, canonical: bool = False) -> None:
    """`canonical`: the appendix figure's terms, each tied to precision or recall, with unmatched modules as squares."""
    if canonical:
        handles = [Line2D([0], [0], color=MF.HIT, lw=1.0, label="matched target edge"),
                   Line2D([0], [0], color=MISS, lw=1.0, label="missed target edge"),
                   Line2D([0], [0], color=MF.COL["extra"], lw=1.0, ls="--", label="spurious edge"),
                   Line2D([0], [0], color=MF.SHORTCUT, lw=1.0, ls=":", label="transitive shortcut"),
                   Line2D([0], [0], color=MF.RELAY, lw=1.0, label="path through unmatched module"),
                   Line2D([0], [0], marker="s", color="w", markerfacecolor=MF.UNNAMED, markeredgecolor="#555555", markeredgewidth=0.4, ms=5.2,
                          label="unmatched module $m_i$"),
                   Circle((0, 0), 1, facecolor="white", edgecolor="#555555", linewidth=0.7, linestyle=(0, (2.2, 1.3)),
                          label="target intermediate without a matched module"),
                   Circle((0, 0), 1, facecolor="none", edgecolor="#333333", linewidth=0.6, linestyle=(0, (1.6, 1.2)),
                          label="module spanning the full layer")]
    else:
        handles = [Line2D([0], [0], color=MF.HIT, lw=0.9, label="module edge = target edge (both ends named)"),
                   Line2D([0], [0], color=MF.SHORTCUT, lw=0.9, ls=":", label="shortcut: ancestor to descendant"),
                   Line2D([0], [0], color=MF.COL["extra"], lw=0.9, ls="--", label="no target relation"),
                   Line2D([0], [0], color=MF.RELAY, lw=0.9, label="through an unnamed module (grey) or an unused input (dot)"),
                   Line2D([0], [0], color=MISS, lw=0.9, label="target edge not recovered (top row)"),
                   Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#555", ms=4.5, label="target node with no named module (hollow)"),
                   Circle((0, 0), 1, facecolor="none", edgecolor="#333333", linewidth=0.6, linestyle=(0, (1.6, 1.2)),
                          label="module that is its whole layer (dashed ring)")]

    def ring(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        orig = orig_handle
        node = canonical and orig.get_facecolor()[3] > 0            # a hollow node is drawn node-sized, a ring larger, as in the graphs
        return Circle((xdescent + width / 2, ydescent + height / 2), height / 2 * (0.8 if node else 1.0), facecolor=orig.get_facecolor(),
                      edgecolor=orig.get_edgecolor(), linewidth=orig.get_linewidth(), linestyle=orig.get_linestyle())
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=fontsize, frameon=False, bbox_to_anchor=(0.5, y),
               handlelength=1.8, handletextpad=0.5, columnspacing=1.1, borderaxespad=0.0, handler_map={Circle: HandlerPatch(patch_func=ring)})


def build(recs: dict, *, width: float, height: float, size: float, font: float, labels: bool, title_size: float, legend_rows: int,
          bottom: float, arrow: float, shrink: float, font3: float | None = None):
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, len(FAMILIES), height_ratios=[1, 1.12], wspace=0.06, hspace=0.32, left=0.005, right=0.995, top=0.93, bottom=bottom)
    for j, fam in enumerate(FAMILIES):
        rec = recs[fam]
        G, whole, sc = graph_of(rec)
        span = max(G["tspan"], G["mspan"])
        axt, axm = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
        draw(axt, G["tpos"], G["tnodes"], G["tedges"], G["tstyle"], G["lab"], G["col"], span, hollow=set(G["missing"]), size=size, font=font,
             font3=font3, labels=labels, arrow=arrow, shrink=shrink)
        draw(axm, G["mpos"], G["mnodes"], G["medges"], G["mstyle"], G["mlab"], G["mcol"], span, small=G["small"], whole=whole, size=size,
             font=font, font3=font3, labels=labels, arrow=arrow, shrink=shrink)
        axt.set_title(f"{TITLE[fam]}\n{rec['n_nodes']} nodes, {len(G['tedges'])} edges", fontsize=title_size, pad=2)
        axm.set_title(f"module graph: $F_1$ {sc['edge']['f1']:.2f}, {len(G['hits'])}/{len(G['tedges'])} edges,\n{sc['n_named_nodes']}/{rec['n_nodes']} nodes named",
                      fontsize=title_size, pad=2)
    return fig


def appendix(recs: dict, *, width: float = 5.5, height: float = 3.32, bottom: float = 0.130):
    """The appendix figure: a row label per row, each extracted module graph under its target, unmatched modules as squares."""
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, len(FAMILIES), height_ratios=[1, 1.12], wspace=0.06, hspace=0.305, left=0.045, right=0.995, top=0.908, bottom=bottom)   # the graphs of the 3.2 in build; 0.117 in more between the rows for the third annotation line
    look = dict(size=52.0, font=4.6, font3=3.8, arrow=4.2, shrink=4.5, lw=(0.6, 0.7))
    for j, fam in enumerate(FAMILIES):
        rec = recs[fam]
        G, whole, sc = graph_of(rec)
        span = max(G["tspan"], G["mspan"])
        axt, axm = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
        draw(axt, G["tpos"], G["tnodes"], G["tedges"], G["tstyle"], G["lab"], G["col"], span, hollow=set(G["missing"]), **look)
        draw(axm, G["mpos"], G["mnodes"], G["medges"], G["mstyle"], G["mlab"], G["mcol"], span, small=G["small"], whole=whole, unmatched="s",
             **look)
        axt.set_title(f"{TITLE[fam]}\n{rec['n_nodes']} nodes · {len(G['tedges'])} edges", fontsize=TYPE, pad=2)
        axm.set_title(f"$F_1 \\;=\\; {sc['edge']['f1']:.2f}$\n{len(G['hits'])}/{len(G['tedges'])} target edges matched\n"   # \\; : TeX's relation space
                      f"{sc['n_named_nodes']}/{rec['n_nodes']} nodes matched", fontsize=TYPE, pad=2)
    for row, name in enumerate(ROWS):
        box = gs[row, 0].get_position(fig)
        fig.text(0.013, (box.y0 + box.y1) / 2, name, rotation=90, ha="center", va="center", fontsize=TYPE + 0.4, fontweight="bold")
    return fig


def caption() -> str:
    """The docstring's caption as one paragraph with LaTeX math (the final report shows it, as it shows world_model_grid's)."""
    text = " ".join(__doc__.split("Caption - ", 1)[1].split("\n\n", 1)[0].split())
    for old, new in (("edge F1", "edge $F_1$"), ("m_i", "$m_i$"), ("d = 8", "$d = 8$"), ("n = 2^15", "$n = 2^{15}$"), ("radius R", "radius $R$")):
        text = text.replace(old, new)
    return text


def main() -> None:
    paper_rc()
    recs = load_examples()
    fig = appendix(recs)
    legend(fig, y=0.0, ncol=3, fontsize=TYPE, canonical=True)
    save(fig, "structure_modules")
    (OUT / "structure_modules.caption.txt").write_text(caption() + "\n")


if __name__ == "__main__":
    main()
