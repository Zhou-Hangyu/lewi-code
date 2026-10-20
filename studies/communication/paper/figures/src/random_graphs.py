"""Figure: eight draws from the held-out graph distribution.

Caption - Eight draws from the held-out graph distribution, in the grammar of
the template figure: light nodes are input leaves, blue discs are local
functions, green is the readout. Every draw has exactly seven local functions
at depth three - matching the depth-4 tree and the depth-3 reuse DAG in
local-function count - while the partition of the eight leaves into strands
(2 + 6, 3 + 5, 4 + 4 or 2 + 2 + 4), the shape of each strand, the number of
readout terms (three; five on seed 6001) and the reuse taps (a leaf or an
intermediate read a second time) vary. The panels are development seeds 6000,
6001, 6003, 6005, 6006, 6007, 6008 and 6011, in that order; seed 6000 is the
draw of the template figure's last panel.

Selection (decision 2026-09-16: eight of the first twelve draws, all different).
By silhouette - the leaf partition and the depth of every readout term - the
first twelve draws fall into eight classes, and each panel is the lowest seed
of its class: 6002 and 6004 repeat 6000's silhouette, 6009 repeats 6003's,
6010 repeats 6006's. The eight topologies are pairwise distinct under the
benchmark's canonical hash, checked at build time.

Built as one row at the full text width (decision 2026-09-16: first a 4 x 2 grid,
then "make it one row, 8 x 1"; the graphs compact, the labels gone, the
titles readable). Each panel is the template figure's eight-leaf panel drawn
by `graph_templates.panel` at 0.90 of its size - the largest at which eight
fit the 5.5 in text width - so discs and edges shrink with the leaf spacing
and the drawing keeps the template's proportions, while the titles go a step
up from the template's 7.8 pt to 8.8 (decision: "a bit bigger so it's more
readable in one row"); one horizontal scale over all eight panels. Node
numbers, input indices, the readout's y and the per-panel count line are
dropped, as in the template. The frame is 5.5 x 1.07 in; included at
\\linewidth.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from csfrbench.tasks.topologies import canonical_topology_hash
from csfrbench.viz import layout_graph

from figstyle import save
from graph_templates import extent, panel, random_graph

SEEDS = (6000, 6001, 6003, 6005, 6006, 6007, 6008, 6011)   # development bank; the final bank stays sealed
PANEL_W, PANEL_H = 0.675, 0.775   # in: the template figure's eight-leaf panels
TITLE_SIZE = 8.8                  # pt: a step above the template's 7.8 (decision: more readable in one row)
TITLE_BAND = 0.25                 # in: room for that title above a panel
WSPACE = 0.10                     # the template's is 0.14; eight panels sit a little closer
FRAME_W = 5.5                     # in: the text width; the tight bounding box adds 0.1 in on each side


def main() -> None:
    graphs = [random_graph(seed) for seed in SEEDS]
    if len({canonical_topology_hash(g) for g in graphs}) != len(SEEDS):
        raise SystemExit("[random_graphs] the caption promises eight distinct topologies")
    half = max(extent(g) for g in graphs) / 2     # one horizontal scale over all eight panels
    axes_w = FRAME_W - 0.2
    scale = axes_w / (len(SEEDS) + (len(SEEDS) - 1) * WSPACE) / PANEL_W   # the template's panel, reduced so eight fit
    height = PANEL_H * scale + TITLE_BAND
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(axes_w, height))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 - TITLE_BAND / height, wspace=WSPACE)
    for ax, seed, graph in zip(axes, SEEDS, graphs):
        panel(ax, f"Seed {seed}", graph, scale)
        ax.title.set_fontsize(TITLE_SIZE)
        xs = [x for x, _y in layout_graph(graph)[0].values()]
        mid = (min(xs) + max(xs)) / 2
        ax.set_xlim(mid - half, mid + half)
    print(f"[random_graphs] eight panels at {scale:.3f} of the template's size")
    save(fig, "random_graphs")


if __name__ == "__main__":
    main()
