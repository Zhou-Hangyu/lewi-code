"""Figure 1, second version (out/teaser-v2): the teaser with a fifth panel
for the transition into the output, a narrower target graph and the
embedding panels packed closer together.

Caption - LeWi learns neuron embeddings that follow the target computation.
Left: the target computation of one depth-4 Fourier-tree instance (active
inputs at the bottom, the seven local functions as numbered intermediates,
the output y on top). Right, in one shared frame, view and cube: the
embeddings in R^3 that a sealed LeWi network trained on this instance
(n = 2^15, test NMSE 0.003) learned for every input coordinate, hidden unit
and the output - their inner products score the connections. Each panel
highlights one layer transition, from the inputs into the first hidden
layer to the last hidden layer into the output: its two levels, coloured
by the target intermediate they are most associated with (inputs by the
local function that reads them, hidden units by the largest absolute
correlation on the diagnostic pool among the intermediates whose inputs
the unit reads through its kept connections, grey below the permutation
floor), and the connections the network keeps between them, each in the
colour of its sending unit; every other level is light grey, and farther
points and connections are drawn fainter. The output's embedding (black
star) is shown in every panel; the last panel draws the connections into
it at the strength of their gate, which stays near one half for every
last-layer unit because the readout is unmasked and the output's embedding
lies near the origin. Marker shape gives the level. The active inputs and
hidden units of one local function align and connect within their module;
the units of the two level-2 functions (5, 6) lie between the modules they
combine, on opposite sides of the space, the deeper layers contract toward
the origin, and the output reads every last-layer unit. Colours use the
known target intermediates for descriptive visualization only; this is not
a graph-recovery measurement.

Data and network: those of the teaser (data/teaser/fourier_f7011_n32768__lewi-v05.npz,
the sealed LeWi winner of Fourier-tree instance 7011 at n = 2^15), with the
teaser's frame, viewpoint, colours, marker shapes and association rule.
Differences from the teaser (design request 2026-09-04): the target panel is
0.8 in wide instead of 1.0 in, the five cubes are packed with a fixed 0.05 in
gap between neighbouring cubes (their axes boxes overlap; the panels are
transparent), and the connections into the output have their own panel,
H4 -> Output, instead of being added to the H3 -> H4 panel, drawn at the
soft gate's own strength (the teaser's decided-gate rule 2g - 1 leaves the
readout empty: its 32 gates lie in 0.47-0.51). The module is
src/teaser_v2.py; the output files are out/teaser-v2.{pdf,svg,png}.
"""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch

import paperstyle as ps
from figstyle import save
from learned_structure import TRANSITIONS, kept
from teaser import (LEVEL, MUTED, NETWORK, POINT_SMALL, TITLE, best_view, cube_extent,
                    draw_geometry, draw_target, fit_cubes, frame, load, marker_handles)

# target: the target panel's width as a fraction of one cube's width; gap and
# gap_target in inches (between neighbouring cubes, between the target panel
# and the first cube); coverage: the share of units inside the cube (None =
# the teaser's cube, 0.8 of the largest norm); output_star: the teaser's
# larger star on top in every panel, else the output renders as any level
# with its marker area scaled by star_scale (a five-point star fills about a
# third of the area of a square of the same marker size)
# faint: the units without an association (the inactive inputs, the dead
# hidden units) are small light-grey markers at this alpha (None = grey as
# in the teaser) and the connections touching them keep the share
# faint_edges of theirs; background: the alpha of the other levels' markers;
# signal_only: leave the units without an association and their connections out
VERSION = {
    "teaser-v2": dict(target=0.94, gap=0.05, gap_target=0.12, coverage=None, output_star=True,
                      star_scale=1.0, faint=None, faint_edges=None, background=1.0,
                      signal_only=False),
    "teaser-final": dict(target=0.75, gap=0.0, gap_target=0.08, coverage=0.98, output_star=False,
                      star_scale=2.5, faint=0.7, faint_edges=0.45, background=0.7,
                      signal_only=False),
    "teaser-v4": dict(target=0.75, gap=0.0, gap_target=0.08, coverage=0.98, output_star=False,
                      star_scale=2.5, faint=None, faint_edges=None, background=1.0,
                      signal_only=True),
}
PANELS = TRANSITIONS + [(4, 5)]              # Input -> H1, ..., H3 -> H4, H4 -> Output
HEADINGS = ["Target"] + [f"{LEVEL[a]} $\\rightarrow$ {LEVEL[b]}" for a, b in PANELS]
WIDTH, HEIGHT = 5.5, 1.75                    # inches, as the teaser
MARGIN = 0.005 * WIDTH                       # the teaser's outer margin
TARGET_MARGIN = 0.7                          # leaf spacings beside the outermost inputs
ROW_CENTER = 0.515                           # of the figure height, as the teaser's row


def strengths(d, boundary: int) -> np.ndarray:
    """Connection strengths of one transition: the teaser's kept-connection
    rule at the hidden transitions (the binary mask at the two masked
    boundaries, the decided part 2g - 1 of the soft gate elsewhere), the
    soft gate g itself into the output. The readout is unmasked and the
    output's embedding lies near the origin, so every readout gate stays
    near one half: the output reads every last-layer unit."""
    if boundary == 4:
        return np.asarray(d["gate4"], dtype=float)
    return kept(d, boundary)


def signal_carriers(d) -> dict:
    """The record restricted to the units that carry signal: the input
    coordinates some local function reads, and the hidden units with a
    non-constant activation on the diagnostic pool (a dead unit correlates
    with no intermediate). Every per-unit array and connection matrix is
    filtered consistently, and the active-input table is re-indexed to the
    kept inputs, which is the index space the association rule compares
    input supports and receptive fields in."""
    d = dict(d)
    active = sorted({int(c) for row in np.asarray(d["active_raw"]) for c in row})
    live = {0: np.array(active)}
    live.update({lv: np.nonzero(~np.asarray(d[f"dead{lv}"]).astype(bool))[0]
                 for lv in range(1, 5)})
    position = {raw: i for i, raw in enumerate(active)}
    d["active_raw"] = np.array([[position[int(c)] for c in row]
                                for row in np.asarray(d["active_raw"])])
    for lv, idx in live.items():
        for key in (f"codes{lv}", f"corr{lv}", f"dead{lv}"):
            d[key] = np.asarray(d[key])[idx]
    for b in range(5):                 # boundary b: rows receive at level b + 1, columns send at b
        for key in (f"gate{b}", f"mask{b}"):
            m = np.asarray(d[key])
            if b + 1 in live:
                m = m[live[b + 1]]
            m = m[:, live[b]]
            d[key] = m
    return d


def half_edge(d, rotation, coverage: float) -> float:
    """The cube half-edge that contains `coverage` of the units: that
    quantile of the largest |coordinate|, in the frame, over every input,
    hidden and output embedding."""
    codes = np.concatenate([np.asarray(d[f"codes{i}"]) @ rotation for i in range(6)])
    return float(np.quantile(np.abs(codes).max(1), coverage))


def rendered_cube(fig, ax) -> tuple[float, float, float, float]:
    """The drawn cube of one panel, in inches from the figure's lower-left
    corner (x0, y0, x1, y1), read from the rendered cube edges."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [line.get_window_extent(renderer) for line in ax.lines]
    return (min(b.x0 for b in boxes) / fig.dpi, min(b.y0 for b in boxes) / fig.dpi,
            max(b.x1 for b in boxes) / fig.dpi, max(b.y1 for b in boxes) / fig.dpi)


def pack(fig, panels, *, target: float, gap: float, gap_target: float):
    """Place the panels so that neighbouring cubes are `gap` apart and the
    row fills the width beside a target panel `target` cubes wide, the last
    box ending at the margin: the cube's footprint in its square axes box
    is read from a rendered panel, the side of the box follows, and the
    boxes overlap where the cubes leave their corners empty. Returns the
    rendered cubes and the target panel's width in inches."""
    fit_cubes(fig, panels)
    box = panels[0].get_position()            # the square a 3-D axes keeps
    side = box.width * WIDTH
    x0, _, x1, _ = rendered_cube(fig, panels[0])
    left = (x0 - box.x0 * WIDTH) / side       # the cube's footprint, in sides
    right = (x1 - box.x0 * WIDTH) / side
    width = right - left
    side = (WIDTH - 2 * MARGIN - gap_target - (len(panels) - 1) * gap) \
        / ((len(panels) + target) * width + 1 - right)
    target_width = target * width * side
    pitch = width * side + gap
    start = MARGIN + target_width + gap_target - left * side
    bottom = ROW_CENTER * HEIGHT - side / 2
    for j, ax in enumerate(panels):
        ax.set_position([(start + j * pitch) / WIDTH, bottom / HEIGHT,
                         side / WIDTH, side / HEIGHT])
    cubes = [rendered_cube(fig, ax) for ax in panels]
    gaps = " ".join(f"{b[0] - a[2]:.3f}" for a, b in zip(cubes, cubes[1:]))
    print(f"[{fig.get_label()}] cubes {width * side:.2f} in wide in {side:.2f} in boxes, "
          f"target {target_width:.2f} in, gaps {gaps} in")
    return cubes, target_width


def build(stem: str) -> None:
    v = VERSION[stem]
    ps.apply_rc()
    lewi = load(NETWORK)
    rotation, lim = frame([lewi])
    if v["coverage"] is not None:
        lim = half_edge(lewi, rotation, v["coverage"])
        print(f"[{stem}] cube half-edge {lim:.2f} ({v['coverage']:.0%} of the units inside)")
    view = best_view(lewi, rotation)
    # frame, cube and viewpoint come from the complete record, so versions
    # differ only in what they draw
    shown = signal_carriers(lewi) if v["signal_only"] else lewi
    if v["signal_only"]:
        print(f"[{stem}] drawn: {shown['codes0'].shape[0]} of {lewi['codes0'].shape[0]} inputs, "
              + ", ".join(f"{shown[f'codes{lv}'].shape[0]} of {lewi[f'codes{lv}'].shape[0]} H{lv}"
                          for lv in range(1, 5)))
    fig = plt.figure(figsize=(WIDTH, HEIGHT), label=stem)
    target = fig.add_axes([MARGIN / WIDTH, 0.17, 0.1, 0.69])
    draw_target(target, lewi, margin=TARGET_MARGIN, label="$y$")
    panels = []
    for transition in PANELS:
        ax = fig.add_axes([0.2, 0.17, 0.15, 0.69], projection="3d")
        ax.set_facecolor("none")              # the boxes overlap
        draw_geometry(ax, shown, rotation, lim, transition=transition, point=POINT_SMALL,
                      view=view, readout=False, weights=strengths,
                      output_star=v["output_star"], star_scale=v["star_scale"],
                      faint=v["faint"], faint_edges=v["faint_edges"],
                      background=v["background"])
        panels.append(ax)
    cubes, target_width = pack(fig, panels, target=v["target"], gap=v["gap"],
                               gap_target=v["gap_target"])
    # the target graph spans exactly the cubes' height, and every heading
    # sits on one line just above them, centred on its panel's cube
    low, high = cube_extent(fig, panels)
    target.set_position([MARGIN / WIDTH, low, target_width / WIDTH, high - low])
    centres = [(MARGIN + target_width / 2) / WIDTH] + [(c[0] + c[2]) / 2 / WIDTH for c in cubes]
    for x, heading in zip(centres, HEADINGS):
        fig.text(x, high + 0.035, heading, ha="center", va="bottom", fontsize=TITLE,
                 color="#2a2a2a")
    # reading cue under the geometry panels, marker key from the left margin
    # (it is wider than the target panel)
    x0 = cubes[0][0] / WIDTH + 0.03
    x1 = cubes[-1][2] / WIDTH - 0.03
    y = low - 0.07
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=5, color="#9a9a9a",
                                   linewidth=0.6, shrinkA=0, shrinkB=0))
    fig.text((x0 + x1) / 2, y - 0.025, "Network depth", ha="center", va="top",
             fontsize=6.4, color=MUTED)
    order = [0, 3, 1, 4, 2, 5]           # legends fill columns first; this reads by rows
    fig.legend([marker_handles(v["star_scale"])[i] for i in order], [LEVEL[i] for i in order],
               loc="upper left", ncol=3, fontsize=5.8, frameon=False, handlelength=0.9,
               handletextpad=0.2, columnspacing=0.6, labelspacing=0.2, borderaxespad=0.0,
               bbox_to_anchor=(MARGIN / WIDTH, low - 0.03))
    save(fig, stem)


def main() -> None:
    build("teaser-v2")


if __name__ == "__main__":
    main()
