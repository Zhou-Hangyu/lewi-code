"""Figure: how much capacity one local function needs.

Reads data/local_fit/a4_grid.csv, frozen from the A-4 local-capacity
calibration study's retained record by src/run_local_fit.py. The plotted
numbers are in that table.

Caption - Local functions are cheap to fit, except the dense node. Each
cell is the median held-out NMSE, on a log color scale, of a small ReLU
network fit to one local function of the named family on its parent
distribution, over the study's 16 sealed draws; panels vary the family,
axes vary the fitted network (width 2-1,024, one to four hidden layers);
each cell is annotated with its median. Two-input local functions of both
families are cheap; the eight-input Fourier node needs moderate width and
depth; the eight-input additive-multiplicative node reaches a median of
only 0.075 anywhere on this grid.

Built at the 5.5 in text width since 2026-09-16 (decision: larger text and
numbers, for reading on paper). Before, a 7.2 in canvas was scaled to the
column at inclusion, which cut every size by a fifth (cell numbers 6 pt
became 4.7 pt). The cells stay square: the figure's height is set from the
panel width, the type is 7.2-8.6 pt, and the cell numbers shrink together
only if the widest of them would fill more than 80% of a cell.
"""
from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from figstyle import DATA, save

WIDTHS = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
HIDDEN = (1, 2, 3, 4)
PANELS = (("(a) 2-input additive–multiplicative", "anova2"),
          ("(b) 2-input Fourier", "fourier2"),
          ("(c) 8-input additive–multiplicative", "anova8"),
          ("(d) 8-input Fourier", "fourier8"))
VMIN, VMAX = -5.0, 0.0
CMAP = plt.get_cmap("viridis")
WIDTH = 5.5                      # inches: the text width the figure is included at (\linewidth)
CELL_TEXT = 7.2                  # pt: the medians in the cells (uniformly reduced if the widest exceeds CELL_FILL of a cell)
CELL_FILL = 0.80
TICK, TITLE, LABEL = 7.4, 8.6, 8.0
CBAR_TICK, CBAR_TITLE = 7.4, 7.6


def load():
    grid = {}
    with (DATA / "local_fit" / "a4_grid.csv").open() as handle:
        for r in csv.DictReader(handle):
            grid[(r["family"], int(r["hidden"]), int(r["width"]))] = \
                float(r["median"])
    return grid


def cell_color(median):
    t = (np.log10(max(median, 10 ** VMIN)) - VMIN) / (VMAX - VMIN)
    return CMAP(min(max(t, 0.0), 1.0))


def _fmt(v):
    if v >= 0.0095:
        return f"{v:.2f}".lstrip("0")
    mantissa, exponent = f"{v:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def draw_panel(ax, grid, family, *, yticks, xticks):
    for yi, h in enumerate(HIDDEN):
        for xi, w in enumerate(WIDTHS):
            value = grid[(family, h, w)]
            color = cell_color(value)
            ax.add_patch(Rectangle((xi, yi), 1, 1, facecolor=color,
                                   edgecolor="#000000", linewidth=0.4,
                                   joinstyle="miter"))
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            ax.text(xi + 0.5, yi + 0.5, _fmt(value), ha="center",
                    va="center", fontsize=CELL_TEXT,
                    color="#000000" if lum > 0.45 else "#ffffff")
    ax.set_xlim(0, len(WIDTHS))
    ax.set_ylim(0, len(HIDDEN))
    ax.set_xticks([i + 0.5 for i in range(len(WIDTHS))])
    ax.set_xticklabels([str(w) for w in WIDTHS] if xticks else [],
                       fontsize=TICK)
    ax.set_yticks([i + 0.5 for i in range(len(HIDDEN))])
    ax.set_yticklabels([str(h) for h in HIDDEN] if yticks else [],
                       fontsize=TICK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_edgecolor("#000000")
        side.set_linewidth(0.5)


def draw_colorbar(ax):
    steps = 128
    for i in range(steps):
        y0 = i / steps
        ax.add_patch(Rectangle((0, y0), 1, 1 / steps,
                               facecolor=CMAP(i / (steps - 1)),
                               edgecolor=CMAP(i / (steps - 1)),
                               linewidth=0.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ticks = range(int(VMIN), int(VMAX) + 1)
    ax.set_yticks([(t - VMIN) / (VMAX - VMIN) for t in ticks])
    ax.set_yticklabels(["1" if t == 0 else f"$10^{{{t}}}$" for t in ticks],
                       fontsize=CBAR_TICK)
    ax.tick_params(length=0, pad=2)
    ax.yaxis.tick_right()
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_edgecolor("#000000")
        side.set_linewidth(0.5)
    ax.set_title("Median\nNMSE", fontsize=CBAR_TITLE, color="#444", pad=5)


GRID = dict(width_ratios=[1, 1, 0.035], wspace=0.08, hspace=0.34,
            left=0.07, right=0.93, top=0.905, bottom=0.115)


def build(height: float):
    """The figure at the text width and a given height; returns it with its
    panels and the slot the first panel was given."""
    grid = load()
    fig = plt.figure(figsize=(WIDTH, height))
    gs = fig.add_gridspec(2, 3, **GRID)
    panels = []
    for i, (title, family) in enumerate(PANELS):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        draw_panel(ax, grid, family, yticks=(col == 0), xticks=(row == 1))
        ax.set_aspect("equal", adjustable="box")      # square cells
        ax.set_title(title, fontsize=TITLE, pad=4)
        panels.append(ax)
    fig.text(0.012, 0.51, "Hidden layers", rotation=90, ha="left",
             va="center", fontsize=LABEL, color="#444")
    fig.text(0.5, 0.016, "Width", ha="center", fontsize=LABEL, color="#444")
    fig.canvas.draw()
    return fig, gs, panels, gs[0, 0].get_position(fig)


def fit_cell_text(fig, panels) -> float:
    """The cell numbers at CELL_TEXT, or all of them smaller by one factor
    when the widest would fill more than CELL_FILL of a cell; returns the size."""
    renderer = fig.canvas.get_renderer()
    cell = panels[0].get_window_extent(renderer).width / len(WIDTHS)
    texts = [t for ax in panels for t in ax.texts]
    widest = max(t.get_window_extent(renderer).width for t in texts)
    size = CELL_TEXT * min(1.0, CELL_FILL * cell / widest)
    for t in texts:
        t.set_fontsize(size)
    return size


def main() -> None:
    # two passes: the panels are width-limited in their slots, so the height that leaves no slack above and below
    # the square cells follows from the first pass's slot
    fig, gs, panels, slot = build(3.0)
    used = slot.width * WIDTH * len(HIDDEN) / len(WIDTHS)          # the panel's height at its full slot width, inches
    height = 3.0 * used / (slot.height * 3.0)
    plt.close(fig)
    fig, gs, panels, slot = build(height)
    size = fit_cell_text(fig, panels)
    # the colorbar spans exactly the two rows of (square-celled) panels
    top = panels[1].get_position().y1
    bottom = panels[3].get_position().y0
    cbar = fig.add_subplot(gs[:, 2])
    cpos = cbar.get_position()
    cbar.set_position([cpos.x0, bottom, cpos.width, top - bottom])
    draw_colorbar(cbar)
    box = panels[0].get_position()
    print(f"[local_fit] {WIDTH} x {height:.2f} in, cells {box.width * WIDTH / len(WIDTHS):.3f} in square, "
          f"cell numbers {size:.2f} pt")
    save(fig, "local_fit")


if __name__ == "__main__":
    main()
