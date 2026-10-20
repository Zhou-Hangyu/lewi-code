"""Figure (main text, the compact alternative; decision 2026-09-23 20:xx): the SWM-16 heat map as one continuous matrix.

The cells of world_model_grid (its load, collect and row set, so the same adopted record and checks): the 17 counted tasks
as one continuous 17-column matrix, black separators between the four task groups with abbreviated group names above,
the seven principal baselines as rows. Tasks are grouped by environment type and ordered within each group by state
dimension (world_model_grid.GROUPS). The filled CAPTION is written beside the figure as world_model_grid_compact.caption.txt.
The colour scale is linear, symmetric about zero and CLIPPED FOR DISPLAY at +-CLIP; the values are not changed. The 119
differences have a gap between 0.66 ($\\ell_1$ on Cartpole; Dense on Cartpole 0.65) and 1.12 (L0 on Reacher); 0.75 is the
smallest round value in that gap, so every other comparison uses the whole scale (+-1 would clip the same cells and resolve
the bulk less). The cells beyond are drawn at the end colour with no printed value (the caption lists them), and the colour
bar's arrow marks the clipped side. Hollow circles: the paired 95% interval contains zero; a key in the plate says so.
White seams of 0.25 pt between the cells only, so the body's outer edge is the cells' own edge.

Decision 2026-09-25: no values in the clipped cells; the group separators black, 0.8 pt, from the body's top edge to its
bottom edge and no further (butt caps, clipped to the body); the colour bar drawn by hand in one coordinate system: the
ramp spans the body's height (matplotlib's colorbar embedded its mesh as a 6 x 91 px raster at 110 dpi, about half a point
off its outline), the outline and the ramp's ends are 0.4 pt lines, the arrow of the clipped side continues the outline
from the ramp's corners, the ticks point inward, so nothing sticks out of the bar; the key of the hollow circle under the
bar. Built at its natural print size (include it unscaled): 7 pt names, 6.5 pt group names, bar and key.
"""
from __future__ import annotations

import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

import paperstyle as ps
from figstyle import DATA, OUT, save
from world_model_grid import BASELINES, CMAP, GROUPS, MAIN_ROWS, SEALED, TASKS, TRIAL, _SUFFIX, collect, load, number

CLIP = 0.75
SHORT = {"Control": "Control", "Navigation": "Navigation", "Manipulation": "Manip.", "Synthetic": "Synth."}
INK, RULE, RING, SEPARATOR = "#1a1a1a", "#8a8a8a", "#303030", "black"
CELL_W, CELL_H, ANGLE = 0.19, 0.13, 40                 # inches, inches, degrees (the task names)
LEFT, HEAD, FOOT, RIGHT = 0.45, 0.16, 0.45, 0.42       # inches: row names, group names, task names, the colour bar
BAR_GAP, BAR_W, ARROW, KEY_Y = 0.1, 0.06, 0.08, -0.02  # inches: the bar's offset from the body, its width, the arrow, the key's height
SEAM, SEPARATOR_W, OUTLINE_W = 0.25, 0.8, 0.4          # points
NAME_SIZE, SMALL_SIZE = 7.0, 6.5
CIRCLE = dict(marker="o", markersize=2.8, markerfacecolor="none", markeredgecolor=RING, markeredgewidth=0.55, linestyle="none")

CAPTION = (
    "LeWi against the seven principal baselines on the 17 counted world-model tasks. Each cell is the paired difference "
    "AULC(LeWi) $-$ AULC(baseline); blue: LeWi better, red: the baseline better. The AULC is the area under the "
    "$\\log_{{10}}$ 16-step open-loop rollout NMSE over $\\log_2 n$ divided by the $\\log_2 n$ range, differenced within "
    "each of the 16 paired seeds and averaged. LeWi wins {win} of the {n} comparisons, ties {tie} and loses {loss}. The "
    "colour scale is linear, symmetric about zero and clipped for display at $\\pm{clip:g}$: the {k} comparisons beyond "
    "it ({beyond}) are drawn at the end colour; no value is changed, and "
    "Figure~\\ref{{fig:world-model-grid-full}} prints them all. A hollow circle marks the {tie} comparisons whose paired "
    "95\\% Student-$t$ interval contains zero. Tasks are grouped by environment type (control, navigation, manipulation "
    "(Manip.) and a synthetic system (Synth.)) and ordered within each group by state dimension. Pendulum (DM): DeepMind Control Suite; Pendulum (Gym): Gymnasium.")


def draw(value: np.ndarray, verdict: np.ndarray, labels: list[str]) -> plt.Figure:
    rows, cols = value.shape
    body_w, body_h = cols * CELL_W, rows * CELL_H
    fig = plt.figure(figsize=(LEFT + body_w + RIGHT, HEAD + body_h + FOOT))
    width, height = fig.get_size_inches()
    ax = fig.add_axes([LEFT / width, FOOT / height, body_w / width, body_h / height])
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)                                                   # cell units, the first baseline on top
    ax.axis("off")
    norm = Normalize(-CLIP, CLIP, clip=True)
    for i in range(rows):
        for j in range(cols):
            assert not (abs(value[i, j]) > CLIP and verdict[i, j] == "tie"), "a clipped cell with an interval containing zero"
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=CMAP(norm(value[i, j])), edgecolor="none"))
            if verdict[i, j] == "tie":
                ax.plot(j + 0.5, i + 0.5, **CIRCLE)
    ax.hlines(range(1, rows), 0, cols, color="white", linewidth=SEAM)       # seams between the cells only
    ax.vlines(range(1, cols), 0, rows, color="white", linewidth=SEAM)
    start = 0
    for g, (name, tasks) in enumerate(GROUPS):
        if g:                                                              # the body's top edge to its bottom edge, no further
            ax.plot([start, start], [0, rows], color=SEPARATOR, linewidth=SEPARATOR_W, solid_capstyle="butt")
        ax.text(start + len(tasks) / 2, -0.3, SHORT[name], ha="center", va="bottom", fontsize=SMALL_SIZE, color=INK)
        start += len(tasks)
    for i, label in enumerate(labels):
        ax.text(-0.25, i + 0.5, label, ha="right", va="center", fontsize=NAME_SIZE, color=INK)
    for j, (_task, name) in enumerate(TASKS):
        ax.text(j + 0.7, rows + 0.25, name, rotation=ANGLE, rotation_mode="anchor", ha="right", va="top",
                fontsize=NAME_SIZE, color=INK)

    # the colour bar by hand, every part in the bar's own units: the ramp exactly as tall as the body, an arrow in the end
    # colour on a clipped side, one outline around both, the ramp's ends as lines of the outline's weight, inward ticks
    cax = fig.add_axes([(LEFT + body_w + BAR_GAP) / width, FOOT / height, BAR_W / width, body_h / height])
    cax.set_xlim(0, 1)
    cax.set_ylim(-CLIP, CLIP)
    cax.imshow(np.linspace(-CLIP, CLIP, 512)[:, None], aspect="auto", origin="lower", cmap=CMAP, norm=norm,
               extent=[0, 1, -CLIP, CLIP], interpolation="nearest", zorder=1)
    arrow = ARROW / body_h * 2 * CLIP                                      # the arrow's length in the bar's units
    low, high = bool((value < -CLIP).any()), bool((value > CLIP).any())
    outline = ([(0, -CLIP)] + ([(0.5, -CLIP - arrow)] if low else []) + [(1, -CLIP), (1, CLIP)]
               + ([(0.5, CLIP + arrow)] if high else []) + [(0, CLIP)])
    cax.add_patch(Polygon(outline, closed=True, fill=False, edgecolor=RULE, linewidth=OUTLINE_W, joinstyle="miter",
                          clip_on=False, zorder=3))
    for end in [e for e, clipped in ((-CLIP, low), (CLIP, high)) if clipped]:
        inner, tip = end - np.sign(end) * arrow, end + np.sign(end) * arrow  # the fill reaches under the ramp: no seam
        cax.add_patch(Polygon([(0, inner), (1, inner), (1, end), (0.5, tip), (0, end)], facecolor=CMAP(norm(end)),
                              edgecolor="none", clip_on=False, zorder=0.5))
        cax.plot([0, 1], [end, end], color=RULE, linewidth=OUTLINE_W, solid_capstyle="butt", clip_on=False, zorder=3)
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks([-CLIP, 0, CLIP], [number(t) for t in (-CLIP, 0, CLIP)])
    cax.tick_params(axis="y", direction="in", length=1.6, width=OUTLINE_W, color=RULE, pad=2.0, labelsize=SMALL_SIZE)
    cax.set_axisbelow(False)                                             # the zero tick above the ramp
    for spine in cax.spines.values():
        spine.set_visible(False)
    cax.yaxis.set_label_position("right")
    cax.set_ylabel(r"$\Delta$AULC", fontsize=SMALL_SIZE, labelpad=1.5)

    # the key of the hollow circle in the free corner under the bar, right-aligned with the bar's title
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    right = cax.yaxis.label.get_window_extent(renderer).x1 / fig.dpi
    key = fig.text(right / width, KEY_Y / height, "95% CI includes 0", ha="right", va="center", fontsize=SMALL_SIZE,
                   color=INK)
    circle_x = right - key.get_window_extent(renderer).width / fig.dpi - 0.06
    fig.add_artist(Line2D([circle_x / width], [KEY_Y / height], transform=fig.transFigure, **CIRCLE))  # moves with the tight crop
    return fig

def main() -> None:
    ps.apply_rc()
    aulc = load(_SUFFIX)
    if aulc is None:
        return
    if not (SEALED or TRIAL):
        meta = json.loads((DATA / "world_model_transfer_all/swm16.json").read_text())
        assert meta.get("in5_adopted") and meta.get("resel_adopted") and meta.get("lrx_adopted"), "paired.csv is not the adopted record: re-run harvest_swm16.py"
        if meta["resel_adopted"].get("pending"):
            print(f"[world_model_grid_compact] pending: {meta['resel_adopted']['pending']}")
    value, verdict, _counts = collect(aulc)
    v, d = value[MAIN_ROWS], verdict[MAIN_ROWS]
    a = np.sort(np.abs(v).ravel())
    inside = a[a <= CLIP]
    print(f"[world_model_grid_compact] |dAULC| median {np.median(a):.3f}, 95th percentile {np.quantile(a, 0.95):.3f}; "
          f"largest inside +-{CLIP:g}: {inside.max():.3f}; smallest beyond: {a[a > CLIP].min() if (a > CLIP).any() else float('nan'):.3f}")
    beyond = sorted(((v[i, j], BASELINES[MAIN_ROWS[i]][1], TASKS[j][1]) for i, j in zip(*np.nonzero(np.abs(v) > CLIP))), key=lambda x: abs(x[0]), reverse=True)
    save(draw(v, d, [BASELINES[k][1] for k in MAIN_ROWS]), "world_model_grid_compact" + _SUFFIX)
    fill = dict(n=v.size, clip=CLIP, k=len(beyond), **{w: int((d == w).sum()) for w in ("win", "tie", "loss")},
                beyond="; ".join(f"{b} on {t}, {number(x, '.2f')}" for x, b, t in beyond))
    print("[world_model_grid_compact] CAPTION:", CAPTION.format(**fill))
    (OUT / f"world_model_grid_compact{_SUFFIX}.caption.txt").write_text(CAPTION.format(**fill) + "\n")


if __name__ == "__main__":
    main()
