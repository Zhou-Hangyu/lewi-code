"""Figure: the graph-oracle masking scheme, as the binary matrices it applies.

Caption - The graph-oracle masking scheme, exactly as applied. Each cell is
the fixed binary mask on one weight matrix of the common width-32,
four-hidden-layer backbone; dark entries are the weights the arm may
train, and the readout mask is the single row it is. (a)
Progressive structure supply on the primary depth-3 tree: GraphOracle(0),
the ladder's bottom rung, keeps only the eight active input columns and is
unrestricted above; GraphOracle(1) also groups the first hidden layer into
one block per level-1 function; GraphOracle(all) enforces the block pattern
through every level, merging blocks where the target merges, so the masks
are nested. (b) The full GraphOracle(all) mask across structural regimes: on
the reuse DAG the shared level-1 functions fan out to several level-2
blocks; on flat modules the mask stays block-diagonal at every depth,
modularity without hierarchy; on the high-arity node there is no compositional
structure to reveal, and GraphOracle(all) reduces to the support
restriction - the same mask as GraphOracle(0).

Both panels share the column structure, so the weight matrices are named
once, above panel (a); the rows are named in the margin by workload and
rung. The drawn instances come from the development bank, chosen for
legibility (contiguous block order on the tree, no adjacent kept input
columns).
Before drawing, the builder rebuilds the masks at the baseline audit's
instance (f7000) and hash-checks them against
``data/graph_oracle_masks/expected_sha.json``, the audit-verified values,
so the construction cannot drift from what the benchmark runs.
"""
from __future__ import annotations

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
from csfrbench.configs import resolve_cell
from csfrbench.pipeline import build_campaign_dict, compose_campaign
from csfrbench.methods.graph_oracle import lift_masks
from csfrbench.methods.maskladder import support_masks
from csfrbench.registry import get_task, load_builtin_registrations
from matplotlib.patches import Rectangle

from figstyle import DATA, save

INK = "#1a1a1a"                 # near-black charcoal, not pure black
# a small plate (decision, 2026-09-14): the matrices carry the pattern, not the page,
# and every label is large against them - 0.166 of a matrix's width for the column
# names, against 0.095 in the first build - so the figure reads at 0.8 of the text width
NAME_SIZE = 7.4                 # the column names, drawn once above panel (a)
ROW_SIZE = 8.2                  # the row's workload
SUB_SIZE = 6.8                  # the rung under it
TAG_SIZE = 9.0                  # the (a) / (b) tags
AUDIT_INSTANCE = 7000           # the baseline audit's hash-verified instance
PRIMARY = "tree_functional_anova"
REGIMES = (("Reuse DAG", "dag_functional_anova"),   # the paper's setting names (decision 2026-09-26)
           ("Flat modules", "flat_functional_anova"),
           ("High-arity node", "dense_fourier"))
# development-bank draws chosen for legibility: the tree's block order is
# contiguous (two diagonal blocks on h1 -> h2) and no instance keeps two
# adjacent input columns
FIG_SEED = {"tree_functional_anova": 6023, "dag_functional_anova": 6002,
            "flat_functional_anova": 6000, "dense_fourier": 6001}


def load_campaign():
    load_builtin_registrations()
    composed, config_dir = compose_campaign("v05")
    campaign = build_campaign_dict(composed, config_dir, "v05")
    return int(composed["d"][0]), campaign


def arch_of(task, campaign):
    arch = campaign["defaults"]["arch"]
    preset = campaign["tasks"][task]["preset"]
    override = (campaign.get("workload_overrides") or {}).get(
        f"{task}:{preset}", {}).get("arch", {})
    return (int(override.get("width", arch["width"])),
            int(override.get("hidden", arch["hidden"])))


def draw_target(task, seed, d, campaign):
    _, cell = resolve_cell({"task": task, "d": d}, campaign)
    return get_task(task).draw(cell, seed)


def mask_sha(masks) -> str:
    h = hashlib.sha256()
    for m in masks:
        a = np.asarray(m, dtype=np.float32)
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def audit_check(d, campaign) -> None:
    """The construction must still produce the audit-verified masks."""
    expected = json.loads(
        (DATA / "graph_oracle_masks" / "expected_sha.json").read_text())
    for key, want in expected.items():
        task, rung = key.split(":")
        width, hidden = arch_of(task, campaign)
        t = draw_target(task, AUDIT_INSTANCE, d, campaign)
        if rung == "go0":
            masks = support_masks(t, width, hidden)
        else:
            levels = "all" if rung == "all" else int(rung[1:])
            masks = lift_masks(t, width, hidden, levels=levels)
        if mask_sha(masks) != want:
            raise SystemExit(
                f"mask drift: {key} no longer matches the audit-verified "
                "value in data/graph_oracle_masks/expected_sha.json")


def _rects(m):
    """Maximal rectangles covering the kept cells, so the drawing is a
    handful of vector shapes with exact edges rather than a raster that
    PDF viewers smooth."""
    rows, cols = m.shape
    out = []
    r = 0
    while r < rows:
        r2 = r
        while r2 + 1 < rows and np.array_equal(m[r2 + 1], m[r]):
            r2 += 1
        c = 0
        while c < cols:
            if m[r, c]:
                c2 = c
                while c2 + 1 < cols and m[r, c2 + 1]:
                    c2 += 1
                out.append((c, r, c2 + 1 - c, r2 + 1 - r))
                c = c2 + 1
            else:
                c += 1
        r = r2 + 1
    return out


def _panel(ax, mask, rows_full: int):
    m = np.asarray(mask) > 0
    rows, cols = m.shape
    if rows < rows_full:
        # the 1 x 32 readout mask is drawn as the single row it is, at the
        # same cell scale as its neighbours and centred on the row
        pos = ax.get_position()
        height = pos.height * rows / rows_full
        ax.set_position([pos.x0, pos.y0 + (pos.height - height) / 2,
                         pos.width, height])
    for x, y, w, h in _rects(m):
        # the hairline same-color edge seals antialiasing seams between
        # abutting rectangles without changing their extent visibly
        ax.add_patch(Rectangle((x, y), w, h, facecolor=INK, edgecolor=INK,
                               linewidth=0.25, joinstyle="miter"))
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_edgecolor("#000000")
        side.set_linewidth(0.5)


def main() -> None:
    d, campaign = load_campaign()
    audit_check(d, campaign)

    width, hidden = arch_of(PRIMARY, campaign)
    t = draw_target(PRIMARY, FIG_SEED[PRIMARY], d, campaign)
    n_layers = hidden + 1

    ladder = [
        ("GraphOracle(0)", support_masks(t, width, hidden)),
        ("GraphOracle(1)", lift_masks(t, width, hidden, levels=1)),
        ("GraphOracle(all)", lift_masks(t, width, hidden, levels="all")),
    ]

    regimes = []
    for label, task in REGIMES:
        rw, rh = arch_of(task, campaign)
        rt = draw_target(task, FIG_SEED[task], d, campaign)
        masks = lift_masks(rt, rw, rh, levels="all")
        regimes.append((label, "GraphOracle(all)", masks))

    fig = plt.figure(figsize=(4.4, 4.5))
    grid = fig.add_gridspec(
        8, n_layers,
        height_ratios=[0.24, 1, 1, 1, 0.20, 1, 1, 1],
        wspace=0.10, hspace=0.08,
        left=0.235, right=0.995, top=0.975, bottom=0.012)

    def header(row: int):
        """The column names, once: both blocks share the column structure."""
        for column in range(n_layers):
            ax = fig.add_subplot(grid[row, column])
            ax.axis("off")
            name = ("Input → H1" if column == 0 else
                    f"H{hidden} → y" if column == n_layers - 1
                    else f"H{column} → H{column + 1}")
            ax.text(0.5, 0.10, name, ha="center", va="bottom", fontsize=NAME_SIZE,
                    color="#444")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    def block_tag(ax, tag: str):
        pos = ax.get_position()
        fig.text(0.012, pos.y1, tag, ha="left", va="top", fontsize=TAG_SIZE,
                 color="#222")

    def row_label(ax, main: str, sub: str):
        # in the figure margin, left of the mask grid
        pos = ax.get_position()
        yc = (pos.y0 + pos.y1) / 2
        lines = [(main, ROW_SIZE, "#333"), (sub, SUB_SIZE, "#999")]
        step = 0.031
        y = yc + step * (len(lines) - 1) / 2
        for text, size, color in lines:
            fig.text(0.223, y, text, ha="right", va="center",
                     fontsize=size, color=color)
            y -= step

    # ------------------------------------------------ (a) tree, the ladder
    header(0)
    for row, (rung, masks) in enumerate(ladder, start=1):
        for column, mask in enumerate(masks):
            ax = fig.add_subplot(grid[row, column])
            _panel(ax, mask, width)
            if column == 0:
                row_label(ax, "Depth-4 tree", rung)
                if row == 1:
                    block_tag(ax, "(a)")

    # ------------------------------- (b) the full oracle across the regimes
    for row, (main, sub, masks) in enumerate(regimes, start=5):
        for column, mask in enumerate(masks):
            ax = fig.add_subplot(grid[row, column])
            _panel(ax, mask, width)
            if column == 0:
                row_label(ax, main, sub)
                if row == 5:
                    block_tag(ax, "(b)")

    save(fig, "graph_oracle_masks")


if __name__ == "__main__":
    main()
