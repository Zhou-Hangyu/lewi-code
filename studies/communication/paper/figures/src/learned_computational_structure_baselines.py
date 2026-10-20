"""Figure (appendix): learned_computational_structure with its second row of curves dropped and three more arms in its place
(decision 2026-09-22: "without the line plots, instead put three more heatmaps for three other representative baselines"; "drop the
second row completely").

The two blocks stand side by side as in the main figure - linear probes on the depth-4 tree (left), causal interventions (right).
Each block is now two rows of three heatmaps: Dense, LeWi and GraphOracle(all) above, and the strongest learned baseline of each
family below - L0 (hard-concrete gates), RigL (dynamic sparse training) and BIMT (geometric modularity); the same three in both
blocks. A heatmap gives the held-out R^2 (linear: the ridge probe; causal: the patched counterfactual R^2 of the joint DAS fit, on
the diverging scale centred at 0) of every target intermediate (rows, grouped by target depth and named once per row on the far
left) at every hidden layer (columns); every cell is the mean over the 16 target instances of the validation-selected winner run.
Data: as learned_computational_structure (the baselines' probes are the study's, their joint fits were run for the paper set);
n = 2^15 here, 2^14 in the sibling stem. Every heatmap has the main figure's size (learned_computational_structure.heatmap_inches,
0.689 x 0.859 in; decision 2026-09-22: the same aspect ratio as the main figure), the two rows ROWS["gap"] apart, the margins of the
compact layout. Output: <stem>.{pdf, svg, png} and the cell means in <stem>.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import learned_computational_structure as L
import paperstyle as ps
from figstyle import OUT, save
from learned_structure import ARMS, N_SHOWN

BASELINES = [("l0hc", ps.NAME["l0hc"]), ("rigl", ps.NAME["rigl"]), ("bimt", ps.NAME["bimt"])]
ARMS6 = list(ARMS) + BASELINES
ROWS = dict(gap=0.40, top=L.COMPACT["top"], bottom=L.COMPACT["bottom"])   # inches: between the rows, the page margins


def main(n: int = N_SHOWN, stem: str = "learned_computational_structure_baselines") -> None:
    ps.apply_rc()
    pv, cells = L.load(n)
    blocks = {"linear": (pv, "r2_trained", L.PROBE_LAYERS, L.SEQUENTIAL, L.LINEAR_NORM),
              "causal": (cells, "test_r2", L.DAS_LAYERS, L.DIVERGING, L.CAUSAL_NORM)}
    for block, (df, *_rest) in blocks.items():
        missing = [m for m, _ in ARMS6 if df[df.method == m].empty]
        if missing:
            raise SystemExit(f"[{stem}] no {block} rows at n = {n} for {missing}: run the fits and the harvest first")
    rows_of = [list(ARMS), BASELINES]
    records = []
    hw, hh = L.heatmap_inches()                # the width follows from L.BLOCKS at the text width, the height sets the page
    height = ROWS["top"] + 2 * hh + ROWS["gap"] + ROWS["bottom"]
    fig = plt.figure(figsize=(L.TWO_ROW["figsize"][0], height))
    for block, (df, value, layers, cmap, norm) in blocks.items():
        left, right = L.BLOCKS[block]
        gs = GridSpec(2, 3, figure=fig, left=left, right=right, top=1 - ROWS["top"] / height,
                      bottom=ROWS["bottom"] / height, wspace=L.WSPACE, hspace=ROWS["gap"] / hh)
        axes = [[fig.add_subplot(gs[r, k]) for k in range(3)] for r in range(2)]
        for r, arms in enumerate(rows_of):
            for k, (method, title) in enumerate(arms):
                table = L.cell_table(df[df.method == method], value, layers)
                L.draw_heatmap(axes[r][k], table, cmap, norm, title=title, row_labels=block == "linear" and k == 0,
                               ylabel=False, group_rows=True)
                for num, row in table.iterrows():
                    for layer, v in row.items():
                        records.append((block, method, f"intermediate {num}", str(layer), v))
        x0, x1 = axes[0][0].get_position().x0, axes[0][-1].get_position().x1
        fig.text((x0 + x1) / 2, axes[0][0].get_position().y1 + L.HEADER_GAP / height, L.HEADER[block], ha="center", va="bottom",
                 fontsize=7.6, color="#3a3a3a")
    y_mid = (axes[0][0].get_position().y1 + axes[1][0].get_position().y0) / 2
    fig.text(L.BLOCKS["linear"][0] - 0.092, y_mid, "Target intermediate", rotation=90, ha="center", va="center", fontsize=L.LABEL_SIZE)
    fig.text((L.BLOCKS["linear"][0] + L.BLOCKS["causal"][1]) / 2, 0.012, "Hidden layer", ha="center", va="bottom", fontsize=L.LABEL_SIZE)
    save(fig, stem)
    pd.DataFrame(records, columns=["block", "method", "series", "layer", "value"]).to_csv(OUT / f"{stem}.csv", index=False)


if __name__ == "__main__":
    main()
