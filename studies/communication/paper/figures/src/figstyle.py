"""Shared style and output rules for the paper figures.

One matplotlib figure produces the three formats the paper needs: PDF for
LaTeX in out/, SVG for the web report and slides in out/svg/, PNG for quick
viewing in out/png/ (design note, 2026-09-04; the tables and verification
notes stay beside the PDFs in out/). Text is
embedded as TrueType in PDF (venues reject Type 3) and as outlines in SVG, so
a figure renders identically wherever it is opened, and the output carries no
build timestamp, so an unchanged figure rebuilds byte-identical.

Three environment switches select the wave (13-main runner, section 7.5):
CSFR_FIGURES_OUT (the output folder under figures/, default ``out``; wave 2
writes ``out-v2``), CSFR_FIGURES_DATA (the data folder, default ``data``;
wave 2 reads ``data-v2``) and, in paperstyle, CSFR_FIGURES_LEWI (the LeWi
arm key, default ``lewi-v05``; wave 2 ``lewi-v1``). The data folder names
the campaign whose records it holds (CAMPAIGN below), so the harvesters and
the campaign-pinned builders follow the same switch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parents[1]
OUT = FIGURES / os.environ.get("CSFR_FIGURES_OUT", "out")
DATA = FIGURES / os.environ.get("CSFR_FIGURES_DATA", "data")
# the campaign whose records a data folder holds: data/ = 08-main (CSFRBench
# v0.5, LeWi-v05), data-v2/ = 13-main (CSFRBench v1.0, LeWi-v1),
# data-main-14/ = 14-main (CSFRBench v1.1, LeWi-v1 with the structural readout)
CAMPAIGN_OF = {"data": "08-main", "data-v2": "13-main", "data-main-14": "14-main",
               "data-final": "08-main",    # the final arm (12-extra wave 5) over the sealed 08-main baselines
               # the paper's setting at 64,000 steps (18-main-d8, 2026-09-17): the 08-main figure schema, every
               # row of the seven d = 32 workloads from that campaign, the depth-5 tree and the studies not
               # rerun kept from data-final (src/harvest_clean.py)
               "data-final-clean": "08-main",
               # the all-active twin of that setting (d = 8, 64,000 steps; 17-main-clean panel main8b read through
               # 18-main-d8): the same schema, rendered into a folder inside out-final-clean
               "data-final-clean-d8": "08-main",
               # the 8-input suite at d = 8 on the record's own instances (7000-7015, 7000-7031 on the random graphs;
               # 17-main-clean panel main8r read through 18-main-d8, decision 2026-09-20): the same schema
               "data-final-clean-7000": "08-main",
               # the nuisance-input setting on the twin's instances (d = 32 on 6000-6015 / 6020-6035; 17-main-clean panels main32s + main32t
               # read through 18-main-d8, decision 2026-09-20): the same schema
               "data-final-clean-6000": "08-main",
               # the paper's final figure set (decision 2026-09-22): the d = 8 setting and the d = 32 setting on the 6000 seeds, each with its
               # depth-5 tree, every LeWi under the input rule in5 and the readout rule, rendered into out-paper/d8 and out-paper/d32
               "data-paper-d8": "08-main", "data-paper-d32": "08-main"}
# the output folder each data folder renders into (build.py refuses any other pairing)
OUT_OF = {"data": "out", "data-v2": "out-v2", "data-main-14": "out-main-14", "data-final": "out-final",
          "data-final-clean": "out-final-clean", "data-final-clean-d8": "out-final-clean/d8-64k", "data-final-clean-7000": "out-final-clean-7000",
          "data-final-clean-6000": "out-final-clean-6000", "data-paper-d8": "out-paper/d8", "data-paper-d32": "out-paper/d32"}
if DATA.name not in CAMPAIGN_OF:
    raise SystemExit(f"[figstyle] CSFR_FIGURES_DATA must be one of {sorted(CAMPAIGN_OF)}, not {DATA.name!r}")
CAMPAIGN = FIGURES.parents[3] / "studies/main-experiments" / CAMPAIGN_OF[DATA.name]

PNG_DPI = 400

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 9.5,
    # Times-compatible type for every figure (decision, 2026-09-04): STIXGeneral
    # for text, the TrueType family matplotlib bundles, and the STIX math set
    # for LaTeX math; both embed as Type 42. Set here, not in paperstyle, so
    # the builders that never call apply_rc get it too
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "path",
    "svg.hashsalt": "paper-figures",
})


# where each format lands: the PDF beside the tables and notes, the two
# viewing formats in folders of their own
FOLDER = {"pdf": OUT, "svg": OUT / "svg", "png": OUT / "png"}


def output(stem: str, ext: str) -> Path:
    return FOLDER[ext] / f"{stem}.{ext}"


def save(fig, stem: str) -> None:
    for folder in FOLDER.values():
        folder.mkdir(parents=True, exist_ok=True)
    # no build timestamp in the output: rebuilding an unchanged figure leaves
    # the files byte-identical, so a diff between two builds is a real change
    for ext, options in (("pdf", {"metadata": {"CreationDate": None}}),
                         ("svg", {"metadata": {"Date": None}}),
                         ("png", {"dpi": PNG_DPI})):
        fig.savefig(output(stem, ext), bbox_inches="tight", **options)
    plt.close(fig)
    # the curve standard's marked means (sample_efficiency.MARKS: a mean above its panel's frame is drawn as a caret):
    # one table per stem beside the figures, rewritten on every build, removed when a build marks nothing
    se = sys.modules.get("sample_efficiency")
    if se is not None and hasattr(se, "MARKS"):
        marks = OUT / "marks" / f"{stem}.csv"
        if se.MARKS:
            import csv
            marks.parent.mkdir(parents=True, exist_ok=True)
            with marks.open("w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(se.MARKS[0]))
                wr.writeheader()
                wr.writerows(se.MARKS)
            se.MARKS.clear()
        elif marks.exists():
            marks.unlink()
    sizes = " ".join(f"{ext} {output(stem, ext).stat().st_size / 1e3:.0f}K" for ext in FOLDER)
    print(f"[fig] {stem}  {sizes}")
