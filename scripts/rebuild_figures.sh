#!/bin/bash
# Rebuild the paper's experimental figures and tables from the packaged records, under
# studies/communication/paper/figures/out-paper/{d8,d32} (PDF, PNG and SVG per figure; .md/.tex/.csv per table).
#
#     bash scripts/rebuild_figures.sh            every figure and table listed in REPRODUCE.md
#     bash scripts/rebuild_figures.sh STEM ...   only the named stems of the d = 8 set (see figures/build.py)
#
# The builders take three environment switches: CSFR_FIGURES_DATA (the data folder), CSFR_FIGURES_OUT (the output
# folder) and CSFR_FIGURES_LEWI (the LeWi arm key). The learned-embedding strips come from the design builder under
# figures/design/teaser-paper and are written through the same switches.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
FIG=$ROOT/studies/communication/paper/figures
PY=${PYTHON:-python}
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 CSFR_FIGURES_LEWI=lewi-v1-rgm
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
cd "$FIG"
mkdir -p out-paper/d8 out-paper/d32 design/teaser-paper/out
if [ $# -gt 0 ]; then
    CSFR_FIGURES_DATA=data-paper-d8 CSFR_FIGURES_OUT=out-paper/d8 $PY build.py "$@"
    exit 0
fi
echo "== Figure 19 (embedding strips per setting; the Fourier-tree panel is Figure 1's network): design/teaser-paper =="
(cd design/teaser-paper && CSFR_FIGURES_DATA=data-paper-d8 CSFR_FIGURES_OUT=out-paper/d8 $PY build.py figure)
echo "== Figures 3-14, 17-23 and Tables 3-5, 7 (d = 8 set) =="
CSFR_FIGURES_DATA=data-paper-d8 CSFR_FIGURES_OUT=out-paper/d8 $PY build.py
echo "== Figures 15 and 16 (d = 32 set) =="
CSFR_FIGURES_DATA=data-paper-d32 CSFR_FIGURES_OUT=out-paper/d32 $PY build.py learning_curves_all sample_efficiency_aulc
echo "REBUILD_OK"
