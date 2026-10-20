#!/bin/bash
# Smoke test of the archive (a few minutes on one core; no scheduler; no network after the install):
#   1. the benchmark package and its registry import;
#   2. one training unit of the paper's LeWi arm and one Dense unit on a fresh target instance, at a reduced step budget;
#   3. the quick record checks (the winner rule and the headline numbers recomputed from the packaged records);
#   4. one figure (Figure 4) rebuilt from the packaged records.
#
#     bash scripts/smoke_test.sh            (PYTHON=... selects the interpreter; SMOKE_OUT=... the output folder)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PYTHON:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
OUT=${SMOKE_OUT:-$ROOT/results/smoke}
echo "[1/4] imports"
PYTHONPATH=src $PY -c "import csfrbench, csfrbench.methods, csfrbench.tasks; from csfrbench.registry import METHODS, TASKS; print('methods', len(METHODS), 'task families', len(TASKS))"
echo "[2/4] two training units (300 steps each; the dataset of instance 6000 is generated on first use)"
$PY scripts/train_unit.py --cell tree8 --arm lewi-v1-rgm --method lewi-v1-rgm-in5 --fs 6000 --n 512 --trial 0 --steps 300 --out "$OUT"
$PY scripts/train_unit.py --cell tree8 --arm noreg --fs 6000 --n 512 --trial 0 --steps 300 --out "$OUT"
echo "[3/4] record checks"
$PY scripts/verify_results.py --quick
echo "[4/4] one figure from the packaged records"
bash scripts/rebuild_figures.sh sample_efficiency
test -s studies/communication/paper/figures/out-paper/d8/sample_efficiency.pdf
echo "SMOKE_OK"
