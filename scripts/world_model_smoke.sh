#!/bin/bash
# World-model smoke test (needs the second environment, environment/requirements-worldmodel.txt):
#   1. collect a tiny corpus of the Pendulum task with the paper's collector (deterministic from SEED0 = 20260831);
#   2. build a cell from it and train LeWi and Dense at 256 x 4 for 200 steps (a smoke value, never a record);
#   3. re-run one final unit of the record as run's Pendulum cell with the step budget capped at 100 (SWM16_SMOKE_STEPS)
#      when the record's corpus pendulum_xl.npz has been regenerated (scripts below print the command).
#
#     bash scripts/world_model_smoke.sh            (PYTHON=... selects the interpreter of the world-model environment)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PYTHON:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=${MUJOCO_GL:-egl} PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
OUT=${SMOKE_OUT:-$ROOT/results/smoke_wm}
mkdir -p "$OUT"
$PY - <<'PY'
import os, sys, time
from pathlib import Path
ROOT = Path(os.getcwd())
sys.path.insert(0, str(ROOT / "studies/main-experiments/16-main-final/src"))
t0 = time.time()
import swm16
lewibench, swm_tasks, v05_port = swm16.harness()          # the package pin, the fixed hyperparameters, the engine
out = Path(os.environ["SMOKE_OUT"]) if os.environ.get("SMOKE_OUT") else ROOT / "results/smoke_wm"
swm_tasks.DATA = out / "data"
swm_tasks.DATA.mkdir(parents=True, exist_ok=True)
print(f"harness imported in {time.time() - t0:.0f} s; csfrbench from {sys.modules['csfrbench'].__file__}")
# 1. a tiny corpus (the collector is deterministic from SEED0; the same call twice gives the same arrays)
swm_tasks.collect("pendulum", 8, 100, "pendulum_smoke")
print("corpus:", sorted(p.name for p in swm_tasks.DATA.iterdir()))
# 2. one cell and two short trainings
cell = lewibench.make_cell_data("pendulum", 8192, "pendulum_smoke")
print("cell: n_actual", cell["n_actual"], "pool", cell["pool_size"], "split_hash", swm16.split_hash(cell))
for arm, hp in (("lewi-v1", {"lr": 0.0012}), ("dense", {"lr": 0.003})):
    seeds = lewibench.final_seeds("pendulum", 8192, 0)
    t1 = time.time()
    result, model, _ = lewibench.train_one("pendulum", 8192, arm, hp, seeds, width=256, hidden=4, batch=lewibench.BATCH, max_steps=200, cell=cell, corpus_name="pendulum_smoke")
    print(f"{arm}: {time.time() - t1:.1f} s, best_val_nmse {result.get('best_val_nmse')}, params {result.get('params')}")
print("WORLD_MODEL_SMOKE_OK")
print("To re-run a record unit: regenerate a corpus (python studies/main-experiments/09-swm-lewi/src/swm_tasks.py --task pendulum --eps 8192 --horizon 128 --name pendulum_xl),")
print("then SWM16_SMOKE_STEPS=100 python studies/main-experiments/16-main-final/src/swm16.py one lewi-v1  (the first pending unit of that arm; see the README)")
PY
