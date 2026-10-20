#!/usr/bin/env python3
"""Train one CSFRBench unit of the paper's harness on this machine (no scheduler).

    python scripts/train_unit.py --cell tree8 --arm lewi-v1-rgm --method lewi-v1-rgm-in5 --fs 6000 --n 512 --trial 0 --steps 300
    python scripts/train_unit.py --panel main32s --cell tree --arm noreg --fs 6003 --n 4096 --trial 7

The unit is built as the campaign built it: the harness composes the campaign (m17.campaign), draws the trial's
hyperparameters from the arm's frozen space (m17.trials; --lr replaces the draw), resolves the run specification
(csfrbench.configs.resolve_run_spec), builds the target instance's dataset if it is not there yet
(csfrbench.pipeline.ensure_campaign_datasets: about 30 s and 9 MB per instance at d = 8) and trains with
csfrbench.train.execute_experiment_run. --steps caps the training budget for a smoke test (never for a record; the paper's
budgets are 64,000 steps for the 32x4 backbone and 128,000 for the depth-5 tree). --method overrides the registry id the
panel would use: the paper's LeWi arm is `lewi-v1-rgm-in5` (the harness selects it for the d = 32 panels and the repaired
d = 8 cells; every other d = 8 LeWi cell is the record's `lewi-v1-rgm`, identical by construction where the rule is idle).

Panels (m17.PANELS): main8b (d = 8, cells tree8 dag8 ftree8 flat8 dense8 fdag8 rand8), main32s (d = 32, cells tree dag ftree
flat dense fdag rand), main32t (depth-5 tree at d = 32, cell tree5x32), main (depth-5 tree at d = 16, cell tree16), the
ablation panels abl8-in5 ..., lms8 / lms5, sens8rad / sens8init. Arm keys: noreg (Dense), l1, l2, l0hc, rigl, bimt, fiedler,
erb_positions, lewi-v1-rgm, go1, go2, graph_oracle, support_oracle (GO(0), d = 32 only).

Outputs in <out>/<unit>/: spec.json, init.pt, best.pt, run.json (metrics and the validation curve); the dataset under
<datasets>/n<n_max>/<target fingerprint>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M17 = ROOT / "studies/main-experiments/17-main-clean"
for p in (ROOT / "src", M17 / "src", ROOT / "studies/main-experiments/12-extra/src", ROOT / "studies/main-experiments/18-main-d8/input-cutoff/src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default="main8b")
    ap.add_argument("--cell", default="tree8")
    ap.add_argument("--arm", default="lewi-v1-rgm")
    ap.add_argument("--method", default=None, help="registry id to train (default: the panel's for this arm)")
    ap.add_argument("--fs", type=int, default=6000, help="target instance (function seed = dataset seed)")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--trial", type=int, default=0, help="index of the scrambled-Sobol draw in the frozen space (0..31)")
    ap.add_argument("--lr", type=float, default=None, help="use this learning rate instead of the trial's draw")
    ap.add_argument("--steps", type=int, default=None, help="cap the training budget (smoke tests only)")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "results/units")
    ap.add_argument("--datasets", type=Path, default=ROOT / "results/datasets")
    a = ap.parse_args(argv)

    t0 = time.time()
    import torch
    torch.set_num_threads(1)
    import m17_tasks  # noqa: F401
    import w5_methods  # noqa: F401
    import cutoff_methods  # noqa: F401
    import abl_methods, abl_in5_methods, lms_methods  # noqa: F401,E401
    import m17 as M
    from csfrbench.configs import resolve_run_spec
    from csfrbench.pipeline import ensure_campaign_datasets
    from csfrbench.train import execute_experiment_run

    camp = M.campaign(a.panel)
    spaces = M.PANELS[a.panel].get("spaces") or M.PANELS[a.panel].get("space_set")
    if a.lr is not None:
        hp = {"lr": a.lr}
    else:
        hp = M.trials(a.arm, a.cell, a.n, spaces)[a.trial]
    raw = M._raw(a.panel, a.cell, a.arm, a.fs, a.n, hp, a.trial)
    if a.method:
        raw["method"] = a.method
    ds_root = a.datasets / f"n{camp['defaults']['n_max']}"
    ensure_campaign_datasets(camp, [raw], ds_root)
    spec = resolve_run_spec(raw, camp)
    spec["torch_threads"] = 1
    spec["retain"] = {"init_pt": True, "best_pt": True}
    if a.steps or a.eval_every:
        val = dict(spec["training"]["validation"])
        if a.eval_every:
            val["eval_every"] = a.eval_every
        elif a.steps:
            val["eval_every"] = max(1, min(val.get("eval_every", 1000), a.steps // 3))
        spec["training"] = {**spec["training"], **({"max_steps": a.steps} if a.steps else {}), "validation": val}
    run_dir = a.out / M.unit_name(a.cell, raw["method"], a.fs, a.n, a.trial)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    execute_experiment_run(spec, run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    m = run.get("metrics", {})
    print(json.dumps({"unit": run_dir.name, "method": raw["method"], "hp": raw["hp"], "steps": spec["training"]["max_steps"],
                      "best_step": m.get("best_step"), "best_val_nmse": m.get("best_val_nmse"), "test_nmse": m.get("test_nmse"),
                      "seconds": round(time.time() - t0, 1), "run_dir": str(run_dir)}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
