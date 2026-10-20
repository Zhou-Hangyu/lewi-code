"""A new method on CSFRBench, from its definition to the comparison with the recorded methods.

    python examples/new_method.py space      1. make the search space on the development instances
    python examples/new_method.py run        2. train 32 configurations per condition
    python examples/new_method.py compare    3. AULC beside the recorded methods

The example uses one workload and two target instances: 384 runs for the search space and 448 runs for the
evaluation, about 5 hours on eight cores. A complete evaluation uses every target instance of the workload.

    python examples/new_method.py smoke      all three steps with 1,000 training steps per run (about 5 minutes):
                                             a test of the pipeline; its numbers are not results
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from csfrbench import analysis, record, runner, search, suite
from csfrbench.seeds import derive
from csfrbench.methods import Method, register
from csfrbench.models import DenseMLP

OUT = Path("runs/group_lasso")
WORKLOAD = suite.workload("primary", "depth4_tree")
INSTANCES = (6000, 6001)
WORKERS = 8
STOP = None             # the smoke test sets it to 1: every run ends at its first validation checkpoint


@register("group_lasso")
class GroupLasso(Method):
    """A penalty on the norm of the incoming weights of each hidden unit."""

    hp_keys = frozenset({"lam"})

    def build(self, cfg, target):
        return DenseMLP(target.d, cfg["width"], cfg["hidden"])

    def penalty(self, model, hp):
        norms = [layer.weight.norm(dim=1) for layer in model.hidden_layers]
        return float(hp["lam"]) * torch.cat(norms).mean()


# A broad starting space. `hard: ["low"]` keeps the lower boundary of the penalty strength.
START = [{"name": "lam", "low": 1e-7, "high": 1e2, "log": True, "hard": ["low"]},
         {"name": "lr", "low": 1e-5, "high": 1.0, "log": True, "hard": []}]


def units(instances, sizes, space, tag):
    out = []
    for n in sizes:
        seed = derive(suite.SEARCH_SEED, "example", tag, "group_lasso", WORKLOAD.family, WORKLOAD.preset, int(n))
        for trial, hp in enumerate(search.sample(space, suite.TRIALS, seed)):
            for instance in instances:
                out.append({"setting": "primary", "workload": WORKLOAD.name, "arm": "group_lasso",
                            "method": "group_lasso", "instance": instance, "n": n, "trial": trial, "hp": hp})
    return out


def space() -> None:
    sizes = (WORKLOAD.profile.sizes[0], WORKLOAD.profile.sizes[3], WORKLOAD.profile.sizes[-1])
    panel = units(suite.DEVELOPMENT_INSTANCES, sizes, START, "start")
    print(runner.run(panel, OUT / "panel", workers=WORKERS, stop_after_evals=STOP, progress=False))
    selected = [w["hp"] for w in analysis.select_winners(runner.read_runs(OUT / "panel"))]
    frozen = search.freeze(START, selected)
    (OUT / "space.json").write_text(json.dumps(frozen, indent=1) + "\n")
    print(json.dumps(frozen, indent=1))


def run() -> None:
    frozen = json.loads((OUT / "space.json").read_text())
    todo = units(INSTANCES, WORKLOAD.profile.sizes, frozen, "frozen")
    print(runner.run(todo, OUT / "evaluation", workers=WORKERS, stop_after_evals=STOP, progress=False))


def compare() -> None:
    mine = analysis.select_winners(runner.read_runs(OUT / "evaluation"))
    recorded = [r for r in record.winners("primary") if r["workload"] == WORKLOAD.name and r["instance"] in INSTANCES]
    inst = analysis.instance_aulc(mine + recorded, {WORKLOAD.name: WORKLOAD.profile.sizes})
    for row in sorted(analysis.aulc_table(inst), key=lambda r: r["mean"]):
        print(f"{row['arm']:14s} AULC {row['mean']:.3f} ± {row['half_width']:.3f}   ({row['n']} instances)")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if step == "smoke":
        OUT, STOP = Path("runs/group_lasso_smoke"), 1
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (("space", "run", "compare") if step == "smoke" else (step,)):
        {"space": space, "run": run, "compare": compare}[name]()
