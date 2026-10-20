"""Representation diagnostics of the paper's selected networks (Figure 5): linear probes and counterfactual interventions.

    python examples/representations.py                          # depth-4 tree, n = 2^14, LeWi / Dense / GraphOracle(all), probes
    python examples/representations.py --das --workers 16       # with joint Boundless DAS (about 15 minutes per network)
    python examples/representations.py --instances 6000 6001    # a quick look at two instances

The selected run of every method and target instance is trained again (``csfrbench run``, states kept), then probed:
the held-out R^2 of a ridge probe predicting each target intermediate from each hidden layer (csfrbench.probes) and,
with ``--das``, the held-out counterfactual R^2 of joint multi-variable Boundless DAS (csfrbench.das). The table gives
the mean over the instances, rows grouped by target level as in Figure 5.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from csfrbench import das, datasets, probes, record, runner, suite


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--workload", default="depth4_tree")
    p.add_argument("--n", type=int, default=16384)
    p.add_argument("--arms", nargs="+", default=["lewi", "dense", "go_all"])
    p.add_argument("--instances", nargs="*", type=int)
    p.add_argument("--das", action="store_true", help="also the counterfactual interventions (slow)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--engine", default="auto", choices=["auto", "exact"])
    p.add_argument("--out", default="runs/representations")
    args = p.parse_args()
    torch.set_num_threads(1)
    w = suite.workload("primary", args.workload)
    instances = args.instances or list(w.instances)
    winners = [r for r in record.winners("primary") if r["workload"] == args.workload and r["arm"] in args.arms
               and r["n"] == args.n and r["instance"] in instances]
    units = runner.units_of_winners(winners)
    print(runner.run(units, args.out, workers=args.workers, keep_states=True, engine=args.engine))
    runs = {r["unit"]: r for r in runner.read_runs(args.out)}
    tables: dict[str, dict[str, list]] = {arm: {"probe": [], "das": []} for arm in args.arms}
    view = None
    for u in units:
        run = runs[runner.unit_id(u)]
        state = torch.load(Path(args.out) / "states" / f"{run['unit']}.pt", map_location="cpu", weights_only=True)
        target, model = runner.model_of(run, state["best"])
        X = datasets.make_dataset(target, 32, pools=("diagnostic",))["X_diagnostic"]
        result = probes.probe(model, target, X)
        view = view or result
        tables[u["arm"]]["probe"].append(np.asarray(result["r2"]))
        if args.das:
            chosen = das.score(model, target)["variables"]
            das_names = [c["var"] for c in chosen]
            tables[u["arm"]]["das"].append(np.asarray([c["test_r2"] for c in chosen]))
    levels = view["levels"]
    order = sorted(range(len(levels)), key=lambda v: (-levels[v], v))
    print(f"\nlinear probe R^2 (rows: target intermediates by level; columns: input, hidden layer 1, 2, ...), "
          f"{w.paper_name}, n = {args.n}, mean over {len(instances)} instance(s)")
    for arm in args.arms:
        mean = np.mean(tables[arm]["probe"], axis=0)
        print(f"  {suite.ARMS[arm].paper_name}")
        for v in order:
            print(f"    level {levels[v]}  {view['intermediates'][v]:>10s}  " + "  ".join(f"{x:+.2f}" for x in mean[v]))
    if args.das:
        print(f"\ncounterfactual R^2 of joint Boundless DAS at the selected layer, mean over {len(instances)} instance(s)")
        for arm in args.arms:
            mean = np.mean(tables[arm]["das"], axis=0)
            print(f"  {suite.ARMS[arm].paper_name}: " + "  ".join(f"{n} {x:+.2f}" for n, x in zip(das_names, mean)))


if __name__ == "__main__":
    main()
