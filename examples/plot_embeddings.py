"""Learned neuron embeddings of the paper's LeWi networks, coloured by the target intermediate each neuron follows.

    python examples/plot_embeddings.py                         # the depth-4 tree network of Figure 19
    python examples/plot_embeddings.py --all                   # the seven networks of Figure 19
    python examples/plot_embeddings.py --workload reuse_dag --instance 6008 --n 32768 --out figures

Each network is the paper's selected LeWi run for a workload, target instance and training-set size (its configuration
and seeds from the records), trained again with the per-network trainer: in the records' environment the network is
the paper's bit for bit (about four minutes on one CPU core at n = 2^15). Hidden units are coloured by the intermediate
with which their activation has the largest absolute correlation on the diagnostic pool, provided it reaches the
permutation threshold (csfrbench.structure); units without an association are left out. Needs matplotlib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from csfrbench import record, runner, structure, suite, viz

#: the networks of Figure 19 (primary setting, n = 2^15)
FIGURE_19 = [("depth4_tree", 6002), ("fourier_tree", 6003), ("reuse_dag", 6008), ("fourier_reuse_dag", 6007),
             ("sampled_dags", 6013), ("flat_modules", 6003), ("high_arity_node", 6010)]


def network(workload: str, instance: int, n: int, setting: str = "primary"):
    """The recorded LeWi winner of a condition, trained again; returns (target, model at its selected checkpoint)."""
    rows = [w for w in record.winners(setting) if w["workload"] == workload and w["arm"] == "lewi"
            and w["instance"] == instance and w["n"] == n]
    if not rows:
        raise SystemExit(f"no recorded LeWi run for {setting} {workload} instance {instance} n {n}")
    unit = runner.units_of_winners(rows)[0]
    run = runner.execute(unit, keep_states=True)
    print(f"{workload} {instance} n = {n}: trial {unit['trial']}, test NMSE {run['test_nmse']:.4f} "
          f"(record {rows[0]['test_nmse']:.4f})", flush=True)
    return runner.model_of(run, run["best_state"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--workload", default="depth4_tree")
    p.add_argument("--instance", type=int, default=6002)
    p.add_argument("--n", type=int, default=32768)
    p.add_argument("--setting", default="primary")
    p.add_argument("--all", action="store_true", help="the seven networks of Figure 19")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for workload, instance in (FIGURE_19 if args.all else [(args.workload, args.instance)]):
        target, model = network(workload, instance, args.n, args.setting)
        corr = structure.correlations(model, target)
        assoc = structure.associations(model, target, corr=corr)
        title = f"{suite.workload(args.setting, workload).paper_name}, instance {instance}, n = {args.n}"
        fig = viz.plot_embeddings(model, target, assoc=assoc, title=title)
        stem = out / f"embeddings_{workload}_{instance}_n{args.n}"
        fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        print(f"  permutation threshold |r| = {corr['floor']:.3f}; associated hidden units per layer "
              f"{[int((a >= 0).sum()) for a in assoc[1:-1]]}; written {stem}.png", flush=True)


if __name__ == "__main__":
    main()
