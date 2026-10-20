"""The computational cost of LeWi and Dense (the paper's Table 6): parameters and seconds per trained network.

    python tools/timing.py --backbone 32x4          # depth-4 tree, instance 6000, every size and trial: 224 + 224 networks
    python tools/timing.py --backbone 64x8          # depth-5 tree, instance 6000 (64x8, 128,000 steps)

The protocol of Table 6: both methods train the same cells (one target instance, every training-set size, the 32
configurations of each) with the stacked engine, each in stacks of its own optimal size, all stacks started at the same
moment as separate single-thread processes on one node. The runtime of a trained network is the wall time of its stack
(training, validation and testing; the build and the compilation included) divided by the stack's size; the figure
of a method is the total over its stacks divided by its number of networks. Absolute times depend on the processor
and on what else runs on the node; the ratio LeWi / Dense is the quantity to compare (1.32 at 32x4 and 1.02 at 64x8 in
the paper, measured on a node shared with other stacks).
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context

from csfrbench import runner, stack

CELLS = {"32x4": ("primary", "depth4_tree"), "64x8": ("primary", "depth5_tree")}
#: the stack sizes of the paper's measurement: 128 networks per stack for the four smallest sizes and 96 for the three
#: largest at 32x4, one stack of 32 networks per size at 64x8
SPLIT = {"32x4": ((512, 1024, 2048, 4096), (8192, 16384, 32768)), "64x8": None}


def _stack(units: list[dict]) -> dict:
    import torch
    torch.set_num_threads(1)
    start = time.perf_counter()
    records = stack.train_stack(units)
    wall = time.perf_counter() - start
    return {"arm": units[0]["arm"], "networks": len(units), "wall_s": wall, "n_params": records[0]["n_params"],
            "engine": records[0]["engine"], "diverged": sum(r["diverged"] for r in records)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backbone", choices=sorted(CELLS), default="32x4")
    p.add_argument("--instance", type=int, default=6000)
    p.add_argument("--out", help="write the result to this JSON file")
    args = p.parse_args()
    setting, workload = CELLS[args.backbone]
    stacks = []
    for arm in ("lewi", "dense"):
        units = runner.plan(setting, [workload], [arm], [args.instance])
        if SPLIT[args.backbone]:
            for sizes in SPLIT[args.backbone]:
                stacks.append([u for u in units if u["n"] in sizes])
        else:
            for n in sorted({u["n"] for u in units}):
                stacks.append([u for u in units if u["n"] == n])
    print(f"{len(stacks)} stacks started together, sizes {[len(s) for s in stacks]}", flush=True)
    with get_context("spawn").Pool(len(stacks)) as pool:
        results = pool.map(_stack, stacks)
    table = {}
    for arm in ("dense", "lewi"):
        mine = [r for r in results if r["arm"] == arm]
        networks = sum(r["networks"] for r in mine)
        table[arm] = {"params": mine[0]["n_params"], "seconds_per_network": sum(r["wall_s"] for r in mine) / networks,
                      "networks": networks, "stacks": len(mine), "engine": mine[0]["engine"],
                      "diverged": sum(r["diverged"] for r in mine)}
    ratio = table["lewi"]["seconds_per_network"] / table["dense"]["seconds_per_network"]
    print(f"\n{args.backbone}: parameters Dense {table['dense']['params']:,}, LeWi {table['lewi']['params']:,}; runtime per "
          f"trained network Dense {table['dense']['seconds_per_network']:.1f} s, LeWi "
          f"{table['lewi']['seconds_per_network']:.1f} s; ratio {ratio:.2f}")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"backbone": args.backbone, "instance": args.instance, "ratio": ratio, "methods": table,
                       "stacks": results}, handle, indent=1)


if __name__ == "__main__":
    main()
