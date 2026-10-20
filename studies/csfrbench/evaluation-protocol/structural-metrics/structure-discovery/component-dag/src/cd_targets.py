"""Per-instance target cache: the computation graph and its node values on the
three probe pools.  Built once per (task, preset, d, function seed); every unit
of that instance reads it.  Evaluation stage only.

    python cd_targets.py --out ../results/targets --workers 16
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cd_view as cv  # noqa: E402

REPO = Path(__file__).resolve().parents[7]
LEDGER = REPO / "studies/main-experiments/16-main-final/ledger/winners.csv"
ARM = "lewi-v1-rgm"
CELLS = ("tree", "dag", "ftree", "flat", "dense", "fdag", "rand")
POOLS = (("train", 8192), ("val", 4096), ("test", 8192))


def instances() -> dict[tuple, dict]:
    """One representative winner per (cell, function seed)."""
    out = {}
    with open(LEDGER) as fh:
        for r in csv.DictReader(fh):
            if r["arm"] != ARM or r["cell"] not in CELLS:
                continue
            key = (r["cell"], int(r["function_seed"]))
            if key not in out or int(r["n"]) > int(out[key]["n"]):
                out[key] = r
    return out


def build(row: dict) -> dict:
    model, target, spec = cv.load(REPO / row["checkpoint"], which="best")
    g = cv.target_graph(target)
    d = int(spec["d"])
    data = {"fingerprint": target.fingerprint, "d": d, **g}
    for tag, n in POOLS:
        X = cv.draw_inputs(target.fingerprint, tag, n, d)
        data[f"X_{tag}"] = X
        data[f"Z_{tag}"] = cv.node_values(target, X, g["nodes"]["indices"])
    return data


def _one(args):
    key, row, out_dir = args
    cell, fs = key
    path = Path(out_dir) / f"{cell}__fs{fs}.npz"
    if path.exists():
        return f"skip {path.name}"
    t0 = time.time()
    d = build(row)
    meta = {k: v for k, v in d.items() if not k[0] in "XZ"}
    np.savez_compressed(path, meta=json.dumps(meta),
                        **{k: v for k, v in d.items() if k[0] in "XZ"})
    return f"wrote {path.name} A={d['A']} S={len(d['node_parents'])} in {time.time()-t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--first-per-cell", action="store_true")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    inst = instances()
    keys = sorted(inst)
    if a.first_per_cell:
        seen, sel = set(), []
        for k in keys:
            if k[0] not in seen:
                seen.add(k[0]); sel.append(k)
        keys = sel
    if a.only:
        keys = [k for k in keys if a.only in k[0]]
    jobs = [(k, inst[k], a.out) for k in keys]
    print(f"{len(jobs)} instances", flush=True)
    if a.workers > 1:
        with Pool(a.workers) as p:
            for m in p.imap_unordered(_one, jobs):
                print(m, flush=True)
    else:
        for j in jobs:
            print(_one(j), flush=True)


if __name__ == "__main__":
    main()
