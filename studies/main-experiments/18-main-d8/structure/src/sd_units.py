"""The structure-discovery records of the clean-input cohort (17-main-clean panel main8b, d = 8):
the modules study's round-3 runner (r3_unit.run: the reference procedure with every registered
variant and control) applied to every sealed lewi-v1-rgm winner of the four structured workloads at
every sample size, best and init states.  The records land in units/main8b/ in the modules study's
layout (<family>__fs<seed>__n<n>__<which>.json; the cell names lose their "8").

    python sd_units.py list                      ledger/d8/winners.csv -> lists/main8b.list (896 states)
    python sd_units.py reuse                     the records reused from the modules study checked against the checkpoints on disk
    python sd_units.py verify [--workers 8]      the checkpoint verification of every state (r3_verify) -> tables/verify.json
    python sd_units.py targets [--workers 8]     the target cache of every listed instance that lacks one (the 2026-09-20 instances)
    python sd_units.py run --workers 24          claim-based workers over the list (a state runs at most once across every pool)
    python sd_units.py status                    records / claims / pending
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool, Process
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
STRUCT = HERE.parent
MO = REPO / "studies/csfrbench/evaluation-protocol/structural-metrics/structure-discovery/modules"
for p in (str(MO / "src"), str(MO.parent / "component-dag" / "src"), str(MO.parent / "recovered-graph" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
CUTOFF_SRC = STRUCT.parents[0] / "input-cutoff" / "src"      # the study-local arm the repaired winners ran as (lewi-v1-rgm-in5; the rule adopted 2026-09-20)


def _register_arms() -> None:
    """The loaders register the campaign's arms through the modules study's stack; the repaired winners' arm is registered here, in the
    parent process before any pool or worker forks."""
    for p in (REPO / "src", REPO / "studies/main-experiments/12-extra/src",
              REPO / "studies/main-experiments/17-main-clean/src", CUTOFF_SRC):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    import cutoff_methods  # noqa: F401

LEDGER = STRUCT.parents[0] / "ledger" / "d8" / "winners.csv"
LIST = STRUCT / "lists" / "main8b.list"
UNITS = STRUCT / "units" / "main8b"
CLAIMS = STRUCT / "logs" / "claims" / "main8b"
LOGS = STRUCT / "logs" / "main8b"
TARGETS = STRUCT / "targets" / "main8b"                  # the modules study's target cache (links; size-independent) + tree fs6004, absent there
ARM = "lewi-v1-rgm"
RULE_ARM = "lewi-v1-rgm-in5"                             # the adopted input-cutoff rule's arm (2026-09-20): the repaired winners' checkpoints
RULE_ROOT = "studies/main-experiments/18-main-d8/input-cutoff/runs/repair/"
CELLS = {"tree8": "tree", "ftree8": "ftree", "dag8": "dag", "rand8": "rand"}
TRIALS = 32


def rows():
    out = []
    with open(LEDGER) as fh:
        for r in csv.DictReader(fh):
            if r["arm"] == ARM and r["cell"] in CELLS and int(float(r["trials"])) >= TRIALS and not int(r["trials_diverged"] or 0):
                out.append(r)
    return out


def states():
    for line in LIST.read_text().splitlines():
        if line.strip():
            run_dir, cell, fs, n, which = line.split()
            yield run_dir, cell, int(fs), int(n), which


def name_of(cell, fs, n, which):
    return f"{cell}__fs{fs}__n{n}__{which}"


def cmd_list():
    lines, missing = [], []
    for r in rows():
        ck = r["checkpoint"]
        if not ck or not (REPO / ck / "best.pt").exists() or not (REPO / ck / "init.pt").exists():
            missing.append(r["unit"]); continue
        for which in ("best", "init"):
            lines.append(f"{ck} {CELLS[r['cell']]} {r['function_seed']} {r['n']} {which}")
    LIST.parent.mkdir(parents=True, exist_ok=True)
    LIST.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} states from {len(rows())} winners; {len(missing)} winners without a checkpoint on disk: {missing[:5]}")


def _positions(args):
    run_dir, which, path = args
    import cd_view as cv
    rec = json.loads(Path(path).read_text())
    model, _, spec = cv.load(REPO / run_dir, which=which)
    E = [c.detach().numpy() for c in model.gate_codes]
    same = len(E) == len(rec["positions"]) and all(np.allclose(np.round(e, 4), np.array(p), atol=1.5e-4) for e, p in zip(E, rec["positions"]))
    return {"record": Path(path).name, "record_checkpoint": rec["checkpoint"], "checkpoint": run_dir, "which": which,
            "unit_equal": Path(rec["checkpoint"]).name == Path(run_dir).name, "positions_equal": bool(same), "version": rec["version"]}


def cmd_reuse(workers):
    _register_arms()
    jobs = []
    for run_dir, cell, fs, n, which in states():
        path = UNITS / f"{name_of(cell, fs, n, which)}.json"
        if path.exists():
            jobs.append((run_dir, which, path))
    with Pool(workers) as p:
        recs = p.map(_positions, jobs)
    ok = sum(r["unit_equal"] and r["positions_equal"] for r in recs)
    out = {"reused": len(recs), "pass": ok, "records": recs}
    (STRUCT / "tables").mkdir(exist_ok=True)
    (STRUCT / "tables" / "reuse_check.json").write_text(json.dumps(out, indent=0))
    print(f"{len(recs)} reused records, {ok} equal to the checkpoint on disk (unit name and positions to 4 dp)")


def cmd_verify(workers):
    _register_arms()
    import r3_verify as rv
    st = [(run_dir, which) for run_dir, _, _, _, which in states()]
    rule = [x for x in st if x[0].startswith(RULE_ROOT)]
    with Pool(workers) as p:
        recs = p.map(rv._one, [x for x in st if not x[0].startswith(RULE_ROOT)])
    if rule:
        # the adopted rule's winners ran as the study-local arm lewi-v1-rgm-in5 (LeWi under the input rule, the same architecture and
        # constants): the modules study's verifier expects the arm's name, so these states are checked with that name expected and
        # every other condition of its rule unchanged (the pool forks after the patch)
        rv.EXPECTED["method"] = RULE_ARM
        with Pool(workers) as p:
            recs += p.map(rv._one, rule)
        rv.EXPECTED["method"] = ARM
    summary = {"states": len(recs), "pass": sum(r["pass"] for r in recs), "errors": sum("error" in r for r in recs),
               "methods": sorted({r.get("method") for r in recs if "method" in r}),
               "radius2": sorted({x for r in recs for x in r.get("radius2", [])}),
               "readout_open_edges_min": min((r["readout_open_edges"] for r in recs if "readout_open_edges" in r), default=None),
               "readout_plain_gap_min_best": min((r["readout_plain_gap"] for r in recs if "readout_plain_gap" in r and r["which"] == "best"), default=None),
               "forward_gap_max": max((r["forward_equals_effective"] for r in recs if "forward_equals_effective" in r), default=None)}
    (STRUCT / "tables").mkdir(exist_ok=True)
    (STRUCT / "tables" / "verify.json").write_text(json.dumps({"summary": summary, "records": recs}, indent=0))
    print(json.dumps(summary))


def work(lines, wid):
    import r3_unit as ru
    import cd_unit as cu
    for run_dir, cell, fs, n, which in lines:
        name = name_of(cell, fs, n, which)
        path = UNITS / f"{name}.json"
        if path.exists():
            continue
        try:
            os.mkdir(CLAIMS / name)
        except FileExistsError:
            continue
        try:
            res = ru.run(run_dir, cell, fs, n, which, TARGETS)
            tmp = path.with_suffix(f".tmp{wid}")
            tmp.write_text(json.dumps(cu._clean(res)))
            os.replace(tmp, path)
            with open(LOGS / "units.log", "a") as fh:
                fh.write(f"wrote {name} in {res['seconds']:.1f}s (worker {wid}, host {os.uname().nodename})\n")
        except Exception:
            with open(LOGS / "errors.log", "a") as fh:
                fh.write(f"FAILED {name}\n{traceback.format_exc()}\n")
            with open(LOGS / "failed.list", "a") as fh:
                fh.write(f"{name}\n")


def cmd_targets(workers):
    """The target cache of every instance in the list that lacks one (the sampled graphs' 16 more instances, 2026-09-20): built
    from the instance's winner at its largest size with the component-DAG builder, as the modules study built the others
    (r3_cohort._one); the target is size-independent."""
    _register_arms()
    import cd_targets as ct
    inst = {}
    for run_dir, cell, fs, n, which in states():
        if (TARGETS / f"{cell}__fs{fs}.npz").exists():
            continue
        if (cell, fs) not in inst or n > inst[(cell, fs)][1]:
            inst[(cell, fs)] = (run_dir, n)
    TARGETS.mkdir(parents=True, exist_ok=True)
    jobs = [(cell, fs, run_dir) for (cell, fs), (run_dir, _) in sorted(inst.items())]
    with Pool(workers) as p:
        for m in p.imap_unordered(_target_one, jobs):
            print(m, flush=True)
    print(f"{len(jobs)} targets built into {TARGETS}")


def _target_one(args):
    import cd_targets as ct
    cell, fs, run_dir = args
    path = TARGETS / f"{cell}__fs{fs}.npz"
    if path.exists():
        return f"skip {path.name}"
    d = ct.build({"checkpoint": str(run_dir)})
    meta = {k: v for k, v in d.items() if not k[0] in "XZ"}
    np.savez_compressed(path, meta=json.dumps(meta), **{k: v for k, v in d.items() if k[0] in "XZ"})
    return f"wrote {path.name} A={d['A']} S={len(d['node_parents'])}"


def cmd_run(workers):
    _register_arms()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    os.chdir(REPO)
    for d in (UNITS, CLAIMS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    lines = [s for s in states() if not (UNITS / f"{name_of(*s[1:])}.json").exists()]
    rng = np.random.default_rng(os.getpid())
    rng.shuffle(lines)                                       # pools claim different states first
    procs = [Process(target=work, args=(lines, w)) for w in range(workers)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print(f"main8b: {len(lines)} pending states walked by {workers} workers in {time.time() - t0:.0f}s")


def cmd_status():
    st = list(states())
    done = sum((UNITS / f"{name_of(*s[1:])}.json").exists() for s in st)
    claims = len(list(CLAIMS.iterdir())) if CLAIMS.exists() else 0
    failed = len((LOGS / "failed.list").read_text().splitlines()) if (LOGS / "failed.list").exists() else 0
    print(f"{done} of {len(st)} records; {claims} claims; {failed} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "reuse", "verify", "run", "status", "targets"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "list":
        cmd_list()
    elif a.cmd == "reuse":
        cmd_reuse(a.workers)
    elif a.cmd == "verify":
        cmd_verify(a.workers)
    elif a.cmd == "run":
        cmd_run(a.workers)
    elif a.cmd == "targets":
        cmd_targets(a.workers)
    else:
        cmd_status()


if __name__ == "__main__":
    main()
