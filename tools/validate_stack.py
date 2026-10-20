"""Validate the stacked engine against the records of the paper.

Re-runs LeWi and Dense on a part of the benchmark with the default engine (stacks with the fused kernels) and compares
the result with the recorded runs at the three levels the protocol works at: single trials, selected runs (winners) and
learning curves (AULC). The recorded runs are the paper's, made with the per-network trainer.

    python tools/validate_stack.py plan --out runs/validate                  # the units (default: the parts below)
    csfrbench run --units runs/validate/units.jsonl --out runs/validate --workers 24 [--shard 0/4]
    python tools/validate_stack.py compare --runs runs/validate              # tables, and validation.json in the folder

The default parts: the depth-4 tree and the reuse DAG of the primary setting on all 16 target instances, the depth-4
tree of the nuisance-input setting on all 16, and the depth-5 tree of the primary setting on instance 6000 (64x8).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from csfrbench import analysis, metrics, record, runner, suite

ARMS = ("lewi", "dense")
PARTS = [("primary", "depth4_tree", None), ("primary", "reuse_dag", None), ("nuisance", "depth4_tree", None),
         ("primary", "depth5_tree", (6000,))]
CLIP = 1.5


def plan(args) -> None:
    units = []
    for setting, workload, instances in PARTS:
        units += runner.plan(setting, [workload], ARMS, instances)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    print(f"{len(units)} units -> {out / 'units.jsonl'}")


def _key(r: dict) -> tuple:
    return (r["setting"], r["workload"], r["arm"], int(r["instance"]), int(r["n"]), int(r["trial"]))


def compare(args) -> dict:
    runs = [r for r in runner.read_runs(args.runs) if not r.get("failed")]
    if not runs:
        raise SystemExit("no runs in the folder")
    report: dict = {"trials": [], "winners": [], "aulc": []}
    by_part = defaultdict(list)
    for r in runs:
        by_part[(r["setting"], r["workload"], r["arm"])].append(r)
    print("\n## Trials: every re-run trial against the recorded trial of the same configuration and seeds\n")
    print("| setting | workload | method | trials | test NMSE corr. (clipped at 1.5) | rank corr. | median abs. diff. | "
          "within 0.01 | diverged (re-run / record) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (setting, workload, arm), rows in sorted(by_part.items()):
        rec = {_key(r): r for r in record.unit_rows(setting, workload, arm)}
        pairs = [(rec[_key(r)], r) for r in rows if _key(r) in rec]
        both = [(a["test_nmse"], b.get("test_nmse")) for a, b in pairs
                if a["test_nmse"] is not None and b.get("test_nmse") is not None]
        x = np.minimum(np.array([a for a, _ in both]), CLIP)
        y = np.minimum(np.array([b for _, b in both]), CLIP)
        d = np.abs(np.array([a for a, _ in both]) - np.array([b for _, b in both]))
        row = {"setting": setting, "workload": workload, "arm": arm, "trials": len(pairs),
               "corr": float(np.corrcoef(x, y)[0, 1]), "rank_corr": float(spearmanr(x, y).statistic),
               "median_abs_diff": float(np.median(d)), "within_0.01": float(np.mean(d < 0.01)),
               "diverged_rerun": sum(int(b.get("diverged", 0)) for _, b in pairs),
               "diverged_record": sum(int(a["diverged"]) for a, _ in pairs)}
        report["trials"].append(row)
        print(f"| {setting} | {workload} | {suite.ARMS[arm].paper_name} | {row['trials']} | {row['corr']:.4f} | "
              f"{row['rank_corr']:.4f} | {row['median_abs_diff']:.4f} | {100 * row['within_0.01']:.0f}% | "
              f"{row['diverged_rerun']} / {row['diverged_record']} |")

    print("\n## Selected runs: the winner of every (instance, n) by validation NMSE\n")
    print("| setting | workload | method | conditions | same trial | test NMSE of the winner, re-run - record: mean [95% CI] | median |")
    print("|---|---|---|---|---|---|---|")
    curves = {}
    for (setting, workload, arm), rows in sorted(by_part.items()):
        mine = analysis.select_winners(rows)
        rec = {(w["instance"], w["n"]): w for w in record.winners(setting) if w["workload"] == workload and w["arm"] == arm}
        pairs = [(rec[(w["instance"], w["n"])], w) for w in mine if (w["instance"], w["n"]) in rec]
        if not pairs:
            continue
        same = np.mean([a["trial"] == b["trial"] for a, b in pairs])
        diff = [b["test_nmse"] - a["test_nmse"] for a, b in pairs if b.get("test_nmse") is not None]
        ci = metrics.t_interval(diff)
        report["winners"].append({"setting": setting, "workload": workload, "arm": arm, "conditions": len(pairs),
                                  "same_trial": float(same), **{k: ci[k] for k in ("mean", "lo", "hi", "median")}})
        print(f"| {setting} | {workload} | {suite.ARMS[arm].paper_name} | {len(pairs)} | {100 * same:.0f}% | "
              f"{ci['mean']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}] | {ci['median']:+.4f} |")
        curves[(setting, workload, arm)] = (pairs, mine)

    print("\n## Learning curves: AULC per target instance, re-run against record\n")
    print("| setting | workload | method | instances | AULC record | AULC re-run | difference: mean [95% CI] | "
          "re-run better / record better | the paper's LeWi - Dense |")
    print("|---|---|---|---|---|---|---|---|---|")
    contrast = {(c["workload"], c["arm"]): c for s in {k[0] for k in by_part} for c in record.contrasts(s)
                if c.get("method", "lewi") == "lewi"}
    for (setting, workload, arm), (pairs, mine) in sorted(curves.items()):
        sizes = {workload: suite.workload(setting, workload).profile.sizes}
        new = {r["instance"]: r["aulc"] for r in analysis.instance_aulc(mine, sizes)}
        old = {r["instance"]: r["aulc"] for r in record.aulc_instances(setting)
               if r["workload"] == workload and r["arm"] == arm and r["instance"] in new}
        keys = sorted(set(new) & set(old))
        if not keys:
            continue
        d = [new[i] - old[i] for i in keys]
        ci = metrics.t_interval(d)
        c = contrast.get((workload, "dense"))
        ref = f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" if c and setting == "primary" else "-"
        report["aulc"].append({"setting": setting, "workload": workload, "arm": arm, "instances": len(keys),
                               "record": float(np.mean([old[i] for i in keys])),
                               "rerun": float(np.mean([new[i] for i in keys])),
                               **{k: ci[k] for k in ("mean", "lo", "hi")},
                               "rerun_better": int(sum(x < 0 for x in d)), "record_better": int(sum(x > 0 for x in d))})
        interval = f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}]" if len(keys) > 1 and math.isfinite(ci["lo"]) else ""
        print(f"| {setting} | {workload} | {suite.ARMS[arm].paper_name} | {len(keys)} | "
              f"{np.mean([old[i] for i in keys]):.4f} | {np.mean([new[i] for i in keys]):.4f} | {ci['mean']:+.4f} {interval} | "
              f"{sum(x < 0 for x in d)} / {sum(x > 0 for x in d)} | {ref} |")
    engines = sorted({r.get("engine", "exact") for r in runs})
    report["engines"] = engines
    print(f"\nengines: {', '.join(engines)}")
    (Path(args.runs) / "validation.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--out", required=True)
    a.set_defaults(func=plan)
    a = sub.add_parser("compare")
    a.add_argument("--runs", required=True)
    a.set_defaults(func=compare)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
