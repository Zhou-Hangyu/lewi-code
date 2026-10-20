"""Command line of CSFRBench.

    csfrbench info                          the settings, workloads, methods and search spaces
    csfrbench verify [--level quick]        check this installation against the records of the paper
    csfrbench record --setting primary      the recorded AULC table
    csfrbench target --workload depth4_tree --instance 6000 [--out target.json]
    csfrbench dataset --workload depth4_tree --instance 6000 --out DIR
    csfrbench unit --workload depth4_tree --arm lewi --instance 6000 --n 512 --trial 0
    csfrbench plan --setting primary --workloads depth4_tree --arms lewi dense --out units.jsonl
    csfrbench run --units units.jsonl --out DIR --workers 8 [--shard 0/4] [--engine exact]
    csfrbench analyze --runs DIR [--with-record]
    csfrbench reproduce --workload depth4_tree --arms lewi dense --out DIR --workers 8 [--engine exact]

LeWi and Dense units train in stacks of many networks with LeWi's fused kernels (csfrbench.stack); ``--engine exact``
trains every unit alone with the per-network trainer, which reproduces a record bit for bit.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from . import __version__


def _add_cell(p, arm: bool = False):
    p.add_argument("--setting", default="primary", choices=["primary", "nuisance"])
    p.add_argument("--workload", required=True)
    p.add_argument("--instance", type=int, default=6000)
    if arm:
        p.add_argument("--arm", required=True)
        p.add_argument("--n", type=int, required=True)
        p.add_argument("--trial", type=int, default=0)


def cmd_info(args) -> int:
    from . import search, suite
    print(f"CSFRBench {__version__}")
    for setting, workloads in suite.SETTINGS.items():
        print(f"\n{suite.SETTING_TITLE[setting]}  [--setting {setting}]")
        print(f"  {'workload':20s}{'paper name':24s}{'d':>4s}{'active':>8s}{'backbone':>10s}{'steps':>9s}"
              f"{'sizes':>18s}{'instances':>11s}  methods")
        for w in workloads.values():
            p = w.profile
            print(f"  {w.name:20s}{w.paper_name:24s}{w.d:>4d}{w.active:>8d}{p.name:>10s}{p.max_steps:>9,d}"
                  f"{f'2^{p.sizes[0].bit_length() - 1} .. 2^{p.sizes[-1].bit_length() - 1}':>18s}"
                  f"{len(w.instances):>11d}  {' '.join(w.arms)}")
    print("\nmethods")
    for a in suite.ARMS.values():
        fixed = ", ".join(f"{k} = {v}" for k, v in a.fixed.items())
        print(f"  {a.name:12s}{a.paper_name:20s}{'graph-informed reference' if a.graph_informed else 'learned':26s}{fixed}")
    if args.spaces:
        print("\nfrozen search spaces (all dimensions are sampled uniformly in log space)")
        for name, block in search.spaces().items():
            print(f"  set {name}: {block['note']}")
            for arm, sp in block["arms"].items():
                dims = "  ".join(f"{d['name']} in [{d['low']:.3g}, {d['high']:.3g}]" for d in sp["dimensions"])
                print(f"    {arm:12s}{dims}")
    return 0


def cmd_verify(args) -> int:
    from . import training, verify
    env = training.environment()
    print("CSFRBench", __version__, "| verification level:", args.level)
    print("environment:", ", ".join(f"{k} {v}" for k, v in env.items()))
    results = verify.run(args.level, progress=lambda line: print(line, flush=True))
    counts = verify.summary(results)
    print("\nresult:", ", ".join(f"{v} {k}" for k, v in counts.items() if v))
    if args.out:
        Path(args.out).write_text(json.dumps({"environment": env, "level": args.level, "summary": counts,
                                              "checks": [r.__dict__ for r in results]}, indent=1) + "\n")
    if counts["FAIL"]:
        print("VERIFICATION FAILED")
        return 1
    print("VERIFIED: every check agrees with the record"
          + (" bit for bit" if not counts["close"] else " (some within tolerance: see docs/REPRODUCE.md, section 1)"))
    return 0


def cmd_record(args) -> int:
    from . import analysis, record, suite
    table = [{**r, "half_width": (r["hi"] - r["lo"]) / 2} for r in record.aulc(args.setting)]
    print(f"Recorded mean AULC ± half-width of the 95% Student-t interval, {suite.SETTING_TITLE[args.setting]}")
    names = list(suite.SETTINGS[args.setting])
    for block in (names[:4], names[4:]):
        print()
        print(analysis.format_table([r for r in table if r["workload"] in block]))
    return 0


def cmd_target(args) -> int:
    from . import suite
    target = suite.workload(args.setting, args.workload).target(args.instance)
    record = target.to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
        print(args.out)
    else:
        g = target.graph
        print(json.dumps({"target_id": target.target_id, "fingerprint": target.fingerprint, "d": target.d,
                          "active_inputs": target.active_inputs, "node_parents": [list(p) for p in g.node_parents],
                          "levels": g.levels(), "readout": list(g.readout_indices),
                          "topology_hash": target.topology_hash}, indent=1))
    return 0


def cmd_dataset(args) -> int:
    from . import datasets, suite
    w = suite.workload(args.setting, args.workload)
    data = datasets.make_dataset(w.target(args.instance), args.n_max or w.profile.n_max, pools=tuple(args.pools))
    print(datasets.save_dataset(data, args.out))
    return 0


def cmd_unit(args) -> int:
    from . import runner, training
    unit = {"setting": args.setting, "workload": args.workload, "arm": args.arm, "instance": args.instance,
            "n": args.n, "trial": args.trial}
    record = runner.execute(unit, keep_states=bool(args.out and args.keep_states), threads=args.threads,
                            device=args.device)
    states = {k: record.pop(k) for k in ("init_state", "best_state") if k in record}
    print(json.dumps({k: record[k] for k in ("unit", "hp", "best_step", "best_val_nmse", "test_nmse", "diverged",
                                             "unit_wallclock_s") if k in record}, indent=1))
    if args.out:
        out = Path(args.out)
        training.write_json(out / f"{record['unit']}.json", record)
        if states:
            import torch
            torch.save({"init": states["init_state"], "best": states["best_state"], "unit": record["unit"],
                        "hp": record["hp"]}, out / f"{record['unit']}.pt")
    return 0


def cmd_plan(args) -> int:
    from . import runner
    units = runner.plan(args.setting, args.workloads, args.arms, args.instances, args.sizes, args.trials)
    with open(args.out, "w") as handle:
        for u in units:
            handle.write(json.dumps(u, sort_keys=True) + "\n")
    print(f"{len(units)} units -> {args.out}")
    return 0


def _shard(text: str | None):
    if not text:
        return None
    index, count = text.split("/")
    return int(index), int(count)


def cmd_run(args) -> int:
    from . import runner
    units = [json.loads(line) for line in Path(args.units).read_text().splitlines() if line.strip()]
    summary = runner.run(units, args.out, workers=args.workers, shard=_shard(args.shard),
                         keep_states=args.keep_states, method_modules=tuple(args.methods or ()),
                         engine=args.engine, stack_size=args.stack_size)
    print(json.dumps(summary))
    return 1 if summary["failed"] else 0


def cmd_analyze(args) -> int:
    from . import analysis, record, runner, suite
    runs = runner.read_runs(args.runs)
    runs = [r for r in runs if not r.get("failed")]
    settings = sorted({r["setting"] for r in runs})
    code = 0
    for setting in settings:
        mine = [r for r in runs if r["setting"] == setting]
        winners = analysis.select_winners(mine, trials=args.trials)
        if args.with_record:
            have = {(w["workload"], w["arm"]) for w in winners}
            wanted = {w["workload"] for w in winners}
            winners += [r for r in record.winners(setting) if r["workload"] in wanted
                        and (r["workload"], r["arm"]) not in have]
        sizes = {w.name: w.profile.sizes for w in suite.SETTINGS[setting].values()}
        inst = analysis.instance_aulc(winners, sizes)
        table = analysis.aulc_table(inst)
        print(f"\n{suite.SETTING_TITLE[setting]}: {len(mine)} runs, {len(winners)} selected runs, "
              f"{len(inst)} complete learning curves")
        if not table:
            print("  no workload has a complete learning curve: a curve needs a selected run at every size of the grid")
            code = 1
            continue
        known = [r for r in table if r["arm"] in suite.ARMS]
        if known:
            print(analysis.format_table(known))
        for r in table:
            if r["arm"] not in suite.ARMS:
                print(f"  {r['workload']:20s}{r['arm']:16s}{r['mean']:.3f}±{r['half_width']:.3f}  ({r['n']} instances)")
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"aulc_{setting}.json").write_text(json.dumps(table, indent=1) + "\n")
            (out / f"aulc_instances_{setting}.json").write_text(json.dumps(inst, indent=1) + "\n")
            (out / f"winners_{setting}.json").write_text(json.dumps(winners, indent=1) + "\n")
            (out / f"contrasts_{setting}.json").write_text(
                json.dumps(analysis.contrasts(inst, args.method), indent=1) + "\n")
    return code


def cmd_reproduce(args) -> int:
    """Run recorded selected configurations again and compare the results with the record."""
    from . import record, runner, suite, verify
    winners = [w for w in record.winners(args.setting) if w["workload"] == args.workload
               and (not args.arms or w["arm"] in args.arms) and (not args.instances or w["instance"] in args.instances)
               and (not args.sizes or w["n"] in args.sizes)]
    if not winners:
        print("no recorded winner matches the selection")
        return 1
    units = runner.units_of_winners(winners)
    summary = runner.run(units, args.out, workers=args.workers, shard=_shard(args.shard), engine=args.engine,
                         stack_size=args.stack_size)
    print(json.dumps(summary))
    done = {r["unit"]: r for r in runner.read_runs(args.out)}
    exact = close = differ = missing = 0
    relative = []
    for w, u in zip(winners, units):
        r = done.get(runner.unit_id(u))
        if r is None or r.get("failed"):
            missing += 1
            continue
        if r.get("test_nmse") is None:          # the new run diverged; the recorded run did not
            differ += 1
            continue
        relative.append(abs(r["test_nmse"] - w["test_nmse"]) / max(abs(w["test_nmse"]), 1e-300))
        if r["best_val_nmse"] == w["best_val_nmse"] and r["best_step"] == w["best_step"] and relative[-1] <= 1e-12:
            exact += 1
        elif verify.factor(r["test_nmse"], w["test_nmse"]) <= verify.TOL_FACTOR:
            close += 1
        else:
            differ += 1
    relative.sort()
    print(f"{len(winners)} recorded selected runs of {suite.workload(args.setting, args.workload).paper_name}: "
          f"{exact} exact, {close} within a factor of {verify.TOL_FACTOR:g}, {differ} different, {missing} not run")
    if relative:
        print(f"relative difference of the test NMSE: median {statistics.median(relative):.2e}, "
              f"largest {relative[-1]:.2e}")
    _aulc_agreement(winners, units, done, args)
    if args.engine == "exact":
        return 1 if differ else 0
    return 1 if missing else 0


def _aulc_agreement(winners, units, done, args) -> None:
    """The AULC of the re-run learning curves against the recorded ones, per method: the mean paired difference with
    its 95% interval over the target instances (the level at which a stacked re-run is compared with the record)."""
    from . import analysis, metrics, runner, suite
    sizes = {args.workload: suite.workload(args.setting, args.workload).profile.sizes}
    rerun = []
    for w, u in zip(winners, units):
        r = done.get(runner.unit_id(u))
        if r is not None and r.get("test_nmse") is not None:
            rerun.append({**w, "test_nmse": r["test_nmse"]})
    old = {(r["arm"], r["instance"]): r["aulc"] for r in analysis.instance_aulc(winners, sizes)}
    new = {(r["arm"], r["instance"]): r["aulc"] for r in analysis.instance_aulc(rerun, sizes)}
    for arm in sorted({a for a, _ in new}):
        keys = sorted(k for k in new if k[0] == arm and k in old)
        if not keys:
            continue
        diff = [new[k] - old[k] for k in keys]
        ci = metrics.t_interval(diff)
        line = (f"AULC of {suite.ARMS[arm].paper_name if arm in suite.ARMS else arm} re-run minus record over "
                f"{len(keys)} instance(s): {ci['mean']:+.4f}")
        if len(keys) > 1:
            line += f" [{ci['lo']:+.4f}, {ci['hi']:+.4f}]"
        print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="csfrbench", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"csfrbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="settings, workloads, methods")
    p.add_argument("--spaces", action="store_true", help="also print the frozen search spaces")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("verify", help="check this installation against the records of the paper")
    p.add_argument("--level", default="quick", choices=["quick", "standard", "full"])
    p.add_argument("--out", help="write the results to this JSON file")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("record", help="the recorded AULC table of a setting")
    p.add_argument("--setting", default="primary", choices=["primary", "nuisance"])
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("target", help="draw a target instance")
    _add_cell(p)
    p.add_argument("--out")
    p.set_defaults(func=cmd_target)

    p = sub.add_parser("dataset", help="write the dataset of a target instance")
    _add_cell(p)
    p.add_argument("--out", required=True)
    p.add_argument("--n-max", type=int, default=0)
    p.add_argument("--pools", nargs="+", default=["train", "val", "test", "diagnostic"])
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("unit", help="train one unit")
    _add_cell(p, arm=True)
    p.add_argument("--out")
    p.add_argument("--keep-states", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_unit)

    p = sub.add_parser("plan", help="list the units of a part of a setting")
    p.add_argument("--setting", default="primary", choices=["primary", "nuisance"])
    p.add_argument("--workloads", nargs="*")
    p.add_argument("--arms", nargs="*")
    p.add_argument("--instances", nargs="*", type=int)
    p.add_argument("--sizes", nargs="*", type=int)
    p.add_argument("--trials", nargs="*", type=int)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="execute a list of units")
    p.add_argument("--units", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--shard", help="i/k: run the units i, i + k, i + 2k, ... of the list")
    p.add_argument("--keep-states", action="store_true")
    p.add_argument("--methods", nargs="*", help="modules that register new methods")
    p.add_argument("--engine", default="auto", choices=["auto", "exact"],
                   help="auto: LeWi and Dense in stacks with the fused kernels; exact: every unit alone, bit for bit")
    p.add_argument("--stack-size", type=int, help="networks per stack (default 128 at width 32, 32 at width 64)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze", help="selection and AULC table of the runs of a folder")
    p.add_argument("--runs", required=True)
    p.add_argument("--out")
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--method", default="lewi", help="the method of the paired comparisons")
    p.add_argument("--with-record", action="store_true", help="add the recorded methods that the folder does not hold")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("reproduce", help="run recorded selected configurations again")
    p.add_argument("--setting", default="primary", choices=["primary", "nuisance"])
    p.add_argument("--workload", required=True)
    p.add_argument("--arms", nargs="*")
    p.add_argument("--instances", nargs="*", type=int)
    p.add_argument("--sizes", nargs="*", type=int)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--shard")
    p.add_argument("--engine", default="auto", choices=["auto", "exact"],
                   help="auto: LeWi and Dense in stacks with the fused kernels; exact: every unit alone, bit for bit")
    p.add_argument("--stack-size", type=int)
    p.set_defaults(func=cmd_reproduce)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
