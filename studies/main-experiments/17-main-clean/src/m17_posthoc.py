"""17-main-clean post-hoc readouts on retained winners (DESIGN amendment 4).

The three readouts of the record's report (16-main-final sections 7-9), applied to
the winners of a panel that retained `init.pt` and `best.pt`:

    probes PANEL [--slot i --slots k]   linear probes: the frozen v0.5 ridge probe
                                        (csfrbench.metrics.probe_r2, amendment-14 support
                                        floor) of every target intermediate from every
                                        layer h0..hL, on the exact initialization and on
                                        the best checkpoint, every arm's winners
    probes-analyze PANEL                -> analysis/PANEL/probes/{probe_values,probe_summary,
                                        probe_levels}.csv
    das-datasets PANEL                  the frozen counterfactual pools per (task, instance)
    das-units PANEL                     one fit line per (winner, hidden layer, restart) for
                                        Dense, GO(all) and LeWi
    das-run PANEL --slot i --slots k    execute the list in-process (csfrbench.das_probe)
    das-harvest PANEL                   fits -> posthoc/PANEL/causal/tidy/das_fits.parquet
    das-analyze PANEL                   -> analysis/PANEL/causal/{das_cells,das_star,das_scores,
                                        das_summary,das_contrasts}.csv
    embeddings PANEL [--fs 6000] [--ns 4096 32768 262144]
                                        the learned positions of LeWi's winners: the
                                        interactive panel and a static view per workload
                                        -> analysis/PANEL/embeddings/
    sbatch PANEL probes|das N [--slots 24 --hours 8]   free_qos packs for the work

Two departures from the record, forced by the setting and disclosed in the report:
the target roster is every local function of the workload (15 on the tree, the DAG
and the random graphs; 8 on the flat modules; 1 on the dense node), where the
record's mechanism workloads had exactly 7; and the backbone has 8 hidden layers,
where the frozen DAS config names layers 1-4 - layers 1-4 are read under the frozen
protocol and 5-8 under the same computation as an extension. Everything else is the
record's: the probe estimator, the frozen DAS constants and selection rule, the
eta^2 tracking and the positions renderer of the wave-5 study.

Outputs live under posthoc/PANEL (records) and analysis/PANEL (tables).
"""
from __future__ import annotations

import argparse
import fcntl
import html
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import m17 as M  # noqa: E402  (also puts csfrbench and 12-extra on sys.path)

POST = M.STUDY / "posthoc"
ANALYSIS = M.STUDY / "analysis"
LOGS = M.STUDY / "logs" / "posthoc"
NEXTGEN_SRC = M.REPO / "studies" / "viewer"   # the embeddings viewer's script is not part of this archive
FROZEN = M.REPO / "studies/csfrbench/final-design/versions/v0.5/configs/causal_das_frozen.json"
MECH_ARMS = ["noreg", "graph_oracle", M.LEWI]     # the record's causal-probe arms
# decision 2026-09-22 (the paper's appendix): the causal fits of other arms on chosen cells, without touching the record's lists -
# M17_MECH_ARMS=l0hc,rigl,bimt M17_MECH_CELLS=tree8 m17_posthoc.py das-units PANEL (the datasets are per task and instance, shared)
MECH_ARMS = [a for a in os.environ.get("M17_MECH_ARMS", "").split(",") if a] or MECH_ARMS
MECH_CELLS = [c for c in os.environ.get("M17_MECH_CELLS", "").split(",") if c] or None
# Arms the linear probes cannot read: the neuron-position arm\'s SpatialMLP has no layer-representation rule in
# csfrbench.metrics (the record never probed it either; amendment 8 brought the arm back at d = 32).
PROBE_SKIP = {"erb_positions"}
EMB_ARMS = [M.LEWI]
DAS_LAYERS = list(range(1, 9))     # the 8 hidden layers of the 64x8 backbone; 1-4 = the frozen set
DAS_K = 3
POOL_N = 8192
TASK_CELL = {task: cell for cell, (task, _p, _d) in M.CELLS.items() if cell in M.SUITE2}
SUITES = M.SUITE2 + ["tree32"] + M.SUITE8 + M.SUITE8N + ["tree40", "tree5x32"]       # the display order of every cell a panel can carry (amendment 8 adds the d = 32 cells; 11 the tree at d = 40; 12 the tree at d = 32)


def das_layers(panel: str) -> list[int]:
    """Every hidden layer of the panel's backbone: 1-8 on the setting's 64x8, 1-4 on the
    8-input suite's 32x4 (amendment 6) - exactly the frozen config's set there."""
    return list(range(1, int(M.PROFILES[M.PANELS[panel]["profile"]]["hidden"]) + 1))


def task_cell(panel: str) -> dict:
    """task family -> cell name within one panel (a family appears once per panel; the
    setting's and the suite's cells share families, so the map is per panel)."""
    return {M.CELLS[c][0]: c for c in M.PANELS[panel]["cells"]}


def panel_cells(panel: str) -> list[str]:
    return [c for c in SUITES if c in M.PANELS[panel]["cells"]]
# Thirty-two colours: the first sixteen for up to 16 teacher nodes (the wave-5 palette has 8).
TEACHER_COLORS = ("#c44e52", "#4c72b0", "#55a868", "#8172b3", "#dd8452", "#12a3a3", "#8c564b", "#e377c2",
                  "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#ad494a", "#e7ba52", "#31a354", "#843c39",
                  # sixteen more for the 32-leaf tree's 31 local functions (the deep object)
                  "#5254a3", "#8ca252", "#bd9e39", "#d6616b", "#a55194", "#6baed6", "#fd8d3c", "#74c476",
                  "#9e9ac8", "#8c6d31", "#e6550d", "#3182bd", "#637939", "#7b4173", "#ce6dbd", "#9c9ede")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _register():
    import m17_tasks  # noqa: F401  (the suite's families)
    import w5_methods  # noqa: F401  (lewi-v1-rgm)
    import abl_methods  # noqa: F401  (the mechanism-ablation arms)
    cutoff = str(M.REPO / "studies" / "main-experiments" / "18-main-d8" / "input-cutoff" / "src")
    if cutoff not in sys.path:
        sys.path.append(cutoff)
    import cutoff_methods  # noqa: F401  (lewi-v1-rgm-in5: the twin's LeWi under the adopted input rule, panel main8b-in5)


def tci(values) -> dict:
    import m17_analysis as A
    return A.tci(values)


# ---------------------------------------------------------------- winners

def winner_dirs(panel: str, arms=None, cells=None) -> list[Path]:
    """Retained winner directories of a panel: every unit of a winners re-run panel,
    otherwise the validation-selected winners of the panel's complete conditions."""
    root = M.RUNS / panel
    if not root.exists():
        return []
    if M.PANELS[panel].get("winners"):
        cands = [d for d in root.iterdir() if d.is_dir() and d.name != "done"]
    else:
        led = M.STUDY / "ledger" / f"winners_{panel}.csv"
        if not led.exists():
            return []
        cands = [root / u for u in pd.read_csv(led, usecols=["unit"])["unit"]]
    out = []
    for d in cands:
        m = M.parse_unit(d.name)
        if m is None or (arms and m["arm"] not in arms) or (cells and m["cell"] not in cells):
            continue
        if (d / "run.json").exists() and (d / "best.pt").exists() and (d / "init.pt").exists():
            out.append(d)
    return sorted(out)


# ---------------------------------------------------------------- linear probes

def probe_one(run_dir: Path, out_dir: Path) -> str:
    """08-main's probe_winners.probe_one with the node count recorded, not asserted:
    the roster here is every local function of the workload (1 to 15 nodes)."""
    import torch
    from csfrbench.datasets import load_splits
    from csfrbench.metrics import probe_r2
    from csfrbench.registry import get_method, get_task

    run_dir = run_dir.resolve()
    name = run_dir.name
    out_npz = out_dir / f"{name}.npz"
    if out_npz.exists() and (out_dir / f"{name}.json").exists():
        return "skipped"
    spec = json.loads((run_dir / "spec.json").read_text())
    target = get_task(spec["task"]).draw(spec["cell"], int(spec["function_seed"]))
    method = get_method(spec["method"])
    splits = load_splits(spec["dataset_dir"], n=int(spec["n"]), verify_hashes=True)
    X = splits["diagnostic"].tensors[0]
    if int(X.shape[0]) != POOL_N:
        raise SystemExit(f"{run_dir}: diagnostic pool has {X.shape[0]} examples, protocol requires {POOL_N}")
    results = {}
    for stem in ("init", "best"):
        model = method.build({"width": int(spec["arch"]["width"]), "hidden": int(spec["arch"]["hidden"]),
                              "hp": spec["hp"], "auxiliary_init_seed": 0}, target)
        state = torch.load(run_dir / f"{stem}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            results[stem] = probe_r2(model, target, X)
    nodes = results["best"]["nodes"]
    if results["init"]["nodes"] != nodes:
        raise SystemExit(f"{run_dir}: init/best probes disagree on nodes")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, r2_trained=results["best"]["matrix"], r2_init=results["init"]["matrix"],
                        delta_r2=results["best"]["matrix"] - results["init"]["matrix"],
                        lambdas_trained=results["best"]["selected_lambdas"],
                        lambdas_init=results["init"]["selected_lambdas"])
    (out_dir / f"{name}.json").write_text(json.dumps({
        "run_dir": str(run_dir.relative_to(M.REPO)), "task": spec["task"], "task_preset": spec["task_preset"],
        "method": spec["method"], "n": int(spec["n"]), "function_seed": int(spec["function_seed"]),
        "hpo_trial": int(spec["hpo_trial"]), "nodes": nodes,
        "node_depths": [int(d) for d in results["best"]["node_depths"]], "layers": results["best"]["layers"],
        "pool_n": POOL_N,
        "features_kept": {stem: results[stem].get("features_kept") for stem in ("init", "best")},
        "features_total": results["best"].get("features_total")}, indent=1))
    return "probed"


def probes(panel: str, slot: int, slots: int) -> None:
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    # The ridge probes run in numpy/BLAS, whose pool defaults to every core: eight
    # workers x 48 threads put a load of 117 on the 24-core interactive node. Share
    # the node's cores among the workers instead.
    threadpool_limits(limits=max(1, (os.cpu_count() or 24) // max(1, slots)))
    _register()
    dirs = [d for d in winner_dirs(panel) if M.parse_unit(d.name)["arm"] not in PROBE_SKIP][slot::max(1, slots)]
    counts = defaultdict(int)
    t0 = time.time()
    for d in dirs:
        m = M.parse_unit(d.name)
        try:
            counts[probe_one(d, POST / panel / "probes" / m["cell"] / m["arm"])] += 1
        except ValueError as exc:            # a model without a layer-representation rule
            counts["unsupported"] += 1
            print(f"unsupported {d.name}: {exc}", flush=True)
        except SystemExit as exc:
            counts["failed"] += 1
            print(f"FAILED {d.name}: {exc}", flush=True)
    print(f"probes {panel} slot {slot}/{slots}: {dict(counts)} in {time.time() - t0:.0f} s", flush=True)


def probes_analyze(panel: str) -> None:
    rows = []
    for js in sorted((POST / panel / "probes").glob("*/*/*.json")):
        meta = json.loads(js.read_text())
        arr = np.load(js.with_suffix(".npz"))
        cell, arm = js.parents[1].name, js.parent.name
        for i, (node, depth) in enumerate(zip(meta["nodes"], meta["node_depths"])):
            for j, layer in enumerate(meta["layers"]):
                rows.append({"cell": cell, "arm": arm, "paper_name": M.PAPER.get(arm, arm), "n": int(meta["n"]),
                             "fs": int(meta["function_seed"]), "node": node, "node_depth": int(depth),
                             "layer": layer, "r2_init": float(arr["r2_init"][i, j]),
                             "r2_trained": float(arr["r2_trained"][i, j]), "delta_r2": float(arr["delta_r2"][i, j])})
    out = ANALYSIS / panel / "probes"
    out.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"{panel}: no probe records")
        return
    pv = pd.DataFrame(rows).sort_values(["cell", "arm", "n", "fs", "node", "layer"], kind="stable")
    _write_probe_tables(panel, out, pv)


def _write_probe_tables(panel: str, out: Path, pv: pd.DataFrame, merged_from: list[str] | None = None) -> None:
    """probe_values.csv (the primary per-winner table) and the two tables that are functions of it."""
    pv.to_csv(out / "probe_values.csv", index=False)
    # the mean intermediate R^2 per layer: mean over the roster, then over instances
    per = pv.groupby(["cell", "arm", "layer", "n", "fs"])[["r2_init", "r2_trained", "delta_r2"]].mean().reset_index()
    summ = []
    for (cell, arm, layer, n), g in per.groupby(["cell", "arm", "layer", "n"]):
        row = dict(cell=cell, arm=arm, paper_name=M.PAPER.get(arm, arm), layer=layer, n=int(n))
        for k in ("r2_init", "r2_trained", "delta_r2"):
            st = tci(g[k])
            row.update({f"{k}_mean": st["mean"], f"{k}_lo": st["ci95_lo"], f"{k}_hi": st["ci95_hi"]})
        row["n_instances"] = int(len(g))
        summ.append(row)
    pd.DataFrame(summ).to_csv(out / "probe_summary.csv", index=False)
    # the best layer per intermediate (max over layers of the trained R^2), then per level
    best = pv.loc[pv.groupby(["cell", "arm", "n", "fs", "node"]).r2_trained.idxmax()]
    lev = []
    for (cell, arm, n, depth), g in best.groupby(["cell", "arm", "n", "node_depth"]):
        st = tci(g.groupby("fs").r2_trained.mean())
        lev.append(dict(cell=cell, arm=arm, paper_name=M.PAPER.get(arm, arm), n=int(n), node_depth=int(depth),
                        best_layer_r2_mean=st["mean"], lo=st["ci95_lo"], hi=st["ci95_hi"],
                        n_instances=st["n_instances"], n_nodes=int(g.node.nunique()),
                        best_layer_mode=g.layer.mode().iloc[0]))
    pd.DataFrame(lev).to_csv(out / "probe_levels.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "panel": panel, "generated_local": M.now_local(), "winners_probed": int(pv.groupby(["cell", "arm", "n", "fs"]).ngroups),
        "cells": sorted(pv.cell.unique()), "arms": sorted(pv.arm.unique()),
        "layers": sorted(pv.layer.unique(), key=lambda s: int(s[1:])), **({"merged_from": merged_from} if merged_from else {})}, indent=1))
    print(f"{panel}: {pv.groupby(['cell', 'arm', 'n', 'fs']).ngroups} winners probed -> {out.relative_to(M.STUDY)}")


PROBE_KEY = ["cell", "arm", "n", "fs", "node", "layer"]


def probes_merge(dst: str, srcs: list[str]) -> None:
    """The probe tables of panels read into DST's record (amendment 9: rand8b's winners join main8b): the per-winner rows of
    each source join DST's probe_values.csv (a row already there for the same winner, node and layer is replaced), and the
    summary and level tables are recomputed over the union, so every mean runs over the record's full instance set."""
    out = ANALYSIS / dst / "probes"
    pv = pd.read_csv(out / "probe_values.csv")
    for src in srcs:
        f = ANALYSIS / src / "probes" / "probe_values.csv"
        if not f.exists():
            raise SystemExit(f"{f} missing: run probes-analyze {src} first")
        sv = pd.read_csv(f)
        pv = pd.concat([pv[~pv.set_index(PROBE_KEY).index.isin(sv.set_index(PROBE_KEY).index)], sv], ignore_index=True)
        print(f"{dst}: probe rows of {src} joined ({len(sv)} rows, {sv.groupby(['cell', 'arm', 'n', 'fs']).ngroups} winners)")
    pv = pv.sort_values(PROBE_KEY, kind="stable").reset_index(drop=True)
    old = json.loads((out / "summary.json").read_text()).get("merged_from", []) if (out / "summary.json").exists() else []
    _write_probe_tables(dst, out, pv, merged_from=sorted(set(old) | set(srcs)))


# ---------------------------------------------------------------- Boundless DAS

def _frozen() -> dict:
    from csfrbench.das_probe import load_frozen_protocol
    return load_frozen_protocol(FROZEN)


def das_datasets(panel: str) -> None:
    _register()
    from csfrbench import das_teacher_cf as tc
    from csfrbench.registry import get_task
    cfg = _frozen()
    cd = cfg["counterfactual_data"]
    sizes = {"das_train": int(cd["train_pairs_per_variable"]), "das_val": int(cd["val_pairs_per_variable"]),
             "das_test": int(cd["test_pairs_per_variable"])}
    seed = int(cd["split_seed_for_the_frozen_run"])
    root = POST / panel / "causal" / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    seen = set()
    for w in winner_dirs(panel, arms=MECH_ARMS):
        spec = json.loads((w / "spec.json").read_text())
        key = f"{spec['task']}|{int(spec['function_seed'])}"
        if key in seen:
            continue
        seen.add(key)
        if key in index and (root / index[key] / "das_manifest.json").exists():
            continue
        target = get_task(spec["task"]).draw(spec["cell"], int(spec["function_seed"]))
        roster = tc.aligned_variables(target, include_readout=True)
        out = tc.ensure_das_dataset(root, target, sizes, seed, include_readout=True)
        index[key] = out.name
        print(f"{key}: {len(roster)} aligned variables -> {out.name}", flush=True)
        index_path.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    print(f"{panel}: {len(index)} DAS datasets indexed")


# The record's two other fit lists (16-main-final section 8; 08-main's frozen DAS plan), for a panel that asks for
# them (amendment 8: the paper's setting at 64,000 steps feeds the appendix stems causal_probe_detail and
# linear_probe_detail): `indep` = one variable per fit on the depth-4 tree at n = 2^13; `init` = the joint fit on
# the exact pre-training initialization (das_probe --control init) on the three mechanism workloads at 2^13 and
# the tree at 2^15. A list is a folder under fits/, and the harvest names each fit by its folder.
EXTRA_LISTS = {
    # one fit per variable (das_probe refuses a roster-wide indep fit); the record ran it for LeWi alone
    "indep": dict(tasks=("tree_functional_anova",), ns=(8192,), args="--formulation indep --no-w", arms=(M.LEWI,), per_var=True),
    "init": dict(tasks=("tree_functional_anova", "dag_functional_anova", "tree_fourier"), ns=(8192,), args="--formulation joint --control init --no-w",
                 extra={"tree_functional_anova": (32768,)}),
}


def das_units(panel: str, ns: list[int] | None = None, lists: tuple[str, ...] = ("joint",)) -> Path:
    cfg = _frozen()
    K = int(cfg["restarts"]["K"])
    root = POST / panel / "causal"
    index = json.loads((root / "datasets" / "index.json").read_text())
    harvested = _harvested_stems(panel, lists)
    panel_d = {int(M.CELLS[c][2]) for c in panel_cells(panel)}      # the extra lists follow the panel's own ambient dimension
    lines, n_w = [], 0
    for lst in lists:
        fits = root / "fits" / lst
        fits.mkdir(parents=True, exist_ok=True)
        plan = EXTRA_LISTS.get(lst)
        for w in winner_dirs(panel, arms=MECH_ARMS, cells=MECH_CELLS):
            spec = json.loads((w / "spec.json").read_text())
            if lst == "joint":
                if ns and int(spec["n"]) not in ns:
                    continue
                args = "--formulation joint --no-w"
            else:
                wanted = set(plan["ns"]) | set(plan.get("extra", {}).get(spec["task"], ()))
                if spec["task"] not in plan["tasks"] or int(spec["n"]) not in wanted or int(spec.get("d", 32)) not in panel_d:
                    continue
                if plan.get("arms") and M.parse_unit(w.name)["arm"] not in plan["arms"]:      # the arm key (a panel may run a key as another registry id: arm_method)
                    continue
                args = plan["args"]
            key = f"{spec['task']}|{int(spec['function_seed'])}"
            if key not in index:
                continue
            n_w += 1
            d = root / "datasets" / index[key]
            stem = f"{spec['task']}__fs{int(spec['function_seed'])}__n{int(spec['n'])}__{spec['method']}"
            variables = [None]
            if plan and plan.get("per_var"):
                variables = _roster_vars(d)
            for layer in das_layers(panel):
                for seed in range(K):
                    for var in variables:
                        tag = f"{stem}__L{layer}__s{seed}" + (f"__v{_slug(var)}" if var else "")
                        if (lst, tag) in harvested or (fits / f"{tag}.json").exists():
                            continue
                        lines.append(f"--frozen-config {FROZEN} --target {w} --dataset {d} --layer {layer} "
                                     f"--das-seed {seed} {args}" + (f" --vars {var}" if var else "") + f" --out {fits / tag}")
    units = root / "units"
    units.mkdir(parents=True, exist_ok=True)
    path = units / "joint.list"
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"{panel}: {len(lines)} pending fits ({', '.join(lists)}) over {n_w} winner-lists -> {path.relative_to(M.STUDY)}")
    return path


def _roster_vars(pool_dir: Path) -> list[str]:
    """The intermediates of a counterfactual pool (its manifest's roster without the readout): the variables the
    record's independent probes were fit on, one fit each."""
    rec = json.loads((pool_dir / "das_manifest.json").read_text())
    return [v["var"] for v in rec["roster"] if not v.get("readout") and v["var"] != "y"]


def _slug(var: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in var)


def das_run(panel: str, slot: int, slots: int) -> None:
    import torch
    torch.set_num_threads(1)
    _register()
    from csfrbench import das_probe
    # The frozen config names the four hidden layers of the record's 32x4 backbone. This
    # backbone has eight; the layer set is the only constant changed, and the fit
    # record still carries the frozen metric's version and hash, so the analysis labels
    # layers 5-8 as the extension.
    _fc = das_probe.frozen_constants

    layers = das_layers(panel)

    def fc8(cfg):
        d = _fc(cfg)
        d["layers"] = list(layers)
        return d
    das_probe.frozen_constants = fc8
    rows = [r.strip() for r in (POST / panel / "causal" / "units" / "joint.list").read_text().splitlines() if r.strip()]
    rows = rows[slot::max(1, slots)]
    os.chdir(M.REPO)
    done = failed = skipped = 0
    t0 = time.time()
    for row in rows:
        args = row.split()
        out = Path(args[args.index("--out") + 1])
        if out.with_suffix(".json").exists():
            skipped += 1
            continue
        t1 = time.time()
        sys.argv = ["das_probe", *args, "--threads", "1"]
        try:
            das_probe.main()
            done += 1
            print(f"fit {out.name}: {time.time() - t1:.0f} s", flush=True)
        except SystemExit as exc:
            if exc.code in (0, None):
                done += 1
            else:
                failed += 1
                print(f"FIT FAILED ({exc}): {row}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FIT FAILED ({type(exc).__name__}: {exc}): {row}", flush=True)
    print(f"das {panel} slot {slot}/{slots}: {done} done, {skipped} skipped, {failed} failed "
          f"in {time.time() - t0:.0f} s", flush=True)


def das_harvest(panel: str) -> pd.DataFrame:
    tidy = POST / panel / "causal" / "tidy"
    tidy.mkdir(parents=True, exist_ok=True)
    with open(tidy / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out = tidy / "das_fits.parquet"
        old = None
        if out.exists():
            try:
                old = pd.read_parquet(out)
            except Exception as exc:  # noqa: BLE001
                print(f"das_fits.parquet unreadable ({exc}); rebuilding from the fit records")
        have = set(zip(old.fit_list, old.fit_stem)) if old is not None else set()
        rows, taken = [], []
        for j in sorted((POST / panel / "causal" / "fits").glob("*/*.json")):
            if (j.parent.name, j.stem) in have:
                taken.append(j)                 # harvested earlier, not yet compacted
                continue
            try:
                rec = json.loads(j.read_text())
            except ValueError:
                continue                        # a worker is still writing it; next pass
            taken.append(j)
            rec.pop("history", None)
            rec["fit_list"] = j.parent.name
            rec["fit_stem"] = j.stem
            for k, v in list(rec.items()):
                if isinstance(v, (dict, list)):
                    rec[k] = json.dumps(v)
            rows.append(rec)
        if not rows:
            print(f"{panel}: no new fits ({len(have)} harvested)")
            _compact_fits(panel, taken)
            return old if old is not None else pd.DataFrame()
        df = pd.DataFrame(rows)
        if old is not None:
            df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["fit_list", "fit_stem"], keep="last")
        tmp = out.with_name("das_fits.tmp.parquet")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        print(f"{panel}: {len(rows)} new fits harvested; {len(df)} total -> {out.relative_to(M.STUDY)}")
        _compact_fits(panel, taken)
        return df


def _harvested_stems(panel: str, lists: tuple[str, ...] = ("joint",)) -> set[tuple[str, str]]:
    """(fit list, stem) pairs already in the tidy parquet: the record of a compacted fit."""
    out = POST / panel / "causal" / "tidy" / "das_fits.parquet"
    if not out.exists():
        return set()
    try:
        df = pd.read_parquet(out, columns=["fit_list", "fit_stem"])
    except Exception:  # noqa: BLE001
        return set()
    df = df[df.fit_list.isin(lists)]
    return set(zip(df.fit_list, df.fit_stem))


def _compact_fits(panel: str, files: list[Path]) -> None:
    """Move harvested fit records into one tar per panel and free their inodes.

    A fit is one JSON of about 150 kB; main2's 2,304 mechanism winners x 24 fits would
    hold 55,000 inodes, 5.5% of the scratch quota, for records the parquet already carries
    in full (only the training history is dropped). The tar keeps every record verbatim
    under posthoc/PANEL/causal/archive/; das_units treats a stem in the parquet as done.
    """
    if not files:
        return
    import tarfile
    arch = POST / panel / "causal" / "archive"
    arch.mkdir(parents=True, exist_ok=True)
    by_list: dict[str, list[Path]] = {}
    for j in files:
        by_list.setdefault(j.parent.name, []).append(j)
    for lst, group in by_list.items():
        tar = arch / f"fits_{lst}.tar"
        with tarfile.open(tar, "a") as tf:
            for j in group:
                tf.add(j, arcname=f"{j.parent.name}/{j.name}")
        for j in group:
            j.unlink()
        print(f"{panel}: {len(group)} harvested fit records compacted into {tar.relative_to(M.STUDY)}")


def _listify(v):
    if isinstance(v, str):
        v = json.loads(v)
    if isinstance(v, dict):
        keys = sorted(v, key=lambda s: int("".join(ch for ch in s if ch.isdigit()) or 0))
        return [float(v[k]) for k in keys]
    return [float(x) for x in v]


def das_cells(df: pd.DataFrame, cell_of: dict | None = None) -> pd.DataFrame:
    """Per-variable cells after the frozen selection: K restarts, the restart with the best
    validation macro per (checkpoint, layer) kept; every variable with its held-out and
    no-patch R^2 (16-main-final port_posthoc.das_cells)."""
    df = df[df.fit_list == "joint"].copy()
    df["das_seed"] = df.das_seed.astype(int)
    df = df[df.das_seed < DAS_K]
    for k in ("layer", "n_train", "function_seed"):
        df[k] = df[k].astype(int)
    group = ["task", "method", "function_seed", "n_train"]
    idx = df.groupby(group + ["layer"], dropna=False).best_val_macro.idxmax()
    rows = []
    for _, r in df.loc[idx].iterrows():
        names = [e["var"] for e in json.loads(r.roster)]
        val, test, none, dims = (_listify(r[k]) for k in ("val_per_var", "test_per_var", "no_edit_per_var", "dims"))
        for i, var in enumerate(names):
            rows.append({"cell": (cell_of or TASK_CELL).get(r.task, r.task), "task": r.task, "method": r.method,
                         "fs": int(r.function_seed), "n": int(r.n_train), "var": var, "layer": int(r.layer),
                         "frozen_layer": int(r.layer) <= 4, "restart": int(r.das_seed), "stem": r.fit_stem,
                         "val_r2": val[i], "test_r2": test[i], "test_r2_nopatch": none[i],
                         "delta_cf": test[i] - none[i], "dim": int(dims[i]),
                         "residual_dim": int(r.residual_dim), "ambient_dim": int(r.ambient_dim)})
    return pd.DataFrame(rows)


def das_analyze(panel: str) -> None:
    src = POST / panel / "causal" / "tidy" / "das_fits.parquet"
    out = ANALYSIS / panel / "causal"
    out.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"{panel}: no DAS fits")
        return
    cells = das_cells(pd.read_parquet(src), task_cell(panel))
    arm_of = {v: k for k, v in M.PANELS[panel].get("arm_method", {}).items()}     # amendment 11: a fit's method is the registry id; the tables carry the arm key
    if arm_of and not cells.empty:
        cells["stem"] = [st.replace(f"__{m}__", f"__{arm_of[m]}__") if m in arm_of else st for st, m in zip(cells.stem, cells.method)]
        cells["method"] = cells.method.map(lambda m: arm_of.get(m, m))
    layers_all = das_layers(panel)
    if cells.empty:
        print(f"{panel}: no selected DAS cells")
        return
    cells["paper_name"] = cells.method.map(lambda a: M.PAPER.get(a, a))
    cells.sort_values(["cell", "method", "n", "fs", "var", "layer"], kind="stable").to_csv(out / "das_cells.csv", index=False)
    key = ["cell", "method", "fs", "n"]
    # the layer per variable chosen on validation (ties to the earlier layer) over all eight
    # hidden layers (decision 2026-09-16: the frozen config's layers 1-4 are not read
    # separately; this backbone is always read on every layer). A winner enters only once
    # every layer has its selected fit on record; a partial winner would be scored on
    # whichever layers happened to finish first.
    n_layers = cells.groupby(key)["layer"].nunique()
    full = n_layers[n_layers >= len(layers_all)].index
    sub = cells.set_index(key).loc[full].reset_index() if len(full) else cells.iloc[0:0]
    for stale in ("das_star_frozen.csv", "das_scores_frozen.csv", "das_summary_frozen.csv", "das_contrasts_frozen.csv",
                  "das_star_all.csv", "das_scores_all.csv", "das_summary_all.csv", "das_contrasts_all.csv"):
        (out / stale).unlink(missing_ok=True)
    if sub.empty:
        print(f"{panel}: no winner has every layer's fit yet")
        return
    star = sub.loc[sub.groupby(key + ["var"]).val_r2.idxmax()]
    star.to_csv(out / "das_star.csv", index=False)
    scores = star.groupby(key).agg(S=("test_r2", "mean"), S_nopatch=("test_r2_nopatch", "mean"),
                                   dS_cf=("delta_cf", "mean"), n_vars=("var", "size"),
                                   layers=("layer", lambda s: json.dumps(sorted(set(int(x) for x in s))))).reset_index()
    scores["paper_name"] = scores.method.map(lambda a: M.PAPER.get(a, a))
    scores.to_csv(out / "das_scores.csv", index=False)
    summ = []
    for (cell, method, n), g in scores.groupby(["cell", "method", "n"]):
        row = dict(cell=cell, method=method, paper_name=M.PAPER.get(method, method), n=int(n))
        for k in ("S", "S_nopatch", "dS_cf"):
            st = tci(g[k])
            row.update({f"{k}_mean": st["mean"], f"{k}_lo": st["ci95_lo"], f"{k}_hi": st["ci95_hi"]})
        row["n_instances"] = int(len(g))
        summ.append(row)
    pd.DataFrame(summ).to_csv(out / "das_summary.csv", index=False)
    con = []
    by = scores.set_index(["cell", "method", "fs", "n"])
    for (cell, n), _g in scores.groupby(["cell", "n"]):
        for ref in ("noreg", "graph_oracle"):
            try:
                a, b = by.loc[(cell, M.LEWI)], by.loc[(cell, ref)]
            except KeyError:
                continue
            a, b = a[a.index.get_level_values("n") == n], b[b.index.get_level_values("n") == n]
            idx = a.index.intersection(b.index)
            if len(idx) == 0:
                continue
            for k in ("S", "S_nopatch", "dS_cf"):
                d = (a.loc[idx, k] - b.loc[idx, k]).values
                con.append(dict(cell=cell, n=int(n), arm=M.LEWI, ref=ref, contrast=f"LeWi - {M.PAPER.get(ref, ref)}",
                                key=k, **tci(d), n_pairs=len(idx), wins=int((d > 0).sum())))
    pd.DataFrame(con).to_csv(out / "das_contrasts.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "panel": panel, "generated_local": M.now_local(), "fits": int(len(pd.read_parquet(src, columns=["fit_stem"]))),
        "winners": int(sub.groupby(key).ngroups), "winners_partial": int(cells.groupby(key).ngroups - sub.groupby(key).ngroups),
        "cells": sorted(sub.cell.unique()), "layers": layers_all}, indent=1))
    print(f"{panel}: DAS {sub.groupby(key).ngroups} whole winners -> {out.relative_to(M.STUDY)}")


def das_merge(dst: str, srcs: list[str]) -> None:
    """The DAS fit records of panels read into DST's record (amendment 9: rand8b's Dense, GO(all) and LeWi winners join
    main8b): each source's tidy parquet joins DST's (a fit is one (list, stem); a duplicate keeps the source's row), then
    DST's tables are rebuilt over the union by das_analyze."""
    tidy = POST / dst / "causal" / "tidy"
    out = tidy / "das_fits.parquet"
    df = pd.read_parquet(out)
    for src in srcs:
        f = POST / src / "causal" / "tidy" / "das_fits.parquet"
        if not f.exists():
            raise SystemExit(f"{f} missing: run das-harvest {src} first")
        sv = pd.read_parquet(f)
        df = pd.concat([df, sv], ignore_index=True).drop_duplicates(subset=["fit_list", "fit_stem"], keep="last")
        print(f"{dst}: DAS fits of {src} joined ({len(sv)} fits)")
    with open(tidy / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        tmp = out.with_name("das_fits.tmp.parquet")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)
    print(f"{dst}: {len(df)} fits in the tidy record")
    das_analyze(dst)


# ---------------------------------------------------------------- a winners panel's readouts over its source panel's (amendment 10)

def own_units(panel: str, base: str) -> dict[tuple, str]:
    """The winners a winners panel carries itself, {(cell, fs, n, method of the unit's spec): arm}: every unit under its run root
    that is not BASE's unit of the same name. A link into BASE's root is BASE's winner, whose readouts stand."""
    out = {}
    for d in winner_dirs(panel):
        b = M.RUNS / base / d.name
        if b.exists() and d.resolve() == b.resolve():
            continue
        m = M.parse_unit(d.name)
        out[(m["cell"], int(m["fs"]), int(m["n"]), json.loads((d / "spec.json").read_text())["method"])] = m["arm"]
    return out


def probes_overlay(dst: str, base: str) -> None:
    """DST's probe tables over BASE's (amendment 10: main8b-in5 = the twin under the input rule): DST's own winners replace
    BASE's rows of the same condition (cell, arm, instance, size); BASE's rows stand for every other condition, including
    those DST carries as links into BASE's root; the summary and level tables are recomputed over the union. DST's re-probes
    of BASE's own winners are compared with BASE's rows and reported, not used. Runs after probes-analyze DST."""
    own = {(c, a, n, fs) for (c, fs, n, _m), a in own_units(dst, base).items()}
    out = ANALYSIS / dst / "probes"
    mine = pd.read_csv(out / "probe_values.csv")
    bv = pd.read_csv(ANALYSIS / base / "probes" / "probe_values.csv")
    cond = ["cell", "arm", "n", "fs"]
    in_own_m = pd.Series(list(zip(mine.cell, mine.arm, mine.n, mine.fs))).isin(own).values
    in_own_b = pd.Series(list(zip(bv.cell, bv.arm, bv.n, bv.fs))).isin(own).values
    rep = mine[~in_own_m].set_index(PROBE_KEY).sort_index()
    ref = bv.set_index(PROBE_KEY).reindex(rep.index)
    agree = int(np.isclose(rep.r2_trained, ref.r2_trained, atol=1e-9, equal_nan=True).sum()) if len(rep) else 0
    pv = pd.concat([bv[~in_own_b], mine[in_own_m]], ignore_index=True).sort_values(PROBE_KEY, kind="stable").reset_index(drop=True)
    _write_probe_tables(dst, out, pv, merged_from=[base])
    print(f"{dst}: {len(own)} own conditions replace {base}'s rows ({int(in_own_b.sum())} rows out, {int(in_own_m.sum())} in); "
          f"{rep.reset_index().groupby(cond).ngroups} of {base}'s winners re-probed here: {agree} of {len(rep)} rows equal to 1e-9")


def das_overlay(dst: str, base: str) -> None:
    """DST's tidy DAS record over BASE's (amendment 10): the fits of DST's own winners replace BASE's fits of the same condition
    and list; their method is relabelled to the unit's arm (the repaired winners ran as the study-local arm lewi-v1-rgm-in5, the
    same architecture under the input rule, so their fit stems name that arm); BASE's fits stand for every other condition.
    DST's re-fits of BASE's own winners are compared with BASE's and reported, not used. Then das_analyze(DST)."""
    own = own_units(dst, base)
    cell_of = task_cell(dst)
    tidy = POST / dst / "causal" / "tidy"
    mine = pd.read_parquet(tidy / "das_fits.parquet")
    bv = pd.read_parquet(POST / base / "causal" / "tidy" / "das_fits.parquet")
    # an own winner's fits match by the unit's registry id or by its arm key: a previous overlay relabelled them to the key, and
    # the overlay must be repeatable (2026-09-22: a second overlay had dropped the earlier repaired winners' fits)
    own_by_cond = {}
    for (c, fs, n, m), a in own.items():
        own_by_cond.setdefault((c, fs, n), set()).update({m, a})
        own_by_cond[(c, fs, n, "arm")] = a
    key_m = [(cell_of.get(t, t), int(fs), int(n), m) for t, fs, n, m in zip(mine.task, mine.function_seed, mine.n_train, mine.method)]
    arm_m = pd.Series([own_by_cond.get((c, fs, n, "arm")) if m in own_by_cond.get((c, fs, n), ()) else None for c, fs, n, m in key_m], index=mine.index)
    own_cond = {(c, a, fs, n) for (c, fs, n, _m), a in own.items()}
    in_own_b = pd.Series([(cell_of.get(t, t), m, int(fs), int(n)) in own_cond
                          for t, m, fs, n in zip(bv.task, bv.method, bv.function_seed, bv.n_train)]).values
    take = mine[arm_m.notna()].copy()
    for i, r in take.iterrows():
        arm = arm_m[i]
        if r.method != arm:
            take.at[i, "fit_stem"] = r.fit_stem.replace(f"__{r.method}__", f"__{arm}__")
            take.at[i, "method"] = arm
    rest = mine[arm_m.isna()]
    ref = bv.set_index(["fit_list", "fit_stem"])
    idx = rest.set_index(["fit_list", "fit_stem"]).index
    have = idx.isin(ref.index)
    agree = int(np.isclose(rest[have].test_macro.values, ref.loc[idx[have]].test_macro.values, atol=1e-9, equal_nan=True).sum()) if have.any() else 0
    df = pd.concat([bv[~in_own_b], take], ignore_index=True).drop_duplicates(subset=["fit_list", "fit_stem"], keep="last")
    with open(tidy / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        tmp = tidy / "das_fits.tmp.parquet"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, tidy / "das_fits.parquet")
    print(f"{dst}: {len(own)} own winners' fits ({len(take)}) replace {base}'s ({int(in_own_b.sum())}); {len(df)} fits in the tidy record; "
          f"{int(have.sum())} re-fits of {base}'s own winners here: {agree} equal to {base}'s in test macro R^2 to 1e-9")
    das_analyze(dst)


# ---------------------------------------------------------------- embeddings

def _w5(panel: str):
    """The wave-5 embedding code (12-extra/src/w5_metrics.py) pointed at this panel's
    winners and outputs: the same tracking, the same panel payload, the same renderer."""
    W5 = _load_module("w5_metrics", M.X12_SRC / "w5_metrics.py")
    out = POST / panel / "embeddings"
    W5.MET = out
    W5.PLOTS = out / "plots"
    W5.ARMS = list(EMB_ARMS)
    W5.NAME = {a: M.PAPER.get(a, a) for a in EMB_ARMS}
    W5.EMB_CELL_LABEL = dict(M.CELL_LABEL)
    W5.EXPECTED_NODES = None
    root = M.RUNS / panel
    keep = {d.name for d in winner_dirs(panel, arms=EMB_ARMS)}

    def unit_dir(cell, arm, fs, n):
        for d in root.glob(f"{cell}__{arm}__fs{fs}__n{n}__t*"):
            if d.name in keep:
                return d
        return None
    W5._unit_dir = unit_dir
    return W5


def embeddings(panel: str, fs: int, ns: list[int]) -> None:
    _register()
    W5 = _w5(panel)
    TR = W5._load_tracking_module()
    TR.TEACHER_COLORS = TEACHER_COLORS
    cells = panel_cells(panel)
    out = ANALYSIS / panel / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    panels, statics, targets = [], [], {}
    for cell in cells:
        cell_panels = []
        for n in ns:
            for arm in EMB_ARMS:
                got = W5._panel(cell, arm, fs, n, TR)
                if got is None:
                    continue
                p, t = got
                cell_panels.append(p)
                targets.setdefault(cell, t)
        if not cell_panels:
            print(f"{cell}: no retained LeWi winner at instance {fs} for sizes {ns} yet", flush=True)
            continue
        panels.extend(cell_panels)
        statics.append((cell, W5._static_cell(cell, fs, cell_panels, targets[cell], f"embeddings_{cell}")))
        print(f"{cell}: {len(cell_panels)} panels", flush=True)
    if not panels:
        return
    cells_present = [c for c in cells if any(p["cell"] == c for p in panels)]
    payload = {"cells": cells_present, "cell_names": {c: M.CELL_LABEL[c] for c in cells_present},
               "ns": sorted({p["n"] for p in panels}), "arms": [{"arm": a, "name": M.PAPER.get(a, a)} for a in EMB_ARMS],
               "fs": fs, "panels": panels}
    (out / "panels.json").write_text(json.dumps(payload))
    rows = []
    for p in panels:
        st = p.get("stats") or {}
        rows.append(dict(cell=p["cell"], label=M.CELL_LABEL[p["cell"]], n=p["n"], arm=p["arm"], unit=p.get("unit"),
                         layers=len(p.get("layers", [])), nodes=len(p.get("nodes", [])),
                         edges=sum(len(e["pairs"]) for e in p.get("edges", [])), floor=p.get("floor"),
                         val_nmse=st.get("val"), lit=json.dumps(st.get("lit")), sizes=json.dumps(st.get("sizes"))))
    pd.DataFrame(rows).to_csv(out / "cells.csv", index=False)
    (out / "static.html").write_text("".join(
        f"<div class='emb-static'><h4>{html.escape(M.CELL_LABEL[c])}</h4>{s}</div>" for c, s in statics))
    (out / "positions3d.js").write_text((NEXTGEN_SRC / "positions3d.js").read_text())
    print(f"{panel}: embeddings of {len(panels)} panels over {cells_present} -> {out.relative_to(M.STUDY)}")


# ---------------------------------------------------------------- SLURM

# The scheduler verb of this script (job packs of the probe and DAS runs) is not part of this archive; the verbs above
# run on one machine (`--slot`/`--slots` split the work).


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", choices=["probes", "probes-analyze", "das-datasets", "das-units", "das-run",
                                     "das-harvest", "das-analyze", "embeddings", "sbatch", "probes-merge", "das-merge",
                                     "probes-overlay", "das-overlay"])
    ap.add_argument("panel")
    ap.add_argument("--from", dest="srcs", nargs="*", default=None,
                    help="probes-merge / das-merge: the panels read into PANEL's record; probes-overlay / das-overlay: the one source panel PANEL's own winners overlay")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--fs", type=int, default=6000)
    ap.add_argument("--ns", type=int, nargs="*", default=None)
    ap.add_argument("--kind", default="probes")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--hours", type=int, default=8)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--lists", nargs="*", default=None, help="DAS fit lists for das-units (default: joint)")
    args = ap.parse_args(argv)
    if args.verb == "probes":
        probes(args.panel, args.slot, args.slots)
    elif args.verb == "probes-analyze":
        probes_analyze(args.panel)
    elif args.verb == "das-datasets":
        das_datasets(args.panel)
    elif args.verb == "das-units":
        das_units(args.panel, args.ns, tuple(args.lists) if args.lists else ("joint",))
    elif args.verb == "das-run":
        das_run(args.panel, args.slot, args.slots)
    elif args.verb == "das-harvest":
        das_harvest(args.panel)
    elif args.verb == "das-analyze":
        das_analyze(args.panel)
    elif args.verb == "embeddings":
        ns = args.ns or sorted(M.PANELS[args.panel]["ns"])
        if not args.ns:
            ns = [ns[0], ns[len(ns) // 2], ns[-1]]
        embeddings(args.panel, args.fs, ns)
    elif args.verb == "sbatch":
        sbatch(args.panel, args.kind, args.jobs, args.workers, args.hours)
    elif args.verb == "probes-merge":
        probes_merge(args.panel, list(args.srcs or []))
    elif args.verb == "das-merge":
        das_merge(args.panel, list(args.srcs or []))
    elif args.verb in ("probes-overlay", "das-overlay"):
        if not args.srcs or len(args.srcs) != 1:
            raise SystemExit(f"{args.verb}: --from names exactly one source panel")
        (probes_overlay if args.verb == "probes-overlay" else das_overlay)(args.panel, args.srcs[0])


if __name__ == "__main__":
    main()
