"""One checkpoint state -> one JSON: the recovered computation graph of a LeWi
checkpoint, its probe correspondence, and the agreement with the target graph,
under the learned wiring and the two rewiring controls.

    python rg_unit.py --checkpoint <run_dir> --cell dag --fs 6000 --n 32768 \
        --which best --targets ../results/targets/d8 --out ../results/units/d8

The extraction, the correspondence and the scores are the component-DAG study's
(../../component-dag/src), imported unchanged, at its registered design:
components = the live units of one hidden stage that the same inputs reach (the
quotient of the wiring by input ancestry); an edge = any open forward connection;
correspondence = ridge probes on held-out pools, one-to-one, admissible only where
the component beats every chance set of the same stage composition.  The seeds
follow the component-DAG runner, so a d = 32 unit reproduces its archived record.

Added here: the direct-connection reading (a target edge counts only when the two
matched components share an open connection, no relay), the per-component probe
table and the positions, for the report's inspection views.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CD = HERE.parents[1] / "component-dag" / "src"
for p in (str(HERE), str(CD)):
    if p not in sys.path:
        sys.path.insert(0, p)
import cd_extract as ce      # noqa: E402
import cd_score as cs        # noqa: E402
import cd_unit as cu         # noqa: E402
import cd_view as cv         # noqa: E402
from csfrbench.utils.seeds import derive  # noqa: E402

VERSION = "rg-v1"
DESIGN = ("field_stage", "any", "significant")
WIRINGS = cu.WIRINGS
POOLS = cu.POOLS


def direct_scores(t, node_of, E):
    """Edge agreement when only a direct open connection between the two matched
    components counts: no path through a component that stands for no node."""
    Ehat = {(node_of[a], node_of[b]) for a, b in E if a in node_of and b in node_of}
    Et = cs.target_edges(t)
    out = {"n_pred_edges": len(Ehat), "edge": cs.prf(len(Ehat & Et), len(Ehat), len(Et))}
    groups = {"first": lambda e: e[0][0] == "leaf", "higher": lambda e: e[0][0] == "node",
              "readout": lambda e: e[1][0] == "y"}
    for name, keep in groups.items():
        T, P = {e for e in Et if keep(e)}, {e for e in Ehat if keep(e)}
        out[name] = cs.prf(len(P & T), len(P), len(T))
    return out


def probe_table(rows, t):
    """Every component against every target node: held-out excess R^2 and p."""
    return [{"comp": r["comp"], "size": r["size"], "stages": r["stages"],
             "excess": [round(float(x), 4) for x in r["test"]],
             "raw": [round(float(x), 4) for x in r["raw"]],
             "p": [round(float(x), 4) for x in r["p"]],
             "sig": [bool(x) for x in r["sig"]]} for r in rows]


def hidden_columns(v, live):
    cols, col_of, stage_cols = [], {}, {}
    for st in range(1, len(live) - 1):
        stage_cols[st] = []
        for u in np.flatnonzero(live[st]):
            col_of[(st, int(u))] = len(cols)
            stage_cols[st].append(len(cols))
            cols.append(v["H"][st - 1][:, u])
        stage_cols[st] = np.array(stage_cols[st], dtype=int)
    return cols, col_of, stage_cols


def run(checkpoint, cell, fs, n, which, targets_dir):
    t0 = time.time()
    tgt = np.load(Path(targets_dir) / f"{cell}__fs{fs}.npz", allow_pickle=True)
    meta = json.loads(str(tgt["meta"]))
    t = cs.canonical_target(meta)
    active_raw = [int(a) for a in meta["active_raw"]]
    X = np.concatenate([tgt[f"X_{p}"] for p in POOLS])
    cuts = np.cumsum([0] + [len(tgt[f"X_{p}"]) for p in POOLS])

    model, _, spec = cv.load(checkpoint, which=which)
    view = cv.learner(model, spec["method"], X)
    radius2 = float(model.hidden_layers[0].gate_offset)
    seed = int(derive(meta["fingerprint"], "cd", which, int(n)))
    comp_rule, edge_rule, matcher = DESIGN

    first = {}
    for raw, c in enumerate(t["canon"]):
        first.setdefault(c, raw)
    order = [first[v] for v in t["match_nodes"]]
    Zp = {p: tgt[f"Z_{p}"][:, order] for p in POOLS}

    out = {"version": VERSION, "design": "|".join(DESIGN), "setting": f"d{int(spec['d'])}",
           "cell": cell, "function_seed": int(fs), "n": int(n), "which": which,
           "checkpoint": str(checkpoint), "method": spec["method"], "d": int(spec["d"]),
           "fingerprint": meta["fingerprint"], "A": t["A"], "n_nodes": t["n_nodes"],
           "n_true_edges": len(cs.target_edges(t)), "radius2": radius2,
           "active_raw": active_raw,
           "target_parents": {str(v): [list(p) for p in ps] for v, ps in t["parents"].items()},
           "target_root": t["root"], "target_match_nodes": t["match_nodes"],
           "raw_node_parents": meta["node_parents"],
           "raw_readout_indices": list(meta["readout_indices"]),
           "target_depth": {str(k): v for k, v in t["depth"].items()},
           "target_readout": t["readout"],
           "positions": None if view["E"] is None else
           [[[round(float(x), 4) for x in row] for row in e] for e in view["E"]],
           "wirings": {}}

    for wiring in WIRINGS:
        v = cu.rewire(view, wiring, seed + WIRINGS.index(wiring) * 977)
        base = ce.extract(v, "layer", radius2)
        cols, col_of, stage_cols = hidden_columns(v, base["live"])
        if not cols:
            continue
        Hall = np.stack(cols, axis=1)
        mom = cs.moments(Hall[:cuts[1]], Zp["train"],
                         {p: (Hall[cuts[i]:cuts[i + 1]], Zp[p]) for i, p in enumerate(POOLS)})
        cache = {}
        # the component-DAG runner fills the chance cache with the `field` rule
        # first; the same order keeps every chance draw identical to the archive
        cs.correspondence(mom, ce.extract(v, "field", radius2)["components"], col_of,
                          stage_cols, np.random.default_rng(seed + 7), cache)
        ex = ce.extract(v, comp_rule, radius2)
        rows = cs.correspondence(mom, ex["components"], col_of, stage_cols,
                                 np.random.default_rng(seed + 7), cache)
        raw_match, _ = cs.assign(rows, t["n_nodes"], matcher)
        match = {("node", t["match_nodes"][i]): r for i, r in raw_match.items()}
        nm = cu.node_map(ex, match, rows, active_raw)
        E = ce.edges(v, ex, edge_rule)
        sc = cs.graph_scores(t, nm, ex["nodes"], E)
        sc["direct"] = direct_scores(t, nm, E)
        sc["n_components"] = len(ex["components"])
        sc["component_sizes"] = [len(c) for c in ex["components"]]
        sc["live"] = [int(l.sum()) for l in ex["live"]]
        sc["n_matched"] = len(match)
        sc["n_edges"] = len(E)
        sc["matched_stages"] = {str(k[1]): rows[r]["stages"] for k, r in match.items()}
        sc["matched_excess"] = {str(t["match_nodes"][i]): float(rows[r]["test"][i])
                                for i, r in raw_match.items()}
        sc["matched_p"] = {str(t["match_nodes"][i]): float(rows[r]["p"][i])
                           for i, r in raw_match.items()}
        sc["node_precision"] = (len(match) / len(ex["components"]) if ex["components"]
                                else float("nan"))
        sc["input_recall"] = float(np.mean([base["live"][0][a] for a in active_raw]))
        sc["input_precision"] = (float(np.mean([a in active_raw for a in ex["inputs"]]))
                                 if ex["inputs"] else float("nan"))
        if wiring == "learned":
            sc["nulls"] = cu.null_scores(t, ex, E, rows, active_raw, match, seed + 11)
            sc["probe_table"] = probe_table(rows, t)
            out["graph"] = cu.graph_record(ex, E, raw_match, rows, nm, t["match_nodes"])
        out["wirings"][wiring] = sc
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("checkpoint", "cell", "fs", "n", "which", "targets", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    name = f"{a.cell}__fs{a.fs}__n{a.n}__{a.which}"
    path = Path(a.out) / f"{name}.json"
    if path.exists():
        print(f"skip {name}"); return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    res = run(a.checkpoint, a.cell, int(a.fs), int(a.n), a.which, a.targets)
    path.write_text(json.dumps(cu._clean(res)))
    print(f"wrote {name} in {res['seconds']:.1f}s")


if __name__ == "__main__":
    main()
