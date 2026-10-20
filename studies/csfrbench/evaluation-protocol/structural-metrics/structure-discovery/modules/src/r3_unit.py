"""Round 3 (intake/structure_discovery_GOAL.md), one checkpoint state -> one JSON record.

Module sets (clusterings of the learned positions, target-blind): per-layer average linkage at
0.75 / 1 / 1.25 R (the reference at R), single linkage (``cc``) and complete linkage at R, and the
cross-layer average linkage at 0.75 R and R (a module may span layers).  Edges: the majority rule
(``coverage@0.5``, the reference), ``coverage@0.25``, every open connection (``coverage@0``), the
weight share and the contribution share at 0.25.  Correspondences, all at p < 0.05 on 200
shape-matched chance draws: one-to-one (the reference), copies (the one-to-one assignment, then every
unnamed module takes its best admissible node: persistence as an extension of the reference),
one-to-many (every module takes its best admissible node, module-centric), many-to-one (a module may
carry several nodes: superposition), many-to-many (every admissible pair).  Controls: wiring rewired, first transition kept, modules shuffled (every
module keeps its per-layer counts), and the multiplicity-matched random correspondence (the name
sets dealt among the modules of the same layer composition); the initialization is its own state.
Per target node its status; per module its names; persistence and superposition counts; the
whole-layer diagnosis (decodable nodes, undecidable target edges); the attribution of every missing
target edge under the reference edges.

    python r3_unit.py --checkpoint <run_dir> --cell tree --fs 7001 --n 32768 --which best \
        --targets ../results/targets/d32 --out ../results/units/r3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
SD = HERE.parents[1]
for p in (str(HERE), str(SD / "component-dag" / "src"), str(SD / "recovered-graph" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
import cd_score as cs        # noqa: E402
import cd_unit as cu         # noqa: E402
import cd_view as cv         # noqa: E402
import mo_extract as mo      # noqa: E402
import mo_score as ms        # noqa: E402
import rg_unit as rg         # noqa: E402
from csfrbench.utils.seeds import derive  # noqa: E402

VERSION = "r3-v1"
MODULE_SETS = (("avg", 0.75), ("avg", 1.0), ("avg", 1.25), ("cc", 1.0), ("complete", 1.0), ("xavg", 0.75), ("xavg", 1.0))
EDGE_RULES = ("coverage@0.5", "coverage@0.25", "coverage@0", "weight_share@0.25", "contrib_share@0.25")
NAMINGS = ("one_to_one", "copies", "one_to_many", "many_to_one", "many_to_many")
REFERENCE = ("avg@1", "coverage@0.5", "one_to_one")
WIRINGS = ("learned", "shuffle_all", "shuffle_higher")
NULL_DRAWS = 16
DECODABLE = 0.5
ADMISSIBLE = 0.95
POOLS = cu.POOLS
LEVELS = {"first": lambda e: e[0][0] == "leaf", "higher": lambda e: e[0][0] == "node", "readout": lambda e: e[1][0] == "y"}


def set_key(rule, scale):
    return f"{rule}@{scale:g}"


# ------------------------------------------------------------------ naming --

def names_of(rows, n_nodes, naming):
    """Module (row index) -> the target node indices it is named after."""
    if not rows:
        return {}
    ok = np.stack([r["frac"] >= ADMISSIBLE for r in rows], axis=1)          # (nodes, modules)
    Ev = np.stack([r["val"] for r in rows], axis=1)
    out = {}
    if naming == "one_to_one":
        raw, _ = ms.assign(rows, n_nodes, "p05")
        for i, j in raw.items():
            out.setdefault(j, []).append(int(i))
    elif naming == "copies":
        raw, _ = ms.assign(rows, n_nodes, "p05")
        for i, j in raw.items():
            out.setdefault(j, []).append(int(i))
        for j in range(len(rows)):
            adm = np.flatnonzero(ok[:, j])
            if j not in out and len(adm):
                out[j] = [int(adm[np.argmax(Ev[adm, j])])]
    elif naming == "one_to_many":
        for j in range(len(rows)):
            adm = np.flatnonzero(ok[:, j])
            if len(adm):
                out[j] = [int(adm[np.argmax(Ev[adm, j])])]
    elif naming == "many_to_one":
        for i in range(n_nodes):
            adm = np.flatnonzero(ok[i])
            if len(adm):
                out.setdefault(int(adm[np.argmax(Ev[i, adm])]), []).append(i)
    elif naming == "many_to_many":
        for j in range(len(rows)):
            adm = np.flatnonzero(ok[:, j])
            if len(adm):
                out[j] = [int(i) for i in adm]
    else:
        raise ValueError(naming)
    return {j: sorted(v) for j, v in out.items()}


def node_map(ex, names, rows, t, active_raw):
    """DAG key -> the target keys it stands for (a list; inputs and the output carry one)."""
    out = {}
    for a, raw in enumerate(active_raw):
        if int(raw) in ex["inputs"]:
            out[("in", int(raw))] = [("leaf", a)]
    for i in ex["inputs"]:
        out.setdefault(("in", i), [("extra", i)])
    for j, ids in names.items():
        out[("c", rows[j]["comp"])] = [("node", t["match_nodes"][i]) for i in ids]
    out[("out", 0)] = [("y", 0)]
    return out


def _closure(rel):
    succ = {}
    for a, b in rel:
        succ.setdefault(a, set()).add(b)
    out = set()
    for a in list(succ):
        seen, front = set(), set(succ[a])
        while front:
            nxt = set()
            for u in front:
                if u not in seen:
                    seen.add(u); nxt |= succ.get(u, set())
            front = nxt
        out |= {(a, b) for b in seen if b != a}
    return out


def relations(node_of, dag_edges):
    """The name-level ancestor relation (closed) and its transitive reduction: reachability between
    the marked DAG nodes, every pair of their names (a pair inside one module does not count, nor a
    name with itself)."""
    marked = set(node_of)
    A = set()
    for a, b in cs.reachability(dag_edges, marked):
        for u in node_of[a]:
            for v in node_of[b]:
                if u != v:
                    A.add((u, v))
    A = _closure(A)
    return A, cs.transitive_reduction(A)


def scores(t, A, E):
    Et, At = cs.target_edges(t), cs.target_ancestors(t)
    out = {"edge": cs.prf(len(E & Et), len(E), len(Et)), "ancestor": cs.prf(len(A & At), len(A), len(At)),
           "n_pred_edges": len(E)}
    for name, keep in LEVELS.items():
        T, P = {e for e in Et if keep(e)}, {e for e in E if keep(e)}
        out[name] = cs.prf(len(P & T), len(P), len(T))
    return out


def edge_set(M, rule):
    m, th = rule.split("@")
    return {pair for pair, v in M.items() if v[m] > float(th)}


def disconnected(ex, E):
    keys = [("c", k) for k in range(len(ex["components"]))]
    has_in = {b for _, b in E}; has_out = {a for a, _ in E}
    return {"modules": len(keys), "no_in": sum(k not in has_in for k in keys), "no_out": sum(k not in has_out for k in keys),
            "either": sum(k not in has_in or k not in has_out for k in keys)}


# --------------------------------------------------------------- diagnosis --

def status(t, rows):
    """Why every target node is or is not named one-to-one: named / not decodable (best validation
    R^2 below 0.5) / whole layer (its best module is an undivided layer: untestable) / module taken
    (its best admissible module is named after another node: not separated) / below chance."""
    raw, _ = ms.assign(rows, t["n_nodes"], "p05")
    out = {}
    for i, v in enumerate(t["match_nodes"]):
        best = int(np.argmax([r["raw_val"][i] for r in rows])); r = rows[best]
        rec = {"best_r2": round(float(r["raw_val"][i]), 4), "best_size": r["size"], "best_stages": r["stages"], "best_whole": bool(r["whole"])}
        if i in raw:
            rec["status"] = "named"
        elif rec["best_r2"] < DECODABLE:
            rec["status"] = "not decodable"
        elif r["whole"]:
            rec["status"] = "whole layer"
        elif any(r_["frac"][i] >= ADMISSIBLE for r_ in rows):
            rec["status"] = "module taken"
        else:
            rec["status"] = "below chance"
        out[str(v)] = rec
    return out


def whole_layers(t, rows):
    """Every undivided layer with the nodes it decodes (validation R^2 >= 0.5): untestable against
    same-size subsets, so decodability is reported, not a name."""
    return [{"comp": r["comp"], "stages": r["stages"], "size": r["size"],
             "decodable": [str(t["match_nodes"][i]) for i in np.flatnonzero(r["raw_val"] >= DECODABLE)]}
            for r in rows if r["whole"]]


def undecidable(t, st):
    """Target edges that touch a node whose best module is an undivided layer."""
    wl = {("node", int(v)) for v, rec in st.items() if rec["status"] == "whole layer"}
    return {e for e in cs.target_edges(t) if e[0] in wl or e[1] in wl}


def persistence(names, rows, E, t):
    """Per node under a naming: the modules named after it, their stages, and whether the modules of
    consecutive stages are joined by an edge (carried through the wiring)."""
    by_node = {}
    for j, ids in names.items():
        for i in ids:
            by_node.setdefault(i, []).append(j)
    out = {}
    for i, js in by_node.items():
        stages = sorted({s for j in js for s in rows[j]["stages"]})
        joined = any((("c", rows[a]["comp"]), ("c", rows[b]["comp"])) in E for a in js for b in js if a != b)
        out[str(t["match_nodes"][i])] = {"modules": len(js), "stages": stages, "joined": bool(joined)}
    return out


def superposition(names, rows, t):
    """Per module with several names: the pairs of names that are ancestor-related in the target
    (serial) and those that are not (parallel); undivided layers listed apart."""
    anc = cs.target_ancestors(t)
    keys = t["match_nodes"]
    out = {"modules": 0, "serial": 0, "parallel": 0, "whole": 0, "detail": []}
    for j, ids in names.items():
        if len(ids) < 2:
            continue
        if rows[j]["whole"]:
            out["whole"] += 1
            continue
        out["modules"] += 1
        ser = par = 0
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                u, v = ("node", keys[ids[a]]), ("node", keys[ids[b]])
                if (u, v) in anc or (v, u) in anc:
                    ser += 1
                else:
                    par += 1
        out["serial"] += ser; out["parallel"] += par
        out["detail"].append({"comp": rows[j]["comp"], "stages": rows[j]["stages"], "names": [str(keys[i]) for i in ids],
                              "serial": ser, "parallel": par})
    return out


def attribution(t, st, node_of, A_ref, E_ref, A_any, live_inputs, active_raw):
    """Every target edge missing from the reduced graph and the reason: an endpoint is unnamed
    (with its status), or both are named and the wiring holds no open path, or a path only below
    the threshold, or a path that the reduction replaced by a longer one through another named
    node."""
    named = {v for vs in node_of.values() for v in vs}
    out = {}
    for e in sorted(cs.target_edges(t) - E_ref):
        p, q = e
        key = f"{p[0]}:{p[1]}->{q[0]}:{q[1]}"
        if p[0] == "leaf" and int(active_raw[p[1]]) not in live_inputs:
            out[key] = "input dead"
        elif q not in named:
            out[key] = "child " + st[str(q[1])]["status"]
        elif p not in named:
            out[key] = "parent " + st[str(p[1])]["status"]
        elif e in A_ref:
            out[key] = "reduced"
        elif e in A_any:
            out[key] = "threshold"
        else:
            out[key] = "no connection"
    return out


# -------------------------------------------------------------------- nulls --

def shuffle_modules(ex, rng):
    """The modules' per-stage counts kept, the units of every stage dealt at random."""
    comps = [list(c) for c in ex["components"]]
    by_stage = {}
    for k, units in enumerate(comps):
        for s, u in units:
            by_stage.setdefault(s, []).append((k, u))
    new = [[] for _ in comps]
    for s, entries in by_stage.items():
        pool = [u for _, u in entries]
        perm = rng.permutation(len(pool))
        for (k, _), j in zip(entries, perm):
            new[k].append((s, int(pool[j])))
    new = [sorted(c) for c in new]
    cfield = [np.any([ex["fields"][s][u] for s, u in c], axis=0) for c in new]
    return {**ex, "components": new, "cfield": cfield, "reads": {}}


def random_names(names, rows, rng):
    """The name sets dealt among the modules of the same stage composition."""
    groups = {}
    for j in range(len(rows)):
        groups.setdefault(tuple(rows[j]["stages"]), []).append(j)
    out = {}
    for js in groups.values():
        sets = [names.get(j, []) for j in js]
        for j, k in zip(js, rng.permutation(len(js))):
            if sets[k]:
                out[j] = sets[k]
    return out


# ---------------------------------------------------------------------- run --

def compact(sc):
    out = {lvl: {k: sc[lvl][k] for k in ("precision", "recall", "f1")} for lvl in ("edge", "first", "higher", "readout", "ancestor")}
    out["n_pred_edges"] = sc["n_pred_edges"]
    return out


def evaluate(t, ex, rows, M, active_raw, st, with_nulls, seed):
    """Every naming x edge rule."""
    res, name_maps = {}, {}
    und = undecidable(t, st)
    Et = cs.target_edges(t)
    live_inputs = set(ex["inputs"])
    for naming in NAMINGS:
        names = names_of(rows, t["n_nodes"], naming)
        name_maps[naming] = {str(rows[j]["comp"]): [str(t["match_nodes"][i]) for i in ids] for j, ids in names.items()}
        nm = node_map(ex, names, rows, t, active_raw)
        rel = {rule: relations(nm, edge_set(M, rule)) for rule in EDGE_RULES}
        for rule, (A, E) in rel.items():
            sc = compact(scores(t, A, E))
            Es = edge_set(M, rule)
            sc["n_edges"] = len(Es); sc["disconnected"] = disconnected(ex, Es)
            sc["n_named_modules"] = len(names); sc["n_names"] = sum(len(v) for v in names.values())
            sc["n_named_nodes"] = len({i for v in names.values() for i in v})
            dec = Et - und
            sc["undecidable_edges"] = len(und)
            sc["decidable_recall"] = len(E & dec) / len(dec) if dec else float("nan")
            res[f"{rule}|{naming}"] = sc
        A_ref, E_ref = rel[REFERENCE[1]]
        A_any, _ = rel["coverage@0"]
        res[f"attribution|{naming}"] = attribution(t, st, nm, A_ref, E_ref, A_any, live_inputs, active_raw)
        if naming in ("copies", "one_to_many"):
            res.setdefault("persistence", {})[naming] = persistence(names, rows, edge_set(M, REFERENCE[1]), t)
        if naming == "many_to_many":
            res["superposition"] = superposition(names, rows, t)
        if with_nulls:
            rng = np.random.default_rng(seed + 11 + NAMINGS.index(naming))
            acc = {rule: [] for rule in EDGE_RULES}
            for _ in range(NULL_DRAWS):
                nm_r = node_map(ex, random_names(names, rows, rng), rows, t, active_raw)
                for rule in EDGE_RULES:
                    A, E = relations(nm_r, edge_set(M, rule))
                    sc = scores(t, A, E)
                    acc[rule].append([sc["edge"]["f1"], sc["higher"]["f1"], sc["ancestor"]["f1"]])
            with np.errstate(invalid="ignore"):
                res[f"random_names|{naming}"] = {rule: {"edge": float(np.nanmean([x[0] for x in v])), "higher": float(np.nanmean([x[1] for x in v])),
                                                        "ancestor": float(np.nanmean([x[2] for x in v]))} for rule, v in acc.items()}
    return res, name_maps


def run(checkpoint, cell, fs, n, which, targets_dir, module_sets=MODULE_SETS):
    t0 = time.time()
    tgt = np.load(Path(targets_dir) / f"{cell}__fs{fs}.npz", allow_pickle=True)
    meta = json.loads(str(tgt["meta"]))
    t = ms.target(meta)
    active_raw = [int(a) for a in meta["active_raw"]]
    X = np.concatenate([tgt[f"X_{p}"] for p in POOLS])
    cuts = np.cumsum([0] + [len(tgt[f"X_{p}"]) for p in POOLS])
    model, target, spec = cv.load(checkpoint, which=which)
    view = cv.learner(model, spec["method"], X)
    radius2 = float(model.hidden_layers[0].gate_offset)
    seed = int(derive(meta["fingerprint"], "cd", which, int(n)))
    first = {}
    for raw, c in enumerate(t["canon"]):
        first.setdefault(c, raw)
    order = [first[v] for v in t["match_nodes"]]
    Zp = {p: tgt[f"Z_{p}"][:, order] for p in POOLS}
    out = {"version": VERSION, "module_sets": [set_key(*m) for m in module_sets], "edge_rules": list(EDGE_RULES),
           "namings": list(NAMINGS), "reference": list(REFERENCE), "setting": f"d{int(spec['d'])}", "cell": cell,
           "function_seed": int(fs), "n": int(n), "which": which, "checkpoint": str(checkpoint), "method": spec["method"],
           "d": int(spec["d"]), "A": t["A"], "n_nodes": t["n_nodes"], "n_true_edges": len(cs.target_edges(t)), "radius2": radius2,
           "active_raw": active_raw, "target_parents": {str(v): [list(p) for p in ps] for v, ps in t["parents"].items()},
           "target_root": None, "target_match_nodes": t["match_nodes"], "raw_node_parents": meta["node_parents"],
           "raw_readout_indices": list(meta["readout_indices"]), "target_depth": {str(k): v for k, v in t["depth"].items()},
           "target_readout": t["readout"], "positions": [[[round(float(x), 4) for x in row] for row in e] for e in view["E"]],
           "wirings": {}}
    for wiring in WIRINGS:
        v = cu.rewire(view, wiring, seed + WIRINGS.index(wiring) * 977)
        base = mo.modules(v, "ancestry", radius2)
        cols, col_of, stage_cols = rg.hidden_columns(v, base["live"])
        if not cols:
            continue
        Hall = np.stack(cols, axis=1)
        mom = cs.moments(Hall[:cuts[1]], Zp["train"], {p: (Hall[cuts[i]:cuts[i + 1]], Zp[p]) for i, p in enumerate(POOLS)})
        cache, res = {}, {}
        for rule, scale in module_sets:
            ex = mo.modules(v, rule, radius2, scale)
            rows = ms.correspondence(mom, ex["components"], col_of, stage_cols, np.random.default_rng(seed + 7), cache)
            M = mo.measures_x(v, ex)
            st = status(t, rows)
            sc, name_maps = evaluate(t, ex, rows, M, active_raw, st, wiring == "learned", seed)
            rec = {"scores": sc, "names": name_maps, "status": st, "whole_layers": whole_layers(t, rows),
                   "n_components": len(ex["components"]), "component_sizes": [len(c) for c in ex["components"]],
                   "component_stages": [rows[k]["stages"] for k in range(len(rows))], "live": [int(l.sum()) for l in ex["live"]]}
            if wiring == "learned":
                rec["graph"] = {"components": [[[int(s_), int(u)] for s_, u in c] for c in ex["components"]],
                                "inputs": [int(i) for i in ex["inputs"]],
                                "measures": [[list(a), list(b)] + [round(vv[m], 4) for m in mo.MEASURES_X] for (a, b), vv in sorted(M.items())]}
                exs = shuffle_modules(ex, np.random.default_rng(seed + 13))
                rows_s = ms.correspondence(mom, exs["components"], col_of, stage_cols, np.random.default_rng(seed + 7), cache)
                sc_s, _ = evaluate(t, exs, rows_s, mo.measures_x(v, exs), active_raw, status(t, rows_s), False, seed)
                rec["shuffle_modules"] = {k: v_ for k, v_ in sc_s.items() if "|" in k and not k.startswith("attribution")}
            res[set_key(rule, scale)] = rec
        out["wirings"][wiring] = res
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
