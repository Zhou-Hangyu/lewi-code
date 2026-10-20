"""One checkpoint -> one JSON: extract the component DAG, score it against the
target computation graph, run the nulls.

    python cd_unit.py --checkpoint <run_dir> --cell tree --fs 7000 --n 32768 \
                      --which best --targets ../results/targets --out ../results/units

Extraction (``cd_extract``) sees the executed masks, the effective weights, the
positions and the hidden activations on unlabeled draws.  The target enters
only through ``cd_score``, after extraction.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cd_extract as ce      # noqa: E402
import cd_score as cs        # noqa: E402
import cd_view as cv         # noqa: E402
from csfrbench.utils.seeds import derive  # noqa: E402

VERSION = "cd-v7"
WIRINGS = ("learned", "shuffle_all", "shuffle_higher")
PRIMARY = ("learned", "field", "any", "significant")
NULL_DRAWS = 200
POOLS = ("train", "val", "test")


# ------------------------------------------------------------- wirings --

def swap_shuffle(mask: np.ndarray, rng, rounds: int = 20) -> np.ndarray:
    """Degree-preserving rewire of a bipartite 0/1 matrix (double-edge swaps)."""
    A = mask.copy()
    rows, cols = np.nonzero(A)
    m = len(rows)
    if m < 2:
        return A
    for _ in range(rounds * m):
        a, b = int(rng.integers(m)), int(rng.integers(m))
        r1, c1, r2, c2 = rows[a], cols[a], rows[b], cols[b]
        if a == b or r1 == r2 or c1 == c2 or A[r1, c2] or A[r2, c1]:
            continue
        A[r1, c1] = A[r2, c2] = 0
        A[r1, c2] = A[r2, c1] = 1
        cols[a], cols[b] = c2, c1
    return A


def rewire(view: dict, kind: str, seed: int) -> dict:
    """A view whose open edges are rewired at fixed degrees; the weight
    magnitudes follow the edges."""
    if kind == "learned":
        return view
    rng = np.random.default_rng(seed)
    first = 1 if kind == "shuffle_higher" else 0
    M, W = list(view["M"]), list(view["W"])
    for b in range(first, len(M)):
        new = swap_shuffle(M[b] > 0, rng)
        vals = np.abs(W[b])[M[b] > 0]
        rng.shuffle(vals)
        Wb = np.zeros_like(W[b])
        Wb[new] = np.resize(vals, int(new.sum()))
        M[b], W[b] = new.astype(np.float64), Wb
    return {**view, "M": M, "W": W}


# --------------------------------------------------------------- mapping --

def node_map(ex, match, rows, active_raw):
    """match maps a target key to a row index."""
    """DAG node -> target key.  Live inputs the target does not use enter under
    their own keys, so every relation they carry counts against precision."""
    out = {}
    for a, raw in enumerate(active_raw):
        if int(raw) in ex["inputs"]:
            out[("in", int(raw))] = ("leaf", a)
    for i in ex["inputs"]:
        out.setdefault(("in", i), ("extra", i))
    for v, r in match.items():
        out[("c", rows[r]["comp"])] = v
    out[("out", 0)] = ("y", 0)
    return out


def null_scores(t, ex, E, rows, active_raw, match, seed):
    """Chance levels for the edge F1: a random one-to-one correspondence, and a
    permutation of the target's leaf labels over the same active inputs."""
    rng = np.random.default_rng(seed)
    nodes = ex["nodes"]
    rm = []
    for _ in range(NULL_DRAWS):
        if len(rows) < t["n_nodes"]:
            rm.append(float("nan")); continue
        pick = rng.choice(len(rows), size=t["n_nodes"], replace=False)
        m = {("node", v): int(c) for v, c in zip(t["match_nodes"], pick)}
        rm.append(cs.graph_scores(t, node_map(ex, m, rows, active_raw),
                                  nodes, E)["edge"]["f1"])
    lp = []
    for _ in range(NULL_DRAWS):
        perm = rng.permutation(len(active_raw))
        lp.append(cs.graph_scores(
            t, node_map(ex, match, rows, [active_raw[p] for p in perm]),
            nodes, E)["edge"]["f1"])
    with np.errstate(invalid="ignore"):
        return {"random_match": float(np.nanmean(rm)),
                "leaf_permutation": float(np.nanmean(lp))}


def graph_record(ex, E, raw_match, rows, nm, nodes):
    """The extracted graph of one design, for the figures."""
    return {"components": [[[int(s_), int(u)] for s_, u in c] for c in ex["components"]],
            "cfield": [[int(i) for i in np.flatnonzero(f)] for f in ex["cfield"]],
            "inputs": [int(i) for i in ex["inputs"]],
            "edges": [[list(a), list(b)] for a, b in sorted(E)],
            "map": {f"{k[0]}:{k[1]}": list(v) for k, v in nm.items()},
            "matched": {str(nodes[i]): [rows[r]["comp"], rows[r]["stages"],
                                        float(rows[r]["test"][i])]
                        for i, r in raw_match.items()}}


# ------------------------------------------------------------------ run --

def run(checkpoint, cell, fs, n, which, targets_dir, graph_design=PRIMARY):
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

    cols = {}
    for raw, c in enumerate(t["canon"]):
        cols.setdefault(c, raw)
    order = [cols[v] for v in t["match_nodes"]]
    Zp = {p: tgt[f"Z_{p}"][:, order] for p in POOLS}

    out = {"version": VERSION, "cell": cell, "function_seed": int(fs), "n": int(n),
           "which": which, "checkpoint": str(checkpoint), "method": spec["method"],
           "fingerprint": meta["fingerprint"], "A": t["A"], "n_nodes": t["n_nodes"],
           "n_true_edges": len(cs.target_edges(t)), "radius2": radius2,
           "active_raw": active_raw,
           "target_parents": {str(v): [list(p) for p in ps]
                              for v, ps in t["parents"].items()},
           "target_root": t["root"], "target_match_nodes": t["match_nodes"],
           "raw_node_parents": meta["node_parents"],
           "raw_readout_indices": list(meta["readout_indices"]),
           "target_depth": {str(k): v for k, v in t["depth"].items()},
           "target_readout": t["readout"], "designs": {}}

    for wiring in WIRINGS:
        v = rewire(view, wiring, seed + WIRINGS.index(wiring) * 977)
        base = ce.extract(v, "layer", radius2)
        live, L = base["live"], base["L"]
        cols, col_of, stage_cols = [], {}, {}
        for st in range(1, L + 1):
            idx = np.flatnonzero(live[st])
            stage_cols[st] = []
            for u in idx:
                col_of[(st, int(u))] = len(cols)
                stage_cols[st].append(len(cols))
                cols.append(v["H"][st - 1][:, u])
            stage_cols[st] = np.array(stage_cols[st], dtype=int)
        if not cols:
            continue
        Hall = np.stack(cols, axis=1)
        mom = cs.moments(Hall[:cuts[1]], Zp["train"],
                         {p: (Hall[cuts[i]:cuts[i + 1]], Zp[p])
                          for i, p in enumerate(POOLS)})
        cache = {}
        for comp_rule in ce.COMPONENT_RULES:
            ex = base if comp_rule == "layer" else ce.extract(v, comp_rule, radius2)
            rng = np.random.default_rng(seed + 7)
            rows = cs.correspondence(mom, ex["components"], col_of, stage_cols,
                                     rng, cache)
            for matcher in cs.MATCHERS:
              raw_match, _ = cs.assign(rows, t["n_nodes"], matcher)
              match = {("node", t["match_nodes"][i]): r for i, r in raw_match.items()}
              nm = node_map(ex, match, rows, active_raw)
              for edge_rule in ce.EDGE_RULES:
                E = ce.edges(v, ex, edge_rule)
                sc = cs.graph_scores(t, nm, ex["nodes"], E)
                sc["n_components"] = len(ex["components"])
                sc["component_sizes"] = [len(c) for c in ex["components"]]
                sc["n_matched"] = len(match)
                sc["n_edges"] = len(E)
                sc["matched_stages"] = {str(k[1]): rows[r]["stages"] for k, r in match.items()}
                sc["matched_excess"] = {str(t["match_nodes"][i]): float(rows[r]["test"][i])
                                        for i, r in raw_match.items()}
                sc["matched_p"] = {str(t["match_nodes"][i]): float(rows[r]["p"][i])
                                   for i, r in raw_match.items()}
                sc["node_precision"] = (len(match) / len(ex["components"])
                                        if ex["components"] else float("nan"))
                sc["input_recall"] = float(np.mean([live[0][a] for a in active_raw]))
                sc["input_precision"] = (
                    float(np.mean([a in active_raw for a in ex["inputs"]]))
                    if ex["inputs"] else float("nan"))
                key = f"{wiring}|{comp_rule}|{edge_rule}|{matcher}"
                if wiring == "learned":
                    sc["nulls"] = null_scores(t, ex, E, rows, active_raw, match,
                                              seed + 11)
                    if (wiring, comp_rule, edge_rule, matcher) == tuple(graph_design):
                        out["graph"] = graph_record(ex, E, raw_match, rows, nm,
                                                    t["match_nodes"])
                out["designs"][key] = sc
    out["seconds"] = time.time() - t0
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    return o


def main():
    ap = argparse.ArgumentParser()
    for k in ("checkpoint", "cell", "fs", "n", "which", "targets", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--graph-design", default="|".join(PRIMARY))
    a = ap.parse_args()
    name = f"{a.cell}__fs{a.fs}__n{a.n}__{a.which}"
    path = Path(a.out) / f"{name}.json"
    if path.exists():
        print(f"skip {name}"); return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    res = run(a.checkpoint, a.cell, int(a.fs), int(a.n), a.which, a.targets,
              tuple(a.graph_design.split("|")))
    path.write_text(json.dumps(_clean(res)))
    print(f"wrote {name} in {res['seconds']:.1f}s")


if __name__ == "__main__":
    main()
