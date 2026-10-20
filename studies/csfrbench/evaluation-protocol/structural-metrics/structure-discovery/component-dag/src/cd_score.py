"""Evaluation: component correspondence on held-out data, then graph agreement.

Applied only AFTER extraction.  ``correspondence`` decides which discovered
component stands for which target intermediate (ridge probes on held-out
pools, chance from same-size unit sets of the same stage, one-to-one
assignment).  ``graph_scores`` compares the directed relations between the
matched components with the target computation graph.

Node keys are ("leaf", a), ("node", v), ("y", 0) on the target side and
(stage, component index) on the learner side.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

LAMBDAS = np.logspace(-3.0, 4.0, 15)
CHANCE_DRAWS = 32


# -------------------------------------------------------------- probes --

def moments(Htr, Ztr, pools):
    """Sufficient statistics per stage: features centred and scaled by the
    train pool, targets centred by the train mean."""
    mu, sd = Htr.mean(0), Htr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    zmu = Ztr.mean(0)
    out = {"mu": mu, "sd": sd, "zmu": zmu}
    for tag, (H, Z) in pools.items():
        A, B = (H - mu) / sd, Z - zmu
        out[tag] = {"G": A.T @ A, "C": A.T @ B, "zz": (B * B).sum(0),
                    "zs": B.sum(0), "n": len(A)}
    return out


def probe(mom, U):
    """Ridge R^2 per target node for unit set ``U``: lambda chosen on val, read
    on test.  One eigendecomposition covers the whole lambda grid."""
    tr = mom["train"]
    G, C = tr["G"][np.ix_(U, U)], tr["C"][U]
    ev, Q = np.linalg.eigh(G)
    QtC = Q.T @ C                                     # (m, S)
    Wl = np.einsum("ij,kjs->kis", Q,
                   QtC[None, :, :] / (ev[None, :, None] + LAMBDAS[:, None, None]))
    r2 = {}
    for tag in ("val", "test"):
        p = mom[tag]
        sst = np.maximum(p["zz"] - p["zs"] ** 2 / p["n"], 1e-12)
        Gp, Cp = p["G"][np.ix_(U, U)], p["C"][U]
        quad = np.einsum("kis,kis->ks", Wl, np.einsum("ij,kjs->kis", Gp, Wl))
        sse = p["zz"][None, :] - 2 * np.einsum("kis,is->ks", Wl, Cp) + quad
        r2[tag] = 1.0 - sse / sst
    pick = r2["val"].argmax(0)
    cols = np.arange(C.shape[1])
    return r2["val"][pick, cols], r2["test"][pick, cols]


def chance(mom, counts, stage_cols, rng):
    """Val / test R^2 of ``CHANCE_DRAWS`` random unit sets with the same number
    of units in each stage as the component."""
    V, T = [], []
    for _ in range(CHANCE_DRAWS):
        U = np.concatenate([rng.choice(stage_cols[s], size=min(k, len(stage_cols[s])),
                                       replace=False)
                            for s, k in counts])
        a, b = probe(mom, U)
        V.append(a); T.append(b)
    return np.stack(V), np.stack(T)


def correspondence(mom, comps, col_of, stage_cols, rng, cache=None):
    """Excess val / test R^2 of every component for every target node.

    ``col_of`` maps a (stage, unit) pair to its column of the pooled hidden
    matrix; ``stage_cols`` lists the columns of each stage.  Chance keeps the
    component's stage composition, so a component is credited only for the way
    it groups units, never for holding more of them or deeper ones.  A match is
    *significant* when the component beats every chance draw (p < 0.05 at 32
    draws).
    """
    rows = []
    cache = {} if cache is None else cache
    for k, units in enumerate(comps):
        U = np.array([col_of[(s, u)] for s, u in units])
        cnt = {}
        for s, _ in units:
            cnt[s] = cnt.get(s, 0) + 1
        key = tuple(sorted(cnt.items()))
        v, t = probe(mom, U)
        if key not in cache:
            cache[key] = chance(mom, key, stage_cols, rng)
        V, T = cache[key]
        rows.append({"comp": k, "size": int(len(U)), "stages": sorted(cnt),
                     "val": v - V.mean(0), "test": t - T.mean(0),
                     "raw": t, "chance": T.mean(0),
                     "sig": v > V.max(0),
                     "p": (1 + (T >= t).sum(0)) / (len(T) + 1)})
    return rows


MATCHERS = ("positive", "significant")


def assign(rows, n_nodes, mode="significant"):
    """One-to-one target node -> row index on the validation excess.  ``mode``
    fixes which (node, component) pairs may be used: any positive excess, or
    only pairs that beat every chance draw."""
    if not rows:
        return {}, np.zeros((n_nodes, 0))
    Ev = np.stack([r["val"] for r in rows], axis=1)          # nodes x components
    ok = (np.stack([r["sig"] for r in rows], axis=1) if mode == "significant"
          else Ev > 0)
    W = np.where(ok, Ev, -1e9)
    pad = np.concatenate([W, np.zeros((n_nodes, n_nodes))], axis=1)
    ri, ci = linear_sum_assignment(pad, maximize=True)
    out = {int(v): int(c) for v, c in zip(ri, ci)
           if c < len(rows) and ok[v, c] and Ev[v, c] > 0}
    return out, Ev


# --------------------------------------------------------- target graph --

def canonical_target(g):
    """The target computation graph.

    Slots with identical parents are one computation.  Under a single-node
    readout the readout is a fixed linear map of that node, so the node *is*
    the output: it carries the key ("y", 0) and is not a separate node to be
    matched.  An additive readout keeps ("y", 0) as its own node.
    """
    A, parents = int(g["A"]), [tuple(int(r) for r in p) for p in g["node_parents"]]
    canon, seen = [], {}
    keys = []
    for refs in parents:
        key = tuple(sorted(("leaf", r) if r < A else ("node", canon[r - A])
                           for r in refs))
        if key not in seen:
            seen[key] = len(seen)
        canon.append(seen[key])
        keys.append(key)
    par = {canon[v]: keys[v] for v in range(len(parents))}
    depth = {}
    for v in sorted(par):
        depth[v] = 1 + max([depth[u] for k, u in par[v] if k == "node"], default=0)
    readout = sorted({canon[int(i)] for i in g["readout_indices"]})
    root = readout[0] if len(readout) == 1 else None
    key = {v: (("y", 0) if v == root else ("node", v)) for v in par}
    match_nodes = [v for v in sorted(par) if v != root]
    return {"A": A, "parents": par, "depth": depth, "readout": readout,
            "canon": canon, "root": root, "key": key,
            "match_nodes": match_nodes, "n_nodes": len(match_nodes)}


def _lift(t, p):
    return t["key"][p[1]] if p[0] == "node" else p


def target_edges(t):
    E = {(_lift(t, p), t["key"][v]) for v, ps in t["parents"].items() for p in ps}
    if t["root"] is None:
        E |= {(("node", v), ("y", 0)) for v in t["readout"]}
    return E


def target_ancestors(t):
    anc = {("leaf", a): set() for a in range(t["A"])}
    for v in sorted(t["parents"], key=lambda v: t["depth"][v]):
        s = set()
        for p in t["parents"][v]:
            q = _lift(t, p)
            s |= {q} | anc[q]
        anc[t["key"][v]] = s
    if t["root"] is None:
        s = set()
        for v in t["readout"]:
            s |= {("node", v)} | anc[("node", v)]
        anc[("y", 0)] = s
    return {(a, b) for b, S in anc.items() for a in S}


# ------------------------------------------------------- graph agreement --

def reachability(edges, marked):
    """Pairs (a, b) of marked nodes with a directed path a -> b."""
    succ = {}
    for a, b in edges:
        succ.setdefault(a, set()).add(b)
    out = set()
    for a in marked:
        seen, front = set(), {a}
        while front:
            nxt = set()
            for u in front:
                for w in succ.get(u, ()):
                    if w not in seen:
                        seen.add(w); nxt.add(w)
            front = nxt
        out |= {(a, b) for b in seen if b in marked and b != a}
    return out


def transitive_reduction(rel):
    """The direct relations of a transitive, acyclic relation."""
    succ = {}
    for a, b in rel:
        succ.setdefault(a, set()).add(b)
    return {(a, b) for a, b in rel
            if not any((c, b) in rel for c in succ.get(a, ()) if c != b)}


def prf(hit, n_pred, n_true):
    p = hit / n_pred if n_pred else float("nan")
    r = hit / n_true if n_true else float("nan")
    f = float("nan") if not n_true else (0.0 if not n_pred or p + r == 0
                                         else 2 * p * r / (p + r))
    return {"precision": p, "recall": r, "f1": f}


def graph_scores(t, node_of, dag_nodes, dag_edges):
    """Edge and ancestor agreement.  The component DAG is read as the ancestor
    relation between the mapped nodes; its transitive reduction is the
    discovered edge set.  Unmapped components take part only by carrying paths;
    live inputs the target does not use enter as keys with no target edge.
    """
    marked = set(node_of)
    Ahat = {(node_of[a], node_of[b]) for a, b in reachability(dag_edges, marked)}
    Ehat = transitive_reduction(Ahat)
    Et, At = target_edges(t), target_ancestors(t)
    out = {"n_pred_edges": len(Ehat), "n_true_edges": len(Et),
           "edge": prf(len(Ehat & Et), len(Ehat), len(Et)),
           "ancestor": prf(len(Ahat & At), len(Ahat), len(At)),
           "exact": bool(Ehat == Et)}
    groups = {"first": lambda e: e[0][0] == "leaf",
              "higher": lambda e: e[0][0] == "node",
              "readout": lambda e: e[1][0] == "y"}
    for name, keep in groups.items():
        T, P = {e for e in Et if keep(e)}, {e for e in Ehat if keep(e)}
        out[name] = prf(len(P & T), len(P), len(T))
    return out
