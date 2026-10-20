"""Extraction: a directed component DAG from a LeWi checkpoint's learned wiring.

Nothing here reads the target.  The input is the executed masks, the effective
weights, the positions and the hidden activations on unlabeled draws.  The same
two steps run for every target family:

  1. **components**  the live hidden units are grouped by a rule that reads only
     the learned wiring or the learned positions;
  2. **edges**       component C feeds component C' if the forward connection
     from C to C' carries signal - any open edge, or a share of the weight mass
     that C' reads.

Stages: 0 = the input units, 1..L = the hidden layers, L+1 = the readout unit.
The input units and the readout unit are given, so each stays a node of its
own; what the extraction discovers is the grouping of the hidden units, which
units are live at all, and the directed relations between the groups.

A component is a list of (stage, unit) pairs.  The ``field`` rule groups every
live hidden unit of the network by the set of inputs that reach it, so one
component may span several stages; the other rules group inside one stage.
Under ``field`` the graph is acyclic by construction: an edge can only run from
a smaller input set to a strictly larger one.

Node keys: ("in", unit), ("c", index), ("out", 0).
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

COMPONENT_RULES = ("field", "field_stage", "avg", "ball", "coparent", "layer")
EDGE_RULES = ("any", "mass90", "mass95", "mass99")


# ------------------------------------------------------------------ live --

def live_units(K, H) -> list[np.ndarray]:
    """Boolean mask per stage: the units of the executed circuit - reachable
    from the inputs, reaching the readout, and not constant on the pool."""
    sizes = [K[0].shape[1]] + [k.shape[0] for k in K]
    fwd = [np.ones(sizes[0], bool)]
    for b, k in enumerate(K):
        fwd.append(k[:, fwd[b]].any(1))
    for i, h in enumerate(H, start=1):
        fwd[i] &= h.std(0) > 0
    bwd = [None] * len(sizes)
    bwd[-1] = np.ones(sizes[-1], bool)
    for b in range(len(K) - 1, -1, -1):
        bwd[b] = K[b][bwd[b + 1] & fwd[b + 1]].any(0)
    return [f & b for f, b in zip(fwd, bwd)]


def receptive_fields(K, live) -> list[np.ndarray]:
    """Boolean (stage units x inputs): the inputs that reach each unit."""
    n_in = K[0].shape[1]
    out = [np.eye(n_in, dtype=bool) & live[0][:, None]]
    for b, k in enumerate(K):
        A = (k & live[b][None, :] & live[b + 1][:, None]).astype(np.float64)
        out.append((A @ out[b]) > 0)
    return out


# ------------------------------------------------------------ components --

def _cc(adj: np.ndarray) -> np.ndarray:
    n = len(adj)
    lab = -np.ones(n, dtype=int)
    c = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        stack, lab[s] = [s], c
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u] & (lab < 0)):
                lab[v] = c
                stack.append(v)
        c += 1
    return lab


def _stage_groups(rule, stage, idx, E, K, live, fields, radius2):
    if len(idx) == 0:
        return []
    if rule == "layer" or len(idx) == 1:
        return [idx]
    if rule in ("ball", "avg"):
        P = E[stage][idx]
        if rule == "ball":
            lab = _cc(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1) < radius2)
        else:
            lab = fcluster(linkage(pdist(P), method="average"),
                           t=np.sqrt(radius2), criterion="distance") - 1
    elif rule == "field_stage":
        keys, lab = {}, np.empty(len(idx), dtype=int)
        for i, row in enumerate(fields[stage][idx]):
            lab[i] = keys.setdefault(row.tobytes(), len(keys))
    elif rule == "coparent":
        A = K[stage - 1][idx][:, live[stage - 1]].astype(np.float64)
        lab = _cc((A @ A.T) > 0)
    else:
        raise ValueError(rule)
    return [idx[lab == c] for c in range(lab.max() + 1)]


def extract(view, rule: str, radius2: float) -> dict:
    """Live units, receptive fields and the components of the hidden stages."""
    K = [m > 0 for m in view["M"]]
    live = live_units(K, view["H"])
    fields = receptive_fields(K, live)
    L = len(K) - 1                                     # hidden stages 1..L
    if rule == "field":
        keys, comps, cfield = {}, [], []
        for s in range(1, L + 1):
            for u in np.flatnonzero(live[s]):
                k = fields[s][u].tobytes()
                if k not in keys:
                    keys[k] = len(comps)
                    comps.append([])
                    cfield.append(fields[s][u])
                comps[keys[k]].append((s, int(u)))
    else:
        comps, cfield = [], []
        for s in range(1, L + 1):
            idx = np.flatnonzero(live[s])
            for g in _stage_groups(rule, s, idx, view["E"], K, live, fields, radius2):
                comps.append([(s, int(u)) for u in g])
                cfield.append(fields[s][g].any(0))
    inputs = [int(i) for i in np.flatnonzero(live[0])]
    nodes = [("in", i) for i in inputs] + [("c", k) for k in range(len(comps))] \
        + [("out", 0)]
    return {"kept": K, "live": live, "fields": fields, "components": comps,
            "cfield": cfield, "inputs": inputs, "nodes": nodes, "L": L}


# ----------------------------------------------------------------- edges --

def edges(view, ex: dict, rule: str) -> set:
    """Directed edges of the component DAG.  ``mass<q>``: every receiver keeps
    the fewest senders whose share of its incoming weight mass reaches q."""
    M, W, comps, L = view["M"], view["W"], ex["components"], ex["L"]
    owner = {}
    for k, units in enumerate(comps):
        for s, u in units:
            owner[(s, u)] = ("c", k)
    for i in ex["inputs"]:
        owner[(0, i)] = ("in", i)
    owner[(L + 1, 0)] = ("out", 0)
    live = ex["live"]

    inflow, present = {}, {}
    for b in range(len(M)):
        open_ = (M[b] > 0) & live[b][None, :] & live[b + 1][:, None]
        mass = np.abs(W[b]) * open_
        for j in np.flatnonzero(live[b + 1]):
            tgt = owner.get((b + 1, j))
            if tgt is None:
                continue
            for i in np.flatnonzero(open_[j]):
                src = owner.get((b, i))
                if src is None or src == tgt:
                    continue
                inflow[(tgt, src)] = inflow.get((tgt, src), 0.0) + float(mass[j, i])
                present.setdefault(tgt, set()).add(src)
    if rule == "any":
        return {(s, t) for t, S in present.items() for s in S}
    q = float(rule[4:]) / 100.0
    out = set()
    for t, S in present.items():
        share = sorted(((inflow[(t, s)], s) for s in S), reverse=True)
        total = sum(v for v, _ in share)
        if total <= 0:
            out |= {(s, t) for s in S}
            continue
        run = 0.0
        for v, s in share:
            out.add((s, t))
            run += v
            if run >= q * total:
                break
    return out
