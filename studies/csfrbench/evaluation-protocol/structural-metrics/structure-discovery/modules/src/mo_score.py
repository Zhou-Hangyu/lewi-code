"""The correspondence with a better-resolved chance test.  The component-DAG rule draws 32 chance
sets and admits a (module, node) pair only when the module beats every draw (p < 0.05 is then the
smallest p the test can give).  Here 200 draws are taken from the same stream; the first 32 are the
registered rule ("significant"), and "p05" admits a pair that beats at least 95% of the 200 draws
on validation - the same nominal level, resolved.  "positive" admits any positive excess.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
CD = HERE.parents[1] / "component-dag" / "src"
if str(CD) not in sys.path:
    sys.path.insert(0, str(CD))
import cd_score as cs  # noqa: E402

CHANCE_DRAWS = 200
REGISTERED = cs.CHANCE_DRAWS          # 32
MATCHERS = ("significant", "p05", "positive")


def chance(mom, counts, stage_cols, rng):
    V, T = [], []
    for _ in range(CHANCE_DRAWS):
        U = np.concatenate([rng.choice(stage_cols[s], size=min(k, len(stage_cols[s])), replace=False)
                            for s, k in counts])
        a, b = cs.probe(mom, U)
        V.append(a); T.append(b)
    return np.stack(V), np.stack(T)


def correspondence(mom, comps, col_of, stage_cols, rng, cache=None):
    rows = []
    cache = {} if cache is None else cache
    for k, units in enumerate(comps):
        U = np.array([col_of[(s, u)] for s, u in units])
        cnt = {}
        for s, _ in units:
            cnt[s] = cnt.get(s, 0) + 1
        key = tuple(sorted(cnt.items()))
        v, t = cs.probe(mom, U)
        if key not in cache:
            cache[key] = chance(mom, key, stage_cols, rng)
        V, T = cache[key]
        V32, T32 = V[:REGISTERED], T[:REGISTERED]
        rows.append({"comp": k, "size": int(len(U)), "stages": sorted(cnt),
                     "whole": all(k_ == len(stage_cols[s_]) for s_, k_ in cnt.items()),
                     "val": v - V32.mean(0), "test": t - T32.mean(0), "raw": t, "raw_val": v, "chance": T32.mean(0),
                     "sig": v > V32.max(0), "p": (1 + (T32 >= t).sum(0)) / (len(T32) + 1),
                     "frac": (v > V).mean(0), "p200": (1 + (T >= t).sum(0)) / (len(T) + 1)})
    return rows


def target(meta):
    """The target computation graph with every local function as a node, the root included (the
    paper's seven rows on the depth-4 tree), and y as the output that reads the readout nodes: under
    an additive readout the several terms, under a standardizing readout the single root.  The
    component-DAG rule folded a single root into y; here the root is matched like every other node
    and its readout edge is scored."""
    t = cs.canonical_target(meta)
    nodes = sorted(t["parents"])
    return {**t, "root": None, "key": {v: ("node", v) for v in nodes}, "match_nodes": nodes, "n_nodes": len(nodes)}


def layer_gain(Hall, Zp, cuts, cols, prev_cols, rng, draws=200):
    """The whole-layer naming statistic of the extension: the gain in validation R^2 of a layer's
    live units over the previous layer's live units, for every target node, with its permutation
    null (the node values permuted within every pool, ``draws`` times).  Returns the gain, the share
    of null gains below it, and the null mean."""
    def r2(U, Ztr, Zva):
        if len(U) == 0:
            return np.zeros(Ztr.shape[1])
        m = cs.moments(Hall[:cuts[1]][:, U], Ztr, {"train": (Hall[:cuts[1]][:, U], Ztr),
                                                    "val": (Hall[cuts[1]:cuts[2]][:, U], Zva),
                                                    "test": (Hall[cuts[1]:cuts[2]][:, U], Zva)})
        return cs.probe(m, np.arange(len(U)))[0]
    Ztr, Zva = Zp["train"], Zp["val"]
    gain = r2(cols, Ztr, Zva) - r2(prev_cols, Ztr, Zva)
    null = np.stack([r2(cols, Ztr[rng.permutation(len(Ztr))], Zva[rng.permutation(len(Zva))])
                     - r2(prev_cols, Ztr[rng.permutation(len(Ztr))], Zva[rng.permutation(len(Zva))])
                     for _ in range(draws)])
    return gain, (null < gain).mean(0), null.mean(0)


def assign(rows, n_nodes, mode="p05"):
    """One-to-one node -> row on the validation excess.  Modes: ``significant`` (beats all 32 chance
    draws), ``p05`` (beats 95% of 200), ``positive`` (any positive excess), ``gain`` (p05, and a
    module that is its whole layer is admissible for the nodes whose decodability it adds over the
    previous layer at p < 0.05 under the permutation null, weighted by that gain: rows carry
    ``gain``, ``gain_frac``), ``oracle`` (the raw validation R^2, no admissibility: a target-side
    diagnostic, not a naming rule)."""
    if not rows:
        return {}, np.zeros((n_nodes, 0))
    Ev = np.stack([r["val"] for r in rows], axis=1)
    if mode == "significant":
        ok = np.stack([r["sig"] for r in rows], axis=1)
    elif mode == "p05":
        ok = np.stack([r["frac"] >= 0.95 for r in rows], axis=1)
    elif mode == "gain":
        ok = np.stack([r["frac"] >= 0.95 for r in rows], axis=1)
        for j, r in enumerate(rows):
            if r.get("whole") and "gain" in r:
                ok[:, j] = r["gain_frac"] >= 0.95
                Ev[:, j] = r["gain"]
    elif mode == "oracle":
        Ev = np.stack([r["raw_val"] for r in rows], axis=1)
        ok = Ev > 0
    else:
        ok = Ev > 0
    W = np.where(ok, Ev, -1e9)
    pad = np.concatenate([W, np.zeros((n_nodes, n_nodes))], axis=1)
    ri, ci = linear_sum_assignment(pad, maximize=True)
    return {int(v): int(c) for v, c in zip(ri, ci) if c < len(rows) and ok[v, c] and Ev[v, c] > 0}, Ev
