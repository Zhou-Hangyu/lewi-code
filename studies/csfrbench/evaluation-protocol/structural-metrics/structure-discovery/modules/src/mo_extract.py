"""Target-blind module extraction from a LeWi checkpoint: the live hidden units of every layer are
grouped into modules by one rule, and the modules are joined by the learned wiring.

Rules (``modules``):
  ancestry   units of one layer that the same inputs reach (the recovered-graph study's rule)
  reads      units of one layer that read the same modules of the previous layer, where a unit
             reads a module when it is connected to more than half of that module's units
             (majority coverage); the modules of layer 1 read single inputs, so they equal the
             ancestry rule there
  cc         connected components of the positions of one layer at distance scale * R
  avg        average-linkage clusters of the positions of one layer cut at scale * R
  gmm        a Gaussian mixture (spherical) on the positions of one layer, components by BIC
  wprof      average-linkage clusters of the units' effective-weight profiles (the incoming row and
             the outgoing column of |W|, each scaled to unit length, joined), cosine distance, cut
             at ``scale`` (the extension of intake/feedback-1.md)
  complete   complete-linkage clusters of the positions of one layer cut at scale * R (every pair of
             a module within the cut)
  xavg       average-linkage clusters of the positions of ALL hidden layers jointly, cut at scale * R:
             a module may span layers (the spatial columns of round 3)

Edges (``edges``): ``any`` = one open connection between the modules; ``majority`` = the
receiving module's units are, on average, connected to more than half of the sending module;
``<measure>@<theta>`` = the measure exceeds theta, where ``coverage`` is the majority rule's
statistic (the share of the sender's units a receiving unit is connected to, averaged over the
receiver), ``mask_share`` the share of a receiving unit's open connections that come from the
sender, and ``weight_share`` the share of its incoming |W| mass that comes from the sender (both
averaged over the receiver); ``coverage@0.5`` equals ``majority``.
Nothing here reads the target.  The output has the shape of ``cd_extract.extract`` so the
correspondence and the scores of the component-DAG study apply unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
CD = HERE.parents[1] / "component-dag" / "src"
if str(CD) not in sys.path:
    sys.path.insert(0, str(CD))
import cd_extract as ce  # noqa: E402

RULES = ("ancestry", "reads", "cc", "avg", "gmm", "wprof", "complete", "xavg")
EDGE_RULES = ("any", "majority")
MEASURES = ("coverage", "mask_share", "weight_share")
MEASURES_X = MEASURES + ("contrib_share",)      # ``measures_x`` adds the contribution share
GMM_MAX = 8


def _labels(rule, P, scale, R):
    n = len(P)
    if n == 1:
        return np.zeros(1, dtype=int)
    if rule == "cc":
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        return ce._cc(D < scale * R)
    if rule in ("avg", "xavg"):
        return fcluster(linkage(pdist(P), method="average"), t=scale * R, criterion="distance") - 1
    if rule == "complete":
        return fcluster(linkage(pdist(P), method="complete"), t=scale * R, criterion="distance") - 1
    if rule == "gmm":
        best, lab = None, np.zeros(n, dtype=int)
        for k in range(1, min(GMM_MAX, n) + 1):
            g = GaussianMixture(k, covariance_type="spherical", n_init=3, random_state=0).fit(P)
            b = g.bic(P)
            if best is None or b < best:
                best, lab = b, g.predict(P)
        return lab
    raise ValueError(rule)


def _profile_labels(W_in, W_out, tau):
    """Units by the similarity of their weight profiles: rows of |W_in| (what a unit reads) and
    columns of |W_out| (who reads it), each scaled to unit length and joined; average linkage on the
    cosine distance, cut at tau."""
    n = W_in.shape[0]
    if n == 1:
        return np.zeros(1, dtype=int)
    def unit(M):
        M = np.abs(M); nrm = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.where(nrm > 0, nrm, 1.0)
    P = np.concatenate([unit(W_in), unit(W_out.T)], axis=1)
    D = pdist(P, metric="cosine")
    D = np.nan_to_num(D, nan=1.0)
    return fcluster(linkage(D, method="average"), t=tau, criterion="distance") - 1


def coverage(K_b, units, prev_modules):
    """(units x previous modules): share of each previous module's units that each unit reads."""
    out = np.zeros((len(units), len(prev_modules)))
    for j, mod in enumerate(prev_modules):
        cols = np.array([u for _, u in mod])
        out[:, j] = K_b[np.ix_(units, cols)].mean(1)
    return out


def modules(view, rule, radius2, scale=1.0):
    K = [m > 0 for m in view["M"]]
    live = ce.live_units(K, view["H"])
    fields = ce.receptive_fields(K, live)
    L = len(K) - 1
    R = float(np.sqrt(radius2))
    inputs = [int(i) for i in np.flatnonzero(live[0])]
    comps, cfield, reads = [], [], {}
    if rule == "xavg":
        units = [(s, int(u)) for s in range(1, L + 1) for u in np.flatnonzero(live[s])]
        if units:
            P = np.stack([view["E"][s][u] for s, u in units])
            lab = _labels("xavg", P, scale, R)
            for c in range(lab.max() + 1):
                g = [units[i] for i in np.flatnonzero(lab == c)]
                comps.append(sorted(g))
                cfield.append(np.any([fields[s][u] for s, u in g], axis=0))
            order = sorted(range(len(comps)), key=lambda k: comps[k][0])
            comps = [comps[k] for k in order]; cfield = [cfield[k] for k in order]
        nodes = [("in", i) for i in inputs] + [("c", k) for k in range(len(comps))] + [("out", 0)]
        return {"kept": K, "live": live, "fields": fields, "components": comps, "cfield": cfield,
                "inputs": inputs, "nodes": nodes, "L": L, "reads": {}, "inputs_index": {-len(inputs) + j: i for j, i in enumerate(inputs)}}
    prev = [[(0, i)] for i in inputs]            # modules of the previous stage
    prev_index = list(range(-len(inputs), 0))    # placeholder ids for the inputs
    for s in range(1, L + 1):
        idx = np.flatnonzero(live[s])
        if len(idx) == 0:
            prev, prev_index = [], []
            continue
        if rule == "ancestry":
            keys, lab = {}, np.empty(len(idx), dtype=int)
            for i, row in enumerate(fields[s][idx]):
                lab[i] = keys.setdefault(row.tobytes(), len(keys))
        elif rule == "reads":
            cov = coverage(K[s - 1], idx, prev)
            keys, lab = {}, np.empty(len(idx), dtype=int)
            for i in range(len(idx)):
                rs = tuple(np.flatnonzero(cov[i] > 0.5))
                if not rs:
                    rs = (int(cov[i].argmax()),)
                lab[i] = keys.setdefault(rs, len(keys))
        elif rule == "wprof":
            W_in = np.asarray(view["W"][s - 1])[np.ix_(idx, np.flatnonzero(live[s - 1]))]
            W_out = np.asarray(view["W"][s])[np.ix_(np.flatnonzero(live[s + 1]), idx)]
            lab = _profile_labels(W_in, W_out, scale)
        else:
            lab = _labels(rule, view["E"][s][idx], scale, R)
        groups = [idx[lab == c] for c in range(lab.max() + 1)]
        this, this_index = [], []
        for g in groups:
            k = len(comps)
            comps.append([(s, int(u)) for u in g])
            cfield.append(fields[s][g].any(0))
            this.append(comps[k]); this_index.append(k)
            if rule == "reads":
                cov_g = coverage(K[s - 1], g, prev).mean(0)
                reads[k] = [prev_index[j] for j in np.flatnonzero(cov_g > 0.5)] or [prev_index[int(cov_g.argmax())]]
        prev, prev_index = this, this_index
    nodes = [("in", i) for i in inputs] + [("c", k) for k in range(len(comps))] + [("out", 0)]
    return {"kept": K, "live": live, "fields": fields, "components": comps, "cfield": cfield,
            "inputs": inputs, "nodes": nodes, "L": L, "reads": reads, "inputs_index": {-len(inputs) + j: i for j, i in enumerate(inputs)}}


def measures(view, ex):
    """Every adjacent pair (sender module, receiver module) with its three connection measures:
    coverage, mask share, weight share (see the module docstring)."""
    K = ex["kept"]; W = view["W"]; L = ex["L"]
    by_stage = {}
    for k, units in enumerate(ex["components"]):
        by_stage.setdefault(units[0][0], []).append(k)
    out = {}
    prev = [("in", i) for i in ex["inputs"]]
    prev_units = [[(0, i)] for i in ex["inputs"]]
    for s in range(1, L + 2):
        if s == L + 1:
            this, this_units = [("out", 0)], [[(L + 1, 0)]]
        else:
            this = [("c", k) for k in by_stage.get(s, [])]
            this_units = [ex["components"][k] for k in by_stage.get(s, [])]
        if this_units and prev_units:
            Kb = K[s - 1].astype(float); Wb = np.abs(np.asarray(W[s - 1])) * Kb
            prev_cols = [np.array([u for _, u in m]) for m in prev_units]
            allprev = np.concatenate(prev_cols)
            for key, units in zip(this, this_units):
                rows = np.array([u for _, u in units])
                deg = Kb[np.ix_(rows, allprev)].sum(1); mass = Wb[np.ix_(rows, allprev)].sum(1)
                deg = np.where(deg > 0, deg, 1.0); mass = np.where(mass > 0, mass, 1.0)
                for j, cols in enumerate(prev_cols):
                    sub = Kb[np.ix_(rows, cols)]
                    out[(prev[j], key)] = {"coverage": float(sub.mean()),
                                           "mask_share": float((sub.sum(1) / deg).mean()),
                                           "weight_share": float((Wb[np.ix_(rows, cols)].sum(1) / mass).mean())}
        prev, prev_units = this, this_units
    return out


def measures_x(view, ex, n_rows=4096):
    """``measures`` for modules of any stage composition (the parts of two modules on every
    transition; a pair's value is the largest over the transitions that join them), with a fourth
    measure, ``contrib_share``: the variance of the sender part's summed contribution to a receiving
    unit's pre-activation (effective weight times the sender's activation on the pool) over the
    variance of the unit's whole pre-activation, averaged over the receiver part.  For modules of one
    layer the first three measures equal ``measures``."""
    K = ex["kept"]; W = view["W"]; L = ex["L"]
    acts = [view["X"]] + list(view["H"])                       # the activations of stage 0 (inputs) .. L
    parts = {}                                                  # (module key, stage) -> unit columns
    for k, units in enumerate(ex["components"]):
        for s, u in units:
            parts.setdefault((("c", k), s), []).append(u)
    for i in ex["inputs"]:
        parts[(("in", i), 0)] = [i]
    parts[(("out", 0), L + 1)] = [0]
    out = {}
    for s in range(1, L + 2):
        senders = [(key, np.array(cols)) for (key, st), cols in parts.items() if st == s - 1]
        receivers = [(key, np.array(rows)) for (key, st), rows in parts.items() if st == s]
        if not senders or not receivers:
            continue
        Kb = K[s - 1].astype(float); Wb = np.abs(np.asarray(W[s - 1])) * Kb
        Weff = np.asarray(W[s - 1]) * Kb
        A = acts[s - 1][:n_rows]
        allprev = np.concatenate([c for _, c in senders])
        for rkey, rows in receivers:
            deg = Kb[np.ix_(rows, allprev)].sum(1); mass = Wb[np.ix_(rows, allprev)].sum(1)
            deg = np.where(deg > 0, deg, 1.0); mass = np.where(mass > 0, mass, 1.0)
            z = A[:, allprev] @ Weff[np.ix_(rows, allprev)].T
            vz = z.var(0); vz = np.where(vz > 0, vz, 1.0)
            for skey, cols in senders:
                if skey == rkey:
                    continue
                sub = Kb[np.ix_(rows, cols)]
                za = A[:, cols] @ Weff[np.ix_(rows, cols)].T
                val = {"coverage": float(sub.mean()), "mask_share": float((sub.sum(1) / deg).mean()),
                       "weight_share": float((Wb[np.ix_(rows, cols)].sum(1) / mass).mean()),
                       "contrib_share": float((za.var(0) / vz).mean())}
                prev = out.get((skey, rkey))
                out[(skey, rkey)] = val if prev is None else {m: max(prev[m], val[m]) for m in val}
    return out


def edges(view, ex, rule):
    """Module graph edges; ``majority`` reads the mask, ``any`` is the component-DAG rule,
    ``<measure>@<theta>`` thresholds one of the three connection measures."""
    if rule == "any":
        return ce.edges(view, ex, "any")
    if "@" in rule:
        m, th = rule.split("@")
        return {pair for pair, v in measures(view, ex).items() if v[m] > float(th)}
    K = ex["kept"]; live = ex["live"]; L = ex["L"]
    owner_stage = {}
    by_stage = {}
    for k, units in enumerate(ex["components"]):
        s = units[0][0]
        by_stage.setdefault(s, []).append(k)
    out = set()
    prev = [("in", i) for i in ex["inputs"]]
    prev_units = [[(0, i)] for i in ex["inputs"]]
    for s in range(1, L + 2):
        if s == L + 1:
            this = [("out", 0)]; this_units = [[(L + 1, 0)]]
        else:
            this = [("c", k) for k in by_stage.get(s, [])]
            this_units = [ex["components"][k] for k in by_stage.get(s, [])]
        if not this_units or not prev_units:
            prev, prev_units = this, this_units
            continue
        for key, units in zip(this, this_units):
            rows = np.array([u for _, u in units])
            cov = coverage(K[s - 1], rows, prev_units).mean(0)
            for j in np.flatnonzero(cov > 0.5):
                out.add((prev[j], key))
        prev, prev_units = this, this_units
    return out
