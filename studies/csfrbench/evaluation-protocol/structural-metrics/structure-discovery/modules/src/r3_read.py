"""Round-3 records -> one JSON read: per module set, edge rule and naming the level scores per
family with 95% confidence intervals over instances (trained, initialization, the rewired
wirings, the modules-shuffled control, the multiplicity-matched random names), node coverage and
module precision, the persistence and superposition counts, the node statuses, the whole-layer
diagnosis, the attribution of the missing edges.

    python r3_read.py --units ../results/units/r3 --out ../results/r3_read.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

FAMILIES = ("tree", "ftree", "dag", "fdag", "flat", "rand", "dense")
STRUCTURED = ("tree", "ftree", "dag", "fdag", "flat", "rand")
LEVELS = ("edge", "first", "higher", "readout", "ancestor")
SCORE_WIRINGS = ("learned", "shuffle_all", "shuffle_higher")
PERSIST = ("copies", "one_to_many")


def ci(values):
    v = np.array([x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = float(v.mean())
    if len(v) < 2:
        return {"mean": m, "lo": m, "hi": m, "n": 1}
    h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return {"mean": m, "lo": m - h, "hi": m + h, "n": int(len(v))}


def node_class(rec, v):
    if str(v) in map(str, rec["target_readout"]) and len(rec["target_readout"]) == 1:
        return "root"
    return "first" if int(rec["target_depth"][str(v)]) == 1 else "higher"


def copy_distances(r, w, naming="copies"):
    """For every node with modules in two adjacent layers under the naming: the mean distance between
    the units of the lower module and those of the upper one, in units of R, and the mean distance from
    the lower module to the other modules of the upper layer."""
    R = float(np.sqrt(r["radius2"])); pos = r["positions"]
    comps = w["graph"]["components"]; names = w["names"].get(naming, {}); stages = w["component_stages"]
    by_node = {}
    for j, ids in names.items():
        for v in ids:
            by_node.setdefault(v, []).append(int(j))
    same, other = [], []
    for v, js in by_node.items():
        for a in js:
            for b in js:
                sa, sb = stages[a], stages[b]
                if len(sa) == 1 and len(sb) == 1 and sb[0] == sa[0] + 1:
                    A = np.array([pos[s][u] for s, u in comps[a]]); B = np.array([pos[s][u] for s, u in comps[b]])
                    same.append(float(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).mean()) / R)
                    for c in range(len(comps)):
                        if c != b and stages[c] == sb and v not in names.get(str(c), []):
                            C = np.array([pos[s][u] for s, u in comps[c]])
                            other.append(float(np.sqrt(((A[:, None, :] - C[None, :, :]) ** 2).sum(-1)).mean()) / R)
    return (float(np.mean(same)) if same else np.nan), (float(np.mean(other)) if other else np.nan)


def load(units_dir, n=None):
    recs = []
    for f in sorted(Path(units_dir).glob("*.json")):
        r = json.loads(f.read_text())
        if n is None or int(r["n"]) == n:
            recs.append(r)
    return recs


def score_source(rec, wiring, mset, key, control=None):
    """The compact scores of one (wiring, module set, rule|naming), or of a control on the learned wiring."""
    w = rec["wirings"].get(wiring)
    if w is None or mset not in w:
        return None
    if control == "shuffle_modules":
        return w[mset].get("shuffle_modules", {}).get(key)
    return w[mset]["scores"].get(key)


def read(recs):
    recs_t = [r for r in recs if r["which"] == "best"]
    recs_i = [r for r in recs if r["which"] == "init"]
    msets = recs[0]["module_sets"]; rules = recs[0]["edge_rules"]; namings = recs[0]["namings"]
    out = {"n_states": len(recs), "n_trained": len(recs_t), "n_init": len(recs_i), "module_sets": msets, "edge_rules": rules,
           "namings": namings, "reference": recs[0]["reference"], "families": list(FAMILIES), "instances": {}}
    out["instances"] = {fam: sorted({r["function_seed"] for r in recs_t if r["cell"] == fam}) for fam in FAMILIES}

    def table(source):
        """source(rec) -> compact scores or None; per family the CI of every level + coverage numbers."""
        tab = {}
        for fam in FAMILIES:
            rows = [(r, source(r)) for r in recs_t if r["cell"] == fam]
            rows = [(r, s) for r, s in rows if s]
            t = {lvl: ci([s[lvl]["f1"] for _, s in rows]) for lvl in LEVELS}
            for lvl in ("edge", "higher"):
                t[lvl + "_precision"] = ci([s[lvl]["precision"] for _, s in rows]); t[lvl + "_recall"] = ci([s[lvl]["recall"] for _, s in rows])
            t["values"] = [round(s["edge"]["f1"], 4) for _, s in rows]
            t["decidable_recall"] = ci([s.get("decidable_recall") for _, s in rows])
            t["undecidable_edges"] = ci([s.get("undecidable_edges") for _, s in rows])
            t["node_coverage"] = ci([s["n_named_nodes"] / r["n_nodes"] for r, s in rows])
            t["module_precision"] = ci([s["n_named_modules"] / max(1, s["disconnected"]["modules"]) for _, s in rows])
            t["names_per_named_module"] = ci([s["n_names"] / max(1, s["n_named_modules"]) for _, s in rows])
            t["disconnected"] = ci([s["disconnected"]["either"] / max(1, s["disconnected"]["modules"]) for _, s in rows])
            t["n_edges"] = ci([s["n_edges"] for _, s in rows]); t["n_pred_edges"] = ci([s["n_pred_edges"] for _, s in rows])
            t["n_modules"] = ci([s["disconnected"]["modules"] for _, s in rows])
            tab[fam] = t
        tab["mean6"] = {lvl: float(np.mean([tab[f][lvl]["mean"] for f in STRUCTURED if tab[f][lvl]["mean"] is not None])) for lvl in LEVELS}
        return tab

    out["tables"] = {}
    for mset in msets:
        out["tables"][mset] = {}
        for rule in rules:
            for naming in namings:
                key = f"{rule}|{naming}"
                entry = {}
                for wiring in SCORE_WIRINGS:
                    entry[wiring] = table(lambda r, w=wiring: score_source(r, w, mset, key))
                entry["shuffle_modules"] = table(lambda r: score_source(r, "learned", mset, key, "shuffle_modules"))
                # initialization: the init states of the same instances
                init_tab = {}
                for fam in FAMILIES:
                    ss = [score_source(r, "learned", mset, key) for r in recs_i if r["cell"] == fam]
                    ss = [s for s in ss if s]
                    init_tab[fam] = {lvl: ci([s[lvl]["f1"] for s in ss]) for lvl in LEVELS}
                    init_tab[fam]["node_coverage"] = ci([s["n_named_nodes"] / r["n_nodes"] for s, r in zip(ss, [r for r in recs_i if r["cell"] == fam])])
                    init_tab[fam]["values"] = [round(s["edge"]["f1"], 4) for s in ss]
                init_tab["mean6"] = {lvl: float(np.mean([init_tab[f][lvl]["mean"] for f in STRUCTURED if init_tab[f][lvl]["mean"] is not None])) for lvl in LEVELS}
                entry["init"] = init_tab
                # multiplicity-matched random names (learned wiring)
                rn = {}
                for fam in FAMILIES:
                    vals = {lvl: [] for lvl in ("edge", "higher", "ancestor")}
                    for r in recs_t:
                        if r["cell"] != fam:
                            continue
                        w = r["wirings"].get("learned", {}).get(mset)
                        if not w:
                            continue
                        x = w["scores"].get(f"random_names|{naming}", {}).get(rule)
                        if x:
                            for lvl in vals:
                                vals[lvl].append(x[lvl])
                    rn[fam] = {lvl: ci(v) for lvl, v in vals.items()}
                rn["mean6"] = {lvl: float(np.mean([rn[f][lvl]["mean"] for f in STRUCTURED if rn[f][lvl]["mean"] is not None])) for lvl in ("edge", "higher", "ancestor")}
                entry["random_names"] = rn
                out["tables"][mset][key] = entry

    # persistence (one-to-many), superposition (many-to-many), statuses, whole layers, attribution: per module set
    out["persistence"], out["superposition"], out["status"], out["whole"], out["attribution"], out["sizes"] = {}, {}, {}, {}, {}, {}
    for mset in msets:
        pers, sup, stt, whl, att, siz = {}, {}, {}, {}, {}, {}
        for fam in FAMILIES:
            rs = [r for r in recs_t if r["cell"] == fam and mset in r["wirings"].get("learned", {})]
            per = {"multi": [], "multi_stage": [], "joined": [], "by_class": {c: [] for c in ("first", "higher", "root")}}
            s_mod, s_ser, s_par, s_whole, s_any_par = [], [], [], [], []
            par_by_stage, ser_by_stage = {}, {}
            counts = {}
            n_whole, top_whole, n_dec = [], [], []
            reasons = {}
            n_missing = []
            per_by = {nm_: {"multi": [], "multi_stage": [], "joined": [], "by_class": {c: [] for c in ("first", "higher", "root")}} for nm_ in PERSIST}
            s_mod, s_ser, s_par, s_whole, s_any_par = [], [], [], [], []
            par_by_stage, ser_by_stage = {}, {}
            counts = {}
            n_whole, top_whole, n_dec = [], [], []
            reasons = {}
            n_missing = []
            for r in rs:
                w = r["wirings"]["learned"][mset]
                n_nodes = r["n_nodes"]
                for nm_ in PERSIST:
                    p = w["scores"].get("persistence", {}).get(nm_, {})
                    per = per_by[nm_]
                    per["multi"].append(sum(v["modules"] >= 2 for v in p.values()) / n_nodes)
                    per["multi_stage"].append(sum(len(v["stages"]) >= 2 for v in p.values()) / n_nodes)
                    mm = [v for v in p.values() if v["modules"] >= 2]
                    per["joined"].append(float(np.mean([v["joined"] for v in mm])) if mm else np.nan)
                    for c in per["by_class"]:
                        vs = [p[str(v)]["modules"] if str(v) in p else 0 for v in r["target_match_nodes"] if node_class(r, v) == c]
                        if vs:
                            per["by_class"][c].append(float(np.mean(vs)))
                s = w["scores"].get("superposition", {})
                s_mod.append(s.get("modules", 0)); s_ser.append(s.get("serial", 0)); s_par.append(s.get("parallel", 0)); s_whole.append(s.get("whole", 0))
                s_any_par.append(float(s.get("parallel", 0) > 0))
                L = len(r["positions"]) - 2
                for d in s.get("detail", []):
                    st = d["stages"][0] if len(d["stages"]) == 1 else -1
                    key = "layer 1" if st == 1 else ("top layer" if st == L else ("middle layers" if st > 0 else "cross-layer"))
                    par_by_stage[key] = par_by_stage.get(key, 0) + d["parallel"]; ser_by_stage[key] = ser_by_stage.get(key, 0) + d["serial"]
                for v, rec in w["status"].items():
                    c = node_class(r, v)
                    counts.setdefault(c, {}); counts[c][rec["status"]] = counts[c].get(rec["status"], 0) + 1
                wl = w["whole_layers"]
                n_whole.append(len(wl)); top_whole.append(float(any(len(x["stages"]) == 1 and x["stages"][0] == len(r["positions"]) - 2 for x in wl)))
                n_dec.append(float(np.mean([len(x["decodable"]) for x in wl])) if wl else np.nan)
                a = w["scores"].get("attribution|one_to_one", {})
                n_missing.append(len(a))
                for reason in a.values():
                    reasons[reason] = reasons.get(reason, 0) + 1
            pers[fam] = {nm_: {"multi": ci(per["multi"]), "multi_stage": ci(per["multi_stage"]), "joined": ci(per["joined"]),
                               "by_class": {c: ci(v) for c, v in per["by_class"].items()}} for nm_, per in per_by.items()}
            cd = [copy_distances(r, r["wirings"]["learned"][mset]) for r in rs]
            pers[fam]["copies"]["copy_dist"] = ci([x for x, _ in cd]); pers[fam]["copies"]["other_dist"] = ci([y for _, y in cd])
            ri = [r for r in recs_i if r["cell"] == fam and mset in r["wirings"].get("learned", {})]
            i_mod, i_par, i_ser, i_any, i_multi = [], [], [], [], []
            for r in ri:
                w = r["wirings"]["learned"][mset]; s = w["scores"].get("superposition", {})
                i_mod.append(s.get("modules", 0)); i_par.append(s.get("parallel", 0)); i_ser.append(s.get("serial", 0)); i_any.append(float(s.get("parallel", 0) > 0))
                p = w["scores"].get("persistence", {}).get("copies", {})
                i_multi.append(sum(v["modules"] >= 2 for v in p.values()) / r["n_nodes"])
            for nm_ in per_by:
                pers[fam][nm_]["init_multi"] = ci(i_multi)
            sup[fam] = {"modules": ci(s_mod), "serial": ci(s_ser), "parallel": ci(s_par), "whole": ci(s_whole), "any_parallel": ci(s_any_par),
                        "init_modules": ci(i_mod), "init_parallel": ci(i_par), "init_serial": ci(i_ser), "init_any_parallel": ci(i_any),
                        "parallel_by_layer": par_by_stage, "serial_by_layer": ser_by_stage}
            stt[fam] = counts
            whl[fam] = {"n_whole": ci(n_whole), "top_whole": ci(top_whole), "decodable_per_whole": ci(n_dec)}
            att[fam] = {"reasons": reasons, "missing": ci(n_missing), "n_true_edges": rs[0]["n_true_edges"] if rs else None}
            siz[fam] = {"n_modules": ci([r["wirings"]["learned"][mset]["n_components"] for r in rs]),
                        "live": ci([sum(r["wirings"]["learned"][mset]["live"][1:-1]) for r in rs])}
        out["persistence"][mset] = {nm_: {fam: pers[fam][nm_] for fam in pers} for nm_ in PERSIST}
        out["superposition"][mset], out["status"][mset] = sup, stt
        out["whole"][mset], out["attribution"][mset], out["sizes"][mset] = whl, att, siz
    # the fit of the winner against its recovery: Spearman correlation per family between the validation NMSE and the reference edge F1
    ref_key = f"{recs[0]['reference'][1]}|{recs[0]['reference'][2]}"; mset0 = recs[0]["reference"][0]
    out["fit"] = {}
    for fam in FAMILIES:
        xs, ys = [], []
        for r in recs_t:
            if r["cell"] != fam:
                continue
            f = Path(__file__).resolve().parents[7] / r["checkpoint"] / "run.json"
            w = r["wirings"].get("learned", {}).get(mset0)
            if not f.exists() or not w:
                continue
            m = json.loads(f.read_text()).get("metrics", {})
            v = m.get("best_val_nmse")
            if v is not None:
                xs.append(float(v)); ys.append(w["scores"][ref_key]["edge"]["f1"])
        if len(xs) > 3:
            rho, pv = stats.spearmanr(xs, ys)
            out["fit"][fam] = {"rho": float(rho), "p": float(pv), "n": len(xs), "val_nmse_median": float(np.median(xs))}
    out["seconds"] = ci([r["seconds"] for r in recs])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    recs = load(a.units, a.n)
    res = read(recs)
    Path(a.out).write_text(json.dumps(res))
    ref = res["reference"]; key = f"{ref[1]}|{ref[2]}"
    t = res["tables"][ref[0]][key]
    print(f"{len(recs)} records; reference mean6 edge {t['learned']['mean6']['edge']:.3f} higher {t['learned']['mean6']['higher']:.3f}; "
          f"init {t['init']['mean6']['edge']:.3f}; shuffled modules {t['shuffle_modules']['mean6']['edge']:.3f}; random names {t['random_names']['mean6']['edge']:.3f}")


if __name__ == "__main__":
    main()
