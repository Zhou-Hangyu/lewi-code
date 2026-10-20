"""Records (units/main8b/) -> tables.  Every state read through the reference alone
(avg@1 | coverage@0.5 | one_to_one) for the trained network and its controls: the initialization
(the init state of the same unit), the modules shuffled at fixed per-layer counts, the wiring rewired
at fixed degrees, the first transition kept, the names dealt at random within layers.

    python sd_read.py            tables/*.csv, tables/*.md, tables/examples.json, tables/examples/;
                                 the paper's data folder (figures/data-final-clean/structure/)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
STRUCT = HERE.parent
UNITS = STRUCT / "units" / "main8b"
DATA = STRUCT / "tables"
PAPER = REPO / "studies/communication/paper/figures/data-final-clean/structure"
MO = REPO / "studies/csfrbench/evaluation-protocol/structural-metrics/structure-discovery/modules"
sys.path.insert(0, str(MO / "src"))
from r3_read import ci  # noqa: E402  (mean and 95% Student-t interval over instances)

MSET, KEY = "avg@1", "coverage@0.5|one_to_one"
FAMILIES = ("tree", "ftree", "dag", "rand")
LABEL = {"tree": "tree (depth 4)", "ftree": "Fourier tree (depth 4)", "dag": "reuse DAG (depth 3)", "rand": "random graph (depth 4)"}
NS = [2 ** k for k in range(9, 16)]
TOP = 32768
CONDITIONS = ("trained", "init", "shuffled", "rewired", "first_kept", "random_names")
CONTROLS = ("init", "shuffled", "rewired", "first_kept", "random_names")
COND_LABEL = {"trained": "trained LeWi", "init": "initialization", "shuffled": "modules shuffled", "rewired": "wiring rewired (degrees kept)",
              "first_kept": "first transition kept, rest rewired", "random_names": "names dealt at random within layers"}
LEVELS = ("edge", "first", "higher", "readout", "ancestor")
PAIRED_METRICS = ("edge_f1", "first_f1", "higher_f1", "ancestor_f1", "node_coverage", "edge_precision", "edge_recall")
CURVE_METRICS = ("edge_f1", "higher_f1", "ancestor_f1", "node_coverage", "first_f1", "edge_precision", "edge_recall", "decidable_recall")


def load():
    recs = {}
    for f in sorted(UNITS.glob("*.json")):
        r = json.loads(f.read_text())
        recs[(r["cell"], int(r["function_seed"]), int(r["n"]), r["which"])] = r
    return recs


def metrics(sc, n_nodes):
    out = {}
    for lvl in LEVELS:
        for k in ("precision", "recall", "f1"):
            out[f"{lvl}_{k}"] = sc[lvl][k]
    d = sc["disconnected"]
    out.update(node_coverage=sc["n_named_nodes"] / n_nodes, module_precision=sc["n_named_modules"] / max(1, d["modules"]),
               n_modules=d["modules"], disconnected=d["either"] / max(1, d["modules"]), n_edges=sc["n_edges"], n_pred_edges=sc["n_pred_edges"],
               undecidable_edges=sc["undecidable_edges"], decidable_recall=sc["decidable_recall"])
    return out


DEAD = {**{f"{lvl}_{k}": (np.nan if k == "precision" else 0.0) for lvl in LEVELS for k in ("precision", "recall", "f1")},
        "node_coverage": 0.0, "module_precision": np.nan, "n_modules": 0, "disconnected": np.nan, "n_edges": 0, "n_pred_edges": 0,
        "undecidable_edges": 0, "decidable_recall": 0.0}      # a network with no live hidden unit: no module, no edge, nothing recovered


def scores_of(r, wiring):
    """The compact reference scores of one wiring of a record, or None when the network has no live unit."""
    W = r["wirings"]
    if wiring not in W or MSET not in W[wiring]:
        return None
    return W[wiring][MSET]


def state_rows(recs):
    rows = []
    for (fam, fs, n, which), r in recs.items():
        if which != "best":
            continue
        L = scores_of(r, "learned")
        base = {"family": fam, "fs": fs, "n": n, "dead": int(L is None)}
        if L is None:
            for cond in ("trained", "shuffled", "rewired", "first_kept"):
                rows.append({**base, "condition": cond, **DEAD})
            rows.append({**base, "condition": "random_names", "edge_f1": 0.0, "higher_f1": 0.0, "ancestor_f1": 0.0})
        else:
            rows.append({**base, "condition": "trained", **metrics(L["scores"][KEY], r["n_nodes"])})
            rows.append({**base, "condition": "shuffled", **metrics(L["shuffle_modules"][KEY], r["n_nodes"])})
            for cond, wiring in (("rewired", "shuffle_all"), ("first_kept", "shuffle_higher")):
                S = scores_of(r, wiring)
                rows.append({**base, "condition": cond, **(metrics(S["scores"][KEY], r["n_nodes"]) if S else DEAD)})
            rn = L["scores"]["random_names|one_to_one"]["coverage@0.5"]
            rows.append({**base, "condition": "random_names", "edge_f1": rn["edge"], "higher_f1": rn["higher"], "ancestor_f1": rn["ancestor"]})
        ri = recs.get((fam, fs, n, "init"))
        if ri is not None:
            Li = scores_of(ri, "learned")
            rows.append({"family": fam, "fs": fs, "n": n, "dead": int(Li is None), "condition": "init", **(metrics(Li["scores"][KEY], ri["n_nodes"]) if Li else DEAD)})
    return pd.DataFrame(rows)


def fmt(c, d=2):
    return "—" if c["mean"] is None or np.isnan(c["mean"]) else f"{c['mean']:.{d}f} [{c['lo']:.{d}f}, {c['hi']:.{d}f}]"


def summary(states):
    rows = []
    metric_cols = [c for c in states.columns if c not in ("family", "fs", "n", "condition", "dead")]
    for (fam, n, cond), g in states.groupby(["family", "n", "condition"]):
        for m in metric_cols:
            v = g[m].dropna()
            if len(v) == 0:
                continue
            c = ci(v.tolist())
            rows.append({"family": fam, "n": n, "condition": cond, "metric": m, "mean": c["mean"], "lo": c["lo"], "hi": c["hi"], "n_instances": c["n"]})
    df = pd.DataFrame(rows)
    m4 = (df[df.family.isin(FAMILIES)].groupby(["n", "condition", "metric"]).agg(mean=("mean", "mean"), families=("family", "size")).reset_index())
    m4 = m4[m4.families == len(FAMILIES)].drop(columns="families")
    m4["family"] = "mean4"; m4["lo"] = np.nan; m4["hi"] = np.nan; m4["n_instances"] = 0
    return pd.concat([df, m4[df.columns]], ignore_index=True)


def paired(states):
    rows = []
    T = states[states.condition == "trained"].set_index(["family", "fs", "n"])
    for cond in CONTROLS:
        C = states[states.condition == cond].set_index(["family", "fs", "n"])
        idx = T.index.intersection(C.index)
        for m in PAIRED_METRICS:
            if m not in C.columns:
                continue
            d = (T.loc[idx, m] - C.loc[idx, m]).dropna()
            for (fam, n), g in d.groupby(level=["family", "n"]):
                c = ci(g.tolist())
                rows.append({"family": fam, "n": n, "control": cond, "metric": m, "mean": c["mean"], "lo": c["lo"], "hi": c["hi"],
                             "wins": int((g > 1e-12).sum()), "ties": int((g.abs() <= 1e-12).sum()), "n_instances": c["n"]})
    return pd.DataFrame(rows)


def precision_recall(states):
    rows = []
    for (fam, n, cond), g in states.groupby(["family", "n", "condition"]):
        for lvl in LEVELS:
            if f"{lvl}_precision" not in g.columns or g[f"{lvl}_f1"].dropna().empty:
                continue
            rec = {"family": fam, "n": n, "condition": cond, "level": lvl}
            for k in ("precision", "recall", "f1"):
                c = ci(g[f"{lvl}_{k}"].dropna().tolist())
                rec.update({f"{k}_mean": c["mean"], f"{k}_lo": c["lo"], f"{k}_hi": c["hi"], "n_instances": c["n"]})
            rows.append(rec)
    return pd.DataFrame(rows)


def wiring_alignment(states):
    """The appendix table's data (figures wiring_alignment_table): at n = 2^15 per family and condition, the mean instance-level
    precision, recall and F1 of every level, the recall on the decidable target edges (both endpoints assignable to a module) and
    the decidable share of the target's edges, each with its 95% Student-t interval over the instances."""
    rows = []
    for (fam, cond), g in states[states.n == TOP].groupby(["family", "condition"]):
        if "edge_f1" not in g.columns or g["edge_f1"].dropna().empty:
            continue
        rec = {"family": fam, "n": TOP, "condition": cond, "n_instances": int(len(g))}
        for lvl in LEVELS:
            for k in ("precision", "recall", "f1"):
                c = ci(g[f"{lvl}_{k}"].dropna().tolist()) if f"{lvl}_{k}" in g.columns else ci([])
                rec.update({f"{lvl}_{k}_mean": c["mean"], f"{lvl}_{k}_lo": c["lo"], f"{lvl}_{k}_hi": c["hi"]})
        c = ci(g["decidable_recall"].dropna().tolist()) if "decidable_recall" in g.columns else ci([])
        rec.update(decidable_recall_mean=c["mean"], decidable_recall_lo=c["lo"], decidable_recall_hi=c["hi"])
        if "n_edges" in g.columns:
            share = ((g.n_edges - g.undecidable_edges) / g.n_edges.where(g.n_edges > 0)).dropna().tolist()
            c = ci(share)
            rec.update(decidable_share_mean=c["mean"], decidable_share_lo=c["lo"], decidable_share_hi=c["hi"],
                       undecidable_edges_mean=float(g.undecidable_edges.mean()), n_edges_mean=float(g.n_edges.mean()))
        rows.append(rec)
    return pd.DataFrame(rows)


def coverage(states):
    rows = []
    cols = ("node_coverage", "module_precision", "n_modules", "disconnected", "undecidable_edges", "decidable_recall", "n_edges", "n_pred_edges")
    for (fam, n, cond), g in states.groupby(["family", "n", "condition"]):
        if "node_coverage" not in g.columns or g["node_coverage"].dropna().empty:
            continue
        rec = {"family": fam, "n": n, "condition": cond}
        for m in cols:
            c = ci(g[m].dropna().tolist())
            rec.update({f"{m}_mean": c["mean"], f"{m}_lo": c["lo"], f"{m}_hi": c["hi"]})
        rec["n_instances"] = int(g["node_coverage"].notna().sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def node_class(r, v):
    if str(v) in map(str, r["target_readout"]) and len(r["target_readout"]) == 1:
        return "root"
    return "first" if int(r["target_depth"][str(v)]) == 1 else "higher"


def diagnosis(recs):
    """Node statuses by class and the reasons of the missing target edges (trained states, one-to-one naming)."""
    st_rows, at_rows, wl_rows = [], [], []
    for (fam, fs, n, which), r in recs.items():
        if which != "best":
            continue
        L = scores_of(r, "learned")
        if L is None:
            continue
        for v, rec in L["status"].items():
            st_rows.append({"family": fam, "fs": fs, "n": n, "node": v, "class": node_class(r, v), "status": rec["status"], "best_r2": rec["best_r2"]})
        for edge, reason in L["scores"]["attribution|one_to_one"].items():
            at_rows.append({"family": fam, "fs": fs, "n": n, "edge": edge, "reason": reason})
        wl = L["whole_layers"]; Ltop = len(r["positions"]) - 2
        wl_rows.append({"family": fam, "fs": fs, "n": n, "n_whole_layers": len(wl), "top_layer_whole": int(any(x["stages"] == [Ltop] for x in wl)),
                        "decodable_per_whole": float(np.mean([len(x["decodable"]) for x in wl])) if wl else np.nan,
                        "n_true_edges": r["n_true_edges"], "n_missing": len(L["scores"]["attribution|one_to_one"])})
    return pd.DataFrame(st_rows), pd.DataFrame(at_rows), pd.DataFrame(wl_rows)


def examples(states):
    """Per family the instance at the median reference edge F1 at the top size (the modules study's rule)."""
    out = {}
    T = states[(states.condition == "trained") & (states.n == TOP)]
    for fam in FAMILIES:
        g = T[T.family == fam].sort_values(["edge_f1", "fs"], kind="stable")
        if g.empty:
            continue
        row = g.iloc[len(g) // 2]
        name = f"{fam}__fs{int(row.fs)}__n{TOP}__best"
        out[fam] = {"record": f"{name}.json", "fs": int(row.fs), "n": TOP, "edge_f1": float(row.edge_f1), "higher_f1": float(row.higher_f1),
                    "node_coverage": float(row.node_coverage), "rank": int((g.edge_f1 < row.edge_f1).sum()) + 1, "of": int(len(g))}
    return out


def md_table(df, cols, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_md(summ, pr, cov, paired_df):
    S = summ[(summ.n == TOP)]
    out = [f"# Structure discovery on the clean-input cohort (d = 8, 64,000 steps), n = 2^15\n",
           "Reference procedure: within-layer average linkage of the learned positions at R, majority edges, one-to-one naming. "
           "Mean over the instances (16 per workload; 32 on the random graphs since 2026-09-20) with the 95% Student-t interval.\n"]
    for cond in CONDITIONS:
        rows = []
        for fam in FAMILIES:
            rec = {"family": LABEL[fam]}
            for lvl in LEVELS:
                for k in ("precision", "recall", "f1"):
                    q = pr[(pr.n == TOP) & (pr.condition == cond) & (pr.family == fam) & (pr.level == lvl)]
                    rec[f"{lvl}_{k}"] = fmt({"mean": q[f"{k}_mean"].iloc[0], "lo": q[f"{k}_lo"].iloc[0], "hi": q[f"{k}_hi"].iloc[0]}) if len(q) else "—"
            rows.append(rec)
        m4 = S[(S.family == "mean4") & (S.condition == cond)].set_index("metric")["mean"]
        rows.append({"family": "mean of four", **{f"{lvl}_{k}": (f"{m4[f'{lvl}_{k}']:.2f}" if f"{lvl}_{k}" in m4 else "—") for lvl in LEVELS for k in ("precision", "recall", "f1")}})
        df = pd.DataFrame(rows)
        out.append(f"\n## {COND_LABEL[cond]}: precision / recall / F1 by level\n")
        cols = ["family"] + [f"{lvl}_{k}" for lvl in ("edge", "first", "higher", "readout", "ancestor") for k in ("precision", "recall", "f1")]
        header = ["family"] + [f"{lvl} {k}" for lvl in ("edge", "first level", "higher level", "readout", "ancestor") for k in ("precision", "recall", "F1")]
        out.append(md_table(df, cols, header))
    out.append("\n## Node coverage, modules, undecidable edges (trained and controls)\n")
    rows = []
    for cond in ("trained", "init", "shuffled", "rewired", "first_kept"):
        for fam in FAMILIES:
            q = cov[(cov.n == TOP) & (cov.condition == cond) & (cov.family == fam)]
            if q.empty:
                continue
            q = q.iloc[0]
            rows.append({"condition": COND_LABEL[cond], "family": LABEL[fam],
                         **{m: fmt({"mean": q[f"{m}_mean"], "lo": q[f"{m}_lo"], "hi": q[f"{m}_hi"]}, 1 if m in ("n_modules", "undecidable_edges", "n_edges", "n_pred_edges") else 2)
                            for m in ("node_coverage", "module_precision", "n_modules", "disconnected", "undecidable_edges", "decidable_recall")}})
    out.append(md_table(pd.DataFrame(rows), ["condition", "family", "node_coverage", "module_precision", "n_modules", "disconnected", "undecidable_edges", "decidable_recall"],
                        ["condition", "family", "node coverage", "module precision", "modules per network", "disconnected modules", "undecidable target edges", "recall on the decidable edges"]))
    out.append("\n## Paired differences, trained minus control, per instance (n = 2^15)\n")
    rows = []
    for cond in CONTROLS:
        for fam in FAMILIES:
            rec = {"control": COND_LABEL[cond], "family": LABEL[fam]}
            for m in ("edge_f1", "higher_f1", "ancestor_f1", "node_coverage"):
                q = paired_df[(paired_df.n == TOP) & (paired_df.control == cond) & (paired_df.family == fam) & (paired_df.metric == m)]
                rec[m] = (fmt({"mean": q["mean"].iloc[0], "lo": q["lo"].iloc[0], "hi": q["hi"].iloc[0]}) + f" ({int(q.wins.iloc[0])}/{int(q.n_instances.iloc[0])})") if len(q) else "—"
            rows.append(rec)
    out.append(md_table(pd.DataFrame(rows), ["control", "family", "edge_f1", "higher_f1", "ancestor_f1", "node_coverage"],
                        ["control", "family", "edge F1 (wins)", "higher-level F1 (wins)", "ancestor F1 (wins)", "node coverage (wins)"]))
    out.append("\n## Edge F1 against the training-set size (trained; initialization in parentheses)\n")
    rows = []
    for fam in FAMILIES + ("mean4",):
        rec = {"family": LABEL.get(fam, "mean of four")}
        for n in NS:
            t = S if False else summ
            a = t[(t.family == fam) & (t.n == n) & (t.condition == "trained") & (t.metric == "edge_f1")]
            b = t[(t.family == fam) & (t.n == n) & (t.condition == "init") & (t.metric == "edge_f1")]
            rec[str(n)] = (f"{a['mean'].iloc[0]:.2f} ({b['mean'].iloc[0]:.2f})") if len(a) and len(b) else "—"
        rows.append(rec)
    out.append(md_table(pd.DataFrame(rows), ["family"] + [str(n) for n in NS], ["family"] + [f"2^{int(np.log2(n))}" for n in NS]))
    (DATA / "tables.md").write_text("\n".join(out) + "\n")


def main():
    DATA.mkdir(exist_ok=True); (DATA / "examples").mkdir(exist_ok=True)
    recs = load()
    states = state_rows(recs)
    states.to_csv(DATA / "states.csv", index=False)
    summ = summary(states); summ.to_csv(DATA / "summary.csv", index=False)
    summ[summ.n == TOP].to_csv(DATA / "summary_2p15.csv", index=False)
    pdf = paired(states); pdf.to_csv(DATA / "paired.csv", index=False)
    pr = precision_recall(states); pr.to_csv(DATA / "precision_recall.csv", index=False)
    cov = coverage(states); cov.to_csv(DATA / "coverage.csv", index=False)
    wa = wiring_alignment(states); wa.to_csv(DATA / "wiring_alignment.csv", index=False)
    st, at, wl = diagnosis(recs)
    st.to_csv(DATA / "node_status.csv", index=False); at.to_csv(DATA / "missing_edges.csv", index=False); wl.to_csv(DATA / "whole_layers.csv", index=False)
    curves = summ[summ.metric.isin(CURVE_METRICS)].copy(); curves.to_csv(DATA / "alignment_vs_n.csv", index=False)
    ex = examples(states)
    (DATA / "examples.json").write_text(json.dumps(ex, indent=1))
    for fam, e in ex.items():
        shutil.copy(UNITS / e["record"], DATA / "examples" / e["record"])
    write_md(summ, pr, cov, pdf)
    # the paper's data folder
    PAPER.mkdir(parents=True, exist_ok=True); (PAPER / "examples").mkdir(exist_ok=True)
    f6 = summ[(summ.n == TOP) & summ.condition.isin(("trained", "init", "shuffled", "rewired")) & summ.metric.isin(("edge_f1", "higher_f1")) & summ.family.isin(FAMILIES)]
    f6[["family", "condition", "metric", "mean", "lo", "hi", "n_instances"]].to_csv(PAPER / "fig6b.csv", index=False)
    curves.to_csv(PAPER / "alignment_vs_n.csv", index=False)
    wa.to_csv(PAPER / "wiring_alignment.csv", index=False)
    (PAPER / "examples.json").write_text(json.dumps(ex, indent=1))
    for fam, e in ex.items():
        shutil.copy(UNITS / e["record"], PAPER / "examples" / e["record"])
    counts = states[states.condition == "trained"].groupby(["family", "n"]).size().unstack("n")
    print(counts)
    ref = summ[(summ.n == TOP) & (summ.metric == "edge_f1")]
    for cond in CONDITIONS:
        print(cond, {fam: round(float(ref[(ref.family == fam) & (ref.condition == cond)]["mean"].iloc[0]), 3) for fam in FAMILIES + ("mean4",)
                     if len(ref[(ref.family == fam) & (ref.condition == cond)])})


if __name__ == "__main__":
    main()
