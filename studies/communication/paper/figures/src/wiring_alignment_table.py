"""Not a figure: the wiring-alignment table of the appendix (out/wiring_alignment_table.tex, label tab:wiring-alignment,
and out/wiring_alignment_table.md; decision 2026-09-22). The module graph read from the trained LeWi network's learned geometry
against the target computation graph at n = 2^15 on the clean-input cohort (d = 8, every input active, 64,000 steps; 16
target instances per setting, 32 sampled depth-4 DAGs): per setting the mean instance-level precision, recall and F1 over all
target edges and over the higher-level edges (from an intermediate node to a node or to the output), the recall restricted
to the assessable target edges (both endpoints can be assigned a module; an edge touching a node whose best module is its
whole hidden layer has no chance reference and cannot be assessed, so a missing dependency and an unassignable endpoint are
told apart) and the assessable share of the target's edges. Means over instances, not metrics computed from pooled edge
counts; the 95% Student-t intervals stand in the data file. Data: data/structure/wiring_alignment.csv (18-main-d8/structure,
src/sd_read.py: states.csv at n = 2^15, condition trained).
"""
from __future__ import annotations

import pandas as pd

from aulc_table import md_table, tex_table
from figstyle import DATA, OUT

FAMILIES = [("tree", "Depth-4 tree"), ("dag", "Reuse DAG"), ("ftree", "Fourier tree"), ("rand", "Sampled depth-4 DAGs")]   # Figure 6's target names (decision 2026-09-23)
COLS = ["edge_precision", "edge_recall", "edge_f1", "higher_precision", "higher_recall", "higher_f1", "decidable_recall", "decidable_share"]
STEM = "wiring_alignment_table"


def caption(counts: dict) -> str:
    n_rand = counts.get("rand", 32)
    return (r"\textbf{Wiring alignment at $n=2^{15}$.}" "\n"
            r"Precision, recall and $F_1$ of the module graph read from the trained LeWi network's learned geometry against the" "\n"
            f"target computation graph, per setting on the clean-input cohort ({counts.get('tree', 16)} target instances per setting," "\n"
            f"{n_rand} for the sampled depth-4 DAGs; every input active, $d=8$, 64{{,}}000 steps): over all target edges, over the" "\n"
            r"higher-level edges (from an intermediate node to a node or to the output), and the recall restricted to the" "\n"
            r"assessable edges (target edges whose two endpoints can be assigned a module; an edge touching a node whose best" "\n"
            r"module is its whole hidden layer cannot be assessed, so a missing dependency is told apart from an unassignable" "\n"
            r"endpoint) with the assessable share of the target's edges. Modules, edges and names as in Figure~6(b).")


def main() -> None:
    df = pd.read_csv(DATA / "structure" / "wiring_alignment.csv")
    df = df[(df.condition == "trained") & (df.n == 32768)].set_index("family")
    missing = [f for f, _ in FAMILIES if f not in df.index]
    if missing:
        raise SystemExit(f"[{STEM}] no trained row at n = 2^15 for {missing} in data/structure/wiring_alignment.csv")
    counts = {f: int(df.loc[f, "n_instances"]) for f, _ in FAMILIES}
    rows = [[name] + [f"{df.loc[f, c + '_mean']:.3f}" for c in COLS] for f, name in FAMILIES]
    note = ("Mean instance-level metrics, not metrics computed from pooled edge counts; 95\\% Student-$t$ intervals over the "
            "instances in \\texttt{data/structure/wiring\\_alignment.csv}.")
    head = [["Setting", r"\multicolumn{3}{c}{All edges}", r"\multicolumn{3}{c}{Higher-level edges}", r"\multicolumn{2}{c}{Assessable edges}"]]
    second = ("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\n"
              + " & ".join(["", "Precision", "Recall", "$F_1$", "Precision", "Recall", "$F_1$", "Recall", "Share"]) + " \\\\\n\\midrule\n")
    (OUT / f"{STEM}.tex").write_text(tex_table(head, [[r[0]] + [f"${v}$" for v in r[1:]] for r in rows], "lcccccccc", caption(counts),
                                              "tab:wiring-alignment", cmidrules=second, colsep="4.5pt", note=note))
    md_head = ["Setting", "Precision", "Recall", "F1", "Higher-level precision", "Higher-level recall", "Higher-level F1",
               "Recall on assessable edges", "Assessable share of edges"]
    md = (f"# Wiring alignment at n = 2^15\n\n{counts.get('tree', 16)} instances per setting ({counts.get('rand', 32)} sampled depth-4 DAGs); "
          "all inputs active (d = 8, 64,000 steps); trained LeWi, the module graph read from its learned geometry against the target graph.\n\n"
          + md_table(md_head, rows)
          + "\nMean instance-level metrics, not metrics computed from pooled edge counts. All edges: every target edge. Higher-level edges: "
            "from an intermediate node to a node or to the output. Assessable edges: target edges whose two endpoints can be assigned a module "
            "(an edge touching a node whose best module is its whole hidden layer cannot be assessed); the share is the mean fraction of the "
            "target's edges that are assessable, so recall over all edges misses both the missing dependencies and the unassessable edges, "
            "the recall on assessable edges the missing dependencies alone. 95% Student-t intervals: data/structure/wiring_alignment.csv.\n")
    (OUT / f"{STEM}.md").write_text(md)
    print(f"[{STEM}] " + "; ".join(f"{name}: edge P/R/F1 {df.loc[f, 'edge_precision_mean']:.3f}/{df.loc[f, 'edge_recall_mean']:.3f}/{df.loc[f, 'edge_f1_mean']:.3f}, "
                                   f"assessable recall {df.loc[f, 'decidable_recall_mean']:.3f} (share {df.loc[f, 'decidable_share_mean']:.3f})" for f, name in FAMILIES))


if __name__ == "__main__":
    main()
