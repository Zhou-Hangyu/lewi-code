"""Not a figure: the AULC numbers behind Figure 3, as LaTeX and Markdown tables.

Writes out/aulc_table.tex (three tables, intake/appendix-1.md) and
out/aulc_table.md:

  Table tab:aulc                  mean AULC over the 16 target instances with its
                                  95% Student-t half-width, the six fixed workloads
  Table tab:aulc-paired-primary   AULC(LeWi) - AULC(method) on the depth-4 tree,
                                  the depth-3 reuse DAG and the depth-4 Fourier tree
  Table tab:aulc-paired-controls  the same on flat modules, the Fourier reuse DAG
                                  and the high-arity node

Paired cells read "mean [95% paired Student-t interval]" over "(instances on
which LeWi has the lower AULC / 16)"; negative favors LeWi. Same data and
definitions as Figure 3 (data/sample_efficiency/curves.csv plus the dense
node's 12-extra roster arms; csfrbench.metrics.aulc over n = 2^9 ... 2^15,
lower is better). The depth-5 tree and the independently sampled depth-4
DAGs use other sample-size ranges and instance counts and are tabulated by
additional_structures_table; the fixed-configuration sensitivity study by
fixed_hyperparameters_table. All three table stems share the cell grammar
defined here, sized for the 5.5 in ICLR text width at the style's \\small
(9 pt): compact "mean+-half-width" cells, two-line column headers and
two-line paired cells (tex/check_tables.tex compiles them under the ICLR
style and reports any overfull line).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import paperstyle as ps
from figstyle import OUT
from sample_efficiency import load, per_instance_aulc

PRIMARY = ["tree_functional_anova", "dag_functional_anova", "tree_fourier"]
CONTROLS = ["flat_functional_anova", "dag_fourier_v2", "dense_fourier"]
ORDER = PRIMARY + CONTROLS
# the appendix's column names (intake/appendix-1.md), as two header lines; the
# figure titles differ on two settings (paperstyle.TITLE: "Depth-3 Fourier
# reuse DAG", "High-arity node")
COLUMN = {"tree_functional_anova": ("Depth-4", "tree"), "dag_functional_anova": ("Depth-3", "reuse DAG"),
          "tree_fourier": ("Depth-4", "Fourier tree"), "flat_functional_anova": ("Flat", "modules"),
          "dag_fourier_v2": ("Fourier", "reuse DAG"), "dense_fourier": ("High-arity", "node")}
ARMS = [ps.LEWI, "noreg", "l1", "l2", "l0hc", "rigl", "bimt", "fiedler",
        "erb_positions", "graph_oracle(levels=0)", "graph_oracle(levels=1)",
        "graph_oracle(levels=all)"]
TEX_LABEL = {**ps.NAME, "l1": r"$\ell_1$", "l2": r"$\ell_2$", "l0hc": r"$L_0$", "input_l0": r"Input $L_0$"}
OFF_SCALE = 2.0    # an interval bound beyond this marks a degenerate winner (the 12-extra report's rule)

CAPTION_MEAN = (r"\textbf{Mean AULC on the fixed CSFRBench settings.}" "\n"
                r"Values are means over 16 target instances with 95\% Student-$t$ confidence" "\n"
                r"half-widths. AULC integrates test NMSE over" "\n"
                r"$n=2^9,\ldots,2^{15}$; lower is better.")
CAPTION_PRIMARY = (r"\textbf{Paired AULC comparisons on the primary compositional settings.}" "\n"
                   r"Each entry reports" "\n"
                   r"$\mathrm{AULC}(\mathrm{LeWi})-\mathrm{AULC}(\mathrm{method})$ as" "\n"
                   r"mean [95\% paired Student-$t$ interval], followed by the number of target" "\n"
                   r"instances on which LeWi has lower AULC. Negative values favor LeWi.")
CAPTION_CONTROLS = (r"\textbf{Paired AULC comparisons on additional fixed settings.}" "\n"
                    r"Reporting follows Table~\ref{tab:aulc-paired-primary}; negative values favor" "\n"
                    r"LeWi.")


def summarize(values: np.ndarray) -> dict:
    """Mean cell statistics of one arm on one setting."""
    m, lo, hi = ps.t_interval(values)
    return {"mean": m, "half": (hi - lo) / 2, "median": float(np.median(values)), "n": len(values)}


def summarize_paired(d: np.ndarray) -> dict:
    """Paired AULC(LeWi) - AULC(arm) statistics (d < 0 favors LeWi)."""
    m, lo, hi = ps.t_interval(d)
    return {"mean": m, "lo": lo, "hi": hi, "median": float(np.median(d)), "wins": int((d < 0).sum()), "n": len(d)}


def off_scale(s: dict) -> bool:
    bounds = (s["mean"] - s["half"], s["mean"] + s["half"]) if "half" in s else (s["lo"], s["hi"])
    return max(abs(b) for b in bounds) > OFF_SCALE


def num(x: float, signed: bool = False) -> str:
    return f"${x:+.3f}$" if signed else f"${x:.3f}$"


def mean_cell(s: dict | None) -> str:
    """"0.680+-0.033"; a degenerate arm shows its median with a dagger."""
    if s is None:
        return "--"
    if off_scale(s):
        return f"${s['median']:.3f}^\\dagger$"
    return f"${s['mean']:.3f}{{\\pm}}{s['half']:.3f}$"


def paired_cell(s: dict | None) -> str:
    """Two lines: "mean [lo, hi]" over "(wins/n)"; a degenerate pair shows its
    median with a dagger over the count."""
    if s is None:
        return "--"
    top = (f"${s['median']:+.3f}^\\dagger$" if off_scale(s)
           else f"{num(s['mean'], True)} [{num(s['lo'], True)}, {num(s['hi'], True)}]")
    return f"\\begin{{tabular}}[t]{{@{{}}c@{{}}}}{top}\\\\({s['wins']}/{s['n']})\\end{{tabular}}"


def md_cell(s: dict | None, paired: bool) -> str:
    if s is None:
        return "--"
    if paired:
        body = f"{s['median']:+.3f} (median)" if off_scale(s) else f"{s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]"
        return f"{body} ({s['wins']}/{s['n']})"
    return f"{s['median']:.3f} (median)" if off_scale(s) else f"{s['mean']:.3f} +- {s['half']:.3f}"


def tex_table(head_rows: list[list[str]], rows: list[list[str]], cols: str, caption: str, label: str,
              *, cmidrules: str = "", colsep: str = "4pt", note: str = "") -> str:
    """One booktabs table at the ICLR style's \\small (9 pt); \\tabcolsep is
    tightened so the six-column table stays inside the 5.5 in text width."""
    head = " \\\\\n".join(" & ".join(r) for r in head_rows)
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    tail = f"\n\\par\\smallskip\\noindent\\footnotesize {note}" if note else ""
    return (f"\\begin{{table}}[tbp]\n\\centering\n\\caption{{\n{caption}\n}}\n\\label{{{label}}}\n"
            f"\\small\n\\setlength{{\\tabcolsep}}{{{colsep}}}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
            + head + (" \\\\\n" + cmidrules if cmidrules else " \\\\\n\\midrule\n") + body
            + " \\\\\n\\bottomrule\n\\end{tabular}" + tail + "\n\\end{table}\n")


def md_table(head: list[str], rows: list[list[str]]) -> str:
    return ("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
            + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n")


def stats(aulc: pd.DataFrame, workloads: list[str], arms: list[str], min_instances: int = 16) -> tuple[dict, dict]:
    """(arm, workload) -> mean statistics and paired-against-LeWi statistics
    (absent where the arm did not run on the setting)."""
    means, pairs = {}, {}
    for wl in workloads:
        a = aulc[aulc.workload == wl]
        lewi = a[a.arm == ps.LEWI].set_index("function_seed").aulc
        for arm in arms:
            mine = a[a.arm == arm].set_index("function_seed").aulc
            if len(mine) < min_instances:
                continue
            means[arm, wl] = summarize(mine.to_numpy(dtype=float))
            if arm != ps.LEWI:
                pairs[arm, wl] = summarize_paired((lewi - mine.reindex(lewi.index)).to_numpy(dtype=float))
    return means, pairs


def two_line_header(workloads: list[str], first: str = "", second: str = "Method") -> list[list[str]]:
    return [[first] + [COLUMN[w][0] for w in workloads], [second] + [COLUMN[w][1] for w in workloads]]


def main() -> None:
    df = load()
    df = df[df.workload.isin(ORDER)]
    means, pairs = stats(per_instance_aulc(df, ps.WINDOW), ORDER, ARMS)
    mean_rows = [[TEX_LABEL[a]] + [mean_cell(means.get((a, w))) for w in ORDER] for a in ARMS]
    # no LeWi row (its own contrast) and no row of dashes only (an arm absent from every column)
    paired_rows = {name: [[TEX_LABEL[a]] + [paired_cell(pairs.get((a, w))) for w in wls]
                          for a in ARMS if a != ps.LEWI and any((a, w) in pairs for w in wls)]
                   for name, wls in (("primary", PRIMARY), ("controls", CONTROLS))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "aulc_table.tex").write_text(
        tex_table(two_line_header(ORDER), mean_rows, "l" + "c" * len(ORDER), CAPTION_MEAN, "tab:aulc", colsep="2pt")
        + "\n" + tex_table(two_line_header(PRIMARY), paired_rows["primary"], "lccc", CAPTION_PRIMARY,
                           "tab:aulc-paired-primary")
        + "\n" + tex_table(two_line_header(CONTROLS), paired_rows["controls"], "lccc", CAPTION_CONTROLS,
                           "tab:aulc-paired-controls"))
    title = lambda wls: ["Method"] + [" ".join(COLUMN[w]) for w in wls]  # noqa: E731
    md_rows = lambda wls, paired: [[TEX_LABEL[a]] + [md_cell((pairs if paired else means).get((a, w)), paired)  # noqa: E731
                                                     for w in wls] for a in ARMS
                                                    if not (paired and (a == ps.LEWI or not any((a, w) in pairs for w in wls)))]
    (OUT / "aulc_table.md").write_text(
        "# Mean AULC on the fixed CSFRBench settings\n\n" + md_table(title(ORDER), md_rows(ORDER, False))
        + "\n# Paired AULC(LeWi) - AULC(method), primary compositional settings\n\n"
        + md_table(title(PRIMARY), md_rows(PRIMARY, True))
        + "\n# Paired AULC(LeWi) - AULC(method), additional fixed settings\n\n"
        + md_table(title(CONTROLS), md_rows(CONTROLS, True)))
    print(f"[table] aulc_table  {len(mean_rows)} arms x {len(ORDER)} workloads; paired tables "
          f"{len(paired_rows['primary'])} x {len(PRIMARY)} and {len(paired_rows['controls'])} x {len(CONTROLS)}")


if __name__ == "__main__":
    main()
