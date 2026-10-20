"""Not a figure: LeWi's learned connection density on the final runs of the two main settings (appendix; decision 2026-09-24), as LaTeX (out/connection_density_table.tex, label tab:connection-density), Markdown and CSV.

Source: 18-main-d8/density/tables (density/src/density.py). summary.csv: per setting and workload, the active-edge fraction of
the final selected LeWi model (lewi-v1-rgm; the selected checkpoint best.pt) of every target instance and training-set size,
averaged over sizes per instance, then the mean and 95% Student-t interval over instances; coverage.csv: the runs on record
per cell; by_size.csv: the instances per size (the coverage note of the partial cells). A transition's density is the mean of
its 0/1 connection mask (shape out x in); the overall density pools the edges of every transition; the hidden column pools the
hidden-to-hidden transitions; the initialization density is the same quantity on init.pt. Rows: the seven d = 8 workloads and
the depth-5 tree at d = 16, then the same at d = 32 (24 nuisance inputs). The CSV keeps every transition's mean and interval.

Checks (PASS / FAIL lines; a FAIL stops the build): 16 rows; every N equals the covered instances and every run count the
covered runs of coverage.csv; every interval is ordered and inside [0, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aulc_table import md_table
from figstyle import FIGURES, OUT

STEM = "connection_density_table"
TABLES = FIGURES.parents[3] / "studies/main-experiments/18-main-d8/density/tables"
SETTINGS = [("d8", "All inputs active", ["tree8", "dag8", "ftree8", "flat8", "dense8", "fdag8", "rand8", "tree5d16"]),
            ("d32s", "Nuisance inputs ($d = 32$)", ["tree32s", "dag32s", "ftree32s", "flat32s", "dense32s", "fdag32s", "rand32s", "tree5d32"])]
NAME = {"tree": "Depth-4 tree", "dag": "Reuse DAG", "ftree": "Fourier tree", "flat": "Flat modules", "dense": "High-arity node",
        "fdag": "Fourier reuse DAG", "rand": "Sampled depth-4 DAGs", "tree5": "Depth-5 tree"}          # the paper's target names (decision 2026-09-23)
COLUMNS = [("density_init_overall", "Initialization"), ("density_end_overall", "End"), ("density_end_input", "Input"),
           ("density_end_hidden", "Hidden"), ("density_end_readout", "Readout")]
TRANSITIONS = [f"density_end_t{k}" for k in range(1, 10)]
CHECKS: list[tuple[bool, str]] = []


def check(ok: bool, text: str) -> None:
    CHECKS.append((bool(ok), text))
    print(f"[{'PASS' if ok else 'FAIL'}] {text}")


def family(cell: str) -> str:
    return "tree5" if cell.startswith("tree5") else cell.rstrip("s").rstrip("0123456789")


def label(cell: str, tex: bool) -> str:
    name = NAME[family(cell)]
    if cell == "tree5d16":
        name += " ($d = 16$)" if tex else " (d = 16)"
    return name


def interval(row: pd.Series, col: str, decimals: int) -> str:
    return f"{row[col + '_mean']:.{decimals}f} [{row[col + '_lo']:.{decimals}f}, {row[col + '_hi']:.{decimals}f}]"


def num(x: int, tex: bool) -> str:
    return f"{x:,}".replace(",", "{,}") if tex else f"{x:,}"


def coverage_note(cov: pd.DataFrame, by_size: pd.DataFrame, tex: bool) -> str:
    parts = []
    for setting, title, cells in SETTINGS:
        c = cov[cov.setting == setting]
        partial = []
        for r in c[c.covered < c.ledger_rows].itertuples():
            b = by_size[(by_size.setting == setting) & (by_size.cell == r.cell)].sort_values("n")
            top, below = b.iloc[-1], b.iloc[:-1].N
            n_top = f"$2^{{{int(np.log2(top.n))}}}$" if tex else f"2^{int(np.log2(top.n))}"
            partial.append(f"{label(r.cell, tex)} {r.covered} of {r.ledger_rows} runs on {r.instances_covered} instances, "
                           f"{int(below.min())}{'--' if tex else '-'}{int(below.max())} instances per size below {n_top} and {int(top.N)} at it")
        parts.append(f"{title if tex else title.replace('$', '')}: {num(int(c.covered.sum()), tex)} of {num(int(c.ledger_rows.sum()), tex)} "
                     "selected runs on record" + (f" ({'; '.join(partial)})" if partial else ""))
    note = "; ".join(parts) + "."
    rerun = int(cov["rerun"].sum()) if "rerun" in cov else 0
    if rerun:
        note += (f" {rerun} of these runs (the Fourier reuse DAG and the depth-5 tree at $d = 16$) had no retained folder and were re-run"
                 " deterministically by the campaign's runner from their recorded configuration; every re-run reproduces its recorded"
                 " selected step, validation and test errors bit for bit.").replace("$d = 16$", "$d = 16$" if tex else "d = 16")
    if int((cov.covered < cov.ledger_rows).sum()):
        note += " $^{\\dagger}$ marks the partial cells." if tex else " (partial) marks the partial cells."
    return note


def main() -> None:
    summary = pd.read_csv(TABLES / "summary.csv")
    cov = pd.read_csv(TABLES / "coverage.csv")
    by_size = pd.read_csv(TABLES / "by_size.csv")
    rows = []
    for setting, _, cells in SETTINGS:
        for cell in cells:
            s = summary[(summary.setting == setting) & (summary.cell == cell)]
            c = cov[(cov.setting == setting) & (cov.cell == cell)]
            check(len(s) == 1 and len(c) == 1, f"{setting} {cell}: one summary row and one coverage row")
            s, c = s.iloc[0], c.iloc[0]
            check(int(s.N) == int(c.instances_covered) and int(s.runs) == int(c.covered),
                  f"{setting} {cell}: N = {int(s.N)} instances and {int(s.runs)} runs match coverage.csv")
            ordered = all(0.0 <= s[q + "_lo"] <= s[q + "_mean"] <= s[q + "_hi"] <= 1.0 for q, _ in COLUMNS)
            check(ordered, f"{setting} {cell}: every interval ordered and inside [0, 1]")
            row = dict(setting=setting, cell=cell, workload=label(cell, False), N=int(s.N), runs=int(s.runs), ledger_rows=int(c.ledger_rows),
                       partial=bool(c.covered < c.ledger_rows))
            for q in [q for q, _ in COLUMNS] + TRANSITIONS:
                for stat in ("mean", "lo", "hi"):
                    row[f"{q}_{stat}"] = s.get(f"{q}_{stat}", float("nan"))
            rows.append(row)
    table = pd.DataFrame(rows)
    check(len(table) == 16, "16 rows: eight workloads in each of the two settings")
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / f"{STEM}.csv", index=False)
    # the pattern the caption states, from the rows themselves
    d8 = table[table.setting == "d8"].set_index("cell")
    d32 = table[table.setting == "d32s"].set_index("cell")
    end_range = lambda t, cells: (t.loc[cells].density_end_overall_mean.min(), t.loc[cells].density_end_overall_mean.max())   # noqa: E731
    comp8 = ["tree8", "dag8", "ftree8", "flat8", "fdag8", "rand8"]
    comp32 = ["tree32s", "dag32s", "ftree32s", "flat32s", "fdag32s", "rand32s"]
    rising = [c for c in comp8 + comp32 if (table.set_index("cell").loc[c, ["density_end_t1_mean", "density_end_t2_mean", "density_end_t3_mean",
                                                                             "density_end_t4_mean"]].diff().dropna() > 0).all()]
    check(len(rising) == len(comp8 + comp32), "the density rises from the input transition through the hidden transitions on every compositional workload")
    check(bool(d8.loc["dense8", ["density_end_t1_mean", "density_end_t2_mean", "density_end_t3_mean", "density_end_t4_mean", "density_end_t5_mean"]]
               .diff().dropna().lt(0).all()), "the high-arity node at d = 8 falls from the input transition to the readout")
    caption = (r"\textbf{LeWi's learned connection density on the final runs.}" "\n"
               r"Active-edge fraction of the final selected LeWi model (the selected checkpoint, \texttt{best.pt}) of every target" "\n"
               r"instance and training-set size: the mean over sizes per instance, then the mean and 95\% Student-$t$ interval" "\n"
               r"over the $N$ instances (brackets). A transition's density is the mean of its $0/1$ connection mask; the overall" "\n"
               r"density (Initialization, End) pools the edges of every transition (input $\to$ hidden 1, the hidden-to-hidden" "\n"
               r"transitions, hidden $\to$ output); Input, Hidden and Readout are the end-of-training densities of the input" "\n"
               r"transition, of the hidden-to-hidden transitions pooled, and of the readout; Initialization is the overall density" "\n"
               r"of the run's initial masks (\texttt{init.pt}). Backbones: 32x4 over 64{,}000 steps ($n = 2^9 \ldots 2^{15}$); the" "\n"
               r"depth-5 trees 64x8 over 128{,}000 steps ($n = 2^{11} \ldots 2^{17}$). Coverage: " + coverage_note(cov, by_size, True) + "\n"
               f"Every initialization sits between {table.density_init_overall_mean.min():.3f} and {table.density_init_overall_mean.max():.3f}; training removes edges on every" "\n"
               f"workload but the high-arity node at $d = 8$: the overall end density is {end_range(d8, comp8)[0]:.2f}--{end_range(d8, comp8)[1]:.2f} on the six" "\n"
               f"compositional workloads at $d = 8$ and {end_range(d32, comp32)[0]:.2f}--{end_range(d32, comp32)[1]:.2f} at $d = 32$, where the 24 nuisance inputs make the input" "\n"
               r"transition the sparsest part of the network. Within a network the density rises with depth, from the input" "\n"
               r"transition through the hidden transitions, on every compositional workload; the high-arity node at $d = 8$ is the" "\n"
               f"exception, its density falling from {d8.loc['dense8', 'density_end_t1_mean']:.2f} at the input transition to {d8.loc['dense8', 'density_end_readout_mean']:.2f} at the readout.")
    head = "Workload & $N$ & " + " & ".join(h for _, h in COLUMNS) + " \\\\"
    body = []
    for setting, title, cells in SETTINGS:
        body.append(f"\\multicolumn{{{2 + len(COLUMNS)}}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for r in table[table.setting == setting].itertuples():
            row = table.loc[r.Index]
            body.append(f"{label(r.cell, True)}{'$^{\\dagger}$' if r.partial else ''} & {r.N} & "
                        + " & ".join(interval(row, q, 2) for q, _ in COLUMNS) + " \\\\")
        body.append("\\midrule")
    body.pop()
    tex = (""
           "% Every number is computed from the final selected runs' init.pt and best.pt (18-main-d8/density/tables/summary.csv).\n"
           f"\\begin{{table}}[tbp]\n\\centering\n\\caption{{\n{caption}\n}}\n\\label{{tab:connection-density}}\n"
           "\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrccccc}\n\\toprule\n" + head + "\n\\midrule\n"
           + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT / f"{STEM}.tex").write_text(tex)
    md_head = ["Setting", "Workload", "N", "runs"] + [h for _, h in COLUMNS]
    md_rows = [[r.setting, label(r.cell, False) + (" (partial)" if r.partial else ""), str(r.N), str(r.runs)]
               + [interval(table.loc[r.Index], q, 3) for q, _ in COLUMNS] for r in table.itertuples()]
    (OUT / f"{STEM}.md").write_text(
        "# LeWi's learned connection density on the final runs\n\n"
        "Active-edge fraction of the final selected LeWi model (best.pt) of every target instance and size: the mean over sizes per "
        "instance, then the mean [95% Student-t interval] over the N instances. Overall = every transition's edges pooled; Input, "
        "Hidden (the hidden-to-hidden transitions pooled) and Readout at the end of training; Initialization = the overall density of "
        f"init.pt. Coverage: {coverage_note(cov, by_size, False)}\n\n" + md_table(md_head, md_rows))
    failed = [t for ok, t in CHECKS if not ok]
    print(f"[table] {STEM}  {len(table)} rows from {TABLES.relative_to(FIGURES.parents[3])}; {len(CHECKS) - len(failed)} of {len(CHECKS)} checks PASS")
    if failed:
        raise SystemExit("FAILED checks: " + "; ".join(failed))


if __name__ == "__main__":
    main()
