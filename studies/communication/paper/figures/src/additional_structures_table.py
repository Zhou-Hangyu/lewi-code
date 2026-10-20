"""Not a figure: the AULC numbers of the two additional computational
structures of Figure 3, as a LaTeX table (out/additional_structures_table.tex,
label tab:additional-structures) and Markdown (out/additional_structures_table.md).

Per method on the depth-5 tree (16 target instances; 64x8 backbone, 128,000
steps, n = 2^11 ... 2^17) and on the 32 independently sampled depth-4
DAG/function instances (common protocol, n = 2^9 ... 2^15): the mean AULC
with its 95% Student-t half-width and the paired AULC(LeWi) - AULC(method)
as mean [95% paired interval] over (instances on which LeWi has the lower
AULC / total); negative favors LeWi. The two settings integrate over
different sample-size ranges, so their AULCs are not comparable with each
other or with the fixed-setting table (aulc_table).

Dense on the depth-5 tree (08-main): one winner (instance 7011 at n = 2^12)
is a collapsed run whose retained weights blow up on rare test inputs (test
NMSE 2.2 x 10^8), entered unclipped by the protocol; its mean and every
paired mean involving it are off scale, so those cells show the median with
a dagger and the instance count, as the executed study reports them (08-main
LOG entry 47, 12-extra report). The dagger sentence of the caption is
derived from the data at build time (the degenerate cells of the arm whose
mean is off scale) and is absent when no arm is off scale. Same cell grammar
and data as aulc_table: data/sample_efficiency/curves.csv (08-main) and
oracle_rungs_12extra.csv (GraphOracle(1) and (2) on these settings, 12-extra
wave 2); under data-v2/ the 13-main ledger alone (every rung is 13-main's own),
the depth-5 profile 64x8 with 256,000 steps over n = 2^12 ... 2^18.
"""
from __future__ import annotations

import numpy as np

import paperstyle as ps
from aulc_table import OFF_SCALE, TEX_LABEL, md_cell, md_table, mean_cell, paired_cell, stats
from figstyle import CAMPAIGN, OUT
from sample_efficiency import load, per_instance_aulc

SETTINGS = [("tree_functional_anova:depth4", "Depth-5 tree", 16),
            ("rand_functional_anova", "Independently sampled depth-4 DAGs", 32)]
ARMS = [ps.LEWI, "noreg", "l1", "l2", "l0hc", "rigl", "bimt", "fiedler", "erb_positions",   # tier B where it ran (the final build: depth-5 tree)
        "graph_oracle(levels=0)", "graph_oracle(levels=1)", "graph_oracle(levels=2)", "graph_oracle(levels=all)"]
DEPTH5_STEPS = {"08-main": "128{,}000", "13-main": "256{,}000", "14-main": "256{,}000"}[CAMPAIGN.name]      # the depth-5 profile's budget
SETTING_TEXT = {"tree_functional_anova:depth4": "depth-5 tree",
                "rand_functional_anova": "independently sampled depth-4 DAGs"}


def caption(dagger_text: str, n_rand: int = 32) -> str:
    lo, hi = int(np.log2(ps.WINDOW_DEPTH5[0])), int(np.log2(ps.WINDOW_DEPTH5[-1]))
    return (r"\textbf{AULC on the depth-5 tree and on independently sampled depth-4 DAGs.}" "\n"
            r"Mean AULC with 95\% Student-$t$ half-widths, and" "\n"
            r"$\mathrm{AULC}(\mathrm{LeWi})-\mathrm{AULC}(\mathrm{method})$ as mean" "\n"
            r"[95\% paired Student-$t$ interval] over the number of instances on which LeWi has" "\n"
            r"lower AULC; negative values favor LeWi. The depth-5 tree (16 target instances;" "\n"
            f"width 64, eight hidden layers, {DEPTH5_STEPS} steps) integrates test NMSE over" "\n"
            f"$n=2^{{{lo}}},\\ldots,2^{{{hi}}}$, the {n_rand} independently sampled depth-4 DAG/function" "\n"
            r"instances over $n=2^9,\ldots,2^{15}$; neither range matches Table~\ref{tab:aulc}, so" "\n"
            r"AULC values are comparable within a column only." + dagger_text)


def dagger_sentence(df, means: dict) -> tuple[str, list]:
    """The caption's dagger clause, from the data: for every (arm, setting)
    whose mean is off scale, the degenerate winners (test NMSE above the
    off-scale bound) that put it there."""
    rows = [(a, w) for a in ARMS for w, _, _ in SETTINGS if (a, w) in means and "dagger" in mean_cell(means[a, w])]
    parts = []
    for arm, wl in rows:
        cells = df[(df.workload == wl) & (df.arm == arm) & (df.test_nmse > OFF_SCALE)]
        n_inst = means[arm, wl]["n"]
        if len(cells) == 1:
            c = cells.iloc[0]
            exp = int(np.floor(np.log10(c.test_nmse)))
            parts.append(f"one {TEX_LABEL[arm]} winner on the" "\n"
                         f"{SETTING_TEXT[wl]} is degenerate (a collapsed run whose retained weights blow up on rare" "\n"
                         f"test inputs; test NMSE ${c.test_nmse / 10 ** exp:.1f}\\times10^{{{exp}}}$ at "
                         f"$n=2^{{{int(np.log2(c.n))}}}$), so the {TEX_LABEL[arm]} mean and the" "\n"
                         f"paired mean are off scale; these entries are medians over the {n_inst} instances, and the" "\n"
                         "instance counts are unaffected")
        else:
            parts.append(f"{len(cells)} {TEX_LABEL[arm]} winners on the {SETTING_TEXT[wl]} are degenerate (collapsed" "\n"
                         f"runs whose retained weights blow up on rare test inputs; the largest test NMSE is" "\n"
                         f"${cells.test_nmse.max():.1e}$), so the {TEX_LABEL[arm]} mean and the paired mean are off" "\n"
                         f"scale; these entries are medians over the {n_inst} instances, and the instance counts are" "\n"
                         "unaffected")
    if not parts:
        return "", rows
    return " " + r"$\dagger$: " + "; ".join(parts) + ".", rows


def main() -> None:
    df = load()
    df = df[df.workload.isin([w for w, _, _ in SETTINGS])]
    aulc = per_instance_aulc(df)                      # each setting's own window (paperstyle.WINDOW_OF)
    means, pairs = stats(aulc, [w for w, _, _ in SETTINGS], ARMS, min_instances=16)
    counts = {w: int(df[df.workload == w].function_seed.nunique()) for w, _, _ in SETTINGS}   # 32 graphs in the record, 16 at d = 8
    for w, _, _ in SETTINGS:                          # every tabulated arm ran on all instances
        for arm in ARMS:
            if (arm, w) in means:
                assert means[arm, w]["n"] == counts[w], (arm, w, means[arm, w]["n"], counts[w])
    rows = [[TEX_LABEL[a]] + sum(([mean_cell(means.get((a, w))), paired_cell(pairs.get((a, w)))]
                                  for w, _, _ in SETTINGS), []) for a in ARMS]
    dagger_text, dagger = dagger_sentence(df, means)
    head = [["", r"\multicolumn{2}{c}{Depth-5 tree}", r"\multicolumn{2}{c}{Independently sampled depth-4 DAGs}"],
            ["Method", "AULC", r"LeWi $-$ method", "AULC", r"LeWi $-$ method"]]
    cmid = "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
    body = (f"\\begin{{table}}[tbp]\n\\centering\n\\caption{{\n{caption(dagger_text, counts["rand_functional_anova"])}\n}}\n\\label{{tab:additional-structures}}\n"
            f"\\small\n\\setlength{{\\tabcolsep}}{{2.5pt}}\n\\begin{{tabular}}{{lcccc}}\n\\toprule\n"
            + " & ".join(head[0]) + " \\\\\n" + cmid + " & ".join(head[1]) + " \\\\\n\\midrule\n"
            + " \\\\\n".join(" & ".join(r) for r in rows) + " \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "additional_structures_table.tex").write_text(body)
    md_head = ["Method"] + sum(([f"{t}: AULC", f"{t}: LeWi - method"] for _, t, _ in SETTINGS), [])
    md_rows = [[TEX_LABEL[a]] + sum(([md_cell(means.get((a, w)), False), md_cell(pairs.get((a, w)), True)]
                                     for w, _, _ in SETTINGS), []) for a in ARMS]
    (OUT / "additional_structures_table.md").write_text(
        "# AULC on the depth-5 tree and on independently sampled depth-4 DAGs\n\n" + md_table(md_head, md_rows))
    print(f"[table] additional_structures_table  {len(rows)} arms x {len(SETTINGS)} settings; "
          f"median-with-dagger rows: {[a for a, _ in dagger]}")


if __name__ == "__main__":
    main()
