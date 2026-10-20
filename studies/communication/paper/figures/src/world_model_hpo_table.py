"""Appendix E.2: the final adopted hyperparameter search spaces of the world-model experiments (decision 2026-09-24), in the
style of the CSFRBench table of Appendix C.12: Method | Parameter | Search range, one row per searched parameter, the canonical method
names, every bound in scientific notation with three significant digits, a one-sentence caption, the fixed hyperparameters that
interpret a space in a note below the table. Only the spaces the reported results used: the record's frozen spaces
(09-swm-lewi/results/spaces_final_v2.json, the file the campaign's sampler read; its sha256 is pinned in the re-search manifest) with
the learning-rate ranges of Dense and l1 replaced by the adopted expanded ranges of the re-search (18-main-d8/swm-lrx/manifests/
spaces_lrx.json, arms.<arm>.expanded; adopted for both arms 2026-09-24). No occupancy, history or superseded range.
Writes world_model_hpo_table.{tex,md,csv}; the csv names the source of every bound.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from figstyle import OUT

STEM = Path(__file__).stem
REPO = Path(__file__).resolve().parents[5]
RECORD_SPACES = REPO / "studies/main-experiments/09-swm-lewi/results/spaces_final_v2.json"
LRX = REPO / "studies/main-experiments/18-main-d8/swm-lrx/manifests/spaces_lrx.json"
ARMS = [("dense", "Dense"), ("l1", r"$\ell_1$"), ("l2", r"$\ell_2$"), ("l0hc", "$L_0$"), ("rigl", "RigL"), ("bimt", "BIMT"), ("fiedler", "Fiedler"),
        ("erb", "Learned neuron positions"), ("lewi", "LeWi")]
PLAIN = {"dense": "Dense", "l1": "l1", "l2": "l2", "l0hc": "L0", "rigl": "RigL", "bimt": "BIMT", "fiedler": "Fiedler", "erb": "Learned neuron positions", "lewi": "LeWi"}
PARAM_ORDER = ["lr", "lam", "density", "delta"]
PARAM = {"lr": "learning rate", "lam": r"$\lambda$", "density": "density", "delta": r"$\delta$"}
PARAM_PLAIN = {"lr": "learning rate", "lam": "lambda", "density": "density", "delta": "delta"}


def sci(x: float) -> str:
    """Three significant digits in scientific notation; an exact 1 stays 1."""
    if x == 1:
        return "1"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, 2) >= 10:                                  # 9.995 -> 10.0
        m, e = m / 10, e + 1
    return rf"{m:.2f}\times10^{{{e}}}"


def tex_range(lo: float, hi: float) -> str:
    return rf"$[{sci(lo)},\,{sci(hi)}]$"


def main() -> None:
    rec = json.loads(RECORD_SPACES.read_text())
    lrx = json.loads(LRX.read_text())
    sha = hashlib.sha256(RECORD_SPACES.read_bytes()).hexdigest()
    assert sha == lrx["source_sha256"], "the frozen-spaces file is not the one the re-search manifest pins"
    assert lrx.get("adoption") and sorted(lrx["adoption"]["arms"]) == ["dense", "l1"], "the re-search is not recorded as adopted for both arms"
    rows, tex_rows, md_rows = [], [], []
    for arm, name in ARMS:
        space = rec[arm]
        keys = [k for k in PARAM_ORDER if k in space and isinstance(space[k], list)]
        for i, k in enumerate(keys):
            lo, hi, log = space[k]
            source = f"{RECORD_SPACES.relative_to(REPO)} [{arm}][{k}]"
            if k == "lr" and arm in lrx["arms"]:
                lo, hi, log = lrx["arms"][arm]["expanded"]["lr"]
                assert hi == rec[arm]["lr"][1] and abs(lo * 10 - rec[arm]["lr"][0]) < 1e-12 * rec[arm]["lr"][0], f"{arm}: the expanded range is not the record's with the lower bound divided by 10"
                source = f"{LRX.relative_to(REPO)} [arms][{arm}][expanded][lr] (adopted {lrx['adoption']['decision_date']})"
            assert log is True, f"{arm} {k} is not log-uniform"
            rows.append(dict(method=PLAIN[arm], arm=arm, parameter=PARAM_PLAIN[k], key=k, low=lo, high=hi, scale="log-uniform", low_tex=sci(lo), high_tex=sci(hi), source=source))
            tex_rows.append(f"{name if i == 0 else ''} & {PARAM[k]} & {tex_range(lo, hi)}")
            md_rows.append(f"| {PLAIN[arm] if i == 0 else ''} | {PARAM_PLAIN[k]} | [{lo:.3g}, {hi:.3g}] | {source} |")
    caption = (r"\textbf{Hyperparameter search spaces for the world-model experiments.} All searched numerical parameters are sampled" "\n"
               r"log-uniformly; every method searches its learning rate, and $\ell_1$, $\ell_2$, $L_0$ and BIMT the penalty coefficient" "\n"
               r"$\lambda$, RigL the density and Fiedler the threshold $\delta$ beside it (32 scrambled-Sobol trials per task, training-set" "\n"
               r"size and method; selection by the validation rollout score).")
    note = (r"Fixed settings: width 256, four hidden layers, 128,000 steps, 16 paired final seeds; for $L_0$, $p_{\mathrm{open,init}} = "
            + f"{rec['l0hc']['p_open_init']:g}" + r"$; LeWi's embedding dimension $r = 3$ and $R^2 = 1.5$ are method constants, not searched.")
    tex = (f"\\begin{{table}}[tbp]\n\\centering\n\\caption{{\n{caption}\n}}\n\\label{{tab:world-model-search-spaces}}\n"
           "\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{llc}\n\\toprule\n"
           "Method & Parameter & Search range \\\\\n\\midrule\n" + " \\\\\n".join(tex_rows) + " \\\\\n\\bottomrule\n\\end{tabular}\n"
           + f"\\par\\smallskip\\noindent\\footnotesize {note}\n\\end{{table}}\n")
    (OUT / f"{STEM}.tex").write_text(tex)
    md = "\n".join(["# World-model (SWM-16) hyperparameter search spaces, the final adopted ones (Appendix E.2)", "",
                    f"Sources: the record's frozen spaces {RECORD_SPACES.relative_to(REPO)} (sha256 {sha[:16]}...; every parameter log-uniform) and, for the learning rates of Dense and l1, "
                    f"the adopted expanded ranges of {LRX.relative_to(REPO)} (the lower bound divided by 10, everything else unchanged; adopted {lrx['adoption']['decision_date']}).", "",
                    "| Method | Parameter | Search range | Source |", "|---|---|---|---|", *md_rows, ""])
    (OUT / f"{STEM}.md").write_text(md)
    with (OUT / f"{STEM}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"[{STEM}] {len(rows)} rows -> {OUT / STEM}.tex / .md / .csv")
    print(tex)


if __name__ == "__main__":
    main()
