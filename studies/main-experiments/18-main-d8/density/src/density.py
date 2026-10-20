"""LeWi's learned connection density on the final CSFRBench runs (decision 2026-09-24).

Rows: the adopted LeWi arm (lewi-v1-rgm) of the two study ledgers, ledger/d8/winners.csv (the d = 8 setting and the depth-5 tree
at d = 16) and ledger/d32s/winners.csv (the d = 32 setting and the depth-5 tree at d = 32): the final selected model of every
target instance and training-set size. Every number comes from that run's own init.pt (the initialization) and best.pt (the
selected checkpoint = the end of training), read from the live run folder (source "ledger checkpoint") or, when the campaign
archived the folder, from the archive tar (results/csfrbench/m17/archive/<root>_runs.tar, input-cutoff/archive/runs.tar;
source "archive:<tar>"). A row whose folder was never retained is read from its deterministic re-run by src/rerun_missing.py
(results/csfrbench/density-rerun/<unit>; source "rerun") when tables/rerun_check.csv marks that re-run as reproducing its record
bit for bit; otherwise the row is counted in tables/coverage.csv and left out. Nothing is imputed.

Connection masks: the buffers hidden_layers.<k>.event_mask and output_layer.event_mask (0/1 per edge, shape out x in). The
transitions in order: t1 = input -> hidden 1, t2 .. t(K-1) = hidden -> hidden, tK = hidden K-1 -> output (the readout); K = 5 on
the 32x4 backbone, 9 on the 64x8 backbone of the depth-5 trees. Density of a transition = mean of its mask; overall density =
open edges / all edges over every transition; the group "hidden" pools the hidden -> hidden transitions the same way.

Tables (tables/): per_run.csv (one row per run), summary.csv (per setting and workload: the mean over target instances of the
per-instance mean over sizes with the 95% Student-t interval over instances), by_size.csv (per setting, workload and n), and
size_trend.csv (per setting and workload: the per-instance least-squares slope of the overall end density on log2 n, its mean
and 95% t interval over instances, and Spearman's rho of the end density against n over every run). The Student-t interval:
mean +- t_{0.975, N-1} sd / sqrt(N) over the N instances.

    OMP_NUM_THREADS=1 python src/density.py
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

torch.set_num_threads(1)

REPO = Path(__file__).resolve().parents[5]
STUDY = REPO / "studies/main-experiments/18-main-d8"
TABLES = STUDY / "density/tables"
LEDGERS = {"d8": STUDY / "ledger/d8/winners.csv", "d32s": STUDY / "ledger/d32s/winners.csv"}
ARM = "lewi-v1-rgm"
WORKLOAD = {"tree8": "depth-4 tree", "dag8": "depth-3 reuse DAG", "ftree8": "depth-4 Fourier tree", "flat8": "flat modules",
            "dense8": "high-arity node", "fdag8": "depth-3 Fourier reuse DAG", "rand8": "sampled depth-4 DAGs",
            "tree5d16": "depth-5 tree (d = 16)",
            "tree32s": "depth-4 tree", "dag32s": "depth-3 reuse DAG", "ftree32s": "depth-4 Fourier tree", "flat32s": "flat modules",
            "dense32s": "high-arity node", "fdag32s": "depth-3 Fourier reuse DAG", "rand32s": "sampled depth-4 DAGs",
            "tree5d32": "depth-5 tree (d = 32)"}
ORDER = list(WORKLOAD)
# where an archived run folder lives: (prefix of the ledger's checkpoint path, the tar of its run root, the member prefix in the tar)
ARCHIVES = (("results/csfrbench/m17/runs/", lambda root: REPO / f"results/csfrbench/m17/archive/{root}_runs.tar", ""),
            ("studies/main-experiments/18-main-d8/input-cutoff/runs/", lambda root: STUDY / "input-cutoff/archive/runs.tar", "runs/"))
RERUN = REPO / "results/csfrbench/density-rerun"                 # the never-retained winners re-run by src/rerun_missing.py
RERUN_CHECK = TABLES / "rerun_check.csv"                         # its bit-for-bit check; a re-run is read only when it passed
FILES = ("init.pt", "best.pt")
MAX_T = 9
_tars: dict[Path, tuple[tarfile.TarFile, set[str]]] = {}


def locate(path: str):
    """The run folder's files as (source, reader): the live folder, else the archive tar that holds both files, else None."""
    live = REPO / path
    if all((live / f).is_file() for f in FILES):
        return "ledger checkpoint", lambda f: (live / f).read_bytes()
    for prefix, tar_of, member_prefix in ARCHIVES:
        if not path.startswith(prefix):
            continue
        rest = path[len(prefix):]
        tar_path = tar_of(rest.split("/")[0])
        if not tar_path.is_file():
            continue
        if tar_path not in _tars:
            tf = tarfile.open(tar_path)
            _tars[tar_path] = (tf, set(tf.getnames()))
        tf, names = _tars[tar_path]
        member = member_prefix + rest
        if all(f"{member}/{f}" in names for f in FILES):
            return f"archive:{tar_path.relative_to(REPO)}", lambda f, tf=tf, member=member: tf.extractfile(f"{member}/{f}").read()
    return None


def rerun_passed() -> set[str]:
    """The re-run units whose check reproduced the record bit for bit (empty until the check has run)."""
    if not RERUN_CHECK.is_file():
        return set()
    chk = pd.read_csv(RERUN_CHECK)
    return set(chk.unit[chk["pass"].astype(str) == "True"])


def locate_rerun(unit: str, passed: set[str]):
    d = RERUN / unit
    if unit in passed and all((d / f).is_file() for f in FILES):
        return "rerun", lambda f: (d / f).read_bytes()
    return None


def masks(blob: bytes) -> list[np.ndarray]:
    """The connection masks of a state dict in transition order (input -> hidden 1 first, the readout last)."""
    sd = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    keys = [k for k in sd if k.endswith(".event_mask")]
    keys.sort(key=lambda k: 10 ** 6 if k.startswith("output_layer") else int(k.split(".")[1]))
    return [sd[k].numpy() for k in keys]


def densities(ms: list[np.ndarray], tag: str) -> dict:
    open_edges = np.array([m.sum() for m in ms], dtype=float)
    edges = np.array([m.size for m in ms], dtype=float)
    hidden = slice(1, len(ms) - 1)
    out = {f"density_{tag}_overall": open_edges.sum() / edges.sum(),
           f"density_{tag}_input": open_edges[0] / edges[0],
           f"density_{tag}_hidden": open_edges[hidden].sum() / edges[hidden].sum(),
           f"density_{tag}_readout": open_edges[-1] / edges[-1]}
    out.update({f"density_{tag}_t{k + 1}": (open_edges[k] / edges[k] if k < len(ms) else np.nan) for k in range(MAX_T)})
    return out


def per_run() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, coverage = [], []
    passed = rerun_passed()
    for setting, ledger in LEDGERS.items():
        led = pd.read_csv(ledger)
        led = led[led.arm == ARM]
        for cell in [c for c in ORDER if c in set(led.cell)]:
            g = led[led.cell == cell]
            found = {"ledger checkpoint": 0, "archive": 0, "rerun": 0}
            for r in g.itertuples():
                hit = locate(r.checkpoint) if isinstance(r.checkpoint, str) else locate_rerun(r.unit.split("/", 1)[1], passed)
                if hit is None:
                    continue
                source, read = hit
                found[source.split(":")[0]] += 1
                init, end = (masks(read(f)) for f in FILES)
                assert [m.shape for m in init] == [m.shape for m in end], r.unit
                row = dict(setting=setting, cell=cell, workload=WORKLOAD[cell], function_seed=int(r.function_seed), n=int(r.n),
                           unit=r.unit, checkpoint=r.checkpoint if isinstance(r.checkpoint, str) else "",
                           source=source, transitions=len(end), edges=int(sum(m.size for m in end)))
                row.update(densities(init, "init"))
                row.update(densities(end, "end"))
                row.update({f"n_edges_t{k + 1}": (int(end[k].size) if k < len(end) else np.nan) for k in range(MAX_T)})
                rows.append(row)
            covered = [x for x in rows if x["setting"] == setting and x["cell"] == cell]
            coverage.append(dict(setting=setting, cell=cell, workload=WORKLOAD[cell], ledger_rows=len(g),
                                 no_checkpoint_path=int(g.checkpoint.isna().sum()),
                                 path_not_found=int(g.checkpoint.notna().sum()) - found["ledger checkpoint"] - found["archive"],
                                 ledger_checkpoint=found["ledger checkpoint"], archived=found["archive"], rerun=found["rerun"],
                                 rerun_missing=int(g.checkpoint.isna().sum()) - found["rerun"], covered=len(covered),
                                 instances_ledger=int(g.function_seed.nunique()), instances_covered=len({x["function_seed"] for x in covered})))
            print(f"[density] {setting} {cell}: {len(covered)} of {len(g)} runs ({found['ledger checkpoint']} ledger checkpoints, "
                  f"{found['archive']} archived, {found['rerun']} re-run)")
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def t_interval(x) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    if len(x) < 2:
        return m, np.nan, np.nan
    h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return m, m - h, m + h


def interval_columns(values: pd.DataFrame, columns: list[str]) -> dict:
    """mean, lo, hi over the rows (one per instance) of every column that has a value."""
    out = {}
    for c in columns:
        v = values[c].dropna()
        if len(v):
            out[f"{c}_mean"], out[f"{c}_lo"], out[f"{c}_hi"] = t_interval(v)
    return out


def summarize(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quantities = [c for c in runs.columns if c.startswith("density_")]
    summary, by_size, trend = [], [], []
    for (setting, cell), g in runs.groupby(["setting", "cell"], sort=False):
        per_instance = g.groupby("function_seed")[quantities].mean()
        summary.append(dict(setting=setting, cell=cell, workload=WORKLOAD[cell], N=len(per_instance), runs=len(g),
                            sizes_per_instance_min=int(g.groupby("function_seed").size().min()),
                            sizes_per_instance_max=int(g.groupby("function_seed").size().max()),
                            **interval_columns(per_instance, quantities)))
        for n, h in g.groupby("n"):
            by_size.append(dict(setting=setting, cell=cell, workload=WORKLOAD[cell], n=int(n), N=len(h),
                                **interval_columns(h, ["density_end_overall", "density_init_overall"])))
        slopes = []
        for _, h in g.groupby("function_seed"):
            if h.n.nunique() >= 2:
                slopes.append(np.polyfit(np.log2(h.n.to_numpy(float)), h.density_end_overall.to_numpy(float), 1)[0])
        rho = stats.spearmanr(g.n, g.density_end_overall)
        m, lo, hi = t_interval(slopes)
        trend.append(dict(setting=setting, cell=cell, workload=WORKLOAD[cell], N=len(slopes), runs=len(g),
                          slope_per_doubling_mean=m, slope_per_doubling_lo=lo, slope_per_doubling_hi=hi,
                          spearman_rho=float(rho.statistic), spearman_p=float(rho.pvalue)))
    return pd.DataFrame(summary), pd.DataFrame(by_size), pd.DataFrame(trend)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    runs, coverage = per_run()
    runs.to_csv(TABLES / "per_run.csv", index=False)
    coverage.to_csv(TABLES / "coverage.csv", index=False)
    summary, by_size, trend = summarize(runs)
    summary.to_csv(TABLES / "summary.csv", index=False)
    by_size.to_csv(TABLES / "by_size.csv", index=False)
    trend.to_csv(TABLES / "size_trend.csv", index=False)
    for s in summary.itertuples():
        print(f"[density] {s.setting:4s} {s.workload:28s} N = {s.N:2d} runs = {s.runs:3d}  init {s.density_init_overall_mean:.3f}  "
              f"end {s.density_end_overall_mean:.3f} [{s.density_end_overall_lo:.3f}, {s.density_end_overall_hi:.3f}]  "
              f"input {s.density_end_input_mean:.3f} hidden {s.density_end_hidden_mean:.3f} readout {s.density_end_readout_mean:.3f}")
    for t in trend.itertuples():
        print(f"[trend] {t.setting:4s} {t.workload:28s} slope per doubling {t.slope_per_doubling_mean:+.4f} "
              f"[{t.slope_per_doubling_lo:+.4f}, {t.slope_per_doubling_hi:+.4f}]  Spearman rho {t.spearman_rho:+.2f} (p = {t.spearman_p:.2g})")
    print(f"[density] {len(runs)} runs of {int(coverage.ledger_rows.sum())} ledger rows covered")


if __name__ == "__main__":
    main()
