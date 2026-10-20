# Reproducing the paper

Every figure and table of *Learning Computational Structure for Sample-Efficient Prediction*: how to regenerate it,
from which records, and at what cost. Commands run from the repository's root after `pip install -e ".[plot]"`.

## 1. Check the installation

```bash
csfrbench verify                     # 4 minutes: targets, data, recorded states, first optimizer steps, tables of the paper
csfrbench verify --level standard    # 20 minutes: every recorded target and state, a run of every method in both settings
```

In the records' environment (Linux x86-64, Python 3.12, `pip install -r requirements-reference.txt`) every check is
`exact`: a run of the per-network trainer equals its record bit for bit. Elsewhere some checks are `close`, which is
expected (floating-point results differ in the last digits between processors and library versions).

## 2. The records

`csfrbench/reference/` holds the outcome of every training run of the paper's CSFRBench campaigns: 354,816 runs in the
primary setting and 387,072 with nuisance inputs (validation and test NMSE, selected step, divergence), the 32
configurations of every condition, the selected run of every condition, the AULC per target instance and the tables,
and a sample of runs with their model states. `csfrbench.record` reads them; every table and curve below can be
recomputed from them without training.

## 3. Training engines

The paper's runs were made one network at a time. This package trains LeWi and Dense in stacks of many networks with
LeWi's fused C++ kernels (`csfrbench/stack.py`, `csfrbench/kernels.py`); every other method trains one network at a
time. A stacked network has the per-network trainer's initialization, minibatch order, learning rate, schedule,
validation, selection and test evaluation; only the floating-point order of the batched kernels differs. Checks
against the per-network trainer (`tests/test_stack.py`): forward pass and gradients to $10^{-8}$, one Adam step to one
float32 unit, the minibatch order identical, exported states loadable by the method's own model; a 2,000-step
trajectory at a stable learning rate agrees to three decimals.

Cost on one CPU core, per network step: LeWi 62 µs in a stack of 128 (C++ kernels, compiled), Dense 44 µs, against
about 4 ms for one LeWi network trained alone, as in the paper's campaign.

Validation against the records (`tools/validate_stack.py`): 21,952 runs of LeWi and Dense with the stacked engine, on
the depth-4 tree and the reuse DAG of the primary setting and the depth-4 tree with nuisance inputs (all 16 target
instances, every size and all 32 configurations), and the depth-5 tree on instance 6000 (64x8). Each run is compared
with the recorded run of the same configuration and seeds.

| | LeWi | Dense |
|---|---|---|
| single runs: correlation of test NMSE with the record (clipped at 1.5) | 0.96 to 0.99 | 0.87 to 0.99 |
| single runs: median absolute difference of test NMSE | 0.0001 to 0.004 | 0.0000 to 0.008 |
| selected runs: same configuration as in the record | 36% to 54% | 29% to 46% |
| AULC, re-run minus record, depth-4 tree | -0.005 [-0.015, +0.005] | +0.000 [-0.004, +0.005] |
| AULC, re-run minus record, reuse DAG | +0.004 [-0.004, +0.012] | -0.000 [-0.004, +0.004] |
| AULC, re-run minus record, depth-4 tree with nuisance inputs | -0.003 [-0.017, +0.011] | see below |
| AULC, re-run minus record, depth-5 tree (one instance) | +0.001 | -0.001 |

Intervals are 95% Student-t intervals over the 16 instances. For scale, the paper's LeWi - Dense differences are
-0.125 on the depth-4 tree and -0.092 on the reuse DAG. Many conditions have several configurations within a small
validation difference, so the selected configuration often changes with the last digits; the selected test NMSE does
not change on average.

Dense with nuisance inputs: at small training sets on some instances no configuration learns (validation NMSE about 1),
and the selection is then a near-tie between failed runs. In the record, one such selected run (instance 6005,
n = 2,048) has a test NMSE of 2.9e8, which puts the recorded mean AULC at 3.0e6 (median 0.89). The stack reproduces that
run (test NMSE 2.9e8) but selects another failed run, by a validation difference of 0.0002. Over the other 15
instances, re-run minus record is +0.046 [-0.022, +0.114] (median +0.003), from the same kind of near-ties on
instances 6012 and 6013.

Cost in this validation (8 single-thread processes per job on shared cluster nodes): 5 to 10 s per run at 32x4 and about
95 s at 64x8, against about 4.3 minutes (LeWi) and 1.8 minutes (Dense) at 32x4 and 16 and 5 minutes at 64x8 for one
network trained alone; 80 core-hours in all.

`--engine exact` trains every unit alone with the per-network trainer (bit for bit the record in the records'
environment). `tools/validate_stack.py` repeats the comparison above.

## 4. Map of the paper

| Item | How | Records or runs | Compute |
|---|---|---|---|
| Table 1 (settings) | `csfrbench info` | `csfrbench/suite.py` | seconds |
| Table 2 (search spaces) | `csfrbench info --spaces` | `csfrbench/config/spaces.json` | seconds |
| Table 3 (mean AULC) | `csfrbench record --setting primary` | `record.aulc` | seconds |
| Table 4 (depth-5 tree, sampled DAGs) | `csfrbench record --setting primary` (those columns), `docs/RESULTS.md` | `record.aulc`, `record.contrasts` | seconds |
| Figure 3 (computation graphs) | `python examples/plot_targets.py graphs` | the target generator | seconds |
| Figure 4, 11 (learning curves) | `python examples/plot_results.py curves` | `record.winners` | seconds |
| Figure 5, 13, 14 (probes, interventions) | `python examples/representations.py --das` | re-trained selected runs (states kept) | probes minutes; DAS about 15 minutes per network |
| Figure 6 (a) (nuisance inputs) | `python examples/plot_results.py inputs` | `record.contrasts` of both settings | seconds |
| Figure 8 (sampled graphs) | `python examples/plot_targets.py sampled` | the graph sampler | seconds |
| Figure 10 (GraphOracle masks) | `python examples/plot_targets.py masks` | `csfrbench/methods/graph_oracle.py` | seconds |
| Figure 12, 16 (paired comparisons) | `python examples/plot_results.py contrasts --setting primary` (and `nuisance`) | `record.contrasts` | seconds |
| Figure 15 (curves with nuisance inputs) | `python examples/plot_results.py curves --setting nuisance` | `record.winners` | seconds |
| Figure 19 (learned embeddings) | `python examples/plot_embeddings.py --all` | the seven recorded LeWi runs, re-trained | 4 minutes each |
| Figure 20 (connection density) | `density_*` in the record of every LeWi run (`csfrbench reproduce` with `--arms lewi`) | re-trained selected runs | see section 5 |
| Figure 21 (embedding dimension and radius) | LeWi's `code_dim` and `radius` hyperparameters in a `csfrbench run` unit | new runs | as a campaign part |
| Table 6 (computational cost) | `python tools/timing.py --backbone 32x4` and `--backbone 64x8` | new runs, one node | about 20 minutes (32x4), 1 hour (64x8) |
| Section 6 numbers (LeWi below Dense by 20%, 34%, 27%, 32%; the 51% and 26% gap shares) | `csfrbench verify` (the paper's values are among its checks) | `record.aulc`, `record.paper_tables` | seconds |

Figure 1 shows a LeWi network trained for the figure (the depth-4 Fourier tree with nuisance inputs, n = 8,192, on a
target instance outside the benchmark's); `examples/plot_embeddings.py` draws the recorded networks of Figure 19.
Figure 2 is a drawing.

The studies that are not part of the benchmark package (the wiring alignment of Table 5 and Figures 6 (b), 17 and 18,
the mechanism ablations of Figure 6 (c), the initialization-scale sensitivity of Figure 21, the local-function fitting
study of Figure 9 and the world-model experiments of Section 6 and Appendix E, Figures 7, 22, 23 and Table 7) are on the
[`paper-code`](https://github.com/Zhou-Hangyu/lewi-code/tree/paper-code) branch: the research code and records exactly as
they produced the paper, with its own reproduction map.

## 5. Campaigns from scratch

```bash
csfrbench plan --setting primary --workloads depth4_tree --out units.jsonl     # every method, instance, size and trial
csfrbench run --units units.jsonl --out runs/tree --workers 24                 # resumable; --shard i/k on a cluster
csfrbench analyze --runs runs/tree --out tables/tree                           # selection, AULC, contrasts
```

| Part | Runs | Compute (one core) |
|---|---:|---|
| one workload, LeWi and Dense, 32x4 | 7,168 | about 20 core-hours (stacked; about 370 trained one at a time) |
| the depth-4 tree, all 11 methods, 32x4 | 39,424 | about 1,800 core-hours (the other methods train one at a time) |
| the depth-5 tree, LeWi and Dense, 64x8 | 7,168 | about 250 core-hours (stacked; about 1,300 one at a time) |
| the primary setting | 354,816 | about 16,000 core-hours (about 20,000 with every network trained alone) |

`examples/slurm_job.sbatch` runs a campaign as an array of cluster jobs with one shard each.
