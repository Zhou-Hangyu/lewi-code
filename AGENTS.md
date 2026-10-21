# Guide for coding agents

This file is for an AI coding agent that installs or uses LeWi and CSFRBench for a person. Every command is relative
to the folder of this file.

## 1. What this repository is

* `lewi/`: LeWi (Learned Wiring), a neural-network layer whose connections open and close with learned neuron
  positions. Use it in any PyTorch model (`lewi/README.md`).
* `csfrbench/`: CSFRBench, a benchmark for compositionally sparse function regression: a target is a composition of
  small local functions on a known computation graph; a learner sees examples only; the benchmark scores the test error
  against the training-set size (AULC, lower is better) and keeps the graph for graph-informed references and
  structural diagnostics (`csfrbench/README.md`).

The package holds the records of the paper's 741,888 training runs, so an installation can be verified against them.
It needs NumPy, SciPy, PyTorch and ninja; no graphics processor. A C++ compiler makes LeWi's training faster.

## 2. Install and verify

Requirements: Linux, macOS, or Windows with the Windows Subsystem for Linux; Python 3.10 or later; a connection to the
package index; 1.5 GB of disk space. On a shared cluster, do this work on a compute node.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
csfrbench verify
```

For the library versions of the paper's records (bit-for-bit runs on Linux x86-64), install them first:
`pip install -r requirements-reference.txt` (Python 3.12).

The installation is correct when the output ends with a `result:` line and a `VERIFIED` line:

```
result: 33 exact
VERIFIED: every check agrees with the record bit for bit
```

| Output | Meaning | Action |
|---|---|---|
| `VERIFIED: every check agrees with the record bit for bit` | every result equals the record | none |
| `VERIFIED: ... (some within tolerance ...)` | some results differ in the last digits; usual with other library versions, processors or operating systems | none |
| `VERIFICATION FAILED` | a result does not agree with the record | stop; report the lines that start with `FAIL` and the line that starts with `environment:` |

## 3. More tests

Set one thread per run: `export OMP_NUM_THREADS=1`.

| Test | Command | Time on one core |
|---|---|---|
| quick verification | `csfrbench verify` | 4 minutes |
| standard verification | `csfrbench verify --level standard` | 20 minutes |
| full verification | `csfrbench verify --level full` | 1 hour |
| test suite | `pip install -e ".[test]"`, then `python -m pytest -q` | 12 minutes |
| the five-minute tutorial | `python examples/quickstart.py` | under a minute |
| recorded results run again | `csfrbench reproduce --workload reuse_dag --arms lewi dense --instances 6003 --sizes 2048 --out runs/check --workers 2` | 2 minutes |

`reproduce` trains LeWi and Dense with the stacked engine: runs agree with their records statistically (the report gives
the AULC difference with its interval), not digit by digit. `--engine exact` trains each run alone; in the records'
environment such a run equals its record (`exact`). Only `FAIL`, and `different` under `--engine exact`, are failures.

## 4. Rules

1. Do not change the files of `csfrbench/` (its `config` and `reference` folders included): the identifiers of the
   target families, the names of the random streams and the constants of the normalization fix the targets and the
   runs. A change gives another benchmark.
2. One thread per training run (`OMP_NUM_THREADS=1`); `--workers` for parallel runs, one per core.
3. A method must not read the graph or the active inputs of a target. Only the graph-informed references (`go0`,
   `go1`, `go2`, `go_all`) do. With the quick interface, never pass `task.reveal()` to the learner.
4. Instances 6000-6015 (and 6020-6035 of `sampled_dags`) are for evaluation only. Develop a method and its search space
   on the development instances 6016-6019.
5. For a result comparable with the paper, use the protocol: 32 configurations from a frozen search space per method,
   instance and size; selection on the validation pool; one evaluation on the test pool. Results of the quick interface
   (`csfrbench.task`, `learning_curve`) are exploratory: say so when you report them.
6. Name the setting with every result (`primary`: active inputs only; `nuisance`: inert inputs added). Compare AULC
   values within one workload only.
7. Tell the person the number of runs and the core hours before you start a campaign.

## 5. Usual tasks

```bash
csfrbench info                                   # settings, workloads, methods
csfrbench info --spaces                          # the frozen search spaces
csfrbench record --setting primary               # the recorded AULC table
csfrbench target --workload depth4_tree --instance 6000
csfrbench unit --workload depth4_tree --arm lewi --instance 6000 --n 512 --trial 12
```

A workload with two methods (7,168 runs; LeWi and Dense train in stacks, about 20 core-hours):

```bash
csfrbench plan --setting primary --workloads depth4_tree --arms lewi dense --out units.jsonl
csfrbench run --units units.jsonl --out runs/tree --workers 8
csfrbench analyze --runs runs/tree --out tables/tree
```

`run` continues an interrupted campaign. For a cluster, see `examples/slurm_job.sbatch`. A new method:
`examples/new_method.py` (`python examples/new_method.py smoke` tests the pipeline in minutes), then
`csfrbench analyze --runs runs/my_method --with-record`.

## 6. Facts

| Item | Value |
|---|---|
| Settings | `primary`: $d = 8$ (16 on `depth5_tree`); `nuisance`: $d = 32$ |
| Workloads | `flat_modules`, `depth4_tree`, `reuse_dag`, `sampled_dags`, `fourier_tree`, `fourier_reuse_dag`, `high_arity_node`, `depth5_tree` |
| Methods | `lewi`, `dense`, `l1`, `l2`, `l0`, `rigl`, `bimt`, `fiedler`, `positions`, `go0`, `go1`, `go2`, `go_all` |
| Training-set sizes | 512 to 32,768 in powers of 2; 2,048 to 131,072 on `depth5_tree` |
| Network and budget | width 32, 4 hidden layers, 64,000 steps; width 64, 8 hidden layers, 128,000 steps on `depth5_tree` |
| One run on one core | 2 to 4 minutes alone (7 to 17 on `depth5_tree`); 5 to 10 s per network in a LeWi or Dense stack (about 95 s on `depth5_tree`) |
| Target instances | 16 per workload; 32 for `sampled_dags`; development instances 6016-6019 |
| Versions | package 1.6.1; benchmark definition 1.5 (`csfrbench.BENCHMARK_VERSION`) |

## 7. Documents

| Document | Content |
|---|---|
| `README.md` | overview, installation, quickstart, reproduction |
| `lewi/README.md` | the LeWi layer and its use |
| `csfrbench/README.md` | the benchmark: catalog, data, metrics, protocol, API |
| `docs/SPEC.md` | the complete definition of the benchmark |
| `docs/RESULTS.md` | the recorded results of every method |
| `docs/REPRODUCE.md` | every figure and table of the paper: command, records, compute |
| `CONTRIBUTING.md` | how to report a problem, add a method, and keep the benchmark definition fixed |

## 8. Problems

| Problem | Cause and solution |
|---|---|
| `the fused LeWi kernels could not be built` | no C++ compiler or no `ninja`; training still works, more slowly. Install a compiler (gcc, clang) for the fast path. |
| `python -c "import csfrbench; print(csfrbench.__file__)"` shows a folder outside this repository | another package of that name is on `PYTHONPATH`; remove it. |
| A training run is slow | set `OMP_NUM_THREADS=1`; use no more workers than cores. |
| `analyze` shows no learning curve | a curve needs a selected run at every size, and a selection all 32 configurations (`--trials N` for fewer). |
| `VERIFICATION FAILED` | do not repair the package; report to the person. |

## 9. Report to the person

After an installation: the folder and the command to activate it; the `environment:` line; the `result:` line and
the line after it; every check that is not `exact`; every step you did not do, with the reason.
