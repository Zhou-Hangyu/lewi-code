# Research code: Learning Computational Structure for Sample-Efficient Prediction

Hangyu Zhou, Constantine Dovrolis, Stephen Mussmann. Paper: <https://arxiv.org/abs/2610.28555>. Project page:
<https://zhou-hangyu.github.io/lewi/>.

This branch holds the research code, final configurations and compact result records as they produced the paper:
the minimum set needed to understand LeWi (learned wiring), regenerate the CSFRBench targets, run the reported methods
with their final configurations, run the world-model experiments, apply the model-selection and evaluation rules, and
regenerate the reported statistics, tables and figures. For new work, use the `main` branch: the same benchmark and
records as a clean package (`lewi` and `csfrbench`) with a short tutorial, faster training of LeWi and Dense, and a
check of an installation against the records.

Generative AI was used in this project for research feedback, interpretation, derivations, literature work and writing
assistance, and an AI coding agent wrote and modified experiment, analysis and verification code; the authors reviewed
the work and take responsibility for it. The manuscript's AI use statement gives the full disclosure.

`REPRODUCE.md` maps every experimental figure and table of the manuscript to its command, configuration, record and
expected output.

## Differences from the arXiv version

This package was assembled for the submitted draft of the paper. Three items of the arXiv version
([2610.28555v1](https://arxiv.org/abs/2610.28555v1)) differ from it:

* Table 6 and Appendix D.7 (computational cost). The arXiv version times optimized implementations of both methods:
  networks trained in stacks, LeWi with fused C++ gate kernels, Dense compiled, both run for the whole budget side by
  side on one node (seconds per trained network: 32x4 Dense 5.2, LeWi 6.8; 64x8 Dense 84.3, LeWi 85.6). The `main`
  branch holds this implementation (`csfrbench/stack.py`, `csfrbench/kernels.py`) and the timing protocol
  (`tools/timing.py`). Section G of `scripts/verify_results.py` checks the earlier timing of one network at a time
  (`18-main-d8/overhead-fast/tables/overhead_fast.csv`).
* Learned-embedding figures (Figure 19 and the appendix embedding figures). A hidden unit takes the colour of the
  target intermediate with which its activation has the largest absolute correlation above a permutation floor. The
  builders here also required the unit to read every input of that intermediate through active connections; the arXiv
  figures drop that condition. The `main` branch implements the rule of the arXiv version
  (`csfrbench/structure.py`, `examples/plot_embeddings.py`).
* The depth-5 tree: the arXiv text gives 32% lower mean AULC than Dense (the records: 31.5%); `verify_results.py`
  prints the draft's 31%.

## Package structure

| folder | contents |
|---|---|
| `src/csfrbench/` | the benchmark package: `tasks/` (graph templates, the depth-4 graph sampler `rand.py`, local-function families `laws.py`, target generation and sequential normalization `graph.py`, `base.py`), `methods/` (LeWi `lewi_v1.py` with `cmask.py`, `lewi_v05.py`, `nns.py`; Dense `noreg.py`; the baselines `l1.py`, `l2.py`, `l0hc.py`, `rigl.py`, `bimt.py`, `fiedler.py`, `erb_positions.py`; GraphOracle `graph_oracle.py`, `maskladder.py`), `train.py` (the trainer), `datasets.py` (data generation and splits), `metrics.py` (NMSE, AULC, linear probes), `das_core.py` / `das_probe.py` / `das_teacher_cf.py` (Boundless-DAS counterfactual interventions), `configs.py` + `configs/` (run-specification schema and defaults), `pipeline.py` (campaign composition, dataset materialization, scrambled-Sobol trial draws) |
| `studies/main-experiments/17-main-clean/src/` | the campaign definitions `m17.py` (settings, arms, backbones, budgets, grids, cohorts, panels; the composition of one unit), `m17_tasks.py` (the study's task registrations), `m17_analysis.py` (the winner rule as run), `m17_posthoc.py` (linear probes and counterfactual interventions on the selected runs), the LeWi study arms (`abl_methods.py` mechanism ablations, `abl_in5_methods.py`, `lms_methods.py` mask-only forward and embedding dimension); `manifests/search_spaces.json` (every frozen hyperparameter range), the two plan files; `results/units_main8b.parquet` (every trial of the primary d = 8 campaign) |
| `studies/main-experiments/12-extra/src/w5_methods.py`, `18-main-d8/input-cutoff/src/cutoff_methods.py` | the paper's LeWi arm: the readout gate and mask (`lewi-v1-rgm`) and the input rule (`lewi-v1-rgm-in5`) |
| `studies/main-experiments/18-main-d8/` | the paper's study: `ledger/` (the selected run of every setting, arm, instance and size), `analysis/` (AULC aggregates, paired contrasts, the boundary audit), `src/` (ledger and contract builders, mechanism-ablation analysis), `structure/` (wiring alignment: extraction driver and its table), `density/` (connection density), `overhead-fast/` (the optimized LeWi implementation and the cost record of Table 6), `input-cutoff/tables/` (the input rule's re-run tables), `swm-in5/`, `swm-rollout/`, `swm-lrx/`, `swm-reconcile/` (the three adopted world-model layers and the counted task set) |
| `studies/csfrbench/evaluation-protocol/structural-metrics/structure-discovery/` | module-graph extraction and scoring used by the wiring-alignment driver and by Figure 17 |
| `studies/main-experiments/09-swm-lewi/src/`, `14-main/src/`, `16-main-final/` | world models: task corpora and collectors (`swm_tasks.py`), the engine (`lewibench.py`), the arm registry (`v05_port.py`), the fixed hyperparameters (`swm_v10.py`), the runner (`swm16.py`: search, selection, finals), the record (`external/analysis/instances.csv`, `contract.json`) |
| `studies/communication/paper/figures/` | `build.py` and `src/` (one builder per figure or table), `data-paper-d8/`, `data-paper-d32/` (the plotting records), `design/` (the learned-embedding figures), `out-paper/` (three table records; every rebuilt output is written here) |
| `scripts/` | `smoke_test.sh`, `train_unit.py`, `select_and_aggregate.py`, `verify_results.py` (+ `paper_claims.py`), `rebuild_figures.sh`, `compare_outputs.py`, `world_model_smoke.sh` |

The folders keep the relative layout the code was developed in, because the scripts compute their paths from their own
location. Folder names such as `17-main-clean` and `18-main-d8` are internal study names without further meaning.

## Environment and setup

Python 3.12 on Linux. Two environments:

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .          # makes csfrbench importable
```

The world-model experiments (Appendix E) need the simulator stack of `requirements-worldmodel.txt` in a second
environment (`pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu`, then the file). No
credentials, no private storage and no network access after the install are needed. Every training run uses one CPU
thread (`OMP_NUM_THREADS=1`), as the reported runs did.

## Smoke test (a few minutes on one core)

```bash
bash scripts/smoke_test.sh
```

Imports the package, trains one LeWi unit (`lewi-v1-rgm-in5`) and one Dense unit of the depth-4 tree (instance 6000,
n = 512) at a reduced budget of 300 steps, runs the quick record checks (`scripts/verify_results.py --quick`) and
rebuilds Figure 4 from the packaged records. `scripts/world_model_smoke.sh` (second environment) collects a tiny
Pendulum corpus, trains LeWi and Dense for 200 steps and prints the command that re-runs one recorded unit.

## A CSFRBench experiment

One unit = one setting, arm, target instance, training-set size and hyperparameter trial:

```bash
python scripts/train_unit.py --panel main8b --cell tree8 --arm lewi-v1-rgm --method lewi-v1-rgm-in5 --fs 6000 --n 4096 --trial 0
python scripts/train_unit.py --panel main8b --cell dag8 --arm noreg --fs 6003 --n 32768 --trial 17
```

The panel fixes backbone, budget and grid (`main8b`: 32x4, 64,000 steps, n = 2^9..2^15, instances 6000-6015, 32 trials
per cell; `main`: the depth-5 tree at d = 16, 64x8, 128,000 steps, n = 2^11..2^17; `main32s` / `main32t`: the
nuisance-input setting at d = 32; `rand8b`: the sampled depth-4 DAGs, instances 6020-6035 with 6000-6015;
`abl8-in5`, `abl8rad-in5`, `abl8init-in5`, `abl8seed-in5`: the mechanism ablations and the sensitivity arms; `lms8`,
`lms5`: mask-only forward and embedding dimension). Cells: `tree8` depth-4 tree, `dag8` reuse DAG, `flat8` flat modules,
`dense8` high-arity node (additive-multiplicative), `ftree8` Fourier tree, `fdag8` Fourier reuse DAG, `rand8` sampled
depth-4 DAGs, `tree16` depth-5 tree (d = 16); the d = 32 cells are `tree`, `dag`, `flat`, `dense`, `ftree`, `fdag`,
`rand`, `tree5x32`. Arms: `noreg` Dense, `l1`, `l2`, `l0hc` (L0), `rigl`, `bimt`, `fiedler`, `erb_positions` (neuron
positions), `lewi-v1-rgm` LeWi, `go1` / `go2` / `graph_oracle` GraphOracle(1) / (2) / (all), `support_oracle`
GraphOracle(0). The trial index selects the scrambled-Sobol draw from the arm's frozen range
(`17-main-clean/manifests/search_spaces.json`; Table 2). `--steps` caps the budget for smoke tests only.

A full campaign is the product of these axes (`m17.plan(panel)` lists the units); `main8b` alone is 96,768 units of
2-4 minutes each on one core. The record's per-trial results are in `17-main-clean/results/units_main8b.parquet` and
the selected runs of every setting in `18-main-d8/ledger/{d8,d32s}/winners.csv`.

## Model selection and evaluation

```bash
python scripts/select_and_aggregate.py        # the winner rule and the AULC aggregates recomputed from the records
python scripts/verify_results.py              # every reported number recomputed and compared with the manuscript
```

The rule: per (setting, arm, instance, size), the trial with the lowest validation NMSE among 32 complete,
non-divergent trials; the LeWi cells changed by the input rule take their winner from the rule's 32-trial re-runs
(`18-main-d8/input-cutoff/tables/`). AULC integrates test NMSE over log2 n on the setting's grid; means carry 95%
Student-t intervals over instances; paired contrasts are LeWi minus the reference per instance.

## A world-model experiment

```bash
bash scripts/world_model_smoke.sh                                                           # second environment
python studies/main-experiments/09-swm-lewi/src/swm_tasks.py --task pendulum --eps 8192 --horizon 128 --name pendulum_xl
python studies/main-experiments/16-main-final/src/swm16.py one lewi-v1                      # one pending unit of that arm
```

`swm16.py` holds the protocol: the 17 counted tasks (`18-main-d8/swm-reconcile/counted_set.csv`; the excluded tasks in
`excluded_tasks.csv`), the grids, the 256x4 backbone at 128,000 steps, the search (`spaces_final_v2.json`, and the
re-searched Dense and L1 learning rates in `swm-lrx/manifests/spaces_lrx.json`), validation-only selection and 16
paired final seeds. The three adopted layers on top of the record: the input rule (`swm-in5/`), the validation-only
re-selection of flagged cells (`swm-rollout/`) and the learning-rate re-search (`swm-lrx/`). A full campaign is about
34,000 search runs and 2,448 finals of 5-40 minutes each.

## Regenerating figures and tables from the records

```bash
bash scripts/rebuild_figures.sh                 # everything in REPRODUCE.md, a few minutes
bash scripts/rebuild_figures.sh fig6            # one stem
```

Outputs go to `studies/communication/paper/figures/out-paper/{d8,d32}`. The plotting records under `data-paper-d8/`
and `data-paper-d32/` are the harvested per-run values (learning curves, probe and counterfactual R^2 per run,
paired contrasts, the world-model cells); `verify_results.py` recomputes them from the ledgers and the raw tables.

## Computational cost

Cheap (minutes, one core): the smoke test, a single unit at d = 8, every figure and table, `verify_results.py`.
Expensive: any campaign (`m17.plan`), the world-model search and finals, the probe and counterfactual analyses of every
selected run (`m17_posthoc.py`, hours per setting), the wiring-alignment extraction (`18-main-d8/structure/src/`) and
the connection-density pass (`density/src/density.py`), which read the selected runs' checkpoints; checkpoints are not
part of this archive, `train_unit.py` regenerates any of them (the training seeds derive from the unit).

## Notes

Table 6 measures the optimized implementation `18-main-d8/overhead-fast/src/fast_lewi.py` (exact: bit-identical
training to the campaign implementation); its record `overhead-fast/tables/overhead_fast.csv` also holds the
compiled and GPU rows the paper does not report. The ledgers' label column says "random graphs (depth 4)" for the
setting the paper calls sampled depth-4 DAGs. Identifiers `cmask*` are internal method names of the LeWi lineage.
