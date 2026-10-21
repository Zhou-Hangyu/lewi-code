# CSFRBench

CSFRBench is a controlled benchmark for **compositionally sparse function regression**. Each target is a composition
of small local functions on a known computation graph. Statistical theory says that a learner that exploits this
structure needs far fewer examples than one that does not; the benchmark measures whether a learning method gets that
benefit in practice.

The learner sees only input–output examples. The benchmark keeps the target's graph and its intermediate values for
two other uses:

* **graph-informed references**: networks wired like the target (GraphOracle) show how much sample efficiency the
  structure is worth;
* **structural evaluation**: linear probes, counterfactual interventions and unit associations measure whether a
  learner's hidden layers and wiring follow the target's computation.

The graph is hidden by protocol, not by access control: `Task.reveal()` works at any time. A learner that reads the
target's graph, its active inputs or its intermediate values while training breaks the protocol; only the
graph-informed references read them.

The complete definition is [docs/SPEC.md](../docs/SPEC.md); the recorded results of the paper's methods are
[docs/RESULTS.md](../docs/RESULTS.md).

## Quickstart

```python
import csfrbench

task = csfrbench.task("depth4_tree", instance=6016, n=1024)   # a development instance: what a learner may see
X, y = task.X_train, task.y_train                             # 1,024 noisy examples (float32 NumPy arrays)
X_val, y_val = task.X_val, task.y_val                         # 4,096 noisy examples, for model selection

predict = my_fit(X, y, X_val, y_val)                          # your learner (../README.md defines a small one)
nmse = task.evaluate(predict)                                 # test NMSE on 65,536 noise-free inputs

target = task.reveal()                                        # after training: the hidden computation
target.graph.node_parents                                     # the parents of every local function
target.intermediates(X_val)                                   # the value of every local function
```

A predictor is a callable on an `(m, d)` array, an object with `.predict`, or a `torch.nn.Module`.
`csfrbench.learning_curve(my_fit, "depth4_tree", instance=6000)` fits at every training-set size of the workload and
`csfrbench.compare([curve])` sets the AULC beside the paper's methods on the same instances. Tune on the development
instances 6016-6019; run the evaluation instances once, with the final learner.
[`examples/quickstart.py`](../examples/quickstart.py) runs the whole sequence in under a minute.

## Catalog

Two settings share the target instances, the methods and the protocol.

| Setting | Inputs | Measures |
|---|---|---|
| `primary` | every input is active: $d = 8$ ($d = 16$ on the depth-5 tree) | the benefit of compositional structure alone |
| `nuisance` | the same targets with inert inputs added: $d = 32$ (24 inert inputs; 16 on the depth-5 tree) | the same, with the selection of the active inputs |

Each setting has eight workloads (target families):

| Workload | Graph | Local functions | What it tests |
|---|---|---|---|
| `flat_modules` | four independent pairs, summed | additive-multiplicative | independent modules without hierarchy |
| `depth4_tree` | balanced binary tree, 7 functions, 3 levels | additive-multiplicative | hierarchical composition |
| `depth5_tree` | balanced binary tree, 15 functions, 4 levels, 16 inputs | additive-multiplicative | a deeper and wider hierarchy (64x8 backbone) |
| `reuse_dag` | two intermediates used twice, readout sums three | additive-multiplicative | intermediate reuse |
| `sampled_dags` | a random graph per instance (32 instances) | additive-multiplicative | structure that varies from target to target |
| `fourier_tree` | balanced binary tree | random Fourier | the tree with another function family |
| `fourier_reuse_dag` | reuse DAG | random Fourier | reuse with another function family |
| `high_arity_node` | one function of all 8 inputs | random Fourier | a target without compositional sparsity (a control) |

Every local function of a compositional workload reads two values. **Additive-multiplicative**:
$g(z) = \sqrt{(1-\rho)/k}\sum_i a_i(z_i) + \sqrt{\rho}\prod_i b_i(z_i)$ with sinusoidal factors and $\rho = 0.5$.
**Random Fourier**: $g(z) = Q^{-1/2}\sum_q a_q \sin(\omega_q^\top z + \phi_q)$. Every intermediate is normalized to
mean 0 and variance 1. The graph figures of all families are on the
[CSFRBench page](https://zhou-hangyu.github.io/lewi/csfrbench/).

Each workload has 16 target instances (6000-6015; 32 for the sampled DAGs: 6000-6015 and 6020-6035) and seven
training-set sizes, $n = 2^9, \dots, 2^{15}$ ($2^{11}, \dots, 2^{17}$ on the depth-5 tree). Instances 6016-6019 are for
method development only; the paper froze its search spaces on them. The evaluation instances give the numbers to
report and must not be used to choose a method's settings.

## Target generation

`suite.workload(setting, name).target(instance)` draws a target: the instance seed fixes the graph (sampled per instance
for `sampled_dags`; a random assignment of children to parents for the trees) and the local functions' parameters.
Inputs are $X_j \sim U[-1, 1]$; every local function is normalized on a fixed pool of $2^{18}$ inputs. Observations are
$Y = f^\star(X) + \epsilon$ with $\sigma^2 = 0.01$ (signal-to-noise ratio 100). A target instance is the same function in
both settings; the nuisance setting only places its 8 active inputs among 32 coordinates.

## Data splits

| Pool | Size | Noise | Use |
|---|---:|---|---|
| training | $n_{\max}$ | yes | the first $n$ examples are the training set of size $n$ (nested sets) |
| validation | 4,096 | yes | checkpoint and configuration selection |
| test | 65,536 | no | the reported test NMSE |
| test audit | 65,536 | no | a second, independent test pool |
| development | 65,536 | no | method development |
| diagnostic | 8,192 | no | probes, interventions, associations |

Every pool has its own seed, derived from the target's fingerprint, so the data are the same for every method and
every machine. `datasets.make_dataset(target, n_max, pools=...)` draws them.

## Metrics

* **Test NMSE**: the squared error summed over the test pool divided by the summed squared deviation of the noise-free
  target from its mean. 1 is the error of predicting the mean.
* **AULC**: the area under the learning curve (test NMSE against $\log_2 n$ over the grid), divided by the width of
  the grid. Lower is better. `metrics.aulc(ns, nmse)`.
* **Reporting**: the arithmetic mean of the AULC over the target instances with a two-sided 95% Student-t interval;
  a comparison of two methods is the mean of their paired AULC differences per instance with its interval
  (`analysis.aulc_table`, `analysis.contrasts`). No p-values.

## The official protocol

The paper's numbers come from this protocol; a result is comparable with the records when it follows it.

| Part | Rule |
|---|---|
| backbone | ReLU MLP, width 32 and 4 hidden layers (width 64 and 8 on the depth-5 tree), one output |
| training | Adam (0.9, 0.999, 1e-8), batch 32, 64,000 steps (128,000 on the depth-5 tree), cosine decay to 1% |
| checkpoints | 64 validation checkpoints; the best one is kept and tested once |
| search | 32 configurations per method, instance and size from a frozen search space (scrambled Sobol) |
| search spaces | frozen on the development instances 6016-6019 (`config/spaces.json`; `csfrbench info --spaces`) |
| selection | the configuration with the lowest validation NMSE; a diverged run is never selected |
| report | mean AULC over all evaluation instances with its 95% interval; paired differences |

For a new method: write a `Method` (below), freeze its search space on the development instances, train 32
configurations per instance and size, and compare with the records. [`examples/new_method.py`](../examples/new_method.py)
does all three steps. `csfrbench.task` and `csfrbench.learning_curve` are the quick, exploratory interface: they fix
the data and the metric but not the learner's training budget or search, so their numbers are indicative.

| | Quick and exploratory | Official |
|---|---|---|
| interface | `csfrbench.task`, `learning_curve`, `compare` | `csfrbench plan / run / analyze`, a `Method` |
| learner | any predictor | a method on the benchmark's backbone and training protocol |
| configurations | your choice | 32 per condition from a space frozen on instances 6016-6019 |
| instances | tune on 6016-6019 | every evaluation instance of the workload |
| compute | minutes | one workload with two methods: 7,168 runs |

## Reporting a result

A result is comparable with the paper's when it follows the official protocol. With every result, give:

| Item | Example |
|---|---|
| benchmark version | CSFRBench 1.5 (`csfrbench.BENCHMARK_VERSION`), package 1.6.1 |
| setting and workload | `primary`, `depth4_tree` |
| target instances | all 16 evaluation instances (6000-6015); 32 for `sampled_dags` |
| training-set sizes | the workload's grid, $n = 2^9, \dots, 2^{15}$ |
| search | the space, frozen on 6016-6019, and 32 configurations per instance and size |
| training budget and compute | 64,000 steps per run; core-hours in total |
| result | mean AULC with its 95% Student-t interval over instances |
| comparison | paired AULC differences per instance against the recorded methods, with their intervals (`csfrbench analyze --runs runs/mine --with-record`) |

A result of the quick interface (`csfrbench.task`, `learning_curve`) is exploratory: say so, and give the learner's
budget and how its settings were chosen.

## Methods of the paper

`lewi` (LeWi), `dense` (Dense), `l1` and `l2` (weight penalties), `l0` (hard-concrete L0, Louizos et al., 2017),
`rigl` (RigL, Evci et al., 2020), `fiedler` (Fiedler regularization, Tam & Dunson, 2020), `bimt` (BIMT, Liu et al.,
2023), `positions` (learned neuron positions, Erb et al., 2025), and the graph-informed references `go0`, `go1`, `go2`,
`go_all` (GraphOracle: connectivity masks from the target's graph for that many levels). A method must not read the
target's graph or active inputs; only the references do.

## Training engines

`csfrbench run` trains LeWi and Dense in **stacks** of many networks (`csfrbench/stack.py`): every network keeps its
own initialization, minibatch order, learning rate, schedule, validation and selection, but the networks of a stack
share every kernel, and LeWi's gate runs as two fused C++ kernels (`csfrbench/kernels.py`). This is 3 to 40 times
faster per core than training the networks one at a time. A batched kernel sums in another order, so a run agrees with
its record to float rounding until the dynamics amplify the difference; trials, winners and learning curves agree with
the records statistically ([docs/REPRODUCE.md](../docs/REPRODUCE.md) gives the validation on 21,952 runs). Every other method, and
every unit with `--engine exact`, trains alone; in the records' environment such a run equals its record bit for bit.

## API

| Operation | Call |
|---|---|
| a task (examples only) | `csfrbench.task(workload, instance, n, setting="primary")` -> `Task` with `X_train`, `y_train`, `X_val`, `y_val`, `d`, `n` |
| score a predictor | `Task.evaluate(predictor, pool="test")` -> test NMSE (`pool="val"` for selection) |
| the hidden target | `Task.reveal()` -> `Target` |
| a learning curve | `csfrbench.learning_curve(fit, workload, instance, sizes=None)` -> `LearningCurve` with `.nmse`, `.aulc` |
| compare with the paper's methods | `csfrbench.compare(curves)` |
| settings and workloads | `suite.SETTINGS`, `suite.workload(setting, name)` -> `Workload` with `.target(i)`, `.profile`, `.instances`, `.arms` |
| a target | `Target.evaluate(X)`, `.intermediates(X)`, `.graph` (`node_parents`, `readout_indices`, `A`, `S`), `.active_inputs` |
| data pools | `datasets.make_dataset(target, n_max, pools=("train", "val", "test"))` |
| a method | subclass `methods.Method`, decorate with `methods.register(name)`; `build`, `penalty`, `after_step` |
| one training run | `runner.execute(unit)`; many: `runner.plan(...)`, `runner.run(units, out_dir, workers=...)` |
| selection and tables | `analysis.select_winners`, `instance_aulc`, `aulc_table`, `contrasts`, `learning_curves` |
| the paper's records | `record.winners`, `units`, `trials`, `aulc`, `aulc_instances`, `contrasts`, `paper_tables` |
| linear probes | `probes.probe(model, target, X)` -> held-out $R^2$ per intermediate and layer |
| interventions | `das.score(model, target)` -> counterfactual $R^2$ (joint Boundless DAS) |
| associations | `structure.correlations(model, target)`, `structure.associations(model, target)` |
| figures | `viz.draw_target(ax, target)`, `viz.plot_embeddings(model, target)` (matplotlib) |

The command line covers the same ground: `csfrbench info`, `verify`, `record`, `target`, `dataset`, `unit`, `plan`,
`run`, `analyze`, `reproduce` (`csfrbench <command> --help`).

## Stability

The benchmark definition has its own version, 1.5 (`csfrbench.BENCHMARK_VERSION`): the paper's targets, data, protocol
and records. The identifiers of the target families, the names of the random streams and the constants of the
normalization fix the targets and the training runs. A change to any of them is a new benchmark version; the package
version (1.6.1) changes with the code. See [CONTRIBUTING.md](../CONTRIBUTING.md) and [CHANGELOG.md](../CHANGELOG.md).
