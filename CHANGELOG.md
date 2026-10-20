# Change record

## 1.6.0: the public release (LeWi and CSFRBench in one repository)

The benchmark is unchanged: targets, data, protocol, methods and records are those of version 1.5, and every check of
`csfrbench verify` gives the same result.

* **LeWi as its own package** (`lewi/`): the layer, the embeddings with their initialization checks and an MLP, for any
  PyTorch model. CSFRBench's `lewi` method is built from it (the same runs bit for bit).
* **The quick interface**: `csfrbench.task`, `learning_curve` and `compare` benchmark any predictor on the paper's data
  and metric; the target stays hidden until `Task.reveal()`.
* **The stacked engine** (`csfrbench.stack`, `csfrbench.kernels`): LeWi and Dense train many networks per kernel with
  LeWi's fused C++ gate kernels, under the per-network protocol; `csfrbench run` and `reproduce` use it by default and
  `--engine exact` keeps the per-network trainer.
* **Associations and figures** (`csfrbench.structure`, `csfrbench.viz`): the target intermediate each unit follows, by
  correlation alone with a permutation threshold (the rule of the paper's embedding figures), the target graphs and the
  learned embeddings.
* Examples for the paper's figures (`examples/`), the reproduction map (`docs/REPRODUCE.md`), the validation of the
  stacked engine (`tools/validate_stack.py`) and the timing of Table 6 (`tools/timing.py`).

## 1.5.0: first stable release

Version 1.5 is the benchmark of the paper "Learning Computational Structure for
Sample-Efficient Prediction". It has the settings, the protocol and the methods
that produced the paper's CSFRBench results, in one package.

Changes from version 1.1, the last development version:

**Settings**

* The primary setting has only active inputs: $d = 8$ on the seven 8-input
  workloads and $d = 16$ on the depth-5 tree. In version 1.1 every workload had
  $d = 32$.
* The nuisance-input setting is the additional setting: the same target
  instances with inert inputs added, $d = 32$. The depth-5 tree has 16 active
  and 16 inert inputs.
* Training budget: 64,000 steps on the 8-input workloads (96,000 in version
  1.1) and 128,000 steps on the depth-5 tree (256,000 in version 1.1), with 64
  validation checkpoints.
* Target instances: seeds 6000 to 6015, and 6020 to 6035 more for the sampled
  depth-4 DAGs (7000 to 7031 in version 1.1).
* The search spaces are frozen again for each setting and backbone, on the
  instances 6016 to 6019.
* The suite has eight workloads. The single local function and the high-arity
  node with an additive-multiplicative function stay available as target
  families; they belong to the local-function fitting study.

**Methods**

* LeWi is one method, `lewi`: the distance gate and the binary mask act on
  every affine layer, the readout included, with two checks at the
  initialization (a layer has an open connection; an input has at least five
  open connections).
* The readout of every baseline is a plain linear layer. The option
  `readout: structural` of version 1.1 is removed: no run of the paper used it.
* GraphOracle(0) is `graph_oracle` with `levels = 0`.
* Removed: the methods and the target families that the evaluation of the
  paper does not use.
* Not part of the package: the variants of LeWi of the paper's ablation,
  sensitivity and cost studies (Section 6.4, Appendices D.6 and D.7), except
  the two arguments of the method itself, `code_dim` and `radius`.

**Package**

* One installable package without a dependency on the experiment repository.
  Dependencies: NumPy, SciPy, PyTorch.
* Public names for workloads and methods. The identifiers of the target
  families stay, because they fix the seeds.
* The records of the paper's runs are part of the package (`csfrbench.record`).
* `csfrbench verify` checks an installation against the records.

**Not changed**

* Target semantics. A target instance has the fingerprint, the graph, the
  local functions and the data that it had in the experiments of the paper.
* The training loop, the seeds, the selection rule, NMSE, AULC, the statistics,
  the linear probes and the counterfactual-intervention protocol.
