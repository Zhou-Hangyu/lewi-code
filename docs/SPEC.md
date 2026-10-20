# CSFRBench: specification

CSFRBench is a controlled benchmark for compositionally sparse function
regression. A target function is a composition of local functions on a known
computation graph. A learner sees only input-output examples. The benchmark
keeps the graph and the intermediate values of the target, so it can measure
three things: the predictive sample efficiency of a learner, the target
intermediates that its hidden layers represent, and the intermediates that its
prediction uses.

This document is the complete definition of the benchmark (version 1.5; version 1.6 of the package implements the
same definition). It agrees with Section 5, Section 6.1 and Appendix C of the paper. The package `csfrbench`
implements it; section 12 gives the module of each part.

## 1. Settings

The benchmark has two settings. They use the same target instances, the same
methods and the same protocol.

| Setting | Inputs | Ambient dimension $d$ |
|---|---|---|
| Primary | every input is active | 8 (16 on the depth-5 tree) |
| Nuisance-input | the same targets with inert inputs added | 32 (24 inert inputs; 16 on the depth-5 tree) |

The primary setting measures the benefit of compositional structure alone. The
nuisance-input setting adds the selection of the active inputs.

Each setting has eight workloads.

| Workload | Identifier | Local functions | Structure | Nonlinear levels | Local functions in the graph | Instances |
|---|---|---|---|---:|---:|---:|
| Flat modules | `flat_modules` | additive-multiplicative | four disjoint pairs, additive readout | 1 | 4 | 16 |
| Depth-4 tree | `depth4_tree` | additive-multiplicative | balanced binary tree | 3 | 7 | 16 |
| Reuse DAG | `reuse_dag` | additive-multiplicative | two intermediates are used twice | 2 | 7 | 16 |
| Sampled depth-4 DAGs | `sampled_dags` | additive-multiplicative | a graph sampled per instance | 3 | 7 | 32 |
| Fourier tree | `fourier_tree` | random Fourier | balanced binary tree | 3 | 7 | 16 |
| Fourier reuse DAG | `fourier_reuse_dag` | random Fourier | two intermediates are used twice | 2 | 7 | 16 |
| High-arity node | `high_arity_node` | random Fourier | one function of 8 inputs | 1 | 1 | 16 |
| Depth-5 tree | `depth5_tree` | additive-multiplicative | balanced binary tree, 16 active inputs | 4 | 15 | 16 |

The depth in a name counts the readout node: the depth-4 tree has three
nonlinear levels and the readout. Every local function of a compositional
workload has arity 2.

Two more target families belong to the local-function fitting study of the
paper (Appendix C.6) and are not part of a setting: a single local function of
arity 2 (additive-multiplicative or Fourier) and the high-arity node with an
additive-multiplicative function of arity 8 (a stress setting). There the model
receives exactly the arguments of the function ($d = k$) and the data have no
noise.

## 2. Computation graphs

A graph has $A$ leaves (the active inputs) and $S$ local functions in
topological order. The parents of a local function have disjoint input
ancestry, so they are independent random variables. The readout is one node or
the sum of several nodes; only the readout adds values that share ancestry.

**Flat modules.** $z_j = g_j(x_{2j-1}, x_{2j})$ for $j = 1, \dots, 4$; the
readout adds $z_1, \dots, z_4$.

**Depth-4 tree and depth-5 tree.** A balanced binary tree with 4, 2, 1 local
functions per level (8, 4, 2, 1 on the depth-5 tree). The assignment of the
children to the parents of each level is a random permutation per instance. The
readout is the root.

**Reuse DAG.** Level 1: $z_1 = g_1(x_1, x_2)$, $z_2 = g_2(x_3, x_4)$,
$z_3 = g_3(x_5, x_6)$, $z_4 = g_4(x_7, x_8)$. Level 2: $z_5 = g_5(z_1, z_2)$,
$z_6 = g_6(z_2, z_3)$, $z_7 = g_7(z_3, z_4)$. The readout adds $z_5$, $z_6$ and
$z_7$. The intermediates $z_2$ and $z_3$ are each used by two local functions.

**High-arity node.** One local function of all 8 active inputs.

**Sampled depth-4 DAGs.** Each instance samples a connected graph with 8
leaves, 7 local functions of arity 2 and 3 nonlinear levels:

1. Draw the number of strands, 2 or 3, with equal probability. Draw a
   partition of the 8 leaves into that number of parts from
   $\{(2,6), (3,5), (4,4)\}$ or $\{(2,2,4), (2,3,3)\}$ and a random order of
   the leaves.
2. Over each part, grow a random binary tree: join two units, drawn uniformly
   from the pairs whose joined unit stays within 3 levels, until one unit is
   left.
3. Reject the draw when the deepest strand does not have 3 levels.
4. Add reuse nodes until the graph has 7 local functions (1 for two strands, 2
   for three strands). A reuse node has a source and a partner. The source is
   drawn uniformly from the intermediates of level 1 or 2; with probability 1/4
   the leaves are candidates too. The partner is drawn uniformly from the
   values whose input ancestry is disjoint from that of the source, such that
   the new node stays within 3 levels and its unordered pair of parents is new.
   The order of the two parents is random. Reject the draw when a reuse node
   has no partner.
5. The readout adds the strand roots and the reuse nodes.

The graph is sampled independently of the local functions. A canonical hash
identifies the topology of a graph up to the labels of its leaves and nodes.

## 3. Local functions

**Additive-multiplicative functions.** For parents $z_1, \dots, z_k$,

$$g(z_1, \dots, z_k) = \sqrt{\frac{1-\rho}{k}} \sum_{i=1}^{k} a_i(z_i) + \sqrt{\rho} \prod_{i=1}^{k} b_i(z_i), \qquad \rho = 0.5 .$$

Each factor $a_i$, $b_i$ is a sinusoid
$\tilde{s}(u) = \sin\!\big(2\pi (f u / 2 + \phi)\big)$ with
$f \sim U[0.25, 0.4]$ and $\phi \sim U[0, 1)$, centred and scaled to unit
variance under the distribution of its parent. An input $x \sim U[-1, 1]$
enters a local function as $u = \sqrt{3}\,x$; at a deeper node $u$ is the
normalized value of the parent.

**Random Fourier functions.**

$$g(z) = \frac{1}{\sqrt{Q}} \sum_{q=1}^{Q} a_q \sin(\omega_q^{\top} z + \phi_q),$$

with $a_q \sim U[0.7, 1]$, $\phi_q \sim U[0, 2\pi)$ and
$\omega_q \sim \mathcal{N}(0, \sigma_\omega^2 I_k / k)$. Functions of arity 2
use $Q = 8$ and $\sigma_\omega = 0.85$. The high-arity node uses $Q = 24$ and
$\sigma_\omega = 1.5$.

## 4. Target instances and sequential normalization

A target instance is fixed by its family and its instance seed. The seed fixes
the graph and the parameters of the local functions. The ambient dimension
fixes only the coordinates of the input that are active. So an instance is the
same function in both settings.

Inputs have independent coordinates $X_j \sim U[-1, 1]$. For each instance a
fixed pool of $2^{18}$ inputs is passed through the graph in topological
order. The output of each local function is centred and scaled to unit variance
on this pool; the stored mean and standard deviation serve all later
computation. The readout is normalized in the same way. A local function is
drawn again only when the standard deviation of its raw output on the pool is
below $10^{-3}$.

The fingerprint of an instance is a hash of its family, dimension, arity, level
profile, noise variance, seed and the constants of its family. The seeds of the
datasets and of the training are derived from the fingerprint.

## 5. Data

Observations are $Y = f^\star(X) + \epsilon$ with
$\epsilon \sim \mathcal{N}(0, \sigma^2)$, $\sigma^2 = 0.01$. The target has
variance 1, so the signal-to-noise ratio is 100.

| Pool | Size | Noise | Use |
|---|---:|---|---|
| training | $n_{\max}$ | yes | the first $n$ examples are the training set of size $n$ |
| validation | 4,096 | yes | selection of the checkpoint and of the configuration |
| test | 65,536 | no | the reported test NMSE |
| test audit | 65,536 | no | a second, independent test pool |
| development | 65,536 | no | development of methods |
| diagnostic | 8,192 | no | linear probes |

The pools are the same for every method. Validation and test pools do not
change with $n$. The training sets are nested.

## 6. Learner and training

Every method trains the same ReLU multilayer perceptron within a workload.

| | Seven 8-input workloads | Depth-5 tree |
|---|---|---|
| Backbone | width 32, four hidden layers | width 64, eight hidden layers |
| Training steps | 64,000 | 128,000 |
| Validation | every 1,000 steps | every 2,000 steps |
| Training-set sizes $n$ | $2^9, \dots, 2^{15}$ | $2^{11}, \dots, 2^{17}$ |

The nuisance-input setting uses the same backbone, budget and sizes.

* Initialization: Kaiming normal (fan-in) for ReLU on the hidden layers and for
  a linear layer on the readout; biases zero.
* Optimizer: Adam, $\beta_1 = 0.9$, $\beta_2 = 0.999$,
  $\epsilon = 10^{-8}$, batch size 32, no weight decay.
* Loss: mean squared error on the noisy targets, plus the penalty of the method.
* Learning rate: selected by the search; cosine decay to 1% of its initial value.
* No warm-up, no gradient clipping, no early stopping.
* 64 validation checkpoints; the checkpoint with the lowest validation NMSE is kept.
* A run whose loss or gradient norm is not finite stops. It is recorded as
  diverged, has no test value and is never selected.

The seeds of the initialization, of the minibatch order and of the training
noise depend on the target instance and on $n$. They do not depend on the
method or on the hyperparameters. So every method sees the same minibatches,
and the methods that keep the initialization of the backbone start from the
same weights.

## 7. Methods

| Identifier | Paper name | Mechanism | Searched beside the learning rate |
|---|---|---|---|
| `lewi` | LeWi | learned embeddings; a connection is open while its two units are closer than $R$ | - |
| `dense` | Dense | none | - |
| `l1` | $\ell_1$ | $\lambda$ times the mean of $\lvert w\rvert$ | $\lambda$ |
| `l2` | $\ell_2$ | $\lambda$ times the mean of $w^2$ | $\lambda$ |
| `l0` | $L_0$ | hard-concrete gate per weight; $\lambda$ times the expected fraction of open gates (Louizos et al., 2017) | $\lambda$ |
| `rigl` | RigL | sparse masks; drop by magnitude and grow by gradient every 100 steps (Evci et al., 2020) | density |
| `bimt` | BIMT | distance-weighted $\ell_1$ cost and exchange of units every 200 steps (Liu et al., 2023) | $\lambda$ |
| `fiedler` | Fiedler | $\delta$ times an upper bound of the Fiedler value of the weight graph (Tam and Dunson, 2020) | $\delta$ |
| `positions` | Neuron positions | weights are inverse distances between learned unit positions (Erb et al., 2025) | - |
| `go0` | GraphOracle(0) | fixed mask: the first layer reads only the active inputs | - |
| `go1`, `go2` | GraphOracle(1), GraphOracle(2) | fixed masks of the first 1 or 2 nonlinear levels of the target graph | - |
| `go_all` | GraphOracle(all) | fixed masks of the full target graph | - |

**Scope of a mechanism.** The mechanism of LeWi acts on every affine layer, the
readout included. The mechanism of each baseline (penalty, gate, mask or
position-defined weight) acts on the transitions from the input to the last
hidden layer; the readout of a baseline is a plain linear layer. Biases carry
no penalty, except in the connection cost of BIMT, as in its source.

**LeWi.** Every unit has an embedding in $\mathbb{R}^r$, $r = 3$, initialized
$\mathcal{N}(0, \sigma_e^2 I)$ with $\sigma_e = 0.5$. With
$s_{ij} = R^2 - \lVert e_i - e_j \rVert^2$ and $R^2 = 2 r \sigma_e^2 = 1.5$,
the effective weight of a connection is
$w_{ij}\,\mathrm{sigmoid}(s_{ij})\,\mathbb{1}[\mathrm{sigmoid}(s_{ij}) > 1/2]$.
The binary mask is computed again after every optimizer step and has no
gradient. Two checks act at the initialization only: a layer without an open
connection has the embeddings of its receiving units drawn again; an input with
fewer than five open outgoing connections has its embedding drawn again. The
constants $r$ and $R^2$ are not searched. For a sensitivity study the method
accepts the arguments `code_dim` ($r$) and `radius` (a multiplier of $R^2$).

**GraphOracle.** For a target with $L$ nonlinear levels and a backbone with
$H \ge L$ hidden layers, level $r$ has
$s_r = \lfloor rH/L \rfloor - \lfloor (r-1)H/L \rfloor$ consecutive hidden
layers. The hidden units of a level are divided into blocks, one per target
value that is alive at that level. A block has at least 1 unit at the level
that computes its value and at least 2 units when it carries a value through a
further ReLU layer. The first layer of a level connects a block to the blocks
of its parents; the further layers of the level connect a block only to itself.
Connections after the supplied levels are not restricted. A masked layer is
initialized at the fan-in that its mask keeps. A rung whose masks equal those
of GraphOracle(all) is not part of a workload. GraphOracle(0) is part of the
nuisance-input setting only: with every input active it equals Dense.

## 8. Hyperparameter search and model selection

Each method has a frozen search space per backbone and setting. Every searched
hyperparameter is sampled uniformly in log space.

*Construction of a search space.* On a development panel (the target instances
6016 to 6019 at the smallest, the middle and the largest training-set size) the
configuration with the lowest validation NMSE is selected in each condition.
For each searched hyperparameter the span of the selected values is expanded by
25% of its width in log space, half on each side, and clipped only at a
boundary that the method defines (a regularization strength has the lower
boundary $10^{-7}$; a density has the upper boundary 1). Every search space was
frozen before the evaluation of its method and was not changed after it.

The panel of a set of search spaces has the workloads of its setting and
backbone. The panel of the primary setting on the backbone of the depth-5 tree
has three workloads with 16 active inputs: the depth-5 tree, a reuse DAG and a
Fourier tree. The function `csfrbench.search.freeze` applies the rule to a new
method. It has two safeguards, which did not act on a search space of the
benchmark: the expansion is at least 0.05 decades on each side, and a panel
with fewer than 12 conditions gives an interval of at least one decade.

*Evaluation.* For each method, target instance and training-set size the
benchmark trains $T = 32$ configurations from a scrambled Sobol sequence. The
sequence depends on the method, the workload and $n$; every target instance of
a workload uses the same 32 configurations. Each run keeps its checkpoint with
the lowest validation NMSE. The configuration with the lowest validation NMSE
is selected and evaluated once on the test pool.

The frozen search spaces of the primary setting are Table 2 of the paper. The
command `csfrbench info --spaces` prints all search spaces.

## 9. Predictive evaluation

At training-set size $n$ the outcome is the normalized mean squared error on
the test pool against the noise-free target,

$$\mathrm{NMSE}(n) = \frac{\mathbb{E}\big[(\hat{f}_n(X) - f^\star(X))^2\big]}{\mathrm{Var}\big(f^\star(X)\big)} .$$

For the sizes $n_1 < \dots < n_K$ the learning curve is summarized by its area
on a logarithmic axis,

$$\mathrm{AULC} = \frac{1}{\log_2 n_K - \log_2 n_1} \sum_{j=1}^{K-1} \frac{\mathrm{NMSE}(n_j) + \mathrm{NMSE}(n_{j+1})}{2} \,\big(\log_2 n_{j+1} - \log_2 n_j\big) .$$

Lower is better. The validation pool has a fixed size, so the curve measures
the performance as a function of the training-set size under a fixed selection
protocol, not the total need of labelled data.

## 10. Statistical reporting

The unit of replication is the target instance: 16 instances per workload, 32
for the sampled depth-4 DAGs. The report gives the arithmetic mean of the AULC
with a two-sided 95% Student-t interval over the instances. A comparison of two
methods uses the paired differences of the AULC per instance and their
interval. A comparison whose interval contains zero is inconclusive. No p-value
is reported. A learning curve shows the mean test NMSE over the instances at
each training-set size. The package gives the arithmetic mean with its 95%
Student-t interval, for a linear axis, and the mean with the 95% Student-t
interval in log-NMSE space, for a logarithmic axis.

## 11. Representation diagnostics

**Linear probes.** The diagnostic pool of 8,192 inputs is split by a fixed
permutation into 4,096 probe-training, 2,048 probe-validation and 2,048
probe-test examples. For each target intermediate and each layer ($h^{(0)}$ is
the input, $h^{(\ell)}$ the output of hidden layer $\ell$ after the ReLU) a
ridge regression predicts the intermediate. A hidden unit that is active on
fewer than 10 probe-training examples is excluded. Features are standardized
and targets centred with probe-training statistics. The ridge strength is
selected from $10^{-4}, 10^{-3.5}, \dots, 10^{7}$ on the probe-validation split.
The probe is evaluated once on the probe-test split. The result is the
held-out $R^2$, not clipped.

**Counterfactual interventions.** Joint multi-variable Boundless DAS
(Wu et al., 2023) learns, for a hidden layer, an orthogonal rotation whose
coordinates are divided into disjoint blocks, one per target intermediate. For
a base input, a source input and an intermediate $v$, the block of $v$ in the
rotated base state is replaced by the block of the source, the rotation is
inverted and the forward pass continues. The prediction is compared with the
counterfactual output of the target. The result is the held-out counterfactual
$R^2$, not clipped.

| Constant | Value |
|---|---|
| Pairs per intermediate | 10,000 training, 1,000 validation, 2,000 test |
| Optimizer | Adam, batch size 64, 18 epochs |
| Learning rates | rotation 0.003, boundary 0.0003 |
| Validation | every 200 updates, with the binary masks |
| Temperature | from 50 to 0.1 in coordinate units at dimension 4096, multiplied by $\text{dim}/4096$ |
| Boundaries | half-offset coordinates; widths in [0, 1], scaled onto the simplex when their sum exceeds 1 |
| Restarts | 3 per hidden layer; the restart with the best validation score is kept |
| Layer | per intermediate, the hidden layer (1 to 4) with the best validation $R^2$ |

The protocol is calibrated for the backbone of width 32 with four hidden
layers. It is not applied to the depth-5 tree. The linear probes do not apply to
the neuron-position method, whose model has no separate layer representation.

## 12. Implementation

| Part | Module |
|---|---|
| Settings, workloads, methods of the suite | `csfrbench.suite` |
| Local functions, graphs, target instances | `csfrbench.targets` |
| Data pools | `csfrbench.datasets` |
| Backbone | `csfrbench.models` |
| Methods | `csfrbench.methods` |
| Training protocol | `csfrbench.training` |
| Search spaces, configurations, selection | `csfrbench.search` |
| NMSE, AULC, intervals | `csfrbench.metrics`, `csfrbench.analysis` |
| Linear probes | `csfrbench.probes` |
| Counterfactual interventions | `csfrbench.das` |
| Execution of runs | `csfrbench.runner` |
| Records of the paper's experiments | `csfrbench.record` |
| Verification of an installation | `csfrbench.verify` |
| LeWi's layer, embeddings and checks | `lewi` |
| Stacked training of LeWi and Dense, fused kernels | `csfrbench.stack`, `csfrbench.kernels` |
| Tasks, learning curves and comparison for any predictor | `csfrbench.benchmark` |
| Unit associations, figures | `csfrbench.structure`, `csfrbench.viz` |

## 13. Identifiers that must not change

The following values enter a seed or a fingerprint. A change gives other
targets, other data or other training runs.

* the identifiers of the target families (`csfrbench.targets.families`);
* the names of the random streams: `structure`, `embedding`,
  `calibration_pool`, `training`, `validation`, `fixed_final_test`,
  `fixed_final_test_audit`, `fixed_development_test`, `fixed_diagnostic`,
  `initialization`, `stochastic`, `batch_order`, `gate_init`, `evaluation`,
  `probe_split`;
* the keys of the embeddings that the two checks of LeWi draw again: `hidden`,
  `output`, `input`;
* the seed of the counterfactual pairs, 11, and the names of their splits:
  `das_train`, `das_val`, `das_test`;
* the keys of the hash functions: `arityseed` (seeds), `aritytask`
  (fingerprints), `aritygraph` (topologies);
* the seed of the search, 20260826, and the keys of the methods and of the
  sets of search spaces in `csfrbench.suite` and `csfrbench.search`;
* the constants of the normalization pool in `csfrbench.targets.graph`.
