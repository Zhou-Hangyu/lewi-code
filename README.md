# LeWi and CSFRBench

Code for **Learning Computational Structure for Sample-Efficient Prediction**
(Hangyu Zhou, Constantine Dovrolis, Stephen Mussmann).
[Project page](https://zhou-hangyu.github.io/lewi/) ·
[Paper](https://zhou-hangyu.github.io/lewi/paper/lewi.pdf) ·
[CSFRBench page](https://zhou-hangyu.github.io/lewi/csfrbench/)

This repository holds two things:

* **LeWi** ([`lewi/`](lewi)), Learned Wiring: every neuron has a learned position in a small space, and only nearby
  neurons connect. Positions and weights train together, so the network learns its own wiring.
* **CSFRBench** ([`csfrbench/`](csfrbench)), a benchmark for compositionally sparse function regression: targets built
  from small local functions on a known computation graph. A learner sees only input–output examples. The benchmark
  keeps the graph to build graph-informed reference networks and to evaluate the structure a learner finds.

## Install

```bash
git clone https://github.com/Zhou-Hangyu/lewi-code.git
cd lewi-code
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip                                                # an old pip cannot read every package
pip install torch --index-url https://download.pytorch.org/whl/cpu      # optional: the CPU build of PyTorch
pip install -e .
csfrbench verify                                                         # about 4 minutes; ends with VERIFIED
```

Requirements: Python 3.10 or later; NumPy, SciPy, PyTorch 2.2 or later and ninja (installed by `pip`). No GPU is
needed. A C++ compiler (gcc or clang) lets LeWi's training use its fused kernels and `torch.compile`; without one the
same training runs more slowly. `pip install -e ".[plot]"` adds matplotlib for the figures.

## Benchmark your model in five minutes

A task gives your learner the training and validation examples of one target. The computation graph stays hidden;
the score is the test NMSE of the paper, on 65,536 noise-free inputs. This example runs as it stands:

```python
import csfrbench, torch

def my_fit(X, y, X_val, y_val):              # any learner: torch, scikit-learn, a plain function ...
    X, y = torch.tensor(X), torch.tensor(y)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 1), torch.nn.Flatten(0))
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(2000):
        i = torch.randint(len(y), (32,))
        loss = ((net(X[i]) - y[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

task = csfrbench.task("depth4_tree", instance=6016, n=1024)       # a development instance: tune here
print(task)                                   # d = 8 inputs, 1,024 noisy training examples, 4,096 validation examples
model = my_fit(task.X_train, task.y_train, task.X_val, task.y_val)
print(task.evaluate(model))                   # test NMSE

curve = csfrbench.learning_curve(my_fit, "depth4_tree", instance=6000)   # an evaluation instance: once, at the end
csfrbench.compare([curve])                    # your AULC beside LeWi, Dense and the other methods of the paper
```

Tune a learner on the development instances 6016-6019 and report on the evaluation instances 6000-6015 (and 6020-6035
of the sampled DAGs). The graph is hidden by protocol, not by access control: `task.reveal()` works at any time, and a
learner that reads the graph, the active inputs or the intermediate values while training breaks the protocol. The
quick interface fixes the data and the metric, not the training budget or the search, so its numbers are exploratory;
[`csfrbench/README.md`](csfrbench/README.md#reporting-a-result) says how to report a result.
`python examples/quickstart.py` runs the whole sequence (under a minute on one CPU core).
[`csfrbench/README.md`](csfrbench/README.md) describes the benchmark: the graph families, the local functions, the
data splits, the metrics, the official evaluation protocol and the API.

## Use LeWi in your network

```python
import torch
from lewi import LeWiMLP

x = torch.rand(4096, 8) * 2 - 1                                  # your data: here a target that uses 4 of 8 inputs
y = torch.sin(3 * x[:, 0]) * x[:, 1] + x[:, 2] * x[:, 3]
model = LeWiMLP(in_features=8, width=32, depth=4)                 # every layer, the readout included, is a LeWi layer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)        # weights and neuron positions together
for step in range(2000):
    i = torch.randint(len(x), (32,))
    loss = torch.nn.functional.mse_loss(model(x[i]).squeeze(-1), y[i])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.update_masks()                                         # connections open and close as neurons move
print(model.connection_density()["density_overall"])             # the fraction of connections still open
```

[`lewi/README.md`](lewi/README.md) gives the rule, the constants, and the building blocks for other architectures.

## Reproduce the paper

| Step | Command | Time |
|---|---|---|
| check the installation against the paper's records | `csfrbench verify` | 4 minutes |
| the recorded results (Tables 3 and 4) | `csfrbench record --setting primary` | seconds |
| run recorded selected configurations again | `csfrbench reproduce --workload depth4_tree --arms lewi dense --out runs/repro --workers 8` | about 1 core-hour |
| one workload of the benchmark from scratch | `csfrbench plan ...`, `csfrbench run ...`, `csfrbench analyze ...` | about 20 core-hours for LeWi and Dense |

The package holds the records of the paper's 741,888 CSFRBench training runs (`csfrbench/reference/`), so every table
can be recomputed and a new method compared without training the recorded ones again. LeWi and Dense train in stacks
of many networks with LeWi's fused C++ kernels: the same protocol and seeds as the per-network trainer, 3 to 40 times
faster per core, with trials, winners and learning curves that agree with the records statistically, not digit by digit.
`--engine exact` trains every network alone and reproduces a record bit for bit in the records' environment
(`requirements-reference.txt`). [`docs/REPRODUCE.md`](docs/REPRODUCE.md) maps every figure and table of the paper to
its command, its records and its compute.

### Original experiment code

The [`paper-code`](https://github.com/Zhou-Hangyu/lewi-code/tree/paper-code) branch holds the research code as it
produced the paper, with its own reproduction map: the world-model experiments (Section 6 and Appendix E), the mechanism
ablations, the sensitivity studies, the wiring alignment and the local-function study. Use `main` for new work: the
same benchmark and records with a cleaner interface. For the world models, start at that branch's README, section
[A world-model experiment](https://github.com/Zhou-Hangyu/lewi-code/tree/paper-code#a-world-model-experiment): the
figures and tables rebuild from the packaged records in minutes, while running the campaign again takes about 34,000
search runs and 2,448 final runs of 5 to 40 minutes each, in a second environment with the simulators
(`requirements-worldmodel.txt`).

## Layout

```
lewi/            LeWi: the layer, the neuron embeddings and their initialization checks, an MLP
csfrbench/       CSFRBench: targets, data, methods, training (per network and stacked), search, analysis,
                 diagnostics (probes, interventions, associations), figures, records of the paper's runs
examples/        quickstart, LeWi in your own network, a new method under the official protocol, embedding figures
docs/            SPEC.md (the benchmark's definition), RESULTS.md (the recorded results), REPRODUCE.md
tools/           validate_stack.py (the stacked engine against the records), timing.py (the paper's Table 6)
tests/           pytest
```

## Contributing

Bug reports, questions and new methods are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md), which also says how the
benchmark's definition is kept stable.

## Citation

```bibtex
@misc{zhou2026lewi,
  title  = {Learning Computational Structure for Sample-Efficient Prediction},
  author = {Zhou, Hangyu and Dovrolis, Constantine and Mussmann, Stephen},
  year   = {2026},
  url    = {https://zhou-hangyu.github.io/lewi/}
}
```

## License

MIT (see [LICENSE](LICENSE)).
