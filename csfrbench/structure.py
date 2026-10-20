"""Which target intermediate each unit of a network follows: correlations on the diagnostic pool and associations.

A unit (an input, a hidden unit or the output) is *associated* with a target intermediate by its activation alone:

* a hidden unit takes the intermediate with which its activation has the largest absolute Pearson correlation on the
  diagnostic pool (8,192 noise-free inputs), provided the correlation reaches the permutation threshold; there is no
  condition on the unit's connections;
* an input takes the local function that reads it (an input that no local function reads has none);
* the output takes the readout's intermediate.

The permutation threshold is the 99th percentile, over 32 random permutations of the pool's rows of the intermediates,
of every unit's largest absolute correlation with the permuted intermediates (one threshold for all layers). This is
the rule of the paper's embedding figures and of the project page. The associations colour the embedding figures
(:mod:`csfrbench.viz`); they use the hidden graph and are a diagnostic, never an input to training.
"""
from __future__ import annotations

import numpy as np
import torch

from . import datasets
from .targets import Target

PERMUTATIONS = 32
QUANTILE = 99.0


def zscore(A: np.ndarray) -> np.ndarray:
    """Columns centred and scaled to unit variance; a constant column (a dead unit) becomes 0."""
    A = A - A.mean(0, keepdims=True)
    sd = A.std(0, keepdims=True)
    live = sd > 1e-8
    return np.where(live, A / np.where(live, sd, 1.0), 0.0)


def diagnostic_inputs(target: Target) -> np.ndarray:
    """The diagnostic pool of a target instance: 8,192 noise-free inputs, the same for every method."""
    return datasets.make_dataset(target, 32, pools=("diagnostic",))["X_diagnostic"].numpy()


def unit_states(model, X) -> list[np.ndarray]:
    """The activation of every unit on ``X``, per level: the inputs, each hidden layer after its ReLU, the output."""
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        hidden = [h.numpy().astype(np.float64) for h in model.hidden_states(Xt)]
        out = model(Xt).reshape(Xt.shape[0], -1).numpy().astype(np.float64)
    model.train(was_training)
    return [Xt.numpy().astype(np.float64), *hidden, out]


def correlations(model, target: Target, X=None, *, permutations: int = PERMUTATIONS, quantile: float = QUANTILE,
                 seed: int = 0) -> dict:
    """Per level, the Pearson correlation of every unit with every target intermediate, and the permutation threshold.

    Returns ``{"corr": [units x intermediates per level], "dead": [bool per unit per level], "floor": float}``.
    """
    X = diagnostic_inputs(target) if X is None else np.asarray(X, dtype=np.float32)
    acts = [zscore(a) for a in unit_states(model, X)]
    Z = target.intermediates(X)
    Zz = zscore(Z)
    m = Zz.shape[0]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(permutations)):
        Zp = zscore(Z[rng.permutation(m)])
        draws += [np.abs((A.T @ Zp) / m).max(1) for A in acts]
    return {"corr": [((A.T @ Zz) / m) for A in acts], "dead": [np.abs(A).sum(0) == 0 for A in acts],
            "floor": float(np.percentile(np.concatenate(draws), quantile))}


def associations(model, target: Target, X=None, *, corr: dict | None = None) -> list[np.ndarray]:
    """Per level (inputs, hidden layers, output), the intermediate each unit is associated with: the index of the
    target's local function (``target.graph.node_parents`` order), or -1 for none."""
    corr = correlations(model, target, X) if corr is None else corr
    g = target.graph
    leaf_input = np.asarray(target.active_raw).reshape(-1)         # leaf r of the graph reads input leaf_input[r]
    readers = np.full(target.d, -1, dtype=int)
    for v, refs in enumerate(g.node_parents):
        for r in refs:
            if r < g.A:
                readers[int(leaf_input[r])] = v
    out = [readers]
    floor = corr["floor"]
    for C in corr["corr"][1:-1]:
        a = np.abs(C)
        best = a.argmax(1)
        out.append(np.where(a[np.arange(len(best)), best] >= floor, best, -1))
    out.append(np.full(corr["corr"][-1].shape[0], g.readout_indices[0], dtype=int))   # the output: the readout
    return out
