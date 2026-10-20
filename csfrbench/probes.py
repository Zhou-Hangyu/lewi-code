"""Linear probes: is a target intermediate linearly decodable from a hidden layer?

Protocol (paper Appendix C.9). Each target instance has a diagnostic pool of
8,192 inputs, split by a fixed permutation into 4,096 probe-training, 2,048
probe-validation and 2,048 probe-test examples. For each target intermediate
``z_v`` and each layer ``h(l)`` (``h(0)`` is the input, ``h(l)`` the output of
hidden layer ``l`` after the ReLU) a ridge regression predicts ``z_v`` from
``h(l)``:

* a hidden unit that is active on fewer than 10 probe-training examples is excluded;
* the features are standardized and the targets centred with probe-training statistics;
* the ridge strength is selected from ``10^-4, 10^-3.5, ..., 10^7`` on the probe-validation split;
* the probe is evaluated once on the probe-test split, without a new fit.

The result is the ordinary held-out ``R^2``. It is not clipped: a negative
value means that the probe predicts worse than the mean.
"""
from __future__ import annotations

import numpy as np
import torch

from .datasets import DIAGNOSTIC_SIZE
from .seeds import derive

FRACTIONS = (0.5, 0.25, 0.25)       # probe-training, probe-validation, probe-test
LAMBDAS = tuple(float(x) for x in np.logspace(-4.0, 7.0, 23))
MIN_SUPPORT = 10


def intermediates(target) -> dict:
    """The probed variables: every local function of the target, the one before the readout included."""
    graph = target.graph
    depths = graph.levels()
    edges = [(r - graph.A, v) for v, refs in enumerate(graph.node_parents) for r in refs if r >= graph.A]
    return {"indices": list(range(graph.S)), "names": [f"z{v}(d{depths[v]})" for v in range(graph.S)],
            "levels": depths, "edges": edges}


def layer_representations(model, X: torch.Tensor) -> list[tuple[str, np.ndarray]]:
    """``h(0) .. h(L)`` as float64 arrays, the model in evaluation mode."""
    if not hasattr(model, "hidden_states"):
        raise ValueError(f"{type(model).__name__} has no rule for the representation of a layer")
    layers = [("h0", X.detach().cpu().numpy().astype(np.float64))]
    model.eval()
    with torch.no_grad():
        for index, h in enumerate(model.hidden_states(X), start=1):
            layers.append((f"h{index}", h.cpu().numpy().astype(np.float64)))
    return layers


def split_indices(fingerprint: str, n: int):
    """The probe split of a target instance; the same for every method."""
    order = np.random.default_rng(derive(fingerprint, "probe_split")).permutation(n)
    n_train, n_val = int(FRACTIONS[0] * n), int(FRACTIONS[1] * n)
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def _features(H_by_split):
    H_train, H_val, H_test = (np.asarray(H, dtype=np.float64) for H in H_by_split)
    keep = (H_train != 0).sum(axis=0) >= MIN_SUPPORT
    H_train, H_val, H_test = H_train[:, keep], H_val[:, keep], H_test[:, keep]
    mean = H_train.mean(axis=0)
    scale = H_train.std(axis=0) + 1e-8
    return tuple((H - mean) / scale for H in (H_train, H_val, H_test)), keep


def ridge_r2(H_by_split, Z_by_split, lambdas=LAMBDAS):
    """Held-out ``R^2`` of every target column and the selected ridge strength.

    ``H_by_split`` and ``Z_by_split`` are (train, validation, test) tuples.
    On a tie of the validation ``R^2`` the smallest strength is selected.
    """
    Z_train, Z_val, Z_test = (np.asarray(Z, dtype=np.float64) for Z in Z_by_split)
    if Z_train.shape[1] == 0:
        return np.zeros(0), np.zeros(0)
    (Ht, Hv, He), _keep = _features(H_by_split)
    z_mean = Z_train.mean(axis=0)
    U, s, Vt = np.linalg.svd(Ht, full_matrices=False)
    UtZ = U.T @ (Z_train - z_mean)
    var_val, var_test = Z_val.var(axis=0), Z_test.var(axis=0)
    val_r2 = np.empty((len(lambdas), Z_train.shape[1]))
    test_r2 = np.empty_like(val_r2)
    for row, lam in enumerate(lambdas):
        beta = (Vt.T * (s / (s * s + lam))) @ UtZ
        val_r2[row] = 1.0 - ((Z_val - (Hv @ beta + z_mean)) ** 2).mean(axis=0) / (var_val + 1e-12)
        test_r2[row] = 1.0 - ((Z_test - (He @ beta + z_mean)) ** 2).mean(axis=0) / (var_test + 1e-12)
    best = val_r2.argmax(axis=0)
    columns = np.arange(val_r2.shape[1])
    return test_r2[best, columns], np.asarray(lambdas, dtype=float)[best]


def probe(model, target, X: torch.Tensor) -> dict:
    """The matrix of held-out ``R^2``: rows are the target intermediates, columns the layers ``h(0) .. h(L)``."""
    if int(X.shape[0]) != DIAGNOSTIC_SIZE:
        raise ValueError(f"the protocol uses the diagnostic pool of {DIAGNOSTIC_SIZE} inputs; received {X.shape[0]}")
    view = intermediates(target)
    Z = target.intermediates(X.cpu().numpy())[:, view["indices"]]
    layers = layer_representations(model, X)
    train, val, test = split_indices(target.fingerprint, X.shape[0])
    r2 = np.zeros((len(view["indices"]), len(layers)))
    lambdas = np.zeros_like(r2)
    kept = []
    for column, (_name, H) in enumerate(layers):
        r2[:, column], lambdas[:, column] = ridge_r2((H[train], H[val], H[test]), (Z[train], Z[val], Z[test]))
        kept.append(int(((H[train] != 0).sum(axis=0) >= MIN_SUPPORT).sum()))
    return {"r2": r2, "lambdas": lambdas, "intermediates": view["names"], "levels": view["levels"],
            "layers": [name for name, _H in layers], "features_kept": kept,
            "features_total": [int(H.shape[1]) for _name, H in layers], "edges": view["edges"]}
