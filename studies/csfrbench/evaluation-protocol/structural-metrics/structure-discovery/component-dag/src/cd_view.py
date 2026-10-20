"""Checkpoint view for the component-DAG study - the only module that touches
checkpoints and targets (barrier boundary).

Learner side (``learner``): positions, executed masks, effective weights and
hidden activations on unlabeled input draws.  Target side (``target_graph``,
``node_values``): the computation graph and its noise-free node values, read
only by the evaluation stage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[7]
SD = Path(__file__).resolve().parents[2]              # structure-discovery
X12 = REPO / "studies/main-experiments/12-extra/src"
for p in (str(SD / "src"), str(X12)):
    if p not in sys.path:
        sys.path.insert(0, p)

import sd_view as sv                                   # noqa: E402  (frozen loader)
import w5_methods                                      # noqa: E402,F401  (lewi-v1-rg / -rgm)
from csfrbench.registry import get_method              # noqa: E402
from csfrbench.metrics import teacher_nodes, teacher_node_values  # noqa: E402
from csfrbench.utils.seeds import derive               # noqa: E402


def load(run_dir, which="best"):
    return sv.load_unit(run_dir, which=which)


def draw_inputs(fingerprint, tag, n, d):
    rng = np.random.default_rng(derive(fingerprint, "cd", tag))
    return rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float32)


def learner(model, method_key, X):
    """Everything the extraction may see."""
    method = get_method(method_key)
    layers = list(model.linear_layers())
    with torch.no_grad():
        W = [w.detach().cpu().numpy().astype(np.float64) for w in method.effective_weights(model)]
        M = [np.asarray(getattr(l, "event_mask", torch.ones_like(l.weight)).detach().cpu().numpy(),
                        dtype=np.float64) for l in layers]
        codes = getattr(model, "gate_codes", None)
        E = None if codes is None else [c.detach().cpu().numpy().astype(np.float64) for c in codes]
        h, H = torch.from_numpy(X), []
        for layer in model.hidden_layers:
            h = torch.relu(layer(h))
            H.append(h.numpy().astype(np.float64))
    return {"W": W, "M": M, "E": E, "H": H, "X": X.astype(np.float64)}


def target_graph(target):
    g = target.graph
    return {"A": int(g.A), "node_parents": [list(r) for r in g.node_parents],
            "active_raw": [int(i) for i in np.asarray(target.active_raw).ravel()],
            "readout_indices": list(g.readout_indices),
            "nodes": teacher_nodes(target)}


def node_values(target, X, indices):
    return teacher_node_values(target, X, indices)
