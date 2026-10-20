"""Checkpoint loading and the learner-only view (barrier boundary).

This module is the ONLY place that touches checkpoints, targets, and the
frozen benchmark stack.  It produces two strictly separated products:

  * ``build_view(model, method)``    the learner-only view consumed by
    ``sd_extract`` (weights, codes, frames, activations on unlabeled inputs).
    Assembled exclusively from the model's own parameters and from input
    draws; no target attribute is read.
  * ``teacher_side(...)``            everything the evaluation stage needs
    (node values, graph, layer representations), used only AFTER discovery.

Checkpoint sources:
  * mdas01 winners  studies/.../boundless-das/checkpoints/<family>/<unit>/
  * planted nets    studies/.../boundless-das/planted/planted_<variant>_s<k>/
  * lv2sel/lv25sel  an earlier internal study{lv2sel,lv25sel,lewi3sel}/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

STUDY = Path(__file__).resolve().parents[1]
DAS = STUDY.parent / "boundless-das"
REPO_SRC = Path(__file__).resolve().parents[6] / "src"

# priority: the repository package, then the DAS study's snapshot
for p in reversed((str(REPO_SRC), str(DAS / "src"))):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from csfrbench.registry import get_method, load_builtin_registrations  # noqa: E402
from csfrbench.utils.seeds import derive                                # noqa: E402

load_builtin_registrations()

N_DISC = 16384          # unlabeled discovery draws
POOL_TRAIN, POOL_VAL, POOL_TEST = 8192, 4096, 8192   # evaluation pools


def _register_for(method: str) -> None:
    """Import the module that registers ``method`` (idempotent)."""
    if method.startswith(("lv25",)):
        import lewiv25_method  # noqa: F401
    elif method.startswith(("lv2",)):
        import lewiv2_method   # noqa: F401
    elif method.startswith(("lewi3", "lewi-v05", "lewi_reg")):
        import lewi3_arms      # noqa: F401
    else:
        import push_methods    # noqa: F401


def load_unit(run_dir: str | Path, which: str = "best"):
    """(model, target, spec) for any campaign checkpoint, frozen."""
    import json
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    _register_for(spec["method"])
    import learner_io
    return learner_io.load_frozen_model(run_dir, which=which)


# ------------------------------------------------------------------- views --

def draw_inputs(fingerprint: str, tag: str, n: int, d: int = 32) -> np.ndarray:
    """Fresh unlabeled inputs from the family-wide input law (uniform on
    [-1,1]^d).  Depends only on (d, law, seed tag), never on the instance's
    graph, so it is discovery-safe; the derived seed makes it reproducible."""
    rng = np.random.default_rng(derive(fingerprint, "sd", tag))
    return rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float32)


def _hidden_activations(model, X: torch.Tensor) -> list[np.ndarray]:
    model.eval()
    with torch.no_grad():
        hs, h = [], X
        for layer in model.hidden_layers:
            h = torch.relu(layer(h))
            hs.append(h.cpu().numpy().astype(np.float64))
    return hs


def build_view(model, method_key: str, X_disc: np.ndarray) -> dict:
    """The learner-only view: everything discovery may see, nothing else."""
    method = get_method(method_key)
    with torch.no_grad():
        W = [w.detach().cpu().numpy().astype(np.float64)
             for w in method.effective_weights(model)]
    widths = [w.shape[0] for w in W[:-1]]
    view = {"L": len(widths), "widths": widths, "W": W,
            "codes": None, "frames": None, "latentW": None}

    codes = getattr(model, "gate_codes", None)
    if codes is not None:
        with torch.no_grad():
            view["codes"] = [codes[b].detach().cpu().numpy().astype(np.float64)
                             for b in range(1, len(widths) + 1)]

    layers = list(model.linear_layers()) if hasattr(model, "linear_layers") else []
    if layers and all(callable(getattr(l, "gate", None)) for l in layers) \
            and getattr(model, "frames", None) is None:
        with torch.no_grad():        # neuron-aligned gate matrices, one per boundary
            view["gates"] = [(l.gate() * getattr(l, "event_mask", 1.0)).detach().cpu().numpy()
                             .astype(np.float64) for l in layers]
    else:
        view["gates"] = None

    frames = getattr(model, "frames", None)
    if frames is not None:
        with torch.no_grad():
            view["frames"] = [f.q().detach().cpu().numpy().astype(np.float64)
                              for f in frames]
            view["latentW"] = [
                layer.latent_weight().detach().cpu().numpy().astype(np.float64)
                for layer in model.linear_layers()]

    view["H"] = _hidden_activations(model, torch.from_numpy(X_disc))
    return view


def view_of_unit(run_dir: str | Path, which: str = "best") -> tuple[dict, dict]:
    """(view, context) for one checkpoint; context feeds the evaluation stage."""
    model, target, spec = load_unit(run_dir, which=which)
    X_disc = draw_inputs(target.fingerprint, "disc", N_DISC, d=int(spec["d"]))
    view = build_view(model, spec["method"], X_disc)
    ctx = {"model": model, "target": target, "spec": spec,
           "fingerprint": target.fingerprint, "which": which}
    return view, ctx


# ------------------------------------------------------- planted networks --

PLANTED_GRAPH = {4: (0, 1), 5: (2, 3), 6: (4, 5)}     # internal-node parents
PLANTED_LEVEL = {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3}


class PlantedTeacher:
    """Target-side interface for a planted net (the net IS the teacher).

    Node values are read exactly through the planted read vectors
    ((H @ c - mu) / sd at the node's live layer), and the graph is the
    compiled tree. Used only by the evaluation stage."""

    def __init__(self, net, meta):
        self.net, self.meta = net, meta
        self.fingerprint = f"planted-{meta['variant']}-s{meta['seed']}"
        self.d = 32

    def node_table(self) -> dict:
        nodes = sorted(PLANTED_LEVEL)
        return {"names": [f"z{v}(d{PLANTED_LEVEL[v]})" for v in nodes],
                "vars": [f"z{v}" for v in nodes],
                "depths": [PLANTED_LEVEL[v] for v in nodes],
                "edges": [(u, v) for v, ps in PLANTED_GRAPH.items() for u in ps]}

    def node_values(self, X: np.ndarray) -> np.ndarray:
        import fb2_planted as fp
        cols = []
        with torch.no_grad():
            for v in sorted(PLANTED_LEVEL):
                e = self.meta["read"][f"z{v}"]
                H = fp._states(self.net, torch.from_numpy(X),
                               int(e["layer"])).numpy().astype(np.float64)
                cols.append((H @ np.asarray(e["c"], np.float64) - e["mu"]) / e["sd"])
        return np.stack(cols, axis=1)

    def true_partition(self, layer: int) -> dict:
        """Known module map at one layer: var -> its feature coordinates."""
        return {f"z{v}": self.meta["feature_rows"][f"z{v}"]
                for v in sorted(PLANTED_LEVEL)
                if int(self.meta["read"][f"z{v}"]["layer"]) == layer}


def view_of_planted(path: str | Path) -> tuple[dict, dict]:
    """(view, context) for one planted net; ``noreg`` supplies the trivial
    effective-weight rule (a planted net is a plain dense ReLU MLP)."""
    import fb2_planted as fp
    net, meta = fp.load(Path(path))
    teacher = PlantedTeacher(net, meta)
    X_disc = draw_inputs(teacher.fingerprint, "disc", N_DISC)
    with torch.no_grad():
        W = [l.weight.detach().cpu().numpy().astype(np.float64)
             for l in list(net.hidden_layers) + [net.output_layer]]
    widths = [w.shape[0] for w in W[:-1]]
    view = {"L": len(widths), "widths": widths, "W": W,
            "codes": None, "frames": None, "latentW": None,
            "H": _hidden_activations(net, torch.from_numpy(X_disc))}
    ctx = {"model": net, "target": teacher, "spec": {"method": "planted",
           "task": f"planted_{meta['variant']}", "d": 32},
           "fingerprint": teacher.fingerprint, "which": "best"}
    return view, ctx
