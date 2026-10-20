"""Counterfactual interventions: joint multi-variable Boundless DAS.

Does the network use a target intermediate in its prediction? Distributed
Alignment Search (Wu et al., 2023) answers with interventions. For a hidden
layer it learns an orthogonal rotation of the hidden state; the rotated
coordinates are divided into disjoint blocks, one per target intermediate, with
learned widths. For a base input ``b``, a source input ``s`` and an
intermediate ``v``, the block of ``v`` in the rotated state of ``b`` is replaced
by the block of ``s``, the rotation is inverted and the forward pass continues.
The prediction is compared with the counterfactual output of the target: the
target output when ``z_v`` takes its value on ``s`` and the descendants of ``v``
are computed again. The weights of the network stay fixed.

Protocol (paper Appendix C.10), fixed on development targets:

* pairs per target intermediate: 10,000 training, 1,000 validation, 2,000 test;
  ``b`` and ``s`` are independent draws from the input distribution;
* Adam, batch size 64, 18 epochs, rotation learning rate 0.003, boundary
  learning rate 0.0003, validation every 200 updates with the binary masks;
* boundaries at half-offset coordinates, widths clamped to [0, 1] and scaled
  onto the simplex when their sum exceeds 1, temperature from 50 to 0.1 in
  coordinate units at dimension 4096, multiplied by ``dim / 4096``;
* three restarts per hidden layer; the restart with the best validation score
  is kept; per intermediate the layer with the best validation ``R^2`` is
  selected; the test pairs are read once.

The score is the ordinary held-out counterfactual ``R^2``, not clipped. The
protocol is calibrated for the backbone of width 32 with four hidden layers.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .seeds import derive
from .targets import sample_inputs
from .targets.graph import SQRT3

TRAIN_PAIRS, VAL_PAIRS, TEST_PAIRS = 10000, 1000, 2000
SPLIT_SEED = 11
SPLITS = {"das_train": TRAIN_PAIRS, "das_val": VAL_PAIRS, "das_test": TEST_PAIRS}
EPOCHS = 18
RESTARTS = 3
ROTATION_LR = 0.003
BOUNDARY_LR = 0.0003
BATCH_SIZE = 64
EVAL_EVERY = 200
TEMPERATURE_AT_4096 = (50.0, 0.10)
LAYERS = (1, 2, 3, 4)
FIDELITY_ROWS = 256
FIDELITY_TOLERANCE = 1e-5


# ------------------------------------------------------------------ target side


def variables(target) -> list[dict]:
    """The aligned variables: every local function, ordered by (nonlinear level, node index)."""
    graph = target.graph
    level = graph.levels()
    roster = [{"var": f"z{v}", "node": v, "level": int(level[v]),
               "parents": sorted(r - graph.A for r in graph.node_parents[v] if r >= graph.A)}
              for v in range(graph.S)]
    roster.sort(key=lambda e: (e["level"], e["node"]))
    return roster


def counterfactual_output(target, Xb: np.ndarray, Xs: np.ndarray, node: int) -> np.ndarray:
    """The target output on ``Xb`` when node ``node`` takes its value on ``Xs``."""
    graph = target.graph
    source = target.forward(Xs)["values"]
    leaves = SQRT3 * np.asarray(Xb, np.float32)[:, target.active_raw.reshape(-1)].astype(np.float64)
    values: list[np.ndarray] = []
    for v, refs in enumerate(graph.node_parents):
        if v == node:
            values.append(source[v])
            continue
        Z = np.stack([leaves[:, r] if r < graph.A else values[r - graph.A] for r in refs], axis=1)
        h = target.law.value(target.params["nodes"][v], Z)
        values.append((h - target.node_mu[v]) / target.node_sd[v])
    raw = sum(w * values[i] for i, w in zip(graph.readout_indices, graph.readout_weights))
    y = (raw - target.out_mu) / target.out_sd
    if not np.all(np.isfinite(y)):
        raise FloatingPointError("the counterfactual output is not finite")
    return y.astype(np.float32)


def pairs(target, split: str, per_variable: int | None = None) -> dict:
    """One split of counterfactual pairs, balanced over the variables.

    Returns ``Xb``, ``Xs``, ``var_idx`` (the variable of each pair) and ``y_cf``.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    per_variable = SPLITS[split] if per_variable is None else int(per_variable)
    roster = variables(target)
    m = len(roster)
    total = per_variable * m
    rng = np.random.default_rng(derive(target.fingerprint, SPLIT_SEED, split))
    Xb = sample_inputs(rng, (total, target.d))
    Xs = sample_inputs(rng, (total, target.d))
    var_idx = np.repeat(np.arange(m), per_variable)
    rng.shuffle(var_idx)
    y_cf = np.empty(total, np.float32)
    for i, entry in enumerate(roster):
        rows = var_idx == i
        y_cf[rows] = counterfactual_output(target, Xb[rows], Xs[rows], entry["node"])
    return {"Xb": Xb, "Xs": Xs, "var_idx": var_idx.astype(np.int64), "y_cf": y_cf, "roster": roster}


def pools(target) -> dict[str, dict]:
    """The three splits of counterfactual pairs of a target instance."""
    return {split: pairs(target, split) for split in SPLITS}


# ------------------------------------------------------------------ the alignment


def _sigmoid_boundary(x, lower, upper, temperature):
    return torch.sigmoid((x - lower) / temperature) * torch.sigmoid((upper - x) / temperature)


class JointBoundlessDAS(nn.Module):
    """One orthogonal rotation and ``m`` ordered blocks with learned widths over ``dim`` coordinates."""

    def __init__(self, dim: int, m: int, seed: int):
        super().__init__()
        torch.manual_seed(int(seed))
        self.rotation = nn.utils.parametrizations.orthogonal(nn.Linear(dim, dim, bias=False))
        self.widths_raw = nn.Parameter(torch.full((m,), 1.0 / (m + 1)))
        self.register_buffer("positions", torch.arange(dim, dtype=torch.float32) + 0.5)
        self.scale = dim / 4096.0
        self.register_buffer("temperature", torch.tensor(TEMPERATURE_AT_4096[0] * self.scale, dtype=torch.float32))
        self.dim, self.m = int(dim), int(m)

    def boundaries(self) -> torch.Tensor:
        return torch.cumsum(self.widths_raw.clamp(0.0, 1.0), 0).clamp(max=1.0)

    def project_(self) -> None:
        with torch.no_grad():
            self.widths_raw.clamp_(0.0, 1.0)
            total = float(self.widths_raw.sum())
            if total > 1.0:
                self.widths_raw.mul_(1.0 / total)

    def soft_masks(self) -> torch.Tensor:
        upper = self.boundaries() * self.dim
        lower = torch.cat([upper.new_zeros(1), upper[:-1]])
        return _sigmoid_boundary(self.positions[None, :], lower[:, None], upper[:, None], self.temperature)

    def snapped(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            upper = (self.boundaries() * self.dim).round().long()
            lower = torch.cat([upper.new_zeros(1), upper[:-1]])
        return lower, upper

    def hard_masks(self) -> torch.Tensor:
        return hard_masks(*self.snapped(), self.dim)

    def matrix(self) -> torch.Tensor:
        return self.rotation.weight


def hard_masks(lower: torch.Tensor, upper: torch.Tensor, dim: int) -> torch.Tensor:
    k = torch.arange(dim)
    return ((k[None, :] >= lower[:, None]) & (k[None, :] < upper[:, None])).float()


def intervene(W: torch.Tensor, masks: torch.Tensor, hb: torch.Tensor, hs: torch.Tensor,
              var_idx: torch.Tensor) -> torch.Tensor:
    M = masks[var_idx]
    rb = hb @ W.T
    rs = hs @ W.T
    return ((1.0 - M) * rb + M * rs) @ W


def resume(model: nn.Module, h: torch.Tensor, layer: int) -> torch.Tensor:
    """The rest of the forward pass after hidden layer ``layer`` (1-based)."""
    for module in model.hidden_layers[layer:]:
        h = F.relu(module(h))
    return model.output_layer(h).squeeze(-1)


@torch.no_grad()
def capture(model: nn.Module, Xb: np.ndarray, Xs: np.ndarray, layer: int, chunk: int = 8192):
    out = []
    for X in (Xb, Xs):
        parts = [model.hidden_states(torch.from_numpy(X[i:i + chunk]))[layer - 1] for i in range(0, len(X), chunk)]
        out.append(torch.cat(parts))
    return out[0], out[1]


def r2_per_variable(y_hat: torch.Tensor, y_star: torch.Tensor, var_idx: torch.Tensor, m: int) -> list[float]:
    out = []
    for v in range(m):
        rows = var_idx == v
        y = y_star[rows]
        sse = float(((y_hat[rows] - y) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum())
        out.append(1.0 - sse / sst if sst > 0 else float("nan"))
    return out


@torch.no_grad()
def evaluate(model, layer: int, W, masks, split: dict, m: int) -> dict:
    y_hat = resume(model, intervene(W, masks, split["hb"], split["hs"], split["var_idx"]), layer)
    per_variable = r2_per_variable(y_hat, split["y"], split["var_idx"], m)
    return {"per_variable": per_variable, "macro": float(np.mean(per_variable))}


def _states(model, pool: dict, layer: int) -> dict:
    hb, hs = capture(model, pool["Xb"], pool["Xs"], layer)
    return {"hb": hb, "hs": hs, "var_idx": torch.from_numpy(pool["var_idx"]),
            "y": torch.from_numpy(np.ascontiguousarray(pool["y_cf"]))}


def fit(model: nn.Module, layer: int, train: dict, val: dict, seed: int) -> dict:
    """One joint alignment at a hidden layer; the checkpoint is selected on the validation pairs."""
    model.eval()
    model.requires_grad_(False)
    dim = train["hb"].shape[1]
    m = int(max(train["var_idx"].max(), val["var_idx"].max())) + 1
    das = JointBoundlessDAS(dim, m, seed)
    optimizer = torch.optim.Adam([{"params": list(das.rotation.parameters()), "lr": ROTATION_LR},
                                  {"params": [das.widths_raw], "lr": BOUNDARY_LR}])
    n = len(train["y"])
    steps_per_epoch = n // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    hi, lo = (t * das.scale for t in TEMPERATURE_AT_4096)
    read_at = set(range(EVAL_EVERY, total_steps + 1, EVAL_EVERY)) | {total_steps}
    generator = torch.Generator().manual_seed(int(seed) + 1)

    best = {"macro": -math.inf, "step": -1, "W": None, "widths": None}
    history = []
    step = 0
    for _epoch in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for b in range(steps_per_epoch):
            rows = order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            fraction = min(1.0, step / max(1, total_steps - 1)) if total_steps > 1 else 1.0
            with torch.no_grad():
                das.temperature.fill_(float(hi + (lo - hi) * fraction))
            y_hat = resume(model, intervene(das.matrix(), das.soft_masks(), train["hb"][rows], train["hs"][rows],
                                            train["var_idx"][rows]), layer)
            loss = ((y_hat - train["y"][rows]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            das.project_()
            step += 1
            if step in read_at:
                scores = evaluate(model, layer, das.matrix(), das.hard_masks(), val, m)
                lower, upper = das.snapped()
                history.append({"step": step, "train_mse": float(loss.detach()), "val_macro": scores["macro"],
                                "val_per_variable": scores["per_variable"], "dims": (upper - lower).tolist()})
                if scores["macro"] > best["macro"]:
                    best = {"macro": scores["macro"], "step": step, "W": das.matrix().detach().clone(),
                            "widths": das.widths_raw.detach().clone()}
    with torch.no_grad():
        das.widths_raw.copy_(best["widths"])
    lower, upper = das.snapped()
    return {"W": best["W"], "lower": lower, "upper": upper, "dims": (upper - lower).tolist(),
            "best_val_macro": best["macro"], "best_step": best["step"], "history": history, "m": m, "dim": dim,
            "total_steps": total_steps}


def fit_layer(model: nn.Module, target, layer: int, seed: int, data: dict | None = None) -> dict:
    """Fit one restart at one hidden layer and read validation and test once.

    ``data`` holds the counterfactual pairs of :func:`pools`; they are drawn here when it is None.
    """
    data = data or pools(target)
    with torch.no_grad():
        X = torch.from_numpy(data["das_val"]["Xb"][:FIDELITY_ROWS])
        fidelity = float((resume(model, model.hidden_states(X)[layer - 1], layer) - model(X)).abs().max())
    if fidelity > FIDELITY_TOLERANCE:
        raise RuntimeError(f"the forward pass from layer {layer} differs from the model output by {fidelity:.3e}")
    splits = {name: _states(model, pool, layer) for name, pool in data.items()}
    m = len(data["das_train"]["roster"])
    result = fit(model, layer, splits["das_train"], splits["das_val"], seed)
    masks = hard_masks(result["lower"], result["upper"], result["dim"])
    test = evaluate(model, layer, result["W"], masks, splits["das_test"], m)
    val = evaluate(model, layer, result["W"], masks, splits["das_val"], m)
    no_edit = evaluate(model, layer, result["W"], torch.zeros_like(masks), splits["das_test"], m)
    return {"layer": int(layer), "seed": int(seed), "variables": [e["var"] for e in data["das_test"]["roster"]],
            "dims": result["dims"], "residual_dim": result["dim"] - int(sum(result["dims"])),
            "best_step": result["best_step"], "best_val_macro": result["best_val_macro"],
            "val_macro": val["macro"], "val_per_variable": val["per_variable"],
            "test_macro": test["macro"], "test_per_variable": test["per_variable"],
            "no_edit_macro": no_edit["macro"], "no_edit_per_variable": no_edit["per_variable"],
            "fidelity": fidelity, "history": result["history"], "W": result["W"],
            "lower": result["lower"], "upper": result["upper"]}


def select(fits: list[dict]) -> dict:
    """The reported score from the fits of every (layer, restart) of one network.

    Per layer the restart with the best validation score is kept. Per variable
    the layer with the best validation ``R^2`` is selected (the earlier layer
    on a tie). ``score`` is the mean test ``R^2`` over the variables.
    """
    by_layer: dict[int, dict] = {}
    for f in sorted(fits, key=lambda f: (f["layer"], f["seed"])):
        kept = by_layer.get(f["layer"])
        if kept is None or f["best_val_macro"] > kept["best_val_macro"]:
            by_layer[f["layer"]] = f
    names = fits[0]["variables"]
    chosen = []
    for i, var in enumerate(names):
        best = None
        for layer in sorted(by_layer):
            f = by_layer[layer]
            if best is None or f["val_per_variable"][i] > best["val_r2"]:
                best = {"var": var, "layer": layer, "restart": f["seed"], "val_r2": f["val_per_variable"][i],
                        "test_r2": f["test_per_variable"][i], "test_r2_no_edit": f["no_edit_per_variable"][i],
                        "dim": int(f["dims"][i])}
        chosen.append(best)
    return {"variables": chosen, "score": float(np.mean([c["test_r2"] for c in chosen])),
            "score_no_edit": float(np.mean([c["test_r2_no_edit"] for c in chosen])),
            "layers": sorted(by_layer), "restarts": {layer: by_layer[layer]["seed"] for layer in by_layer}}


def score(model: nn.Module, target, layers=LAYERS, restarts: int = RESTARTS) -> dict:
    """The complete protocol for one trained network: every layer, every restart, the selection."""
    data = pools(target)
    fits = [fit_layer(model, target, layer, seed, data) for layer in layers for seed in range(restarts)]
    out = select(fits)
    out["fits"] = [{k: v for k, v in f.items() if k not in ("W", "lower", "upper", "history")} for f in fits]
    return out
