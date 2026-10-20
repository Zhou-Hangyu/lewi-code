"""Joint Boundless DAS for CSFR learners (Wu et al. 2023, arXiv:2305.08809).

Faithful to the released pyvene parameterization (sigmoid_boundary, learned
boundaries in coordinate-index units, temperature annealed linearly, snap to
binary masks at evaluation), generalized from the released single block to m
ordered per-variable blocks with independently learned widths that may snap
to exactly zero dimensions (goal.md 2.2). Learner weights stay frozen.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Paper A.2: temperature 50.0 -> 0.10 in coordinate-index units at d = 4096,
# annealed linearly per training step. We preserve the normalized schedule
# (fraction of the axis), so at dimension d the schedule is scaled by d/4096.
TEMP_SCHEDULE_4096 = (50.0, 0.10)
ROTATION_LR = 1e-3
BOUNDARY_LR = 1e-2
BATCH_SIZE = 64
EPOCHS = 3
EVAL_EVERY = 200


def sigmoid_boundary(_input, boundary_x, boundary_y, temperature):
    """Verbatim pyvene models/basic_utils.py (paper Eq. 3)."""
    return torch.sigmoid((_input - boundary_x) / temperature) * torch.sigmoid(
        (boundary_y - _input) / temperature
    )


class JointBoundlessDAS(nn.Module):
    """One orthogonal rotation + m ordered boundary blocks over d coordinates."""

    def __init__(self, dim: int, m: int, das_seed: int, *,
                 positions: str = "half", clamp_lo: float = 0.0,
                 simplex: bool = True, temp_scaled: bool = True):
        """``positions``/``clamp_lo``/``simplex``/``temp_scaled`` name the four
        CSFR departures from the released single-block implementation; the
        defaults are the CSFR protocol and the source-literal setting is
        ``positions="integer", clamp_lo=1e-3, simplex=False,
        temp_scaled=False`` (fb2 section 2.3)."""
        super().__init__()
        torch.manual_seed(int(das_seed))
        rotate = nn.Linear(dim, dim, bias=False)
        self.rotation = nn.utils.parametrizations.orthogonal(rotate)
        # Equal shares over m variables plus the residual; clamped to
        # [clamp_lo, 1] in boundaries() (pyvene clamps to [1e-3, 1]).
        self.widths_raw = nn.Parameter(torch.full((m,), 1.0 / (m + 1)))
        # Half-offset positions: the T -> 0 soft mask limit then equals the
        # round() integer snap exactly, and no coordinate sits at a pinned
        # 0.5 mask when a boundary coincides with a coordinate index.
        offset = 0.5 if positions == "half" else 0.0
        self.register_buffer("positions",
                             torch.arange(dim, dtype=torch.float32) + offset)
        scale = dim / 4096.0 if temp_scaled else 1.0
        self.register_buffer(
            "temperature",
            torch.tensor(TEMP_SCHEDULE_4096[0] * scale, dtype=torch.float32))
        self.dim, self.m = int(dim), int(m)
        self.clamp_lo, self.simplex = float(clamp_lo), bool(simplex)
        self.temp_scale = float(scale)

    def boundaries(self) -> torch.Tensor:
        """Cumulative upper boundaries c_1..c_m, normalized to [0, 1]."""
        widths = self.widths_raw.clamp(self.clamp_lo, 1.0)
        return torch.cumsum(widths, 0).clamp(max=1.0)

    def project_(self) -> None:
        """Keep raw widths inside the region where gradients flow.

        Clamps each width to [0, 1] and rescales the vector when the widths
        sum past 1. Without the rescale one runaway width pushes the cumsum
        chain past 1, the tail variables are crushed against the axis end,
        and the clamp there kills their gradient path -- an absorbing
        degenerate state the paper's 1-2 tied boundaries could never reach.
        """
        with torch.no_grad():
            self.widths_raw.clamp_(self.clamp_lo, 1.0)
            if not self.simplex:
                return
            total = float(self.widths_raw.sum())
            if total > 1.0:
                self.widths_raw.mul_(1.0 / total)

    def soft_masks(self) -> torch.Tensor:
        """(m, d) product-of-sigmoids masks in coordinate-index units."""
        upper = self.boundaries() * self.dim
        lower = torch.cat([upper.new_zeros(1), upper[:-1]])
        return sigmoid_boundary(self.positions[None, :], lower[:, None],
                                upper[:, None], self.temperature)

    def snapped_boundaries(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Integer (lower, upper) per variable; coincident => 0 dimensions."""
        with torch.no_grad():
            upper = (self.boundaries() * self.dim).round().long()
            lower = torch.cat([upper.new_zeros(1), upper[:-1]])
        return lower, upper

    def hard_masks(self) -> torch.Tensor:
        lower, upper = self.snapped_boundaries()
        k = torch.arange(self.dim)
        return ((k[None, :] >= lower[:, None])
                & (k[None, :] < upper[:, None])).float()

    def rotation_matrix(self) -> torch.Tensor:
        """W with rotate(h) = h @ W.T; row k of W is rotated coordinate k."""
        return self.rotation.weight

    def set_temperature(self, value: float) -> None:
        with torch.no_grad():
            self.temperature.fill_(float(value))


def intervene(rotation_W: torch.Tensor, masks: torch.Tensor,
              hb: torch.Tensor, hs: torch.Tensor,
              var_idx: torch.Tensor) -> torch.Tensor:
    """Weighted distributed interchange (paper Eq. 4), one variable per row."""
    M = masks[var_idx]
    rb = hb @ rotation_W.T
    rs = hs @ rotation_W.T
    return ((1.0 - M) * rb + M * rs) @ rotation_W


def layer_states(model: nn.Module, X: torch.Tensor) -> list[torch.Tensor]:
    """Post-activation hidden states, matching ReLUMLP.forward exactly (BIMT's forward first relocates the input
    coordinates by its persistent permutation, methods/bimt.py attach_permutations; the states follow it)."""
    if hasattr(model, "bimt_in_perm"):
        X = X[:, model.bimt_in_perm]
    states, h = [], X
    for layer in model.hidden_layers:
        h = F.relu(layer(h))
        states.append(h)
    return states


def resume_from(model: nn.Module, h: torch.Tensor, layer_index: int) -> torch.Tensor:
    """Finish the forward pass from AFTER hidden layer ``layer_index`` (1-based)."""
    for layer in model.hidden_layers[layer_index:]:
        h = F.relu(layer(h))
    out = model.output_layer(h)
    return out.squeeze(-1)


@torch.no_grad()
def capture_pair(model: nn.Module, Xb: np.ndarray, Xs: np.ndarray,
                 layer_index: int, chunk: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    outs = []
    for X in (Xb, Xs):
        parts = []
        for i in range(0, len(X), chunk):
            parts.append(layer_states(model, torch.from_numpy(X[i:i + chunk]))[layer_index - 1])
        outs.append(torch.cat(parts))
    return outs[0], outs[1]


def per_variable_r2(y_hat: torch.Tensor, y_star: torch.Tensor,
                    var_idx: torch.Tensor, m: int) -> list[float]:
    """CF-R^2 per variable; negatives preserved; NaN only if a variable is absent."""
    out = []
    for v in range(m):
        rows = var_idx == v
        y = y_star[rows]
        sse = float(((y_hat[rows] - y) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum())
        out.append(1.0 - sse / sst if sst > 0 else float("nan"))
    return out


@torch.no_grad()
def evaluate_das(model: nn.Module, layer_index: int, rotation_W: torch.Tensor,
                 masks: torch.Tensor, hb: torch.Tensor, hs: torch.Tensor,
                 var_idx: torch.Tensor, y_star: torch.Tensor, m: int) -> dict:
    y_hat = resume_from(model, intervene(rotation_W, masks, hb, hs, var_idx),
                        layer_index)
    per_var = per_variable_r2(y_hat, y_star, var_idx, m)
    return {"per_var": per_var, "macro": float(np.mean(per_var))}


def fit_das(model: nn.Module, layer_index: int, train: dict, val: dict,
            das_seed: int, *, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
            eval_every: int = EVAL_EVERY,
            temp_index_units: tuple[float, float] | None = None,
            variant: dict | None = None,
            fixed_rank: int | None = None,
            anneal: str = "stretched",
            objective: str = "pooled",
            eval_reads: int | None = None,
            val_pools: dict | None = None,
            rot_lr: float = ROTATION_LR,
            bnd_lr: float = BOUNDARY_LR) -> dict:
    """Fit one joint alignment; select on DAS-val (snapped masks) only.

    ``variant``      constructor overrides naming the four CSFR departures.
    ``fixed_rank``   freeze every variable at exactly r dimensions (fb2 3.3):
                     the boundary stops learning and only the rotation moves.
    ``anneal``       "stretched" (temperature spans the whole run), "hold3"
                     (the 3-epoch trajectory, then hold), "hold23" (reach the
                     final temperature at 2/3 of the run, then hold).
    ``objective``    "pooled" raw MSE, or "varnorm" equal-per-variable
                     variance-normalized MSE (fb2 7.2).
    ``eval_reads``   a fixed number of evenly spaced validation reads instead
                     of the every-``eval_every`` cadence (fb2 4.4).
    ``val_pools``    extra named validation pools scored at every read, so
                     checkpoint selection under a smaller pool can be
                     re-derived offline without refitting (fb2 5.2).
    """
    model.eval()
    model.requires_grad_(False)
    dim = train["hb"].shape[1]
    m = int(max(train["var_idx"].max(), val["var_idx"].max())) + 1

    das = JointBoundlessDAS(dim, m, das_seed, **(variant or {}))
    groups = [{"params": list(das.rotation.parameters()), "lr": rot_lr}]
    if fixed_rank is None:
        groups.append({"params": [das.widths_raw], "lr": bnd_lr})
    else:
        with torch.no_grad():
            das.widths_raw.fill_(float(fixed_rank) / dim)
        das.widths_raw.requires_grad_(False)
    optimizer = torch.optim.Adam(groups)

    n = len(train["y"])
    steps_per_epoch = n // batch_size
    total_steps = epochs * steps_per_epoch
    if temp_index_units is None:
        scale = getattr(das, "temp_scale", dim / 4096.0)
        hi, lo = (t * scale for t in TEMP_SCHEDULE_4096)
    else:
        hi, lo = temp_index_units
    # The anneal horizon is the step at which the final temperature is
    # reached; after it the temperature is held (fb2 4.3).
    if anneal == "stretched":
        horizon = total_steps
    elif anneal == "hold3":
        horizon = 3 * steps_per_epoch
    elif anneal == "hold23":
        horizon = max(1, (2 * total_steps) // 3)
    else:
        raise ValueError(f"unknown anneal schedule {anneal!r}")
    horizon = min(horizon, total_steps)

    weights = None
    if objective == "varnorm":
        w = torch.ones(m)
        for v in range(m):
            rows = train["var_idx"] == v
            if int(rows.sum()) > 1:
                w[v] = 1.0 / float(train["y"][rows].var().clamp_min(1e-12))
        w = w * (m / w.sum())          # mean weight 1: same overall loss scale
        weights = w
    elif objective != "pooled":
        raise ValueError(f"unknown objective {objective!r}")

    if eval_reads is None:
        read_at = {s_ for s_ in range(eval_every, total_steps + 1, eval_every)}
    else:
        read_at = {int(round(total_steps * (i + 1) / eval_reads))
                   for i in range(eval_reads)}
    read_at.add(total_steps)
    generator = torch.Generator().manual_seed(int(das_seed) + 1)

    best = {"macro": -math.inf, "step": -1, "state": None}
    history = []
    step = 0
    for epoch in range(epochs):
        order = torch.randperm(n, generator=generator)
        for b in range(steps_per_epoch):
            rows = order[b * batch_size:(b + 1) * batch_size]
            frac = min(1.0, step / max(1, horizon - 1)) if horizon > 1 else 1.0
            das.set_temperature(hi + (lo - hi) * frac)
            y_hat = resume_from(
                model,
                intervene(das.rotation_matrix(), das.soft_masks(),
                          train["hb"][rows], train["hs"][rows],
                          train["var_idx"][rows]),
                layer_index)
            se = (y_hat - train["y"][rows]) ** 2
            loss = se.mean() if weights is None else (
                se * weights[train["var_idx"][rows]]).mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norms = {
                "rotation": float(torch.cat([p.grad.reshape(-1) for p in
                                             das.rotation.parameters()
                                             if p.grad is not None]).norm())
                if any(p.grad is not None for p in das.rotation.parameters())
                else float("nan"),
                "boundary": float(das.widths_raw.grad.norm())
                if das.widths_raw.grad is not None else float("nan"),
            }
            optimizer.step()
            das.project_()
            step += 1
            if step in read_at:
                scores = evaluate_das(model, layer_index, das.rotation_matrix(),
                                      das.hard_masks(), val["hb"], val["hs"],
                                      val["var_idx"], val["y"], m)
                lower, upper = das.snapped_boundaries()
                entry = {
                    "step": step, "train_mse": float(loss.detach()),
                    "val_macro": scores["macro"], "val_per_var": scores["per_var"],
                    "dims": (upper - lower).tolist(),
                    "residual_dim": dim - int(sum((upper - lower).tolist())),
                    "temperature": float(das.temperature),
                    "grad_norm": grad_norms,
                }
                if val_pools:
                    entry["pools"] = {
                        name: evaluate_das(model, layer_index,
                                           das.rotation_matrix(), das.hard_masks(),
                                           pool["hb"], pool["hs"], pool["var_idx"],
                                           pool["y"], m)["macro"]
                        for name, pool in val_pools.items()}
                history.append(entry)
                if scores["macro"] > best["macro"]:
                    best = {"macro": scores["macro"], "step": step,
                            "state": {
                                "W": das.rotation_matrix().detach().clone(),
                                "widths_raw": das.widths_raw.detach().clone(),
                            }}
    assert best["state"] is not None, "no validation checkpoint was recorded"
    with torch.no_grad():
        das.widths_raw.copy_(best["state"]["widths_raw"])
    lower, upper = das.snapped_boundaries()
    return {
        "W": best["state"]["W"], "widths_raw": best["state"]["widths_raw"],
        "lower": lower, "upper": upper, "dims": (upper - lower).tolist(),
        "best_val_macro": best["macro"], "best_step": best["step"],
        "history": history, "m": m, "dim": dim, "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch, "examples": total_steps * batch_size,
        "anneal": anneal, "anneal_horizon": horizon, "objective": objective,
        "fixed_rank": fixed_rank, "batch_size": batch_size,
        "rot_lr": rot_lr, "bnd_lr": bnd_lr,
        "train_pairs": int(n), "eval_reads": sorted(read_at),
    }


def hard_masks_from(lower: torch.Tensor, upper: torch.Tensor, dim: int) -> torch.Tensor:
    k = torch.arange(dim)
    return ((k[None, :] >= lower[:, None]) & (k[None, :] < upper[:, None])).float()


def haar_orthogonal(dim: int, seed: int) -> torch.Tensor:
    """Haar-distributed orthogonal matrix (QR with sign correction)."""
    generator = torch.Generator().manual_seed(int(seed))
    A = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R))[None, :]
    return Q.to(torch.float32)


def axis_metrics(W: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> dict:
    """Leverage / coherence / effective support / top-r concentration per block,
    pairwise coordinate-energy overlaps, and the orthogonality invariant."""
    d = W.shape[0]
    m = len(lower)
    blocks = [W[int(lo):int(up), :] for lo, up in zip(lower, upper)]  # rows: basis
    leverage = []
    per_var = []
    for i, B in enumerate(blocks):
        r = B.shape[0]
        if r == 0:
            leverage.append(np.zeros(d))
            per_var.append({"r": 0, "coherence": None, "n_eff": None,
                            "c_r": None, "leverage": [0.0] * d})
            continue
        a = (B ** 2).sum(dim=0).cpu().numpy()          # diag(U U^T), sums to r
        mu = float(d / r * a.max())
        n_eff = float(r * r / float((a ** 2).sum()))
        c_r = float(np.sort(a)[-r:].sum() / r)
        leverage.append(a)
        per_var.append({"r": int(r), "coherence": mu, "n_eff": n_eff,
                        "c_r": c_r, "leverage": [float(x) for x in a]})

    coord_overlap = np.full((m, m), np.nan)
    jaccard = np.full((m, m), np.nan)
    intersect = np.full((m, m), np.nan)
    ortho_max = 0.0
    tops = []
    for i, B in enumerate(blocks):
        r = B.shape[0]
        tops.append(set(np.argsort(leverage[i])[-r:].tolist()) if r else set())
    for i in range(m):
        for j in range(m):
            ai, aj = leverage[i], leverage[j]
            ni, nj = np.linalg.norm(ai), np.linalg.norm(aj)
            if ni > 0 and nj > 0:
                coord_overlap[i, j] = float(ai @ aj / (ni * nj))
            if per_var[i]["r"] and per_var[j]["r"]:
                inter = len(tops[i] & tops[j])
                union = len(tops[i] | tops[j])
                jaccard[i, j] = inter / union
                intersect[i, j] = inter
            if i != j and per_var[i]["r"] and per_var[j]["r"]:
                cross = blocks[i] @ blocks[j].T
                ortho_max = max(ortho_max, float(cross.abs().max()))
    reuse = np.zeros(d, dtype=int)
    for t in tops:
        for k in t:
            reuse[k] += 1
    return {"per_var": per_var, "coord_overlap": coord_overlap,
            "jaccard": jaccard, "intersect": intersect,
            "ortho_max_offdiag": ortho_max, "top_sets": [sorted(t) for t in tops],
            "neuron_reuse": reuse.tolist()}
