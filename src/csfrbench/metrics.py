"""Predictive, structural, and inferential metrics.

The PART1 block and the functions under "frozen protocol, part 1" implement the
frozen first part of the CSFRBench evaluation protocol — R(n), AULC, and the
Subfunction Accessibility Matrix A — as specified in
``studies/csfrbench/evaluation-protocol/protocol-freeze/report/evaluation-protocol-part1.md``.
Changing any PART1 value re-opens the protocol.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import t as _student_t
from scipy.stats import wilcoxon

from .utils.seeds import derive

PART1 = {
    # Probe data: the target's fixed diagnostic pool, one split for every
    # method, checkpoint, and seed of the same target instance. Pool size from
    # pilot P3: 8192 halves the between-pool SD of probe R^2 vs 4096 (median
    # 0.025 -> 0.014; width-256 0.030 -> 0.016); 16384 adds little.
    "probe_pool_n": 8192,
    "probe_fractions": (0.5, 0.25, 0.25),          # train / val / test
    # Ridge family: 2 points per decade over the active regime (pilot P1: the
    # selected-lambda p99 is 10^3.5 and the widest informative-cell deficit vs
    # a 35-point reference grid is 0.011; endpoints never bind on informative
    # cells). With z-scored features, lambda <= 1e-4 is numerically OLS and
    # lambda >= 1e7 is inert at probe-train 2048 x width 256.
    "ridge_lambdas": tuple(float(x) for x in np.logspace(-4.0, 7.0, 23)),
    "ridge_selection": "per (node, layer) argmax of probe-val R^2; ties -> smallest lambda",
    # Feature support (amendment 14): a unit active on fewer than this many
    # probe-train points is excluded before z-scoring. A coefficient fitted
    # from a handful of points cannot be validated on a split that contains
    # none of them, and z-scoring such a unit by its near-zero train std
    # sends its rare test activations to z ~ 1e4..1e7 (08-main probes: 55 of
    # 62,720 cells read R^2 -50..-70 from one test point each).
    "probe_min_support": 10,
    # Layer rule: h(0) = raw input; h(l) = post-activation output of hidden
    # block l, eval mode. A future residual-stream architecture must expose the
    # residual stream after block l as its layer-l representation.
    "layer_rule": "input+postactivation",
    # Probe-R^2 standard error: bootstrap over probe-test items at the frozen
    # split and fit. Resplit-refit SEs are not used: no unbiased resplit
    # variance estimator exists (Bengio & Grandvalet 2004), and pilot P3
    # measured resplit means blowing up 18x on degenerate cells.
    "probe_se_bootstrap_B": 1000,
    # Instance-level inference: percentile bootstrap over paired per-instance values.
    "bootstrap_B": 10000,
}


def nmse(model, loader) -> float:
    """Return MSE / Var[y] over a ``DataLoader``, in one streaming pass."""
    try:
        model_device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model must contain at least one parameter") from exc

    was_training = model.training
    model.eval()
    squared_error_sum = torch.zeros((), dtype=torch.float64, device=model_device)
    target_sum = torch.zeros((), dtype=torch.float64, device=model_device)
    target_square_sum = torch.zeros((), dtype=torch.float64, device=model_device)
    num_targets = 0

    try:
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(model_device, non_blocking=True)
                y_batch = y_batch.to(model_device, non_blocking=True)
                prediction = model(X_batch)

                if prediction.shape != y_batch.shape:
                    raise ValueError(
                        f"prediction shape {tuple(prediction.shape)} does not "
                        f"match target {tuple(y_batch.shape)}"
                    )

                targets = y_batch.to(torch.float64)
                squared_error_sum += (prediction.to(torch.float64) - targets).square().sum()
                target_sum += targets.sum()
                target_square_sum += targets.square().sum()
                num_targets += y_batch.numel()
    finally:
        model.train(was_training)

    if num_targets == 0:
        raise ValueError("evaluation data must be nonempty")

    mse = (squared_error_sum / num_targets).item()
    mean = (target_sum / num_targets).item()
    variance = max(0.0, (target_square_sum / num_targets).item() - mean * mean)
    if not np.isfinite(mse) or not np.isfinite(variance):
        raise FloatingPointError("NMSE components are nonfinite")
    return mse / (variance + 1e-12)


# ----------------------------------------------- frozen protocol, part 1 --
# Task performance: R(n) is nmse() per (instance, n) on the target's fixed
# noise-free test pool at the best-validation checkpoint; AULC summarizes it.

def aulc(ns, nmse_values) -> float:
    """Normalized area under the R(n) curve over log sample size.

    Trapezoid of R(n) against log n divided by (log n_max - log n_min): the
    average curve height over the declared n window, invariant to the log
    base. A constant curve R(n) = c has AULC = c.
    """
    x = np.log2(np.asarray(ns, dtype=float))
    y = np.asarray(nmse_values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("ns and nmse_values must be matching one-dimensional arrays of length >= 2")
    if np.any(x[1:] <= x[:-1]):
        raise ValueError("ns must be strictly increasing")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("AULC inputs must be finite")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def teacher_nodes(target) -> dict:
    """The frozen structural node view of one graph target.

    Rows of A are ALL of the teacher's local-function nodes, the pre-readout
    root included (benchmark rule, amendment 11: the primary
    depth-4 tree templates (three nonlinear levels)
    expose seven rows, never six). Under a single-node readout the root row
    scores linear decodability of the final computation itself - disclosed as
    such, not as recovery of a hidden intermediate. An additive readout is a
    fixed linear map, so every law node is internal either way.
    """
    graph = getattr(target, "graph", None)
    if graph is None:
        raise ValueError(f"target family {target.family!r} exposes no GraphSpec")

    internal = list(range(graph.S))

    depths = _node_depths(graph.node_parents, graph.A)
    position = {v: i for i, v in enumerate(internal)}
    edges = []
    for v, refs in enumerate(graph.node_parents):
        if v not in position:
            continue
        for r in refs:
            u = r - graph.A
            if r >= graph.A and u in position:
                edges.append((position[u], position[v]))
    return {
        "indices": internal,
        "names": [f"z{v}(d{depths[v]})" for v in internal],
        "depths": [depths[v] for v in internal],
        "edges": edges,
    }


def _node_depths(node_parents, num_leaves: int) -> list[int]:
    depths = []
    for refs in node_parents:
        parent_depths = [depths[r - num_leaves] for r in refs if r >= num_leaves]
        depths.append(1 + max(parent_depths, default=0))
    return depths


def teacher_node_values(target, X: np.ndarray, indices) -> np.ndarray:
    """Noise-free teacher node values z_v(x), one column per node in ``indices``."""
    values = target.forward(np.asarray(X, dtype=np.float32))["values"]
    if not list(indices):
        return np.zeros((np.asarray(X).shape[0], 0))
    return np.stack([np.asarray(values[v], dtype=np.float64) for v in indices], axis=1)


def layer_representations(model, X: torch.Tensor) -> list[tuple[str, np.ndarray]]:
    """The frozen layer view h(l), l = 0..L, in forward order, eval mode.

    h(0) is the raw input. For the ReLU MLP family h(l) is the post-activation
    output of hidden block l (gated layers contribute their deterministic
    eval-mode forward). For the oracle graph network, node outputs grouped by
    teacher depth stand in for hidden layers. A future residual-stream
    architecture must expose the residual stream after block l here.
    """
    layers = [("h0", X.detach().cpu().numpy().astype(np.float64))]
    model.eval()
    with torch.no_grad():
        if hasattr(model, "node_outputs"):           # oracle graph network
            values = model.node_outputs(X).cpu().numpy().astype(np.float64)
            depths = _node_depths(model.node_parents, model.num_leaves)
            for level in sorted(set(depths)):
                columns = [i for i, d in enumerate(depths) if d == level]
                layers.append((f"h{level}", values[:, columns]))
            return layers
        if hasattr(model, "hidden_layers"):          # ReLU MLP family
            h = X[:, model.bimt_in_perm] if hasattr(model, "bimt_in_perm") else X   # BIMT's forward relocates the inputs first (methods/bimt.py)
            for index, layer in enumerate(model.hidden_layers, start=1):
                h = torch.relu(layer(h))
                layers.append((f"h{index}", h.cpu().numpy().astype(np.float64)))
            return layers
    raise ValueError(f"no layer-representation rule for {type(model).__name__}")


def probe_split_indices(fingerprint: str, n: int):
    """Deterministic probe train/val/test split, shared across methods."""
    fractions = PART1["probe_fractions"]
    rng = np.random.default_rng(derive(fingerprint, "probe_split"))
    order = rng.permutation(n)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def probe_feature_support(H_train, min_support: int | None = None) -> np.ndarray:
    """Boolean mask of the features the probe may use: active (nonzero) on at
    least ``PART1['probe_min_support']`` probe-train points (amendment 14).
    Dense representations (the raw input) keep every feature."""
    m = PART1["probe_min_support"] if min_support is None else int(min_support)
    return (np.asarray(H_train) != 0).sum(axis=0) >= m


def probe_features(H_by_split):
    """Frozen feature preparation shared by every probe: drop features below
    the support floor, then z-score on probe-train moments."""
    H_train, H_val, H_test = (np.asarray(H, dtype=np.float64) for H in H_by_split)
    keep = probe_feature_support(H_train)
    H_train, H_val, H_test = H_train[:, keep], H_val[:, keep], H_test[:, keep]
    mean = H_train.mean(axis=0)
    scale = H_train.std(axis=0) + 1e-8
    return tuple((H - mean) / scale for H in (H_train, H_val, H_test)), keep


def ridge_r2_path(H_by_split, Z_by_split, lambdas=None):
    """Val and test R^2 along the ridge path, for every probe target.

    ``H_by_split`` and ``Z_by_split`` are (train, val, test) tuples. Features
    below the support floor are dropped (amendment 14), the rest are z-scored
    and targets centered on probe-train moments. Returns (val_r2, test_r2),
    each of shape (len(lambdas), num_targets). R^2 is the ordinary held-out
    coefficient of determination against the evaluation split's own mean, and
    is never clipped: negative values mean the decoder generalizes worse than
    the constant predictor.
    """
    lambdas = PART1["ridge_lambdas"] if lambdas is None else tuple(lambdas)
    Z_train, Z_val, Z_test = (np.asarray(Z, dtype=np.float64) for Z in Z_by_split)
    num_targets = Z_train.shape[1]
    if num_targets == 0:
        return np.zeros((len(lambdas), 0)), np.zeros((len(lambdas), 0))

    (Ht, Hv, He), _keep = probe_features(H_by_split)
    z_mean = Z_train.mean(axis=0)

    U, s, Vt = np.linalg.svd(Ht, full_matrices=False)
    UtZ = U.T @ (Z_train - z_mean)

    var_val = Z_val.var(axis=0)
    var_test = Z_test.var(axis=0)
    val_r2 = np.empty((len(lambdas), num_targets))
    test_r2 = np.empty((len(lambdas), num_targets))
    for row, lam in enumerate(lambdas):
        beta = (Vt.T * (s / (s * s + lam))) @ UtZ
        val_pred = Hv @ beta + z_mean
        test_pred = He @ beta + z_mean
        val_r2[row] = 1.0 - ((Z_val - val_pred) ** 2).mean(axis=0) / (var_val + 1e-12)
        test_r2[row] = 1.0 - ((Z_test - test_pred) ** 2).mean(axis=0) / (var_test + 1e-12)
    return val_r2, test_r2


def ridge_probe_r2(H_by_split, Z_by_split, lambdas=None):
    """Held-out probe R^2 with the frozen selection rule.

    The ridge strength is selected per probe target on probe-val R^2 (ties
    resolve to the smallest lambda); the reported number is probe-test R^2 at
    the selected lambda. Returns (test_r2, selected_lambda), each (num_targets,).
    """
    lambdas = PART1["ridge_lambdas"] if lambdas is None else tuple(lambdas)
    val_r2, test_r2 = ridge_r2_path(H_by_split, Z_by_split, lambdas)
    if val_r2.shape[1] == 0:
        return np.zeros(0), np.zeros(0)
    best = val_r2.argmax(axis=0)                    # first max -> smallest lambda
    columns = np.arange(val_r2.shape[1])
    return test_r2[best, columns], np.asarray(lambdas, dtype=float)[best]


def ridge_test_predictions(H_by_split, Z_by_split, lambdas=None) -> np.ndarray:
    """Probe-test predictions at the frozen selection rule, per probe target."""
    lambdas = PART1["ridge_lambdas"] if lambdas is None else tuple(lambdas)
    Z_train, Z_val, Z_test = (np.asarray(Z, dtype=np.float64) for Z in Z_by_split)
    (Ht, Hv, He), _keep = probe_features(H_by_split)
    z_mean = Z_train.mean(axis=0)
    U, s, Vt = np.linalg.svd(Ht, full_matrices=False)
    UtZ = U.T @ (Z_train - z_mean)
    var_val = Z_val.var(axis=0)
    best_val = np.full(Z_train.shape[1], -np.inf)
    preds = np.zeros_like(Z_test)
    for lam in lambdas:
        beta = (Vt.T * (s / (s * s + lam))) @ UtZ
        val_pred = Hv @ beta + z_mean
        val_r2 = 1.0 - ((Z_val - val_pred) ** 2).mean(axis=0) / (var_val + 1e-12)
        better = val_r2 > best_val
        best_val = np.where(better, val_r2, best_val)
        preds[:, better] = (He @ beta + z_mean)[:, better]
    return preds


def probe_r2(model, target, X: torch.Tensor) -> dict:
    """The subfunction probe R^2 matrix for one model state (draft notation
    R^2_{v,l}).

    matrix[v, l] is held-out probe R^2 for teacher internal node v read from
    h(l), l = 0..L with l = 0 the raw-input row. Empty when the target has no
    internal nodes.
    """
    view = teacher_nodes(target)
    Z = teacher_node_values(target, X.cpu().numpy(), view["indices"])
    layers = layer_representations(model, X)
    train_idx, val_idx, test_idx = probe_split_indices(target.fingerprint, X.shape[0])

    A = np.zeros((len(view["indices"]), len(layers)))
    lambdas = np.zeros_like(A)
    kept = []
    for column, (_name, H) in enumerate(layers):
        A[:, column], lambdas[:, column] = ridge_probe_r2(
            (H[train_idx], H[val_idx], H[test_idx]),
            (Z[train_idx], Z[val_idx], Z[test_idx]),
        )
        kept.append(int(probe_feature_support(H[train_idx]).sum()))
    return {
        "matrix": A,
        "selected_lambdas": lambdas,
        "nodes": view["names"],
        "node_depths": view["depths"],
        "layers": [name for name, _H in layers],
        "features_kept": kept,
        "features_total": [int(H.shape[1]) for _name, H in layers],
        "edges": view["edges"],
    }


# Pre-rename alias (the frozen metric-validation studies call it by this name;
# draft.pdf's canonical term is the probe R^2 matrix).
accessibility = probe_r2


def probe_se(model, target, X: torch.Tensor, *, B: int | None = None) -> np.ndarray:
    """Per-cell probe-R^2 SE via bootstrap over probe-test items, shaped like
    the probe R^2 matrix.

    The probes stay fixed at the frozen split and selection; only the test
    items resample, so the estimate carries the test-sampling component of the
    R^2 uncertainty (the resplit alternative has no unbiased variance and
    degenerates on pathological cells; pilot P3). Bootstrap seeds derive from
    the target fingerprint: identical across methods.
    """
    B = PART1["probe_se_bootstrap_B"] if B is None else int(B)
    view = teacher_nodes(target)
    Z = teacher_node_values(target, X.cpu().numpy(), view["indices"])
    layers = layer_representations(model, X)
    tr, va, te = probe_split_indices(target.fingerprint, X.shape[0])

    se = np.zeros((len(view["indices"]), len(layers)))
    for column, (_name, H) in enumerate(layers):
        preds = ridge_test_predictions((H[tr], H[va], H[te]),
                                       (Z[tr], Z[va], Z[te]))
        rng = np.random.default_rng(derive(target.fingerprint, column, "probe_se"))
        idx = rng.integers(0, len(te), size=(B, len(te)))
        r2 = np.zeros((B, Z.shape[1]))
        for b in range(B):
            zt, pt = Z[te][idx[b]], preds[idx[b]]
            r2[b] = 1.0 - ((zt - pt) ** 2).mean(axis=0) / (zt.var(axis=0) + 1e-12)
        se[:, column] = r2.std(axis=0)
    return se


def instance_t_interval(values, *, level: float = 0.95) -> dict:
    """Student-t CI over target instances — the v0.5 reporting interval.

    For AULCs x_1..x_M on independently sampled target instances, returns
    mean(x) +- t_{(1+level)/2, M-1} * s / sqrt(M). For a comparison, pass the
    within-instance differences d_j = AULC_A - AULC_B: the same interval then
    describes the paired effect, and pairs are never broken by handling the
    arms separately.

    This is an interval for an effect estimate. It carries no null-hypothesis
    test and no p-value.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    m = int(v.size)
    if m == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")],
                "half_width": float("nan"), "sd": float("nan"),
                "n_instances": 0, "level": level}
    mean = float(v.mean())
    if m == 1:
        return {"mean": mean, "ci95": [float("nan"), float("nan")],
                "half_width": float("nan"), "sd": float("nan"),
                "n_instances": 1, "level": level}
    sd = float(v.std(ddof=1))
    half = float(_student_t.ppf(0.5 * (1.0 + level), m - 1) * sd / np.sqrt(m))
    return {"mean": mean, "ci95": [mean - half, mean + half],
            "half_width": half, "sd": sd, "n_instances": m, "level": level}


def instance_distribution_shape(values) -> dict:
    """Skew and outlier-domination diagnostic for a set of instance values.

    A sanity check on the target-level AULC distribution (or on paired
    differences), not an acceptance gate: a Student-t interval on a badly
    skewed or single-instance-dominated sample still computes, and this is
    what says so.

    ``top_share`` is the fraction of the total absolute deviation from the
    mean contributed by the single most extreme instance; near 1 means one
    draw is carrying the estimate.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    m = int(v.size)
    if m < 2:
        return {"n_instances": m, "skew": float("nan"),
                "excess_kurtosis": float("nan"), "max_abs_z": float("nan"),
                "top_share": float("nan")}
    dev = v - v.mean()
    sd = v.std(ddof=1)
    total = float(np.abs(dev).sum())
    scaled = dev / sd if sd > 0 else np.zeros_like(dev)
    return {"n_instances": m,
            "skew": float((scaled ** 3).mean()),
            "excess_kurtosis": float((scaled ** 4).mean() - 3.0),
            "max_abs_z": float(np.abs(scaled).max()),
            "top_share": float(np.abs(dev).max() / total) if total > 0 else 0.0}


def paired_instance_bootstrap(values, *, B: int | None = None, seed: int = 0) -> dict:
    """Percentile bootstrap CI over target instances.

    Retained as an optional diagnostic. v0.5 reports Student-t intervals
    (``instance_t_interval``); this does not define v0.5 reporting.

    ``values``: one number per instance — for paired comparisons (trained vs
    init, method vs baseline) pass the within-instance differences, never
    separately resampled arms.
    """
    B = PART1["bootstrap_B"] if B is None else int(B)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")],
                "n_instances": 0, "B": B}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(B, v.size))
    means = v[idx].mean(axis=1)
    return {"mean": float(v.mean()),
            "ci95": [float(np.quantile(means, 0.025)),
                     float(np.quantile(means, 0.975))],
            "n_instances": int(v.size), "B": B}


# ------------------------------------------------------ weight diagnostics --

def effective_fanin(W):
    """Return participation-ratio fan-in for each output unit."""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or not np.all(np.isfinite(W)):
        raise ValueError("W must be a finite two-dimensional matrix")
    numerator = np.abs(W).sum(axis=1) ** 2
    denominator = np.square(W).sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def structure(method, model, active_tol: float = 1e-8) -> dict[str, float]:
    """Summarize first-layer fan-in among active units and report activity."""
    if active_tol < 0:
        raise ValueError("active_tol must be nonnegative")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            weights = tuple(method.effective_weights(model))
            if not weights:
                raise ValueError("effective_weights returned no matrices")
            first = weights[0]
            if not torch.is_tensor(first):
                raise TypeError("effective_weights must return tensors")
            W = first.detach().cpu().numpy()
    finally:
        model.train(was_training)

    row_norms = np.linalg.norm(np.asarray(W, dtype=np.float64), axis=1)
    active = row_norms > active_tol
    active_fraction = float(np.mean(active))
    # Effective, not raw: comparable across gated and ungated arms, so a small gate
    # cannot be read as shrinkage without checking what the weights did in response.
    absmean = float(np.abs(np.asarray(W, dtype=np.float64)).mean())
    if not np.any(active):
        return {
            "first_layer_active_fraction": active_fraction,
            "first_layer_efi_median": 0.0,
            "first_layer_efi_p90": 0.0,
            "first_layer_weight_absmean": absmean,
        }
    efi = effective_fanin(W)[active]
    return {
        "first_layer_active_fraction": active_fraction,
        "first_layer_efi_median": float(np.median(efi)),
        "first_layer_efi_p90": float(np.quantile(efi, 0.90)),
        "first_layer_weight_absmean": absmean,
    }


# ---------------------------------------------------------------- inference --

def holm(pvals) -> np.ndarray:
    """Compute Holm-Bonferroni adjusted p-values in original order."""
    try:
        p = np.asarray(pvals, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("pvals must contain numeric values") from exc
    if p.ndim != 1 or not np.all(np.isfinite(p)):
        raise ValueError("pvals must be a finite one-dimensional array")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("all p-values must lie in [0, 1]")
    if p.size == 0:
        return np.empty(0, dtype=float)

    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (p.size - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def wilcoxon_signed_rank_less_pvalue(a, b) -> float:
    """One-sided paired Wilcoxon p-value for ``a`` tending smaller than ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("a and b must be nonempty matching one-dimensional arrays")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("a and b must be finite")
    differences = a - b
    if np.all(differences == 0.0):
        return 1.0
    return float(
        wilcoxon(differences, alternative="less", zero_method="wilcox", method="auto").pvalue
    )
