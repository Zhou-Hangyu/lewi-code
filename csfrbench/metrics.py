"""Prediction metrics and statistics over target instances."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import t as _student_t

_trapezoid = getattr(np, "trapezoid", None) or np.trapz      # numpy 2 / numpy 1


def nmse(model, loader) -> float:
    """``MSE / Var[y]`` over a data loader, accumulated in float64."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    squared_error = torch.zeros((), dtype=torch.float64, device=device)
    target_sum = torch.zeros((), dtype=torch.float64, device=device)
    target_square_sum = torch.zeros((), dtype=torch.float64, device=device)
    count = 0
    try:
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                prediction = model(X)
                if prediction.shape != y.shape:
                    raise ValueError(f"prediction shape {tuple(prediction.shape)} != target {tuple(y.shape)}")
                targets = y.to(torch.float64)
                squared_error += (prediction.to(torch.float64) - targets).square().sum()
                target_sum += targets.sum()
                target_square_sum += targets.square().sum()
                count += y.numel()
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("the evaluation data must be non-empty")
    mse = (squared_error / count).item()
    mean = (target_sum / count).item()
    variance = max(0.0, (target_square_sum / count).item() - mean * mean)
    if not np.isfinite(mse) or not np.isfinite(variance):
        raise FloatingPointError("the NMSE components are not finite")
    return mse / (variance + 1e-12)


def pointwise_squared_errors(model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Squared errors and targets of every point, in the order of the loader, as float64."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    errors, targets = [], []
    try:
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                prediction = model(X)
                if prediction.shape != y.shape:
                    raise ValueError(f"prediction shape {tuple(prediction.shape)} != target {tuple(y.shape)}")
                y64 = y.to(torch.float64)
                errors.append((prediction.to(torch.float64) - y64).square().reshape(-1).cpu().numpy())
                targets.append(y64.reshape(-1).cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(errors), np.concatenate(targets)


def nmse_from_errors(errors: np.ndarray, targets: np.ndarray) -> float:
    """``sum of squared errors / sum of squared deviations of the targets from their mean``."""
    denominator = targets @ targets - targets.sum() ** 2 / targets.size
    return float(errors.sum() / denominator)


def aulc(ns, nmse_values) -> float:
    """Area under the learning curve on a logarithmic training-set-size axis.

    The trapezoid of NMSE over ``log2 n`` divided by ``log2 n_K - log2 n_1``: the
    mean height of the curve over the grid. Lower is better.
    """
    x = np.log2(np.asarray(ns, dtype=float))
    y = np.asarray(nmse_values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("ns and nmse_values must be one-dimensional arrays of the same length >= 2")
    if np.any(x[1:] <= x[:-1]):
        raise ValueError("ns must increase strictly")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("the AULC inputs must be finite")
    return float(_trapezoid(y, x) / (x[-1] - x[0]))


def t_interval(values, *, level: float = 0.95) -> dict:
    """Mean and two-sided Student-t interval over target instances.

    For a comparison of two methods, pass the paired differences per instance.
    The interval is an uncertainty estimate of an effect; it is not a test.
    """
    v = np.asarray(list(values), dtype=float)
    n = int(v.size)
    mean = float(v.mean()) if n else float("nan")
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    half = float(_student_t.ppf(0.5 * (1.0 + level), n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "half_width": half, "sd": sd, "n": n,
            "median": float(np.median(v)) if n else float("nan"),
            "n_negative": int((v < 0).sum()), "n_positive": int((v > 0).sum())}
