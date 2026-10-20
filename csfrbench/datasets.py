"""Datasets of a target instance.

Every target instance has six pools. The training pool and the validation pool
carry observation noise, ``Y = f*(X) + eps`` with ``eps ~ N(0, sigma^2)``. The
other pools hold the noise-free target and serve as fixed evaluation grids.

=================  =======  ==========  ====================================
pool               size     noise       use
=================  =======  ==========  ====================================
``train``          n_max    yes         the first ``n`` rows are the training set of size ``n``
``val``            4,096    yes         checkpoint and configuration selection
``test``           65,536   no          the reported test NMSE
``test_audit``     65,536   no          a second, independent test pool
``dev``            65,536   no          method development
``diagnostic``     8,192    no          linear probes
=================  =======  ==========  ====================================

Each pool has its own seed, derived from the target fingerprint. The inputs and
the noise of a pool use two independent streams, so the training set of size
``n`` is a prefix of the training set of every larger size.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .seeds import derive
from .targets import Target, sample_inputs

VAL_SIZE = 4096
TEST_SIZE = 65536
DIAGNOSTIC_SIZE = 8192
POOLS = ("train", "val", "dev", "test", "test_audit", "diagnostic")


def pool_seeds(target: Target, dataset_seed: int) -> dict[str, int]:
    """The seed of every pool. Training and validation follow ``dataset_seed``."""
    fp = target.fingerprint
    return {
        "train": int(derive(fp, dataset_seed, "training")),
        "val": int(derive(fp, dataset_seed, "validation")),
        "dev": int(derive(fp, "fixed_development_test")),
        "test": int(derive(fp, "fixed_final_test")),
        "test_audit": int(derive(fp, "fixed_final_test_audit")),
        "diagnostic": int(derive(fp, "fixed_diagnostic")),
    }


def draw_pool(target: Target, n: int, seed: int, *, noisy: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(X, y, f)`` of one pool as float32 tensors; ``y = f`` for a noise-free pool."""
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive; received {n}")
    feature_seed, noise_seed = np.random.SeedSequence(int(seed)).spawn(2)
    X = sample_inputs(np.random.default_rng(feature_seed), (n, target.d))
    f = target.evaluate(X)
    if noisy:
        noise = np.random.default_rng(noise_seed).standard_normal(n).astype(np.float32)
        y = f + np.float32(np.sqrt(target.sigma_sq)) * noise
    else:
        y = f.copy()
    if not np.all(np.isfinite(y)):
        raise FloatingPointError("the generated targets hold non-finite values")
    return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in (X, y, f))


def make_dataset(target: Target, n_max: int, *, dataset_seed: int | None = None,
                 pools: tuple[str, ...] = ("train", "val", "test", "diagnostic")) -> dict[str, Any]:
    """Draw the requested pools of a target instance.

    ``dataset_seed`` defaults to the instance seed, as in the paper's experiments.
    The result maps ``X_<pool>``, ``y_<pool>`` (noisy pools) and ``f_<pool>`` to
    tensors, and ``meta`` to the identity of the dataset.
    """
    dataset_seed = int(target.function_seed if dataset_seed is None else dataset_seed)
    unknown = set(pools) - set(POOLS)
    if unknown:
        raise ValueError(f"unknown pools: {sorted(unknown)}")
    seeds = pool_seeds(target, dataset_seed)
    sizes = {"train": int(n_max), "val": VAL_SIZE, "dev": TEST_SIZE, "test": TEST_SIZE,
             "test_audit": TEST_SIZE, "diagnostic": DIAGNOSTIC_SIZE}
    out: dict[str, Any] = {}
    for pool in pools:
        noisy = pool in ("train", "val")
        X, y, f = draw_pool(target, sizes[pool], seeds[pool], noisy=noisy)
        out[f"X_{pool}"] = X
        out[f"f_{pool}"] = f
        if noisy:
            out[f"y_{pool}"] = y
    out["meta"] = {
        "target_id": target.target_id, "fingerprint": target.fingerprint, "family": target.family,
        "d": target.d, "function_seed": target.function_seed, "dataset_seed": dataset_seed,
        "sigma_sq": target.sigma_sq, "n_max": int(n_max), "pools": list(pools),
        "seeds": {p: seeds[p] for p in pools}, "sizes": {p: sizes[p] for p in pools},
    }
    return out


def tensor_sha256(tensor: torch.Tensor) -> str:
    """SHA-256 of the bytes of a contiguous CPU tensor."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def dataset_hashes(dataset: dict[str, Any]) -> dict[str, str]:
    return {k: tensor_sha256(v) for k, v in dataset.items() if torch.is_tensor(v)}


def save_dataset(dataset: dict[str, Any], directory: str | Path) -> Path:
    """Write ``tensors.pt`` and ``manifest.json`` (shapes and SHA-256 of every tensor)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {k: v for k, v in dataset.items() if torch.is_tensor(v)}
    temporary = directory / "tensors.pt.tmp"
    torch.save(tensors, temporary)
    os.replace(temporary, directory / "tensors.pt")
    manifest = dict(dataset["meta"])
    manifest["tensors"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype).removeprefix("torch."),
                               "sha256": tensor_sha256(v)} for k, v in tensors.items()}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return directory


def load_dataset(directory: str | Path, *, verify: bool = True) -> dict[str, Any]:
    """Read a saved dataset; with ``verify`` the SHA-256 of every tensor is checked."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    tensors = torch.load(directory / "tensors.pt", map_location="cpu", weights_only=True)
    for key, record in manifest["tensors"].items():
        if key not in tensors:
            raise ValueError(f"{directory}: tensor {key} is missing")
        if verify and tensor_sha256(tensors[key]) != record["sha256"]:
            raise RuntimeError(f"{directory}: tensor {key} does not match its recorded hash")
    out: dict[str, Any] = dict(tensors)
    out["meta"] = {k: v for k, v in manifest.items() if k != "tensors"}
    return out
