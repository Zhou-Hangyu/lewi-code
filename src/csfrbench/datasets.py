"""Master-dataset artifacts: one per (target, dataset seed), built once and loaded by
every run. Layout: ``<root>/<target_id>/{target.json, dseed_<dataset_seed>/{tensors.pt,
manifest.json}}``.

Training and validation are finite-data replicates that follow the dataset seed. The
development, test, test-audit and diagnostic pools are fixed per calibrated target and
serve as common Monte-Carlo evaluation grids.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, default_collate

from .utils.seeds import derive
from .tasks.base import sample_features

SCHEMA_VERSION = 2

TENSOR_KEYS = (
    "Xtr_master", "ytr_master", "ftr_master",
    "Xva", "yva", "fva",
    "Xdv", "fdv",
    "Xte", "fte",
    "Xta", "fta",
    "Xdg", "fdg",
)
SPLIT_SEEDS = ("training", "validation", "development_test", "test", "test_audit", "diagnostic")


class TensorPool(TensorDataset):
    """In-memory pool of aligned tensors, fetched a batch at a time."""
    def __getitems__(self, indices):
        """Batched fetch: one fancy-index per minibatch instead of B slices."""
        return tuple(tensor[indices] for tensor in self.tensors)

    def to(self, device) -> "TensorPool":
        """Return the same pool with every tensor moved to ``device``."""
        return TensorPool(*(tensor.to(device) for tensor in self.tensors))


def _collate(batch):
    """Pass through an already-batched fetch; collate ordinary sample lists."""
    return batch if isinstance(batch, tuple) else default_collate(batch)


def make_loader(
    dataset,
    batch_size: int,
    *,
    shuffle: bool = False,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Standard single-process loader; any Dataset works, batched fetch or not."""
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
        num_workers=0,
        collate_fn=_collate,
    )


def iterate_batches(loader: DataLoader):
    """Yield batches forever, reshuffling on each pass."""
    while True:
        yield from loader


def target_id(target) -> str:
    """Directory-safe identity of a calibrated target."""
    digest = target.fingerprint.split("-", 1)[-1][:8]
    return f"{target.family}_d{target.d}_fs{target.function_seed}_{digest}"


def dataset_dir(root: str | Path, target, dataset_seed: Hashable) -> Path:
    """``<root>/<target_id>/dseed_<dataset_seed>``."""
    tag = (f"{int(dataset_seed):04d}" if isinstance(dataset_seed, (int, np.integer))
           else str(dataset_seed))
    return Path(root) / target_id(target) / f"dseed_{tag}"


def derive_split_seeds(target, dataset_seed: Hashable) -> dict[str, int]:
    """Learning pools follow the dataset seed; evaluation pools are fixed per target."""
    return {
        "training": int(derive(target.fingerprint, dataset_seed, "training")),
        "validation": int(derive(target.fingerprint, dataset_seed, "validation")),
        "development_test": int(derive(target.fingerprint, "fixed_development_test")),
        "test": int(derive(target.fingerprint, "fixed_final_test")),
        "test_audit": int(derive(target.fingerprint, "fixed_final_test_audit")),
        "diagnostic": int(derive(target.fingerprint, "fixed_diagnostic")),
    }


def draw_split(target, n: int, seed: int, *, noisy: bool):
    """Return (X, y, f) for one pool. Features and noise use separate child streams,
    so growing ``n`` leaves existing prefixes byte-identical."""
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive; received {n}")

    d = int(target.d)
    if d <= 0:
        raise ValueError(f"target.d must be positive; received {target.d}")

    sigma_sq = float(target.sigma_sq)
    if sigma_sq < 0.0 or not np.isfinite(sigma_sq):
        raise ValueError(f"target.sigma_sq must be finite and nonnegative; received {sigma_sq}")

    feature_seed, noise_seed = np.random.SeedSequence(int(seed)).spawn(2)
    feature_rng = np.random.default_rng(feature_seed)
    noise_rng = np.random.default_rng(noise_seed)

    X = sample_features(feature_rng, (n, d), target.input_distribution)
    f = np.asarray(target.evaluate(X), dtype=np.float32)
    if f.shape == (n, 1):
        f = f[:, 0]
    if f.shape != (n,):
        raise ValueError(
            f"target.evaluate(X) must return shape ({n},) or ({n}, 1); received {f.shape}"
        )
    if not np.all(np.isfinite(f)):
        raise FloatingPointError("target.evaluate(X) returned nonfinite values")

    if noisy:
        noise = noise_rng.standard_normal(n).astype(np.float32)
        y = f + np.float32(np.sqrt(sigma_sq)) * noise
    else:
        y = f.copy()

    if not np.all(np.isfinite(y)):
        raise FloatingPointError("generated targets contain nonfinite values")

    return tuple(
        torch.from_numpy(np.ascontiguousarray(array))
        for array in (X, y, f)
    )


def build_master_dataset(
    target,
    *,
    n_max: int,
    val_size: int,
    dev_test_size: int,
    test_size: int,
    test_audit_size: int,
    diag_size: int,
    seeds: Mapping[str, int],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Draw every pool for one dataset seed; return (tensors, manifest metadata)."""
    sizes = {"n_max": int(n_max), "n_val": int(val_size),
             "n_development_test": int(dev_test_size), "n_test": int(test_size),
             "n_test_audit": int(test_audit_size), "n_diagnostic": int(diag_size)}
    for name, size in sizes.items():
        if size <= 0:
            raise ValueError(f"{name} must be positive; received {size}")

    missing = set(SPLIT_SEEDS) - set(seeds)
    extra = set(seeds) - set(SPLIT_SEEDS)
    if missing:
        raise ValueError(f"missing split seeds: {sorted(missing)}")
    if extra:
        raise ValueError(f"unknown split seeds: {sorted(extra)}")
    if len({int(seeds[name]) for name in SPLIT_SEEDS}) != len(SPLIT_SEEDS):
        raise ValueError("train, validation, test, and diagnostic seeds must differ")

    Xtr, ytr, ftr = draw_split(target, sizes["n_max"], seeds["training"], noisy=True)
    Xva, yva, fva = draw_split(target, sizes["n_val"], seeds["validation"], noisy=True)
    Xdv, _, fdv = draw_split(target, sizes["n_development_test"], seeds["development_test"], noisy=False)
    Xte, _, fte = draw_split(target, sizes["n_test"], seeds["test"], noisy=False)
    Xta, _, fta = draw_split(target, sizes["n_test_audit"], seeds["test_audit"], noisy=False)
    Xdg, _, fdg = draw_split(target, sizes["n_diagnostic"], seeds["diagnostic"], noisy=False)

    tensors = {
        "Xtr_master": Xtr, "ytr_master": ytr, "ftr_master": ftr,
        "Xva": Xva, "yva": yva, "fva": fva,
        "Xdv": Xdv, "fdv": fdv,
        "Xte": Xte, "fte": fte,
        "Xta": Xta, "fta": fta,
        "Xdg": Xdg, "fdg": fdg,
    }
    record = target_record(target)
    metadata = {
        "target_id": record["target_id"],
        "target_family": target.family,
        "target_fingerprint": target.fingerprint,
        "target_config": record["target_config"],
        "input_distribution": record["input_distribution"],
        "calibration": record["calibration"],
        "observation_noise": {
            "distribution": "normal", "mean": 0.0, "variance": float(target.sigma_sq),
        },
        "sizes": sizes,
        "split_seeds": {name: int(seeds[name]) for name in SPLIT_SEEDS},
        "random_stream_policy": {
            "generator": "numpy.random.Generator",
            "bit_generator": "PCG64",
            "feature_noise_streams": "independent_seedsequence_children",
        },
        "storage": {"dtype": "float32", "layout": "contiguous_cpu_tensors"},
        "software": _software_record(),
    }
    return tensors, metadata


def target_record(target) -> dict[str, Any]:
    """The calibrated-target record written to ``target.json``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "target_id": target_id(target),
        "target_name": target.family,
        "target_fingerprint": target.fingerprint,
        "target_config": {
            "d": int(target.d),
            "k": int(target.k),
            "layer_profile": list(target.layer_profile),
            "function_seed": int(target.function_seed),
            "task_options": target.task_options,
            "wiring": target.wiring,
            "sigma_sq": float(target.sigma_sq),
        },
        "input_distribution": {"name": target.input_distribution, "dimension": int(target.d)},
        "calibration": {
            "size": int(target.calibration.get("mc_pool", 0)),   # 0 = analytic, no MC pool
            "accumulation_dtype": "float64",
            "variance_convention": "population_ddof_0",
            "output_mean": float(np.asarray(target.node_mu[-1]).reshape(-1)[0]),
            "output_std": float(np.asarray(target.node_sd[-1]).reshape(-1)[0]),
            "target_variance": float(target.target_variance),
            "config": target.calibration,
        },
        "software": _software_record(),
    }


def _software_record() -> dict[str, Any]:
    """Library versions and, when available, the git commit."""
    record = {"numpy_version": np.__version__, "torch_version": torch.__version__}
    try:
        from runstore import capture_provenance

        provenance = capture_provenance()
        if provenance.get("git_commit"):
            record["git_commit"] = provenance["git_commit"]
    except Exception:  # noqa: BLE001
        pass
    return record


def tensor_sha256(tensor: torch.Tensor) -> str:
    """SHA-256 over a contiguous CPU tensor's raw bytes."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    """Shape, dtype and hash of one saved tensor."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sha256": tensor_sha256(tensor),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON via a temporary file and an atomic rename."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_master_dataset(
    output_directory: str | Path,
    *,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically write one immutable master-dataset artifact."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    missing = set(TENSOR_KEYS) - set(tensors)
    extra = set(tensors) - set(TENSOR_KEYS)
    if missing:
        raise ValueError(f"missing tensors: {sorted(missing)}")
    if extra:
        raise ValueError(f"unknown tensors: {sorted(extra)}")

    saved: dict[str, torch.Tensor] = {}
    for name in TENSOR_KEYS:
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains nonfinite values")
        saved[name] = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()

    manifest = dict(metadata)
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["tensors"] = {name: _tensor_record(tensor) for name, tensor in saved.items()}

    temporary = output_directory / "tensors.pt.tmp"
    torch.save(saved, temporary)
    os.replace(temporary, output_directory / "tensors.pt")
    _atomic_write_json(output_directory / "manifest.json", manifest)
    return output_directory


def load_master_dataset(
    dataset_directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a saved dataset, checking shape, dtype and (by default) hashes."""
    dataset_directory = Path(dataset_directory)
    tensor_path = dataset_directory / "tensors.pt"
    manifest_path = dataset_directory / "manifest.json"
    if not tensor_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"no master dataset at {dataset_directory} (build it with "
            "`python -m csfrbench.pipeline pilot|sample <campaign> --runs DIR`)"
        )

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported dataset schema version: {manifest.get('schema_version')}")

    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    missing = set(TENSOR_KEYS) - set(tensors)
    extra = set(tensors) - set(TENSOR_KEYS)
    if missing:
        raise ValueError(f"saved dataset is missing tensors: {sorted(missing)}")
    if extra:
        raise ValueError(f"saved dataset has unknown tensors: {sorted(extra)}")

    for name in TENSOR_KEYS:
        tensor = tensors[name].contiguous()
        expected = manifest["tensors"][name]
        if list(tensor.shape) != expected["shape"]:
            raise ValueError(
                f"{name} shape mismatch: expected {expected['shape']}, received {list(tensor.shape)}"
            )
        dtype = str(tensor.dtype).removeprefix("torch.")
        if dtype != expected["dtype"]:
            raise ValueError(f"{name} dtype mismatch: expected {expected['dtype']}, received {dtype}")
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains nonfinite values")
        if verify_hashes and tensor_sha256(tensor) != expected["sha256"]:
            raise RuntimeError(f"{name} failed hash verification in {dataset_directory}")
        tensors[name] = tensor

    return tensors, manifest


def load_splits(
    dataset_directory: str | Path,
    *,
    n: int,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Training-safe view of one artifact: ftr_master and fva are withheld."""
    tensors, manifest = load_master_dataset(dataset_directory, verify_hashes=verify_hashes)

    n = int(n)
    n_max = int(manifest["sizes"]["n_max"])
    if not 1 <= n <= n_max:
        raise ValueError(f"require 1 <= n <= n_max={n_max}; received n={n}")

    return {
        "training": TensorPool(tensors["Xtr_master"][:n], tensors["ytr_master"][:n]),
        "validation": TensorPool(tensors["Xva"], tensors["yva"]),
        "development_test": TensorPool(tensors["Xdv"], tensors["fdv"]),
        "test": TensorPool(tensors["Xte"], tensors["fte"]),
        "test_audit": TensorPool(tensors["Xta"], tensors["fta"]),
        "diagnostic": TensorPool(tensors["Xdg"], tensors["fdg"]),
        "n_train": n,
        "n_val": int(tensors["Xva"].shape[0]),
        "n_test": int(tensors["Xte"].shape[0]),
        "n_diag": int(tensors["Xdg"].shape[0]),
        "target_id": manifest["target_id"],
        "target_fingerprint": manifest["target_fingerprint"],
    }


def ensure_dataset(
    root: str | Path,
    target,
    dataset_seed: Hashable,
    *,
    n_max: int,
    val_size: int,
    dev_test_size: int,
    test_size: int,
    test_audit_size: int,
    diag_size: int,
    dry: bool = False,
) -> Path:
    """Artifact directory for (target, dataset_seed), built if absent, stale or too small."""
    directory = dataset_dir(root, target, dataset_seed)
    required = {"n_max": int(n_max), "n_val": int(val_size),
                "n_development_test": int(dev_test_size), "n_test": int(test_size),
                "n_test_audit": int(test_audit_size), "n_diagnostic": int(diag_size)}
    seeds = derive_split_seeds(target, dataset_seed)

    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        sizes = manifest.get("sizes", {})
        # A seed-policy change must rebuild, not silently reuse the old realization.
        reusable = (
            manifest.get("schema_version") == SCHEMA_VERSION
            and manifest.get("target_fingerprint") == target.fingerprint
            and manifest.get("split_seeds") == seeds
            and all(int(sizes.get(key, -1)) >= value for key, value in required.items())
        )
        if reusable:
            return directory
    if dry:
        return directory

    tensors, metadata = build_master_dataset(
        target, n_max=n_max, val_size=val_size, dev_test_size=dev_test_size,
        test_size=test_size, test_audit_size=test_audit_size, diag_size=diag_size,
        seeds=seeds,
    )
    save_master_dataset(directory, tensors=tensors, metadata=metadata)
    _atomic_write_json(directory.parent / "target.json", target_record(target))
    return directory
