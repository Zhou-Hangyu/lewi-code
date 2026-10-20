"""The records of the experiments of the paper.

The package holds the results of the two campaigns that produced the paper's
CSFRBench tables and figures: 741,888 training runs (354,816 in the primary
setting, 387,072 in the nuisance-input setting). Use them to verify an
installation and to compare a new method with the recorded methods on the same
target instances, without a new run of the recorded methods.

``winners(setting)``          the selected run of every (workload, arm, instance, n)
``units(setting)``            every run: validation NMSE, test NMSE, selected step, diverged
``trials(setting)``           the 32 configurations of every (workload, arm, n)
``aulc(setting)``             mean AULC and interval per (workload, arm)
``aulc_instances(setting)``   AULC of every (workload, arm, instance)
``contrasts(setting)``        paired differences LeWi - arm
``targets()``                 fingerprints, normalization constants and tensor hashes of the datasets
``target_values()``           graph, normalization and values of every target on a fixed input
``curves()``                  a sample of runs with their validation curves
``checkpoints()``             the runs of the sample with their model states
``diagnostics()``             probe matrices and DAS fits of runs of the sample
``paper_tables()``            the values printed in Tables 3 and 4 of the paper
"""
from __future__ import annotations

import csv
import gzip
import io
import json
from functools import lru_cache
from importlib import resources

import numpy as np

from . import suite


def _open(name: str):
    return resources.files("csfrbench").joinpath("reference").joinpath(name)


def _text(name: str) -> str:
    data = _open(name).read_bytes()
    return gzip.decompress(data).decode() if name.endswith(".gz") else data.decode()


def _rows(name: str, ints=(), floats=()) -> list[dict]:
    rows = []
    for row in csv.DictReader(io.StringIO(_text(name))):
        for k in ints:
            row[k] = int(row[k])
        for k in floats:
            row[k] = float(row[k])
        rows.append(row)
    return rows


def _check(setting: str) -> None:
    if setting not in suite.SETTINGS:
        raise KeyError(f"unknown setting {setting!r}; available: {sorted(suite.SETTINGS)}")


@lru_cache(maxsize=4)
def winners(setting: str) -> list[dict]:
    _check(setting)
    rows = _rows(f"winners_{setting}.csv.gz", ints=("instance", "n", "trial", "best_step", "trials", "trials_diverged"),
                 floats=("best_val_nmse", "test_nmse"))
    for r in rows:
        r["hp"] = {**suite.ARMS[r["arm"]].fixed, **json.loads(r["hp"])}      # with the fixed arguments of the arm
        r["setting"] = setting
        r["diverged"] = 0
    return rows


@lru_cache(maxsize=4)
def units(setting: str) -> dict[str, np.ndarray]:
    """Arrays of equal length, one entry per run; ``workload`` and ``arm`` are names."""
    _check(setting)
    with resources.as_file(_open(f"units_{setting}.npz")) as path:
        z = np.load(path)
        out = {k: z[k] for k in ("instance", "n", "trial", "best_val_nmse", "test_nmse", "best_step", "diverged")}
        out["workload"] = z["workloads"][z["workload"]]
        out["arm"] = z["arms"][z["arm"]]
    return out


def unit_rows(setting: str, workload: str | None = None, arm: str | None = None) -> list[dict]:
    """The runs of a setting as a list of mappings, as :mod:`csfrbench.analysis` reads them."""
    u = units(setting)
    keep = np.ones(len(u["n"]), dtype=bool)
    if workload is not None:
        keep &= u["workload"] == workload
    if arm is not None:
        keep &= u["arm"] == arm
    index = np.flatnonzero(keep)
    return [{"setting": setting, "workload": str(u["workload"][i]), "arm": str(u["arm"][i]),
             "instance": int(u["instance"][i]), "n": int(u["n"][i]), "trial": int(u["trial"][i]),
             "best_val_nmse": float(u["best_val_nmse"][i]),
             "test_nmse": None if np.isnan(u["test_nmse"][i]) else float(u["test_nmse"][i]),
             "best_step": int(u["best_step"][i]), "diverged": int(u["diverged"][i])} for i in index]


@lru_cache(maxsize=4)
def trials(setting: str) -> dict[tuple[str, str, int], list[dict]]:
    _check(setting)
    raw = json.loads(_text(f"trials_{setting}.json.gz"))
    out = {}
    for key, hps in raw.items():
        workload, arm, n = key.split("|")
        fixed = dict(suite.ARMS[arm].fixed)
        out[(workload, arm, int(n))] = [{**fixed, **hp} for hp in hps]
    return out


@lru_cache(maxsize=4)
def aulc(setting: str) -> list[dict]:
    _check(setting)
    return _rows(f"aulc_{setting}.csv", ints=("n",), floats=("mean", "lo", "hi", "sd", "median"))


@lru_cache(maxsize=4)
def aulc_instances(setting: str) -> list[dict]:
    _check(setting)
    return _rows(f"aulc_instances_{setting}.csv.gz", ints=("instance",), floats=("aulc",))


@lru_cache(maxsize=4)
def contrasts(setting: str) -> list[dict]:
    _check(setting)
    return _rows(f"contrasts_{setting}.csv", ints=("n", "wins"), floats=("mean", "lo", "hi"))


@lru_cache(maxsize=1)
def targets() -> dict[str, dict]:
    return json.loads(_text("targets.json.gz"))


@lru_cache(maxsize=1)
def target_values() -> dict[str, dict]:
    """Graph, normalization and values on a fixed input of every target instance of both settings and of the
    single local functions; the key is ``name|d|instance|noise variance``."""
    return json.loads(_text("target_values.json.gz"))


@lru_cache(maxsize=1)
def curves() -> dict[str, dict]:
    """A sample of recorded runs with hyperparameters, seeds and the validation curve of 64 checkpoints.

    The key is ``setting|workload|arm|instance|n|trial``. Two entries of a run are not records of the campaign
    but measurements with this package in the environment of the records: ``rounding_sensitivity`` and
    ``first_steps`` (see :mod:`csfrbench.verify`).
    """
    import torch
    with resources.as_file(_open("checkpoints.pt")) as path:
        return torch.load(path, map_location="cpu", weights_only=True)


def checkpoints() -> dict[str, dict]:
    """The runs of :func:`curves` with the model state at the initialization (``init``) and at the selected
    checkpoint (``best``). The record of the depth-5 tree in the primary setting has no states."""
    return {key: run for key, run in curves().items() if "best" in run}


@lru_cache(maxsize=1)
def paper_tables() -> dict:
    return json.loads(_text("paper_tables.json"))


@lru_cache(maxsize=1)
def diagnostics() -> dict:
    """Recorded linear-probe matrices and DAS fits of a sample of selected runs."""
    import torch
    with resources.as_file(_open("diagnostics.pt")) as path:
        out = torch.load(path, map_location="cpu", weights_only=True)
    for entry in out.values():
        for key, value in entry.get("probe", {}).items():
            if torch.is_tensor(value):
                entry["probe"][key] = value.numpy()
    return out
