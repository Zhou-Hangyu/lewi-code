"""Teacher-side machinery: aligned-variable roster, single-variable interchange
counterfactuals on the target graph, and balanced DAS datasets (goal.md 1, 3, 4).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from csfrbench.methods.graph_oracle import node_levels
from csfrbench.registry import METHODS, TASKS, load_builtin_registrations
from csfrbench.tasks.base import sample_features
from csfrbench.utils.seeds import derive

load_builtin_registrations()


def aligned_variables(target, include_readout: bool = False) -> list[dict]:
    """Canonical non-output intermediates, ordered by (teacher depth, node id).

    A canonical node owns every slot that aliases it (same function object and
    parents: DAG reuse). For a single-node readout the readout node is the
    output computation and is excluded by default; ``include_readout=True``
    keeps it as the deepest aligned variable (the node right before the
    readout). An additive readout keeps every node (the terminal local
    computations feed the readout, goal.md 1.2).
    """
    graph = target.graph
    level, canon = node_levels(target)
    excluded = set()
    if not graph.additive_readout and not include_readout:
        excluded.add(canon[graph.readout_indices[0]])
    roster = []
    for v in range(graph.S):
        if canon[v] != v or v in excluded:
            continue
        slots = [w for w in range(graph.S) if canon[w] == v]
        parents = sorted({canon[r - graph.A] for s_ in slots
                          for r in graph.node_parents[s_] if r >= graph.A})
        roster.append({"var": f"z{v}", "canon_slot": v, "slots": slots,
                       "level": int(level[v]), "parents": parents})
    roster.sort(key=lambda e: (e["level"], e["canon_slot"]))
    return roster


def _standardized_node(target, v: int, Z: np.ndarray) -> np.ndarray:
    h = target.law.value(target.params["nodes"][v], Z)
    return (h - target.node_mu[v]) / target.node_sd[v]


def counterfactual_frame(target, Xb: np.ndarray, Xs: np.ndarray,
                         slots: list[int]) -> list[np.ndarray]:
    """Per-slot standardized values under z_v(b) <- z_v(s), v = the slot set.

    The overridden slots take the source value; every other slot is recomputed
    from its parents in topological order, so descendants (through any fan-out)
    pick up the change and non-descendants reproduce their base values exactly.
    """
    graph = target.graph
    frame_s = target.forward(Xs)
    leaves = target.leaf_scale * np.asarray(Xb, np.float32)[
        :, target.active_raw.reshape(-1)].astype(np.float64)
    override = set(slots)
    values: list[np.ndarray] = []
    for v, refs in enumerate(graph.node_parents):
        if v in override:
            values.append(frame_s["values"][v])
            continue
        Z = np.stack([leaves[:, r] if r < graph.A else values[r - graph.A]
                      for r in refs], axis=1)
        values.append(_standardized_node(target, v, Z))
    return values


def counterfactual_outputs(target, Xb: np.ndarray, Xs: np.ndarray,
                           slots: list[int]) -> np.ndarray:
    """y*_cf for z_v(b) <- z_v(s) on every row, v given by its slot set."""
    graph = target.graph
    values = counterfactual_frame(target, Xb, Xs, slots)
    raw = sum(w * values[i]
              for i, w in zip(graph.readout_indices, graph.readout_weights))
    y = (raw - target.out_mu) / target.out_sd
    if not np.all(np.isfinite(y)):
        raise FloatingPointError("counterfactual readout produced nonfinite values")
    return y.astype(np.float32)


def teacher_node_values(target, X: np.ndarray) -> np.ndarray:
    """Standardized canonical-node values, shape (n, S) (float32)."""
    return np.stack(target.forward(X)["values"], axis=1).astype(np.float32)


DAS_SPLITS = ("das_train", "das_val", "das_test")


def build_das_split(target, split: str, per_var: int, split_seed: int = 0,
                    include_readout: bool = False) -> dict:
    """One exactly variable-balanced split with its own RNG stream.

    b and s are fresh independent draws from the target's input distribution;
    nothing is conditioned on outputs, correctness, or intervention size.
    """
    if split not in DAS_SPLITS:
        raise ValueError(f"unknown DAS split {split!r}")
    roster = aligned_variables(target, include_readout)
    m = len(roster)
    if m == 0:
        raise ValueError(f"{target.fingerprint}: no aligned variables (DAS N/A)")
    total = per_var * m
    rng = np.random.default_rng(derive(target.fingerprint, int(split_seed), split))
    Xb = sample_features(rng, (total, target.d), target.input_distribution)
    Xs = sample_features(rng, (total, target.d), target.input_distribution)
    var_idx = np.repeat(np.arange(m), per_var)
    rng.shuffle(var_idx)
    y_cf = np.empty(total, np.float32)
    for i, entry in enumerate(roster):
        rows = var_idx == i
        y_cf[rows] = counterfactual_outputs(target, Xb[rows], Xs[rows],
                                            entry["slots"])
    # Mismatched-variable control labels (goal.md 10.4): the same (b, s) pair
    # answered for a different teacher variable. A permutation of balanced
    # group ids is NOT a control -- the joint fit is equivariant to block
    # relabeling -- so the mismatch must change the counterfactual function.
    wrong_idx = (var_idx + 1 + rng.integers(0, m - 1, total)) % m if m > 1         else var_idx.copy()
    y_cf_wrong = np.empty(total, np.float32)
    for i, entry in enumerate(roster):
        rows = wrong_idx == i
        y_cf_wrong[rows] = counterfactual_outputs(target, Xb[rows], Xs[rows],
                                                  entry["slots"])
    return {"Xb": Xb, "Xs": Xs, "var_idx": var_idx.astype(np.int64),
            "y_cf": y_cf, "wrong_idx": wrong_idx.astype(np.int64),
            "y_cf_wrong": y_cf_wrong, "roster": roster}


def ensure_das_dataset(root: str | Path, target, sizes: dict[str, int],
                       split_seed: int = 0, include_readout: bool = False) -> Path:
    """Build (once) and return the DAS dataset directory for one target."""
    out = Path(root) / f"{target.fingerprint}_s{int(split_seed)}"
    manifest = out / "das_manifest.json"
    if manifest.exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    roster = aligned_variables(target, include_readout)
    for split in DAS_SPLITS:
        data = build_das_split(target, split, int(sizes[split]), split_seed,
                               include_readout)
        np.savez_compressed(out / f"{split}.npz", Xb=data["Xb"], Xs=data["Xs"],
                            var_idx=data["var_idx"], y_cf=data["y_cf"],
                            wrong_idx=data["wrong_idx"],
                            y_cf_wrong=data["y_cf_wrong"])
    payload = {
        "fingerprint": target.fingerprint, "family": target.family,
        "function_seed": int(target.function_seed),
        "d": target.d, "input_distribution": target.input_distribution,
        "split_seed": int(split_seed),
        "include_readout": bool(include_readout),
        "sizes": {k: int(v) for k, v in sizes.items()},
        "roster": roster,
        "rng": {s: f"derive(fingerprint, {int(split_seed)}, '{s}')"
                for s in DAS_SPLITS},
    }
    (out / "das_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return out


def load_das_split(dataset_dir: str | Path, split: str) -> dict:
    z = np.load(Path(dataset_dir) / f"{split}.npz")
    manifest = json.loads((Path(dataset_dir) / "das_manifest.json").read_text())
    return {"Xb": z["Xb"], "Xs": z["Xs"], "var_idx": z["var_idx"],
            "y_cf": z["y_cf"], "wrong_idx": z["wrong_idx"],
            "y_cf_wrong": z["y_cf_wrong"], "roster": manifest["roster"]}
