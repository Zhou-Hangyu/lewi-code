"""Benchmark any predictor on CSFRBench: a task, a learning curve, and a comparison with the paper's methods.

A *task* is one target instance of a workload and one training-set size ``n``. The learner receives the noisy
training and validation examples, nothing else. The target's computation graph and intermediate values stay inside
the task; :meth:`Task.reveal` returns them for structural diagnostics after training.

    import csfrbench

    task = csfrbench.task("depth4_tree", instance=6000, n=1024)
    predict = my_fit(task.X_train, task.y_train, task.X_val, task.y_val)       # any learner, any framework
    print(task.evaluate(predict))       # test NMSE on 65,536 noise-free inputs, as reported in the paper

    curve = csfrbench.learning_curve(my_fit, "depth4_tree", instance=6000)    # n = 2^9 ... 2^15
    print(curve)                                                              # NMSE per n and the AULC
    csfrbench.compare([curve])          # AULC beside the paper's methods on the same target instances

The examples, the test pool and the NMSE are those of the paper's runs: the training set of size ``n`` is the first
``n`` rows of the instance's training pool, and the test NMSE is the squared error summed over the test pool divided
by the summed squared deviation of the noise-free target from its mean. A learning curve covers the workload's grid
of training-set sizes, so its AULC is directly comparable with the recorded AULC of every method.

This is the quick, exploratory interface: it fixes the data and the metric, not the learner's protocol. The paper's
protocol (a search space frozen on the development instances 6016-6019, 32 configurations per instance and size,
selection on the validation pool, every evaluation instance) is described in docs/SPEC.md and run with the
``csfrbench`` command (see examples/new_method.py).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from . import datasets, metrics, record, suite
from .targets import Target

#: A fitted predictor: maps an ``(m, d)`` float32 array of inputs to ``m`` predictions. An object with a
#: ``predict`` method (scikit-learn style) or a ``torch.nn.Module`` works too.
Predictor = Any
#: A learner: ``fit(X_train, y_train, X_val, y_val) -> predictor``. The arrays are float32 NumPy arrays.
Fit = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Predictor]

EVAL_BATCH = 8192


@lru_cache(maxsize=2)
def _pools(setting: str, workload: str, instance: int) -> tuple[Target, dict[str, np.ndarray]]:
    w = suite.workload(setting, workload)
    target = w.target(int(instance))
    data = datasets.make_dataset(target, w.profile.n_max, pools=("train", "val", "test"))
    return target, {k: v.numpy() for k, v in data.items() if k != "meta"}


def _predict(predictor: Predictor, X: np.ndarray) -> np.ndarray:
    """Predictions of any supported predictor on ``X``, in batches, as a float64 vector."""
    out = []
    for start in range(0, X.shape[0], EVAL_BATCH):
        batch = X[start:start + EVAL_BATCH]
        if hasattr(predictor, "predict") and callable(predictor.predict):
            p = predictor.predict(batch)
        else:
            try:
                import torch
            except ImportError:                     # pragma: no cover - torch is a dependency
                torch = None
            if torch is not None and isinstance(predictor, torch.nn.Module):
                was_training = predictor.training
                predictor.eval()
                device = next((q.device for q in predictor.parameters()), torch.device("cpu"))
                with torch.no_grad():
                    p = predictor(torch.from_numpy(np.ascontiguousarray(batch)).to(device))
                predictor.train(was_training)
            else:
                p = predictor(batch)
        if hasattr(p, "detach"):
            p = p.detach().cpu().numpy()
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        if p.shape[0] != batch.shape[0]:
            raise ValueError(f"the predictor returned {p.shape[0]} values for {batch.shape[0]} inputs; it must return "
                             "one prediction per input")
        out.append(p)
    return np.concatenate(out)


class Task:
    """One target instance and one training-set size of a CSFRBench workload.

    Attributes the learner may use: ``X_train`` and ``y_train`` (``n`` noisy examples), ``X_val`` and ``y_val``
    (4,096 noisy examples for model selection), ``d`` (the input dimension) and ``n``. Everything else about the
    target is hidden until :meth:`reveal`.
    """

    def __init__(self, workload: str, instance: int, n: int, *, setting: str = "primary"):
        self.workload = suite.workload(setting, workload)
        self.setting, self.instance, self.n = setting, int(instance), int(n)
        if self.n < 1 or self.n > self.workload.profile.n_max:
            raise ValueError(f"n must lie between 1 and {self.workload.profile.n_max} for {workload}; received {n}")
        if self.instance not in self.workload.instances and self.instance not in suite.DEVELOPMENT_INSTANCES:
            raise ValueError(f"instance {instance} is neither an evaluation instance of {workload} "
                             f"({self.workload.instances[0]}-{self.workload.instances[-1]}) nor a development "
                             f"instance {suite.DEVELOPMENT_INSTANCES}")
        target, pools = _pools(setting, self.workload.name, self.instance)
        self._target = target
        self._pools = pools
        self.d = int(target.d)
        self.X_train, self.y_train = pools["X_train"][:self.n].copy(), pools["y_train"][:self.n].copy()
        self.X_val, self.y_val = pools["X_val"].copy(), pools["y_val"].copy()

    def __repr__(self) -> str:
        return (f"Task({self.workload.name!r}, instance={self.instance}, n={self.n}, setting={self.setting!r}: "
                f"d = {self.d}, {self.n} training and {len(self.y_val)} validation examples)")

    def evaluate(self, predictor: Predictor, pool: str = "test") -> float:
        """The NMSE of a predictor on the noise-free test pool (65,536 inputs), as reported in the paper.

        ``pool="val"`` scores the noisy validation pool instead, the one to use for model selection.
        """
        if pool == "test":
            X, f = self._pools["X_test"], self._pools["f_test"]
        elif pool == "val":
            X, f = self.X_val, self.y_val
        else:
            raise ValueError("pool must be 'test' or 'val'")
        prediction = _predict(predictor, X)
        if not np.all(np.isfinite(prediction)):
            return float("inf")
        target = f.astype(np.float64)
        return metrics.nmse_from_errors((prediction - target) ** 2, target)

    def reveal(self) -> Target:
        """The hidden target: its computation graph, its local functions and its intermediate values.

        For diagnostics after training (for example with :mod:`csfrbench.probes` or :mod:`csfrbench.das`). A learner
        that reads it is graph-informed and is not comparable with the methods that learn from examples only.
        ``target.graph.node_parents``, ``target.active_inputs``, ``target.evaluate(X)`` and
        ``target.intermediates(X)`` are the main entry points.
        """
        return self._target


def task(workload: str, instance: int = suite.INSTANCES[0], n: int = 1024, *, setting: str = "primary") -> Task:
    """The task of a workload, a target instance and a training-set size (see :class:`Task`)."""
    return Task(workload, instance, n, setting=setting)


@dataclass
class LearningCurve:
    """The test NMSE of one learner on one target instance over a grid of training-set sizes."""
    workload: str
    instance: int
    setting: str
    nmse: dict[int, float]
    name: str = "mine"
    full_grid: bool = field(init=False)

    def __post_init__(self):
        self.full_grid = tuple(sorted(self.nmse)) == suite.workload(self.setting, self.workload).profile.sizes

    @property
    def aulc(self) -> float:
        """The area under the learning curve over ``log2 n``, divided by the width of the grid (lower is better)."""
        ns = sorted(self.nmse)
        return metrics.aulc(ns, [self.nmse[n] for n in ns])

    def __str__(self) -> str:
        rows = "\n".join(f"  n = {n:>6d}   test NMSE {self.nmse[n]:.4f}" for n in sorted(self.nmse))
        grid = "" if self.full_grid else "   (not the workload's full grid: not comparable with the records)"
        return f"{self.name} on {self.workload}, instance {self.instance}:\n{rows}\n  AULC {self.aulc:.4f}{grid}"


def learning_curve(fit: Fit, workload: str, instance: int = suite.INSTANCES[0], *, sizes: Sequence[int] | None = None,
                   setting: str = "primary", name: str = "mine", verbose: bool = True) -> LearningCurve:
    """Fit a learner at every training-set size of a workload's grid and score it on the test pool.

    ``fit(X_train, y_train, X_val, y_val)`` returns a predictor (see :data:`Predictor`). By default the sizes are the
    workload's grid (``n = 2^9 ... 2^15``; ``2^11 ... 2^17`` on the depth-5 tree), the grid of the recorded AULC.
    """
    w = suite.workload(setting, workload)
    nmse = {}
    for n in (sizes or w.profile.sizes):
        t = Task(w.name, instance, int(n), setting=setting)
        nmse[int(n)] = t.evaluate(fit(t.X_train, t.y_train, t.X_val, t.y_val))
        if verbose:
            print(f"{name}: {w.name} instance {instance} n = {n}: test NMSE {nmse[int(n)]:.4f}", flush=True)
    return LearningCurve(w.name, int(instance), setting, nmse, name)


def compare(curves: Iterable[LearningCurve], arms: Sequence[str] | None = None, *, show: bool = True) -> list[dict]:
    """The AULC of your learner beside the recorded AULC of the paper's methods on the same target instances.

    ``curves`` hold one learning curve per target instance of one workload (each on the full grid). For every
    recorded method the result gives its mean AULC over these instances and the mean paired difference
    ``yours - method`` with a 95% Student-t interval when there are at least two instances (negative favours yours).
    """
    curves = list(curves)
    if not curves:
        raise ValueError("no learning curve to compare")
    keys = {(c.setting, c.workload, c.name) for c in curves}
    if len(keys) != 1:
        raise ValueError("compare the curves of one learner on one workload at a time")
    setting, workload, name = keys.pop()
    if not all(c.full_grid for c in curves):
        raise ValueError("every curve must cover the workload's full grid of training-set sizes")
    mine = {c.instance: c.aulc for c in curves}
    recorded: dict[str, dict[int, float]] = {}
    for row in record.aulc_instances(setting):
        if row["workload"] == workload and row["instance"] in mine:
            recorded.setdefault(row["arm"], {})[row["instance"]] = row["aulc"]
    order = [a for a in (arms or suite.ARM_ORDER) if a in recorded]
    rows = [{"method": name, "paper_name": name, "instances": len(mine),
             "aulc": float(np.mean(list(mine.values()))), "difference": None}]
    for arm in order:
        shared = sorted(set(recorded[arm]) & set(mine))
        diff = np.asarray([mine[i] - recorded[arm][i] for i in shared])
        rows.append({"method": arm, "paper_name": suite.ARMS[arm].paper_name, "instances": len(shared),
                     "aulc": float(np.mean([recorded[arm][i] for i in shared])),
                     "difference": metrics.t_interval(diff) if len(shared) >= 2 else
                     {"mean": float(diff.mean()), "lo": float("nan"), "hi": float("nan")}})
    if show:
        print(f"AULC on {workload} ({setting} setting), {len(mine)} target instance(s); lower is better")
        for r in sorted(rows, key=lambda r: r["aulc"]):
            line = f"  {r['paper_name']:<18s} {r['aulc']:.4f}"
            if r["difference"] is not None:
                d = r["difference"]
                line += f"    yours - this: {d['mean']:+.4f}"
                if np.isfinite(d.get("lo", float("nan"))):
                    line += f"  [{d['lo']:+.4f}, {d['hi']:+.4f}]"
            print(line)
    return rows
