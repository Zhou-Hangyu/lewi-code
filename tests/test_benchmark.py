"""The quick interface: tasks hide the target, evaluate any predictor, learning curves and the comparison."""
import numpy as np
import pytest

import csfrbench
from csfrbench import datasets, suite


@pytest.fixture(scope="module")
def task():
    return csfrbench.task("depth4_tree", instance=6000, n=1024)


def test_task_gives_the_paper_data(task):
    w = suite.workload("primary", "depth4_tree")
    data = datasets.make_dataset(w.target(6000), w.profile.n_max, pools=("train", "val"))
    assert np.array_equal(task.X_train, data["X_train"][:1024].numpy())
    assert np.array_equal(task.y_train, data["y_train"][:1024].numpy())
    assert np.array_equal(task.X_val, data["X_val"].numpy())
    assert task.d == 8 and task.n == 1024
    assert not any(hasattr(task, name) for name in ("graph", "target", "intermediates"))


def test_evaluate_any_predictor(task):
    target = task.reveal()
    assert task.evaluate(target.evaluate) == pytest.approx(0.0, abs=1e-10)       # the noise-free target itself
    assert task.evaluate(lambda X: np.zeros(len(X))) == pytest.approx(1.0, abs=0.02)

    class Mean:
        def predict(self, X):
            return np.full(len(X), 0.5)
    assert task.evaluate(Mean()) > 1.0
    assert 0.0 < task.evaluate(lambda X: np.zeros(len(X)), pool="val") < 1.2


def test_learning_curve_and_comparison():
    def fit(X, y, X_val, y_val):
        mean = float(y.mean())
        return lambda X: np.full(len(X), mean)
    curve = csfrbench.learning_curve(fit, "flat_modules", instance=6000, verbose=False)
    assert curve.full_grid and sorted(curve.nmse) == list(suite.workload("primary", "flat_modules").profile.sizes)
    assert curve.aulc == pytest.approx(1.0, abs=0.02)
    rows = csfrbench.compare([curve], show=False)
    arms = {r["method"] for r in rows}
    assert {"mine", "lewi", "dense", "go_all"} <= arms
    lewi = next(r for r in rows if r["method"] == "lewi")
    assert lewi["aulc"] < 0.1 and lewi["difference"]["mean"] > 0.5


def test_unknown_names_are_refused():
    with pytest.raises(KeyError):
        csfrbench.task("no_such_workload")
    with pytest.raises(ValueError):
        csfrbench.task("depth4_tree", instance=1234)
    with pytest.raises(ValueError):
        csfrbench.task("depth4_tree", n=10**9)


def test_default_task_is_a_development_instance():
    assert csfrbench.task("depth4_tree").instance in suite.DEVELOPMENT_INSTANCES
