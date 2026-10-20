"""Target functions and datasets."""
import numpy as np
import pytest
from conftest import assert_verified

from csfrbench import datasets, suite, targets, verify


def test_values_of_every_target_instance():
    assert_verified(verify.check_target_values())


def test_single_local_functions():
    for name, (_paper, _family, _preset, arity) in suite.LOCAL_FUNCTIONS.items():
        target = suite.local_function(name, 6100)
        assert (target.d, target.A, target.graph.S, target.sigma_sq) == (arity, arity, 1, 0.0)


def test_a_sample_of_recorded_datasets():
    assert_verified(verify.check_targets(12))


@pytest.mark.slow
def test_every_recorded_dataset():
    assert_verified(verify.check_targets(None))


@pytest.mark.parametrize("name", list(suite.SETTINGS["primary"]))
def test_target_is_normalized_and_depends_only_on_active_inputs(name):
    w = suite.workload("nuisance", name)
    target = w.target(6000)
    X = targets.sample_inputs(np.random.default_rng(0), (20000, w.d))
    f = target.evaluate(X)
    assert abs(float(f.mean())) < 0.05 and abs(float(f.var()) - 1.0) < 0.05
    Z = target.intermediates(X)
    assert Z.shape == (20000, target.graph.S)
    assert np.all(np.abs(Z.mean(axis=0)) < 0.05) and np.all(np.abs(Z.var(axis=0) - 1.0) < 0.05)
    inert = [j for j in range(w.d) if j not in target.active_inputs]
    assert len(target.active_inputs) == w.active and len(inert) == w.d - w.active
    X2 = X.copy()
    X2[:, inert] = targets.sample_inputs(np.random.default_rng(1), (20000, len(inert)))
    assert np.array_equal(target.evaluate(X2), f)


@pytest.mark.parametrize("name", list(suite.SETTINGS["primary"]))
def test_the_two_settings_share_graph_and_local_functions(name):
    a = suite.workload("primary", name).target(6003)
    b = suite.workload("nuisance", name).target(6003)
    assert a.graph.node_parents == b.graph.node_parents and a.graph.readout_indices == b.graph.readout_indices
    assert a.node_mu == b.node_mu and a.node_sd == b.node_sd
    X = targets.sample_inputs(np.random.default_rng(0), (512, b.d))
    Xa = np.zeros((512, a.d), dtype=np.float32)
    Xa[:, a.active_raw.reshape(-1)] = X[:, b.active_raw.reshape(-1)]
    assert np.array_equal(a.evaluate(Xa), b.evaluate(X))


def test_graph_shapes_of_the_paper():
    shape = {name: (w.target(6000).graph.S, w.levels, max(w.target(6000).graph.levels()))
             for name, w in suite.SETTINGS["primary"].items()}
    assert shape == {"flat_modules": (4, 1, 1), "depth4_tree": (7, 3, 3), "reuse_dag": (7, 2, 2),
                     "sampled_dags": (7, 3, 3), "fourier_tree": (7, 3, 3), "fourier_reuse_dag": (7, 2, 2),
                     "high_arity_node": (1, 1, 1), "depth5_tree": (15, 4, 4)}
    dag = suite.workload("primary", "reuse_dag").target(6000).graph
    assert dag.node_parents[4:] == ((8, 9), (9, 10), (10, 11)) and dag.readout_indices == (4, 5, 6)


def test_sampled_dags_satisfy_the_constraints():
    w = suite.workload("primary", "sampled_dags")
    hashes = set()
    for instance in w.instances:
        t = w.target(instance)
        g = t.graph
        assert g.A == 8 and g.S == 7 and all(len(p) == 2 for p in g.node_parents)
        assert max(g.levels()) == 3 and len(g.readout_indices) >= 2
        assert len({frozenset(p) for p in g.node_parents}) == 7           # no pair of parents is used twice
        assert t.topology_hash == targets.canonical_topology_hash(g)
        hashes.add(t.topology_hash)
    assert len(hashes) >= 24                                             # about 30 distinct topologies in 32 instances


def test_the_sampler_gives_the_number_of_topologies_of_the_paper():
    """Appendix C.2 of the paper: 354 distinct topologies in 4,096 draws of the graph sampler."""
    hashes = {targets.canonical_topology_hash(targets.sample_graph(np.random.default_rng(seed)))
              for seed in range(4096)}
    assert len(hashes) == 354


def test_gradient_is_the_derivative_of_the_target():
    target = suite.workload("primary", "depth4_tree").target(6001)
    X = targets.sample_inputs(np.random.default_rng(0), (64, 8)).astype(np.float64)
    grad = target.gradients(X.astype(np.float32))
    eps = 1e-3
    for j, column in enumerate(target.active_raw.reshape(-1)[:3]):
        step = np.zeros_like(X)
        step[:, column] = eps
        numeric = (target.forward((X + step).astype(np.float32))["raw_readout"]
                   - target.forward((X - step).astype(np.float32))["raw_readout"]) / (2 * eps) / target.out_sd
        assert np.allclose(grad[:, j], numeric, atol=5e-3)


def test_training_sets_are_nested_and_pools_are_independent():
    target = suite.workload("primary", "reuse_dag").target(6000)
    small = datasets.make_dataset(target, 512, pools=("train", "val", "test"))
    large = datasets.make_dataset(target, 2048, pools=("train", "val", "test"))
    assert np.array_equal(small["X_train"].numpy(), large["X_train"].numpy()[:512])
    assert np.array_equal(small["y_train"].numpy(), large["y_train"].numpy()[:512])
    assert np.array_equal(small["X_test"].numpy(), large["X_test"].numpy())
    noise = (large["y_train"] - large["f_train"]).numpy()
    assert abs(float(noise.var()) - 0.01) < 0.002
    assert np.array_equal(large["f_test"].numpy(), target.evaluate(large["X_test"].numpy()))
    assert tuple(large["X_val"].shape) == (4096, 8) and tuple(large["X_test"].shape) == (65536, 8)


def test_saved_dataset_is_verified_on_load(tmp_path):
    target = suite.workload("primary", "flat_modules").target(6000)
    data = datasets.make_dataset(target, 256, pools=("train", "val"))
    datasets.save_dataset(data, tmp_path)
    loaded = datasets.load_dataset(tmp_path)
    assert datasets.dataset_hashes(loaded) == datasets.dataset_hashes(data)
    assert loaded["meta"]["fingerprint"] == target.fingerprint
