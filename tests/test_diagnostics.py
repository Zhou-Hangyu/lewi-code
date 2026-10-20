"""Linear probes and counterfactual interventions."""
import numpy as np
import pytest
import torch
from conftest import assert_verified

from csfrbench import das, datasets, probes, record, suite, training, verify


def test_probes_of_recorded_states():
    assert_verified(verify.check_probes(3))


@pytest.mark.slow
def test_probes_of_every_recorded_state():
    assert_verified(verify.check_probes())


def test_probe_protocol_constants():
    assert len(probes.LAMBDAS) == 23 and probes.LAMBDAS[0] == 1e-4 and probes.LAMBDAS[-1] == 1e7
    assert probes.LAMBDAS[1] == pytest.approx(10 ** -3.5) and probes.MIN_SUPPORT == 10
    train, val, test = probes.split_indices("any", datasets.DIAGNOSTIC_SIZE)
    assert (len(train), len(val), len(test)) == (4096, 2048, 2048)
    assert len(set(train) | set(val) | set(test)) == 8192


def test_probe_reads_an_intermediate_that_the_layer_holds():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((8192, 2))
    H = np.concatenate([Z @ rng.standard_normal((2, 6)), rng.standard_normal((8192, 4))], axis=1)
    r2, _lam = probes.ridge_r2((H[:4096], H[4096:6144], H[6144:]), (Z[:4096], Z[4096:6144], Z[6144:]))
    assert np.all(r2 > 0.999)
    noise = rng.standard_normal((8192, 10))
    r2, _lam = probes.ridge_r2((noise[:4096], noise[4096:6144], noise[6144:]), (Z[:4096], Z[4096:6144], Z[6144:]))
    assert np.all(r2 < 0.05)


def test_counterfactual_pairs():
    assert_verified(verify.check_pairs(3))


def test_das_protocol_constants():
    assert (das.TRAIN_PAIRS, das.VAL_PAIRS, das.TEST_PAIRS) == (10000, 1000, 2000)
    assert (das.EPOCHS, das.RESTARTS, das.BATCH_SIZE, das.EVAL_EVERY) == (18, 3, 64, 200)
    assert (das.ROTATION_LR, das.BOUNDARY_LR, das.LAYERS) == (0.003, 0.0003, (1, 2, 3, 4))


def test_counterfactual_of_the_readout_variable_replaces_the_output():
    target = suite.workload("primary", "depth4_tree").target(6000)
    rng = np.random.default_rng(0)
    Xb, Xs = (rng.uniform(-1, 1, (128, 8)).astype(np.float32) for _ in range(2))
    root = das.variables(target)[-1]
    assert root["level"] == 3 and root["node"] == 6
    assert np.array_equal(das.counterfactual_output(target, Xb, Xs, root["node"]), target.evaluate(Xs))
    leaf = das.variables(target)[0]
    out = das.counterfactual_output(target, Xb, Xs, leaf["node"])
    assert not np.array_equal(out, target.evaluate(Xb)) and not np.array_equal(out, target.evaluate(Xs))


def test_das_selection_rule():
    fits = [{"layer": layer, "seed": seed, "variables": ["z0", "z1"], "best_val_macro": 0.1 * layer + 0.01 * seed,
             "val_per_variable": [0.5 if layer == 2 else 0.1, 0.2], "test_per_variable": [0.4, 0.3 + 0.01 * layer],
             "no_edit_per_variable": [0.0, 0.0], "dims": [3, 2]} for layer in (1, 2, 3) for seed in (0, 1, 2)]
    out = das.select(fits)
    assert out["restarts"] == {1: 2, 2: 2, 3: 2}                       # the restart with the best validation score
    assert [v["layer"] for v in out["variables"]] == [2, 1]            # a tie goes to the earlier layer
    assert out["score"] == pytest.approx((0.4 + 0.31) / 2)


def test_intervention_with_the_identity_mask_is_the_forward_pass():
    run = next(r for r in record.checkpoints().values() if r["arm"] == "dense" and r["setting"] == "primary")
    w = suite.workload(run["setting"], run["workload"])
    _m, model, _s = training.build_model("dense", w.target(run["instance"]), width=32, hidden=4, hp=run["hp"], n=run["n"])
    model.load_state_dict(run["best"])
    model.eval()
    x = torch.randn(64, w.d)
    for layer in das.LAYERS:
        h = model.hidden_states(x)[layer - 1]
        assert torch.allclose(das.resume(model, h, layer), model(x), atol=1e-6)
        W = torch.linalg.qr(torch.randn(32, 32))[0]
        none = das.intervene(W, torch.zeros(1, 32), h, h.flip(0), torch.zeros(64, dtype=torch.long))
        assert torch.allclose(none, h, atol=1e-5)
        full = das.intervene(W, torch.ones(1, 32), h, h.flip(0), torch.zeros(64, dtype=torch.long))
        assert torch.allclose(full, h.flip(0), atol=1e-5)


@pytest.mark.slow
def test_das_fit_of_a_recorded_state():
    assert_verified(verify.check_das(1, layers=(1,), seeds=(0,)))
