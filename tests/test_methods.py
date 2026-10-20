"""The methods: construction, mechanism and the recorded states."""
import pytest
import torch
from conftest import assert_verified

from csfrbench import record, suite, training, verify
from csfrbench.methods import METHODS, Method, get_method, graph_oracle, lewi, register, rigl
from csfrbench.models import DenseMLP

HP = {"dense": {}, "l1": {"lam": 0.01}, "l2": {"lam": 0.01}, "l0": {"lam": 0.01, "p_open_init": 0.8318},
      "rigl": {"density": 0.2}, "bimt": {"lam": 0.01}, "fiedler": {"delta": 0.01}, "positions": {}, "lewi": {},
      "graph_oracle": {"levels": "all"}}


def build(method, workload="depth4_tree", setting="nuisance", instance=6000, n=512, hp=None):
    w = suite.workload(setting, workload)
    target = w.target(instance)
    m, model, seeds = training.build_model(method, target, width=w.profile.width, hidden=w.profile.hidden,
                                           hp={**(HP[method] if hp is None else hp), "lr": 1e-3}, n=n)
    return w, target, m, model


def test_registered_methods():
    assert set(METHODS) == set(HP)
    assert {a.method for a in suite.ARMS.values()} == set(HP)


def test_recorded_states():
    assert_verified(verify.check_states())


@pytest.mark.parametrize("method", sorted(HP))
def test_methods_share_the_backbone_weights_and_are_deterministic(method):
    _w, _t, _m, a = build(method)
    _w, _t, _m, b = build(method)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    if method in ("dense", "l1", "l2", "bimt", "fiedler", "l0", "lewi"):
        _w, _t, _m, dense = build("dense")
        for i, layer in enumerate(a.hidden_layers):
            assert torch.equal(layer.weight, dense.hidden_layers[i].weight)       # the same raw weights


def test_unknown_hyperparameter_is_refused():
    with pytest.raises(ValueError):
        build("l1", hp={"lam": 0.1, "lambda": 0.1})


def test_a_method_object_holds_no_state():
    with pytest.raises(AttributeError):
        get_method("dense").anything = 1


def test_penalties_act_on_the_hidden_transitions_only():
    for method in ("l1", "l2"):
        _w, _t, m, model = build(method)
        m.penalty(model, {"lam": 1.0}).backward()
        assert all(layer.weight.grad is not None for layer in model.hidden_layers)
        assert model.output_layer.weight.grad is None
    _w, _t, m, model = build("l1")
    expected = sum(float(layer.weight.detach().abs().sum()) for layer in model.hidden_layers) / sum(
        layer.weight.numel() for layer in model.hidden_layers)
    assert m.penalty(model, {"lam": 1.0}).item() == pytest.approx(expected, rel=1e-6)


def test_lewi_constants_and_mask():
    w, target, m, model = build("lewi")
    assert (lewi.EMBEDDING_DIM, lewi.EMBEDDING_STD, lewi.MIN_OPEN_INPUT) == (3, 0.5, 5)
    layers = model.linear_layers()
    assert len(model.gate_codes) == len(layers) + 1 and model.gate_codes[-1].shape == (1, 3)
    for layer in layers:                                        # the readout included
        assert float(layer.gate_offset) == 1.5 and float(layer.gate_scale) == 1.0
        assert not layer.gate_offset.requires_grad and not layer.gate_scale.requires_grad
        assert torch.equal(layer.event_mask, (layer.gate() > 0.5).float())
        assert bool(layer.event_mask.any())
    assert int(layers[0].event_mask.sum(dim=0).min()) >= 5      # every input has at least 5 open connections
    density = lewi.connection_density(model)["density_overall"]
    assert 0.5 < density < 0.72                                 # 61% of the connections are open on average
    x = torch.randn(16, target.d)
    manual = x
    for layer in model.hidden_layers:
        manual = torch.relu(torch.nn.functional.linear(manual, layer.weight * layer.gate() * layer.event_mask, layer.bias))
    out = model.output_layer
    manual = torch.nn.functional.linear(manual, out.weight * out.gate() * out.event_mask, out.bias).squeeze(-1)
    assert torch.equal(model(x), manual)


def test_lewi_initial_density_is_61_percent():
    open_edges = edges = 0
    for instance in suite.INSTANCES:
        _w, _t, _m, model = build("lewi", setting="primary", instance=instance)
        for layer in model.linear_layers():
            open_edges += float(layer.event_mask.sum())
            edges += layer.event_mask.numel()
    assert open_edges / edges == pytest.approx(0.61, abs=0.015)


def test_lewi_mask_follows_the_embeddings_after_a_step():
    _w, target, m, model = build("lewi")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    x, y = torch.randn(64, target.d), torch.randn(64)
    for step in range(1, 6):
        optimizer.zero_grad()
        ((model(x) - y) ** 2).mean().backward()
        optimizer.step()
        m.after_step(model, optimizer, step=step, total_steps=10)
        for layer in model.linear_layers():
            assert torch.equal(layer.event_mask, (layer.gate() > 0.5).float())
    closed = model.hidden_layers[1].event_mask == 0
    assert model.hidden_layers[1].effective_weight()[closed].abs().max().item() == 0.0


def test_lewi_input_check_is_recorded():
    runs = [r for r in record.checkpoints().values() if r["arm"] == "lewi" and r["input_redraws"] > 0]
    assert runs, "the package holds recorded runs in which the input check acts"
    for run in runs:
        w = suite.workload(run["setting"], run["workload"])
        _m, model, _s = training.build_model("lewi", w.target(run["instance"]), width=w.profile.width,
                                             hidden=w.profile.hidden, hp=run["hp"], n=run["n"])
        assert model.lewi_input_redraws == run["input_redraws"]
        assert all(torch.equal(v, run["init"][k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize("workload", list(suite.SETTINGS["primary"]))
def test_graph_oracle_masks(workload):
    w = suite.workload("nuisance", workload)
    target = w.target(6000)
    width, hidden = w.profile.width, w.profile.hidden
    assert sum(graph_oracle.level_lengths(hidden, w.levels)) == hidden
    support = graph_oracle.graph_masks(target, width, hidden, 0)
    assert int(support[0].sum()) == width * w.active and all(bool(m.all()) for m in support[1:])
    full = graph_oracle.graph_masks(target, width, hidden, "all")
    assert [tuple(m.shape) for m in full] == [(width, w.d)] + [(width, width)] * (hidden - 1) + [(1, width)]
    inert = [j for j in range(w.d) if j not in target.active_inputs]
    assert float(full[0][:, inert].sum()) == 0.0
    for rung in (a for a in w.arms if a in ("go1", "go2")):
        level = suite.ARMS[rung].fixed["levels"]
        part = graph_oracle.graph_masks(target, width, hidden, level)
        assert not graph_oracle.is_full(target, width, hidden, level)        # every rung of the suite differs from the full one
        assert all(bool((p >= f).all()) for p, f in zip(part, full))         # a rung restricts less than the full oracle
        assert bool((part[0] <= support[0]).all())
    if w.levels == 1 and workload == "high_arity_node":
        assert all(torch.equal(a, b) for a, b in zip(full, support))         # one node over all inputs: GO(all) = GO(0)


def test_graph_oracle_is_initialized_at_the_kept_fan_in():
    _w, _t, _m, model = build("graph_oracle")
    for layer in model.hidden_layers:
        assert (layer.weight * (1 - layer.mask)).abs().max().item() == 0.0
    first = model.hidden_layers[0]
    fan_in = first.mask.sum(dim=1)
    assert float(fan_in.min()) >= 1


def test_rigl_density_and_update_steps():
    _w, _t, _m, model = build("rigl", hp={"density": 0.25})
    masks = [layer.mask for layer in model.hidden_layers]
    total = sum(float(m.sum()) for m in masks) / sum(m.numel() for m in masks)
    assert total == pytest.approx(0.25, abs=0.01)
    assert isinstance(model.output_layer.weight, torch.nn.Parameter) and not hasattr(model.output_layer, "mask")
    steps = [s for s in range(1, 64001) if rigl._is_update_step(s, int(0.75 * 64000))]
    assert steps[0] == 1 and steps[1] == 100 and steps[-1] == 48000 and len(steps) == 481
    assert rigl.erk_densities([(32, 8), (32, 32)], 1.0) == [1.0, 1.0]


def test_bimt_permutation_keeps_the_function():
    _w, target, m, model = build("bimt")
    x = torch.randn(32, target.d)
    before = model(x).clone()
    m._swap(model, list(model.linear_layers()), 0, 1, 5)                      # exchange two inputs
    m._swap(model, list(model.linear_layers()), 2, 3, 7)                      # exchange two hidden units
    assert torch.allclose(model(x), before, atol=1e-6)
    assert int((model.bimt_in_perm != torch.arange(target.d)).sum()) == 2


def test_a_new_method_can_be_registered_and_trained():
    @register("test_only_dense_copy")
    class Copy(Method):
        def build(self, cfg, target):
            return DenseMLP(target.d, cfg["width"], cfg["hidden"])

    w = suite.workload("primary", "flat_modules")
    a = training.run_unit(w.target(6000), method="test_only_dense_copy", hp={"lr": 0.01}, n=512, width=32, hidden=4,
                          max_steps=64000, eval_every=1000, stop_after_evals=1)
    b = training.run_unit(w.target(6000), method="dense", hp={"lr": 0.01}, n=512, width=32, hidden=4,
                          max_steps=64000, eval_every=1000, stop_after_evals=1)
    assert a["val_curve"] == b["val_curve"]
    METHODS.pop("test_only_dense_copy")
