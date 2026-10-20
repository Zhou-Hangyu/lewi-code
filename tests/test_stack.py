"""The stacked engine against the per-network trainer: the same networks, data order, step, records."""
import copy
import itertools

import pytest
import torch
import torch.nn.functional as F

from csfrbench import runner, search, suite, training
from csfrbench.kernels import ops
from csfrbench.stack import Order, Stack, stacks_of


def units(arm="lewi", n=(512, 1024), trials=range(3), workload="reuse_dag", setting="primary"):
    w = suite.workload(setting, workload)
    return [{"setting": setting, "workload": workload, "arm": arm, "instance": 6000, "n": k, "trial": t,
             "hp": search.configurations(w, arm, k)[t]} for k in n for t in trials]


@pytest.mark.parametrize("arm", ["lewi", "dense"])
def test_stack_holds_the_per_network_models(arm):
    s = Stack(units(arm), kernels=False, compile=False)
    x = torch.randn(s.B, 32, s.d)
    with torch.no_grad():
        mine = s.forward(x)
        ref = torch.stack([m.model(x[i]) for i, m in enumerate(s.members)])
    assert float((mine - ref).abs().max()) < 1e-6


@pytest.mark.parametrize("arm,kernels", [("lewi", False), ("lewi", True), ("dense", False)])
def test_one_step_equals_torch_adam(arm, kernels):
    if kernels and ops() is None:
        pytest.skip("the fused kernels cannot be built here")
    s = Stack(units(arm), kernels=kernels, compile=False)
    s._prepare()
    idx = torch.stack([o.take(1)[0] for o in s.orders])
    inst = s.inst.unsqueeze(1).expand(s.B, 32)
    x, y = s.X_tr[inst, idx], s.y_tr[inst, idx]
    s._step(x, y, s.lr_at(1))
    for i, m in enumerate(s.members):
        model = copy.deepcopy(m.model)
        opt = torch.optim.Adam(model.parameters(), lr=float(m.hp["lr"]), betas=training.ADAM_BETAS, eps=training.ADAM_EPS)
        F.mse_loss(model(x[i]), y[i]).backward()
        opt.step()
        assert torch.allclose(s.params["W0"][i].detach(), model.hidden_layers[0].weight, atol=1e-6)
        assert torch.allclose(s.params["Wo"][i].detach(), model.output_layer.weight, atol=1e-6)
        if arm == "lewi":
            assert torch.allclose(s.params["Ph"][i, 0].detach(), model.gate_codes[1], atol=1e-6)


@pytest.mark.parametrize("n,seed", [(512, 11), (777, 3)])
def test_minibatch_order_equals_the_data_loader(n, seed):
    loader = training.make_loader(training.TensorPool(torch.arange(n)), 32, shuffle=True, drop_last=True,
                                  generator=torch.Generator().manual_seed(seed))
    ref = torch.stack([b[0] for b in itertools.islice(training._cycle(loader), 120)])
    assert torch.equal(Order(n, seed).take(120), ref)


def test_records_and_exported_states():
    s = Stack(units("lewi", n=(512,), trials=range(2)), compile=False)
    s.train(stop_after_evals=1)
    records = s.records(keep_states=True)
    assert [r["unit"] for r in records] == [runner.unit_id(u) for u in units("lewi", n=(512,), trials=range(2))]
    for r in records:
        assert r["steps_run"] == 1000 and r["best_step"] == 1000 and r["test_nmse"] > 0 and "density_overall" in r["diagnostics"]
        target, model = runner.model_of(r, r["best_state"])
        assert set(r["best_state"]) == set(model.state_dict())


def test_stacks_group_by_method_and_workload():
    mixed = units("lewi") + units("dense") + units("l1")
    stacks, rest = stacks_of(mixed, stack_size=4)
    assert {u["arm"] for u in rest} == {"l1"}
    assert all(len({(u["arm"], u["workload"]) for u in s}) == 1 and len(s) <= 4 for s in stacks)
    assert sum(len(s) for s in stacks) == 12
