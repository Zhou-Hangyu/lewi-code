"""The lewi package: the layer, the embeddings and their checks, the MLP, and its use by CSFRBench's LeWi arm."""
import random

import torch

from csfrbench.seeds import derive
from lewi import LeWiLinear, LeWiMLP, connection_density, derive_seed, init_embeddings, update_masks


def test_seed_derivation_equals_the_benchmarks():
    rng = random.Random(3)
    for _ in range(500):
        parts = (rng.randrange(2**62), rng.choice(["input", "hidden", "output"]), rng.randrange(64), rng.randrange(40))
        assert derive_seed(*parts) == derive(*parts)


def test_mlp_shapes_and_initialization_checks():
    torch.manual_seed(0)
    model = LeWiMLP(in_features=8, width=32, depth=4, out_features=3, seed=7)
    assert model(torch.randn(5, 8)).shape == (5, 3)
    assert [e.shape for e in model.embeddings] == [(8, 3), (32, 3), (32, 3), (32, 3), (32, 3), (3, 3)]
    first = model.layers[0]
    assert int((first.score() > 0).sum(dim=0).min()) >= 5          # every input has at least five open connections
    for layer in model.layers:
        assert bool((layer.score() > 0).any())                     # every layer has an open connection
    density = model.connection_density()["density_overall"]
    assert 0.3 < density < 0.9                                      # about 61% open at the initialization


def test_masks_follow_the_embeddings():
    torch.manual_seed(0)
    model = LeWiMLP(in_features=4, width=8, depth=2, seed=1)
    layer = model.layers[0]
    before = layer.event_mask.clone()
    with torch.no_grad():
        model.embeddings[1].add_(10.0)                              # the first hidden layer moves far away
    assert torch.equal(layer.event_mask, before)                    # the mask changes only on update_masks
    model.update_masks()
    assert float(layer.event_mask.sum()) == 0.0


def test_gradients_reach_weights_and_embeddings():
    torch.manual_seed(0)
    model = LeWiMLP(in_features=8, width=16, depth=3, seed=2)
    loss = model(torch.randn(64, 8)).square().mean()
    loss.backward()
    assert all(e.grad is not None and torch.isfinite(e.grad).all() for e in model.embeddings)
    assert all(layer.weight.grad is not None for layer in model.layers)


def test_custom_chain_of_layers():
    layers = [LeWiLinear(6, 10), LeWiLinear(10, 10), LeWiLinear(10, 2, is_output=True)]
    embeddings, layer_redraws, input_redraws = init_embeddings(layers, seed=11)
    assert len(embeddings) == 4 and layer_redraws >= 0 and input_redraws >= 0
    update_masks(layers)
    assert set(connection_density(layers)) == {"density_overall", "density_input", "density_hidden", "density_readout"}


def test_benchmark_arm_uses_the_package():
    from csfrbench import suite, training
    w = suite.workload("primary", "depth4_tree")
    _method, model, seeds = training.build_model("lewi", w.target(6000), width=32, hidden=4, hp={"lr": 1e-3}, n=512)
    assert all(isinstance(layer, LeWiLinear) for layer in model.linear_layers())
    again, _a, _b = init_embeddings([LeWiLinear(l.in_features, l.out_features, is_output=l.is_output)
                                     for l in model.linear_layers()], seed=seeds["gate_init"])
    for mine, theirs in zip(again, model.gate_codes):
        assert torch.equal(mine, theirs)
