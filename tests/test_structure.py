"""Associations by correlation alone, on recorded networks."""
import numpy as np
import pytest

from csfrbench import record, runner, structure, viz


@pytest.fixture(scope="module")
def network():
    run = next(r for r in record.checkpoints().values() if r["arm"] == "lewi" and r["setting"] == "primary")
    return runner.model_of(run, run["best"])


def test_correlations_and_threshold(network):
    target, model = network
    corr = structure.correlations(model, target)
    assert len(corr["corr"]) == 6                            # inputs, four hidden layers, output
    assert [c.shape[1] for c in corr["corr"]] == [target.graph.S] * 6
    assert 0.0 < corr["floor"] < 0.2
    assert all(np.all(np.abs(c) <= 1 + 1e-9) for c in corr["corr"])


def test_associations_follow_the_largest_correlation(network):
    target, model = network
    corr = structure.correlations(model, target)
    assoc = structure.associations(model, target, corr=corr)
    for C, a in zip(corr["corr"][1:-1], assoc[1:-1]):
        best = np.abs(C).argmax(1)
        strong = np.abs(C)[np.arange(len(best)), best] >= corr["floor"]
        assert np.array_equal(a, np.where(strong, best, -1))
    read = {int(np.asarray(target.active_raw).reshape(-1)[r]) for refs in target.graph.node_parents for r in refs
            if r < target.graph.A}
    assert {i for i, v in enumerate(assoc[0]) if v >= 0} == read
    assert set(assoc[-1]) == {target.graph.readout_indices[0]}


def test_numbering_is_left_to_right_bottom_to_top(network):
    target, _model = network
    num = viz.numbering(target.graph)
    depths = viz.node_depths(target.graph)
    order = sorted(num, key=num.get)
    assert [depths[v] for v in order] == sorted(depths)
