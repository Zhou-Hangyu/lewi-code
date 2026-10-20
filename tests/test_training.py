"""The training protocol against the recorded validation curves."""
import pytest
from conftest import assert_verified

from csfrbench import datasets, metrics, record, runner, suite, training, verify


def test_first_optimizer_steps_of_recorded_runs():
    (result,) = verify.check_first_steps()
    assert result.status in ("exact", "close"), str(result)
    runs = [record.curves()[k] for k in verify.first_step_runs()]
    have = {(r["setting"], r["workload"] == "depth5_tree", r["arm"]) for r in runs}
    assert have == {(w.setting, w.name == "depth5_tree", a) for s in suite.SETTINGS.values() for w in s.values()
                    for a in w.arms}                    # every method in both settings and on both backbones
    assert all([c[0] for c in r["first_steps"]] == [1, 2, 3] for r in runs)


def test_a_sensitive_run_is_compared_for_a_gross_difference_only():
    runs = record.curves()
    assert all("rounding_sensitivity" in r for r in runs.values())
    stable = [k for k, r in runs.items() if verify.is_stable(r)]
    assert stable and len(stable) < len(runs)
    assert verify.factor(0.2, 0.5) == 2.5 and verify.factor(0.5, 0.2) == 2.5 and verify.factor(0.0, 0.2) == float("inf")


def test_first_checkpoints_of_recorded_runs():
    keys = verify.one_run_per_method(arms=("lewi", "dense")) + verify.one_run_per_method("primary", ("lewi",), deep=True)
    assert sorted(k.split("|")[:3] for k in keys) == [
        ["nuisance", "depth4_tree", "dense"], ["nuisance", "depth4_tree", "lewi"], ["primary", "depth4_tree", "dense"],
        ["primary", "depth4_tree", "lewi"], ["primary", "depth5_tree", "lewi"]]
    assert_verified(verify.check_training(keys, checkpoints=2))


@pytest.mark.slow
def test_first_checkpoints_of_a_run_of_every_method():
    keys = verify.one_run_per_method()
    assert {record.curves()[k]["arm"] for k in keys} == set(suite.ARMS)
    deep = verify.one_run_per_method(deep=True)
    assert {k.split("|")[0] for k in deep} == {"primary", "nuisance"}
    assert_verified(verify.check_training(keys + deep, checkpoints=3))


@pytest.mark.slow
def test_complete_runs():
    keys = verify.one_run_per_method(arms=("lewi", "dense", "go_all"))
    assert len(keys) == 6
    assert_verified(verify.check_training(keys, complete=True))


def test_the_model_of_a_run_is_rebuilt_from_its_state():
    unit = {"setting": "primary", "workload": "reuse_dag", "arm": "lewi", "instance": 6001, "n": 512, "trial": 3}
    run = runner.execute(unit, stop_after_evals=1, keep_states=True)
    target, model = runner.model_of(run, run["best_state"])
    assert not model.training and target.fingerprint == run["fingerprint"]
    data = datasets.make_dataset(target, 32, pools=("val",))
    loader = training.make_loader(training.TensorPool(data["X_val"], data["y_val"]), training.EVAL_BATCH_SIZE)
    assert float(metrics.nmse(model, loader)) == run["best_val_nmse"]
    _target, initial = runner.model_of(run)
    assert all(bool((v == run["init_state"][k]).all()) for k, v in initial.state_dict().items())


def test_seeds_do_not_depend_on_the_method():
    target = suite.workload("primary", "depth4_tree").target(6000)
    a = training.unit_seeds(target.fingerprint, 512, 6000)
    assert set(a) == set(training.STREAMS) and len(set(a.values())) == 5
    assert training.unit_seeds(target.fingerprint, 1024, 6000) != a
    assert training.unit_seeds(target.fingerprint, 512, 6000, seed_replicate=1) != a
    assert training.unit_seeds(target.fingerprint, 512, 6000, seed_replicate=0) == a


def test_learning_rate_schedule():
    import torch
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64000, eta_min=0.1 * 0.01)
    for _ in range(64000):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.001, rel=1e-6)


def test_runner_writes_one_record_per_unit_and_continues(tmp_path):
    units = runner.plan("primary", ["flat_modules"], ["dense"], [6000], [512], [0, 1])
    assert len(units) == 2 and units[0]["hp"] == record.trials("primary")[("flat_modules", "dense", 512)][0]
    first = runner.execute(units[0], stop_after_evals=1)
    assert first["unit"] == "primary.flat_modules.dense.i6000.n512.t00" and len(first["val_curve"]) == 1
    (tmp_path / "runs-test-0.jsonl").write_text(__import__("json").dumps(
        {k: v for k, v in first.items() if k not in ("init_state", "best_state")}) + "\n")
    assert [r["unit"] for r in runner.read_runs(tmp_path)] == [first["unit"]]
    summary = runner.run(units[:1], tmp_path, workers=1, progress=False)
    assert summary == {"units": 1, "on_record": 1, "run": 0, "failed": 0, "diverged": 0}


def test_recorded_winners_carry_the_configuration_of_their_trial():
    """The hyperparameters of a recorded winner, the fixed arguments of its arm included, are those of its trial."""
    for setting in suite.SETTINGS:
        trials = record.trials(setting)
        winners = record.winners(setting)
        assert {w["arm"] for w in winners} == {a for w in suite.SETTINGS[setting].values() for a in w.arms}
        for w in winners:
            assert w["hp"] == trials[(w["workload"], w["arm"], w["n"])][w["trial"]], w
        for unit, w in zip(runner.units_of_winners(winners[:2000:37]), winners[:2000:37]):
            assert unit["hp"] == w["hp"]


def test_the_fixed_arguments_of_an_arm_always_apply():
    unit = {"setting": "nuisance", "workload": "depth4_tree", "arm": "go0", "instance": 6000, "n": 512, "trial": 12,
            "hp": {"lr": 0.01}}                                  # a record of GraphOracle(0) without its level
    a = runner.execute(unit, stop_after_evals=1, keep_states=True)
    assert a["hp"] == {"levels": 0, "lr": 0.01}
    first = a["init_state"]["hidden_layers.0.mask"]
    assert int(first.sum()) == 32 * 8 and bool(a["init_state"]["hidden_layers.1.mask"].all())
    with pytest.raises(ValueError):
        runner.execute({**unit, "hp": {"lr": 0.01, "levels": "all"}}, stop_after_evals=1)


def test_plan_covers_a_setting():
    assert len(runner.plan("primary")) == 354816
    assert len(runner.plan("nuisance")) == 387072
