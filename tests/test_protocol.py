"""The constants of the benchmark: settings, protocol, search spaces, selection rule."""
import math

import pytest
from conftest import assert_verified

from csfrbench import analysis, metrics, search, suite, verify


def test_settings_and_protocol_constants():
    assert_verified(verify.check_suite())


def test_search_spaces_equal_table_2_of_the_paper():
    assert_verified(verify.check_spaces())


def test_every_arm_of_every_workload_has_a_search_space():
    for setting in suite.SETTINGS.values():
        for w in setting.values():
            for arm in w.arms:
                dims = search.space(w.space_set_of(arm), arm)
                assert dims and dims[-1]["name"] == "lr"
                assert {d["name"] for d in dims} - {"lr"} <= suite.METHOD_KEYS[arm]


def test_configurations_equal_the_record():
    assert_verified(verify.check_configurations())


def test_configurations_do_not_depend_on_the_instance_and_lie_in_the_space():
    w = suite.workload("primary", "depth4_tree")
    for arm in w.arms:
        dims = {d["name"]: d for d in search.space(w.space_set_of(arm), arm)}
        for hp in search.configurations(w, arm, 512):
            for name, dim in dims.items():
                assert dim["low"] <= hp[name] <= dim["high"]
            assert {k: v for k, v in hp.items() if k not in dims} == suite.ARMS[arm].fixed


def test_construction_rule_of_a_search_space():
    assert_verified(verify.check_freeze_rule())


def test_selection_rule():
    runs = [{"trial": t, "best_val_nmse": 1.0 - 0.01 * t, "diverged": 0} for t in range(32)]
    assert search.select(runs)["trial"] == 31
    runs[31]["diverged"] = 1                        # a diverged run is never selected
    assert search.select(runs)["trial"] == 30
    runs[30]["best_val_nmse"] = runs[29]["best_val_nmse"]
    assert search.select(runs)["trial"] == 29       # a tie goes to the lowest trial
    assert search.select(runs[:31]) is None         # an incomplete condition is not selected


def test_aulc_is_the_mean_height_over_log2_n():
    assert metrics.aulc([512, 1024, 2048], [0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert metrics.aulc([512, 1024, 2048], [1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert metrics.aulc([512, 2048, 4096], [1.0, 0.0, 0.0]) == pytest.approx(1.0 / 3.0)


def test_student_t_interval():
    out = metrics.t_interval([1.0, 2.0, 3.0, 4.0])
    assert out["mean"] == pytest.approx(2.5)
    assert out["half_width"] == pytest.approx(3.182446305284263 * math.sqrt(5.0 / 3.0) / 2.0)


def test_tables_from_the_recorded_runs_equal_the_record_and_the_paper():
    assert_verified(verify.check_analysis())


def test_search_space_audit_of_the_paper():
    assert_verified(verify.check_search_audit())
    dimension = {"name": "lr", "low": 1e-4, "high": 1.0, "log": True, "hard": []}
    audit = search.boundary_occupancy(dimension, [1e-4, 2e-4, 1e-2, 0.5, 1.0])      # one decade is 25% of the space
    assert audit == {"at_low": 0.4, "at_high": 0.4, "median_position": 0.5, "n": 5}


def test_paired_comparison_counts_wins():
    inst = [{"workload": "depth4_tree", "arm": a, "instance": i, "aulc": v}
            for a, values in (("lewi", (0.1, 0.2, 0.3)), ("dense", (0.2, 0.2, 0.2))) for i, v in enumerate(values)]
    (row,) = analysis.contrasts(inst)
    assert row["wins"] == 1 and row["mean"] == pytest.approx(0.0)
