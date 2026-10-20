"""The SWM-16 record under the input rule in5 (a check of 2026-09-23: are the world-model results run with the same
initialization rules as CSFRBench's LeWi - every input at least five open first-transition connections, the output at least one?).

The record (16-main-final/external) built LeWi as the v1.1 package's ``lewi-v1`` with the structural readout: the readout rule was
in force (every record carries readout_redraws; none fired); no input rule existed in that stack. tables/init_census.csv rebuilds
every LeWi initialization of the record (133 cells x [the search seed + 16 final seeds]) and counts each input's open connections:
the rule changes 30 final-seed initializations (23 cells, 13 tasks) and no search-seed initialization, so every cell's selected
configuration stands. This study re-runs exactly those 30 finals under the rule, plus two controls whose initialization the rule
leaves alone (they must reproduce the record), as TRIAL data: the record as run is read only; the adoption came afterwards (README).

    python src/swm_in5.py manifest                              # tables/init_census.csv -> jobs/finals_in5.jsonl (30 + 2 controls)
    python src/swm_in5.py work <manifest> <deadline_epoch_s> <worker>   # the SWM venv: the arm lewi-v1-in5 registered on the pinned v1.1
                                                                #   package, then the rollout study's claim-and-run loop -> results/csfrbench/swmin5/<id>.json
    python src/swm_in5.py read                                  # the repo venv: tables/instances_in5.csv, cells_in5.csv, paired_in5.csv,
                                                                #   paired_in5_changes.csv, tally_in5.csv, controls_in5.csv

The arm ``lewi-v1-in5`` = the package's LeWiV1 (readout: structural) with the input rule of 18-main-d8/input-cutoff/src/cutoff_methods.py:
after the build, an input with fewer than five open first-transition connections has its position drawn again by indexed attempts from
derive(auxiliary seed, "input", i, attempt); attempt 0 is the record's own draw, so a unit without a redraw is the record bit for bit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
STUDY = HERE.parents[1]
REPO = HERE.parents[5]
ROLLOUT = REPO / "studies/main-experiments/18-main-d8/swm-rollout/src"
sys.path.insert(0, str(ROLLOUT))
sys.path.insert(0, str(REPO / "studies/main-experiments/16-main-final/src"))
OUT = REPO / "results/csfrbench/swmin5"
TABLES = STUDY / "tables"
JOBS = STUDY / "jobs"
RECORD = REPO / "studies/main-experiments/16-main-final/external"
ARM = "lewi-v1-in5"
MIN_OPEN = 5
CONTROLS = [("cheetah", 65536, 0), ("quadruped", 32768, 1)]      # affected cells, seeds the rule leaves alone: must reproduce the record


def register_arm(v05_port):
    """The study arm on the pinned package (after swm16.harness() put it first)."""
    import torch
    import csfrbench.registry as registry
    from csfrbench.methods.lewi_v1 import LeWiV1
    from csfrbench.methods.cmask import CODE_INIT
    from csfrbench.utils.seeds import derive

    class LeWiV1In5(LeWiV1):
        min_open = MIN_OPEN

        def build(self, cfg, target):
            model = super().build(cfg, target)
            layer, codes = model.hidden_layers[0], model.gate_codes[0]
            model.lewi_input_redraws = 0
            with torch.no_grad():
                for i in range(codes.shape[0]):
                    attempt = 0
                    while int((layer.score()[:, i] > 0).sum()) < self.min_open:
                        attempt += 1
                        g = torch.Generator().manual_seed(int(derive(int(cfg["auxiliary_init_seed"]), "input", i, attempt)))
                        codes[i].copy_(torch.randn(codes.shape[1], generator=g) * CODE_INIT)
                        model.lewi_input_redraws += 1
            if model.lewi_input_redraws:
                self._rederive_masks(model)
            return model

        def extra_diagnostics(self, model):
            out = dict(super().extra_diagnostics(model))
            out["input_redraws"] = int(getattr(model, "lewi_input_redraws", 0))
            return out

    registry.METHODS[ARM] = LeWiV1In5
    v05_port.REGISTRY_KEY[ARM] = ARM
    v05_port.FIXED_HP[ARM] = {"readout": "structural"}
    assert v05_port.hp_defaults(ARM) == {"readout": "structural"}
    assert str(Path(sys.modules["csfrbench"].__file__).resolve()).startswith(str(REPO / "studies/csfrbench/final-design/versions/v1.1/implementation"))
    return LeWiV1In5


def winner_hp(task: str, n: int) -> dict:
    return dict(json.loads((RECORD / "results/final/hpo" / task / f"n{n}" / "lewi-v1.json").read_text())["winner_hp"])


def cmd_manifest() -> None:
    import pandas as pd
    c = pd.read_csv(TABLES / "init_census.csv")
    fin = c[c.stream == "final"]
    hit = fin[fin.inputs_below_5 > 0].sort_values(["task", "n", "seed"])
    assert int((c[c.stream == "hpo"].inputs_below_5 > 0).sum()) == 0, "a search-seed initialization changes: the cell's selection would change"
    units = []
    for r in hit.itertuples():
        units.append(dict(id=f"in5/{r.task}/n{int(r.n)}/lewi-v1/s{int(r.seed):02d}", task=r.task, n=int(r.n), arm=ARM, kind="custom", index=int(r.seed),
                          stream="final", hp=winner_hp(r.task, int(r.n)), test=True, trace=False, flush=False, role="rule",
                          inputs_below_5=int(r.inputs_below_5), min_open_per_input=int(r.min_open_per_input)))
    for task, n, s in CONTROLS:
        row = fin[(fin.task == task) & (fin.n == n) & (fin.seed == s)].iloc[0]
        assert int(row.inputs_below_5) == 0, (task, n, s)
        units.append(dict(id=f"in5ctrl/{task}/n{n}/lewi-v1/s{s:02d}", task=task, n=n, arm=ARM, kind="custom", index=s, stream="final",
                          hp=winner_hp(task, n), test=True, trace=False, flush=False, role="control", inputs_below_5=0, min_open_per_input=int(row.min_open_per_input)))
    JOBS.mkdir(exist_ok=True)
    (JOBS / "finals_in5.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    print(len(units), "units ->", JOBS / "finals_in5.jsonl", f"({len(hit)} under the rule + {len(CONTROLS)} controls)")


def cmd_work(manifest: str, deadline: float, worker: int) -> None:
    import swmfix
    import swm16
    lewibench, _, v05_port = swm16.harness()
    register_arm(v05_port)
    import torch
    swmfix.OUT = OUT
    units = [json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip()]
    units = units[worker % max(len(units), 1):] + units[: worker % max(len(units), 1)]
    done = 0
    for u in units:
        path = OUT / f"{u['id']}.json"
        if path.exists():
            continue
        if time.time() > deadline - swmfix.RESERVE_S["lewi-v1"]:
            continue
        if not swmfix.claim(path):
            continue
        t_start = time.time()
        rec, model, cell_obj = swmfix.run_one(u)
        try:                                                        # the initialization under the rule: redraws and open connections per input
            cell = swmfix.unit_cell(u)
            hp, seeds = swmfix.unit_hp_seeds(u, lewibench)
            in_dim, out_dim = cell_obj["train"].tensors[0].shape[1], cell_obj["train"].tensors[1].shape[1]
            torch.manual_seed(seeds["init"])
            m0 = v05_port.build_model(u["arm"], in_dim, out_dim, dict(v05_port.hp_defaults(u["arm"]) | dict(hp)), seeds["aux"], cell["width"], cell["hidden"])
            rec.setdefault("init", {})
            rec["init"]["input_redraws"] = int(getattr(m0, "lewi_input_redraws", 0))
            rec["init"]["per_input_open"] = swmfix.per_input_open(m0)
        except Exception as exc:  # noqa: BLE001
            rec["init_error"] = f"{type(exc).__name__}: {exc}"
        rec["role"] = u.get("role")
        rec["wall_total_s"] = time.time() - t_start
        rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        swmfix.write(path, rec)
        done += 1
        print(f"{time.strftime('%H:%M:%S')} {u['id']} redraws={rec.get('init', {}).get('input_redraws')} test16={(rec.get('test_rollout') or {}).get('16')} {rec['wall_s']:.0f}s {rec.get('error') or ''}", flush=True)
    print(f"worker {worker}: {done} units done", flush=True)


def _load(u: dict) -> dict | None:
    p = OUT / f"{u['id']}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cmd_read() -> None:
    import numpy as np
    import pandas as pd
    import swm16
    units = [json.loads(l) for l in (JOBS / "finals_in5.jsonl").read_text().splitlines() if l.strip()]
    inst = pd.read_csv(RECORD / "analysis" / "instances.csv")
    old = inst.set_index(["task", "n", "arm", "seed"])
    rows = []
    for u in units:
        r = _load(u)
        srow = old.loc[(u["task"], u["n"], "lewi-v1", u["index"])]
        rec_json = json.loads((RECORD / "results/final/final" / u["task"] / f"n{u['n']}" / "lewi-v1" / f"s{u['index']:02d}.json").read_text())
        row = dict(task=u["task"], n=u["n"], seed=u["index"], role=u["role"], lr=u["hp"].get("lr"), inputs_below_5=u["inputs_below_5"], min_open_per_input=u["min_open_per_input"],
                   record_log10=float(srow["log10"]), record_nmse=float((rec_json.get("test_rollout") or {}).get("16", srow["nmse"])),   # the record's own JSON (the CSV holds 16 digits)
                   record_one_step=float((rec_json.get("test_one_step") or {}).get("nmse", srow["one_step"])), record_best_step=int(rec_json.get("best_step") or 0),
                   state="pending" if r is None else ("error" if r.get("error") else "done"))
        if r is not None and not r.get("error"):
            row.update(log10=swm16.log10_nmse(r), nmse=(r.get("test_rollout") or {}).get("16"), one_step=(r.get("test_one_step") or {}).get("nmse"),
                       best_step=r.get("best_step"), val_rollout_score=r.get("val_rollout_score"), diverged=int(bool(r.get("diverged"))),
                       input_redraws=(r.get("init") or {}).get("input_redraws"), min_open_after=min((r.get("init") or {}).get("per_input_open") or [np.nan]),
                       open_edges_init=json.dumps((r.get("init") or {}).get("open_edges_init")), wall_s=r.get("wall_s"))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "instances_in5.csv", index=False)
    print(df[["task", "n", "seed", "role", "state", "record_nmse", "nmse", "input_redraws", "min_open_per_input", "min_open_after"]].to_string())
    ctrl = df[(df.role == "control") & (df.state == "done")]
    ctrl_ok = ctrl.assign(nmse_equal=lambda d: np.isclose(d.nmse, d.record_nmse, rtol=0, atol=0), one_step_equal=lambda d: np.isclose(d.one_step, d.record_one_step, rtol=0, atol=0),
                          best_step_equal=lambda d: d.best_step == d.record_best_step)
    ctrl_ok[["task", "n", "seed", "record_nmse", "nmse", "nmse_equal", "record_one_step", "one_step", "one_step_equal", "best_step_equal", "input_redraws"]].to_csv(TABLES / "controls_in5.csv", index=False)
    rule = df[(df.role == "rule") & (df.state == "done")]
    if rule.empty:
        print("no rule unit read yet"); return
    # every cell with a changed final: the cell's 16 finals with the rule's rows in place; the record's other seeds unchanged
    inst2 = inst.set_index(["task", "n", "arm", "seed"]).copy()
    for r in rule.itertuples():
        inst2.loc[(r.task, r.n, "lewi-v1", r.seed), "log10"] = r.log10
    inst2 = inst2.reset_index()
    cells_new = []
    for (task, n), g in rule.groupby(["task", "n"]):
        allg = inst2[(inst2.task == task) & (inst2.n == n) & (inst2.arm == "lewi-v1")]
        rec_g = inst[(inst.task == task) & (inst.n == n) & (inst.arm == "lewi-v1")]
        a, b = swm16.student_t(allg.log10.values), swm16.student_t(rec_g.log10.values)
        cells_new.append(dict(task=task, n=n, arm="lewi-v1", seeds_changed=len(g), record_gmean=10 ** b["mean"], in5_gmean=10 ** a["mean"], in5_glo=10 ** a["lo"], in5_ghi=10 ** a["hi"],
                              dlog10_cell=a["mean"] - b["mean"], seeds_better=int((g.log10 < g.record_log10).sum()), dlog10_changed_seeds_mean=float((g.log10 - g.record_log10).mean())))
    cells_df = pd.DataFrame(cells_new)
    cells_df.to_csv(TABLES / "cells_in5.csv", index=False)
    print(cells_df.round(4).to_string())
    old_paired = pd.read_csv(RECORD / "analysis" / "contrasts.csv") if (RECORD / "analysis" / "contrasts.csv").exists() else None
    rows = []
    for task in sorted(inst2.task.unique()):
        grid = swm16.grid_of(task)
        au = {}
        for arm, g in inst2[inst2.task == task].groupby("arm"):
            by = g.set_index(["seed", "n"]).log10
            au[arm] = {s_: swm16.aulc_log(grid, [by.loc[(s_, n)] for n in grid]) for s_ in sorted(g.seed.unique()) if all((s_, n) in by.index for n in grid)}
        for ref in swm16.ARMS:
            if ref == "lewi-v1" or ref not in au or "lewi-v1" not in au:
                continue
            idx = sorted(set(au["lewi-v1"]) & set(au[ref]))
            d = np.array([au["lewi-v1"][i] - au[ref][i] for i in idx])
            st = swm16.student_t(d)
            rows.append(dict(task=task, reference=ref, dlog10_mean=st["mean"], dlog10_lo=st["lo"], dlog10_hi=st["hi"], better=int((d < 0).sum()), seeds=len(idx),
                             verdict="win" if st["hi"] < 0 else "loss" if st["lo"] > 0 else "tie"))
    paired = pd.DataFrame(rows)
    paired.to_csv(TABLES / "paired_in5.csv", index=False)
    if old_paired is not None:
        m = paired.merge(old_paired.rename(columns={"mean": "dlog10_mean", "lo": "dlog10_lo", "hi": "dlog10_hi"}), on=["task", "reference"], suffixes=("", "_record"))
        chg = m[(m.dlog10_mean - m.dlog10_mean_record).abs() > 1e-9]
        cols = [c for c in ["task", "reference", "dlog10_mean_record", "verdict_record", "dlog10_mean", "dlog10_lo", "dlog10_hi", "verdict"] if c in chg.columns]
        chg[cols].to_csv(TABLES / "paired_in5_changes.csv", index=False)
        print(chg[cols].round(4).to_string())
        counted = set(json.loads((RECORD / "analysis" / "read.json").read_text())["claim_bearing"])
        tally = []
        for ref in [a_ for a_ in swm16.ARMS if a_ != "lewi-v1"]:
            row = {"reference": ref}
            for name, d0 in (("record", old_paired), ("in5", paired)):
                for scope, d_ in (("all19", d0[d0.reference == ref]), ("counted17", d0[(d0.reference == ref) & d0.task.isin(counted)])):
                    row[f"{name}_{scope}"] = "-".join(str(int((d_.verdict == v).sum())) for v in ("win", "tie", "loss"))
            tally.append(row)
        pd.DataFrame(tally).to_csv(TABLES / "tally_in5.csv", index=False)
        print(pd.DataFrame(tally).to_string())


if __name__ == "__main__":
    verb = sys.argv[1]
    if verb == "manifest":
        cmd_manifest()
    elif verb == "work":
        cmd_work(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
    elif verb == "read":
        cmd_read()
    else:
        raise SystemExit(__doc__)
