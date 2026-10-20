"""The world-model HPO re-run of Dense and L1 with the learning-rate lower bound extended by one decade (18-main-d8/swm-lrx;
README.md holds the rule, recorded before any unit ran; manifests/spaces_lrx.json the expanded spaces).

Units (results/csfrbench/swmlrx/<id>.json, the rollout study's custom-unit runner swmfix.run_one on the record's harness, the in5
arm registered as in swm-in5): per cell (task, n, arm) the 32 trials lrx/<task>/n<n>/<arm>/t<k>.json (kind custom, stream hpo: the
record's HPO seed tuple; hp = lewibench.sobol_configs(task, arm, n, expanded space, 32, "final")[k]; no test) and, once the 32 are on
record and the winner is chosen by lewibench.winner_key (no divergence, validation rollout score, one-step validation NMSE, trial),
the 16 finals lrx/<task>/n<n>/<arm>/final-s<s>.json (stream final, the record's paired seed s, test once). selection.json in the cell
folder records the choice. the record as run is read only.
Amendment 1 (README.md; recorded 2026-09-23): once every cell of a (task, arm) has its 16 finals, swm-rollout's
validation-only second stage runs unchanged - the trigger (fragility >= log10 1.5 or an exploded final, and a validation bump >= 0.10
log10 above both neighbour sizes; lrx/<task>/stage2/<arm>.json), the top K = 4 trials on M = 8 seeds of the bank "resel"
(resel-c<trial>-r<r>.json, no test), the selection by the mean log10 validation rollout score (selection2.json), and the 16 finals of a
changed winner (resel-final-s<s>.json, test once). A cell whose selection does not change keeps its first-stage finals.

    python src/lrx.py manifest                 jobs/hpo_units.jsonl (every trial unit with its hyperparameters) + a count check
    python src/lrx.py status                   counts per stage; writes tables/status.json
    python src/lrx.py work <deadline> <worker> the SWM venv: claim-and-run pending units (trials first, then the finals of selected cells)
    python src/lrx.py read                     .venv: the comparison against the record (src/lrx_read.py) -> tables/*.csv, read.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
STUDY = HERE.parents[1]
REPO = HERE.parents[5]
D18 = REPO / "studies/main-experiments/18-main-d8"
for p in (REPO / "studies/main-experiments/16-main-final/src", D18 / "swm-rollout/src", D18 / "swm-in5/src"):
    sys.path.insert(0, str(p))       # swm16 (the record's harness, read only), swmfix (the custom-unit runner), swm_in5 (the in5 arm)
OUT = REPO / "results/csfrbench/swmlrx"
TABLES, JOBS = STUDY / "tables", STUDY / "jobs"
MANIFEST = json.loads((STUDY / "manifests/spaces_lrx.json").read_text())
RECORD = REPO / "studies/main-experiments/16-main-final/external"
ARMS = ["dense", "l1"]
TRIALS, SEEDS = 32, 16
RESERVE_S = 2700                     # a Dense unit at 256x4 takes at most 1,706 s in the record; the rollout study's default reserve
STAGE2_K, STAGE2_M, STAGE2_BANK = 4, 8, "resel"          # amendment 1: swm-rollout's second stage, unchanged (resel.py K, M, BANK)
FRAGILITY_MIN, VAL_BUMP_MIN, EXPLODED, CAP = math.log10(1.5), 0.10, 1.0, 1e6
_CONFIGS: dict = {}


def counted_tasks() -> list[str]:
    return list(json.loads((RECORD / "analysis/read.json").read_text())["claim_bearing"])


def expanded_space(arm: str) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in MANIFEST["arms"][arm]["expanded"].items()}


def cells() -> list[tuple[str, int, str]]:
    import swm16
    return [(t, int(n), a) for t in counted_tasks() for n in swm16.grid_of(t) for a in ARMS]


def cell_root(task: str, n: int, arm: str) -> Path:
    return OUT / "lrx" / task / f"n{n}" / arm


def configs(task: str, n: int, arm: str, lewibench) -> list[dict]:
    key = (task, n, arm)
    if key not in _CONFIGS:
        _CONFIGS[key] = lewibench.sobol_configs(task, arm, n, expanded_space(arm), TRIALS, "final")
    return _CONFIGS[key]


def trial_units(task: str, n: int, arm: str, lewibench) -> list[dict]:
    return [dict(id=f"lrx/{task}/n{n}/{arm}/t{k:02d}", task=task, n=n, arm=arm, kind="custom", index=k, stream="hpo", hp=dict(hp),
                 test=False, trace=False, flush=False) for k, hp in enumerate(configs(task, n, arm, lewibench))]


def final_units(task: str, n: int, arm: str, hp: dict) -> list[dict]:
    return [dict(id=f"lrx/{task}/n{n}/{arm}/final-s{s:02d}", task=task, n=n, arm=arm, kind="custom", index=s, stream="final", hp=dict(hp),
                 test=True, trace=False, flush=False) for s in range(SEEDS)]


def select(task: str, n: int, arm: str, lewibench) -> dict | None:
    """The winner of a cell by the record's key once its 32 trials are on record; written to selection.json (idempotent)."""
    import swm16
    root = cell_root(task, n, arm)
    sel = root / "selection.json"
    if sel.exists():
        return json.loads(sel.read_text())
    recs = []
    for k in range(TRIALS):
        p = root / f"t{k:02d}.json"
        if not p.exists():
            return None
        r = json.loads(p.read_text())
        r["trial"] = k
        recs.append(r)
    w = min(recs, key=lewibench.winner_key)
    out = dict(task=task, n=n, arm=arm, winner_trial=int(w["trial"]), winner_hp=w["hp"], winner_val_rollout_score=w.get("val_rollout_score"),
               winner_best_val_nmse=w.get("best_val_nmse"), winner_best_step=w.get("best_step"), diverged_trials=int(sum(int(r.get("diverged") or 0) for r in recs)),
               trials_summary=[dict(trial=int(r["trial"]), hp=r["hp"], val_rollout_score=r.get("val_rollout_score"), best_val_nmse=r.get("best_val_nmse"),
                                    best_step=r.get("best_step"), diverged=int(r.get("diverged") or 0), error=r.get("error"), wall_s=r.get("wall_s")) for r in recs],
               space=expanded_space(arm), selected_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    tmp = sel.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, default=swm16._json_default))
    os.replace(tmp, sel)
    return out


def _lv(rec: dict, key: str = "val_rollout_score") -> float:
    """log10 of a validation value under the screen's conventions: a diverged or errored run, a missing or NaN value at the cap 1e6."""
    v = rec.get(key)
    if rec.get("diverged") or rec.get("error") or v is None or v != v:
        v = CAP
    return math.log10(min(max(float(v), 1e-12), CAP))


def _write_json(path: Path, obj: dict) -> None:
    import swm16
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(obj, default=swm16._json_default))
    os.replace(tmp, path)


def stage2_group(task: str, arm: str) -> dict | None:
    """The trigger of every cell of (task, arm) on the finals' validation values (swm-rollout screen.py / resel.py cells: fragility,
    exploded finals, the validation bump against both neighbour sizes), evaluated once all cells have their 16 finals; cached."""
    import swm16
    p = OUT / "lrx" / task / "stage2" / f"{arm}.json"
    if p.exists():
        return json.loads(p.read_text())
    grid = swm16.grid_of(task)
    rows = {}
    for n in grid:
        root = cell_root(task, n, arm)
        sp = root / "selection.json"
        if not sp.exists():
            return None
        sel = json.loads(sp.read_text())
        finals = []
        for k in range(SEEDS):
            fp = root / f"final-s{k:02d}.json"
            if not fp.exists():
                return None
            finals.append(json.loads(fp.read_text()))
        lv = sorted(_lv(f) for f in finals)
        mean = sum(lv) / len(lv)
        med = (lv[len(lv) // 2 - 1] + lv[len(lv) // 2]) / 2 if len(lv) % 2 == 0 else lv[len(lv) // 2]
        w = sel.get("winner_val_rollout_score")
        w = CAP if w is None or w != w else float(w)
        rows[str(n)] = dict(n=n, winner_trial=sel["winner_trial"], winner_val_score=w, finals_val_gmean=10 ** mean, finals_val_max=10 ** max(lv),
                            fragility=mean - math.log10(max(w, 1e-12)), exploded_val=sum(1 for x in lv if x > med + EXPLODED))
    v = [math.log10(rows[str(n)]["finals_val_gmean"]) for n in grid]
    for i, n in enumerate(grid):
        r = rows[str(n)]
        r["val_bump_rule"] = None if i == 0 else (v[i] - v[i - 1] if i == len(grid) - 1 else v[i] - max(v[i - 1], v[i + 1]))
        r["fragile"] = bool(r["fragility"] >= FRAGILITY_MIN or r["exploded_val"] > 0)
        r["flagged"] = bool(r["fragile"] and r["val_bump_rule"] is not None and r["val_bump_rule"] >= VAL_BUMP_MIN)
    out = dict(task=task, arm=arm, grid=grid, cells=rows, flagged=[n for n in grid if rows[str(n)]["flagged"]],
               rule=dict(K=STAGE2_K, M=STAGE2_M, bank=STAGE2_BANK, fragility_min=FRAGILITY_MIN, val_bump_min=VAL_BUMP_MIN, exploded=EXPLODED),
               evaluated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _write_json(p, out)
    return out


def stage2_candidates(task: str, n: int, arm: str) -> list[dict] | None:
    """The top K trials of the cell by the record's key; the first-stage winner must rank first (else None: reported, not run)."""
    sel = json.loads((cell_root(task, n, arm) / "selection.json").read_text())
    inf = float("inf")
    key = lambda t: (int(t.get("diverged") or 0), t["val_rollout_score"] if t.get("val_rollout_score") is not None else inf,  # noqa: E731
                     t["best_val_nmse"] if t.get("best_val_nmse") is not None else inf, t["trial"])
    ranked = sorted(sel["trials_summary"], key=key)[:STAGE2_K]
    if ranked[0]["trial"] != sel["winner_trial"]:
        return None
    return [dict(rank=i, trial=int(t["trial"]), hp=dict(t["hp"]), trial_val_score=t.get("val_rollout_score"), trial_val_one_step=t.get("best_val_nmse"))
            for i, t in enumerate(ranked)]


def stage2_sel_units(task: str, n: int, arm: str, cands: list[dict]) -> list[dict]:
    return [dict(id=f"lrx/{task}/n{n}/{arm}/resel-c{c['trial']:02d}-r{r:02d}", task=task, n=n, arm=arm, kind="custom", index=r, stream=STAGE2_BANK,
                 hp=dict(c["hp"]), test=False, trace=False, flush=False, cand=c["trial"]) for c in cands for r in range(STAGE2_M)]


def stage2_final_units(task: str, n: int, arm: str, hp: dict, trial: int) -> list[dict]:
    return [dict(id=f"lrx/{task}/n{n}/{arm}/resel-final-s{s:02d}", task=task, n=n, arm=arm, kind="custom", index=s, stream="final", hp=dict(hp),
                 test=True, trace=False, flush=False, cand=trial) for s in range(SEEDS)]


def stage2_select(task: str, n: int, arm: str, cands: list[dict], have: set) -> dict | None:
    """The second-stage selection once the K x M records are on record (resel.py select, unchanged); selection2.json, idempotent."""
    root = cell_root(task, n, arm)
    p = root / "selection2.json"
    if p.exists():
        return json.loads(p.read_text())
    names = [(c, [f"resel-c{c['trial']:02d}-r{r:02d}.json" for r in range(STAGE2_M)]) for c in cands]
    if any(f not in have for _, fs in names for f in fs):
        return None
    rows = []
    for c, fs in names:
        recs = [json.loads((root / f).read_text()) for f in fs]
        score = [_lv(r) for r in recs]
        one = [_lv(r, "best_val_nmse") for r in recs]
        m = sum(score) / len(score)
        rows.append(dict(c, mean_log10_val_score=m, sd_log10_val_score=(sum((x - m) ** 2 for x in score) / (len(score) - 1)) ** 0.5,
                         max_val_score=10 ** max(score), mean_log10_val_one_step=sum(one) / len(one), scores=[round(10 ** x, 6) for x in score],
                         diverged=int(sum(int(r.get("diverged") or 0) for r in recs))))
    best = min(rows, key=lambda r: (r["mean_log10_val_score"], r["mean_log10_val_one_step"], r["rank"]))
    out = dict(task=task, n=n, arm=arm, first_stage_trial=cands[0]["trial"], selected_trial=best["trial"], selected_rank=best["rank"], selected_hp=best["hp"],
               changed=bool(best["rank"] != 0), candidates=rows, selected_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _write_json(p, out)
    return out



def pending(lewibench) -> tuple[list[dict], dict]:
    """The units still to run (trials of incomplete cells first, then the finals of selected cells, then amendment 1's second-stage units of
    the (task, arm) groups whose cells all have their finals) and the stage counts."""
    import swm16
    units, count = [], dict(cells=0, trials_done=0, cells_selected=0, finals_done=0, stage2_groups=0, stage2_flagged=0, stage2_mismatch=0,
                            stage2_selected=0, stage2_changed=0, stage2_finals_done=0)
    complete: dict[tuple[str, str], set[int]] = {}
    for task, n, arm in cells():
        count["cells"] += 1
        root = cell_root(task, n, arm)
        have = set(os.listdir(root)) if root.exists() else set()
        done = sum(1 for k in range(TRIALS) if f"t{k:02d}.json" in have)
        count["trials_done"] += done
        if done < TRIALS:
            units += [u for u in trial_units(task, n, arm, lewibench) if Path(u["id"]).name + ".json" not in have]
            continue
        sel = select(task, n, arm, lewibench)
        count["cells_selected"] += 1
        fin = sum(1 for s in range(SEEDS) if f"final-s{s:02d}.json" in have)
        count["finals_done"] += fin
        if fin < SEEDS:
            units += [u for u in final_units(task, n, arm, sel["winner_hp"]) if Path(u["id"]).name + ".json" not in have]
        else:
            complete.setdefault((task, arm), set()).add(n)
    for (task, arm), ns in sorted(complete.items()):
        if len(ns) < len(swm16.grid_of(task)):
            continue
        g = stage2_group(task, arm)
        if g is None:
            continue
        count["stage2_groups"] += 1
        for n in g["flagged"]:
            count["stage2_flagged"] += 1
            cands = stage2_candidates(task, n, arm)
            if cands is None:
                count["stage2_mismatch"] += 1
                continue
            root = cell_root(task, n, arm)
            have = set(os.listdir(root))
            s2 = stage2_select(task, n, arm, cands, have)
            if s2 is None:
                units += [u for u in stage2_sel_units(task, n, arm, cands) if Path(u["id"]).name + ".json" not in have]
                continue
            count["stage2_selected"] += 1
            if s2["changed"]:
                count["stage2_changed"] += 1
                done2 = sum(1 for s in range(SEEDS) if f"resel-final-s{s:02d}.json" in have)
                count["stage2_finals_done"] += done2
                if done2 < SEEDS:
                    units += [u for u in stage2_final_units(task, n, arm, s2["selected_hp"], s2["selected_trial"]) if Path(u["id"]).name + ".json" not in have]
    count["pending"] = len(units)
    return units, count


def cmd_status() -> None:
    lewibench, _, _ = _harness()
    units, count = pending(lewibench)
    count["trials_total"] = TRIALS * count["cells"]
    count["finals_total"] = SEEDS * count["cells"]
    count["status_local"] = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    TABLES.mkdir(exist_ok=True)
    (TABLES / "status.json").write_text(json.dumps(count, indent=1))
    print(json.dumps(count))


def cmd_manifest() -> None:
    lewibench, _, _ = _harness()
    units = [u for t, n, a in cells() for u in trial_units(t, n, a, lewibench)]
    assert len(units) == TRIALS * 2 * 119, len(units)
    for u in units:
        lo, hi, _ = expanded_space(u["arm"])["lr"]
        assert lo <= u["hp"]["lr"] <= hi, u
    JOBS.mkdir(exist_ok=True)
    (JOBS / "hpo_units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    below = sum(1 for u in units if u["hp"]["lr"] < MANIFEST["arms"][u["arm"]]["lr_low_record"])
    print(f"{len(units)} trial units -> jobs/hpo_units.jsonl; {below} draws below the record's lower bound")


def _harness():
    import swm16
    import swm_in5
    lewibench, swm_tasks, v05_port = swm16.harness()
    swm_in5.register_arm(v05_port)                     # the in5-enabled stack of the adopted LeWi results (no effect on unmasked arms)
    return lewibench, swm_tasks, v05_port


def cmd_work(deadline: float, worker: int) -> None:
    import swmfix
    lewibench, _, _ = _harness()
    swmfix.OUT = OUT
    done, idle = 0, 0
    while time.time() < deadline - RESERVE_S:
        units, _ = pending(lewibench)
        if not units:
            break
        k = worker % len(units)
        units = units[k:] + units[:k]
        claimed = None
        for u in units:
            path = OUT / f"{u['id']}.json"
            if path.exists():
                continue
            if swmfix.claim(path):
                claimed = (u, path)
                break
        if claimed is None:                           # every pending unit is claimed; finals appear when other workers finish cells
            idle += 1
            if idle > 20:
                break
            time.sleep(60)
            continue
        idle = 0
        u, path = claimed
        t_start = time.time()
        rec, _, _ = swmfix.run_one(u)
        if u["stream"] == "hpo":
            rec["trial"] = u["index"]
        rec["wall_total_s"] = time.time() - t_start
        rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        swmfix.write(path, rec)
        done += 1
        print(f"{time.strftime('%H:%M:%S')} {u['id']} val_score={rec.get('val_rollout_score')} test16={(rec.get('test_rollout') or {}).get('16')} "
              f"{rec['wall_s']:.0f}s {rec.get('error') or ''}", flush=True)
    print(f"worker {worker}: {done} units done", flush=True)


if __name__ == "__main__":
    verb = sys.argv[1]
    if verb == "manifest":
        cmd_manifest()
    elif verb == "status":
        cmd_status()
    elif verb == "work":
        cmd_work(float(sys.argv[2]), int(sys.argv[3]))
    elif verb == "read":
        import lrx_read
        lrx_read.main()
    else:
        raise SystemExit(__doc__)
