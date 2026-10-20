"""Study-local runner of the SWM rollout study (18-main-d8/swm-rollout). The campaign source (16-main-final/src/swm16.py)
and the sealed records are read only; every unit of this study writes one JSON record under
results/csfrbench/swmfix/<id>.json (plus <id>.pt when the unit is traced).

    python swmfix.py work <manifest.jsonl> <deadline_epoch_s> <worker_index>   claim-and-run the manifest's units
    python swmfix.py initcounts <out.csv> task:n [task:n ...]                  open connections per input at initialization
    python swmfix.py probe <manifest.jsonl>                                    the probes again on saved traced models

A unit (one JSON line): id, task, n, arm, kind (final | hpo | custom), index, stream, hp (custom only), test (bool),
trace (bool), flush (bool). kind final / hpo take the hyperparameters and the seeds of the record as run's unit, so
the run must reproduce the record; kind custom trains the given hyperparameters on seed `index` of bank `stream`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "studies/main-experiments/16-main-final/src"))
import swm16  # noqa: E402  (read only)

OUT = REPO / "results/csfrbench/swmfix"
RESERVE_S = {"lewi-v1": 3300, "erb": 3300, "l0hc": 3300}
RESERVE_DEFAULT_S = 2700


def unit_cell(u: dict) -> dict:
    c = swm16.contract()
    return dict(task=u["task"], n=int(u["n"]), arm=u["arm"], corpus=swm16.corpus_of(u["task"]), width=c["width"], hidden=c["hidden"],
                max_steps=int(u.get("max_steps") or c["max_steps"]), mode="final", setting=None)


def unit_hp_seeds(u: dict, lewibench) -> tuple[dict, dict]:
    cell = unit_cell(u)
    if u["kind"] == "hpo":
        hp = lewibench.sobol_configs(u["task"], u["arm"], cell["n"], swm16.spaces()[u["arm"]], swm16.TRIALS, "final")[u["index"]]
        return hp, lewibench.trial_seeds(u["task"], cell["n"])
    if u["kind"] == "final":
        hp = dict(json.loads(swm16.winner_path(cell).read_text())["winner_hp"])
        return hp, lewibench.final_seeds(u["task"], cell["n"], u["index"], "final")
    seeds = lewibench.trial_seeds(u["task"], cell["n"]) if u["stream"] == "hpo" else lewibench.final_seeds(u["task"], cell["n"], u["index"], u["stream"])
    return dict(u["hp"]), seeds


def per_input_open(model) -> list[int] | None:
    layers = getattr(model, "structural_layers", None)
    layers = layers() if callable(layers) else []
    mask = getattr(layers[0], "event_mask", None) if layers else None
    return None if mask is None else [int(v) for v in mask.sum(dim=0).tolist()]      # columns = senders = inputs


def run_one(u: dict):
    """swm16.run_unit with explicit hyperparameters and seeds; returns the record, the model and the cell."""
    lewibench, _, v05_port = swm16.harness()
    import torch
    torch.set_num_threads(1)
    torch.set_flush_denormal(bool(u.get("flush")))
    t0 = time.time()
    cell = unit_cell(u)
    cell_obj = swm16.cell_data(lewibench, cell["task"], cell["n"], cell["corpus"])
    hp, seeds = unit_hp_seeds(u, lewibench)
    in_dim, out_dim = cell_obj["train"].tensors[0].shape[1], cell_obj["train"].tensors[1].shape[1]
    full_hp = dict(v05_port.hp_defaults(u["arm"]) | dict(hp))
    rec = {"id": u["id"], "task": u["task"], "n": cell["n"], "arm": u["arm"], "kind": u["kind"], "index": u["index"], "stream": u.get("stream"),
           "hp": hp, "hp_full": full_hp, "seeds": seeds, "width": cell["width"], "hidden": cell["hidden"], "max_steps": cell["max_steps"],
           "batch": lewibench.BATCH, "corpus": cell["corpus"], "split_hash": swm16.split_hash(cell_obj), "flush_denormal": bool(u.get("flush")),
           "package": str(Path(sys.modules["csfrbench"].__file__).parent), "job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))}
    model = None
    try:
        rec["init"] = swm16.init_open_edges(v05_port, u["arm"], in_dim, out_dim, full_hp, seeds, cell["width"], cell["hidden"])
        if u.get("trace"):
            torch.manual_seed(seeds["init"])
            rec["init"]["per_input_open"] = per_input_open(v05_port.build_model(u["arm"], in_dim, out_dim, full_hp, seeds["aux"], cell["width"], cell["hidden"]))
        result, model, _ = lewibench.train_one(u["task"], cell["n"], u["arm"], hp, seeds, width=cell["width"], hidden=cell["hidden"],
                                               batch=lewibench.BATCH, max_steps=cell["max_steps"], cell=cell_obj, corpus_name=cell["corpus"])
        rec.update(result)
        if u.get("test"):
            rec["test_one_step"] = lewibench.one_step_nmse(model, cell_obj["test_ds"], cell_obj["keep_dims"])
            rec["test_rollout"] = {str(h): v["nmse"] for h, v in lewibench.rollout_nmse(
                u["task"], model, cell_obj["test_windows"], cell_obj["norm"], cell_obj["obs_dim"], keep=cell_obj["keep_dims"]).items()}
        rec["error"] = None
    except Exception as exc:  # noqa: BLE001
        rec.update({"error": f"{type(exc).__name__}: {exc}", "diverged": 1, "val_rollout_score": None, "best_val_nmse": None})
    rec["wall_s"] = time.time() - t0
    return rec, model, cell_obj


# ---------------------------------------------------------------- probes of a trained model (validation windows only)

def rollout_states(task, model, windows, norm, obs_dim, h_max=16, hold=None, clip=None):
    """The harness's open-loop rollout (lewibench.rollout_nmse) returning the predicted states; `hold` = feature
    indices kept at their start value, `clip` = (low, high) bounds applied to the state after every step."""
    import torch
    from swm_tasks import feature_projectors, project_features_
    s0, acts, _ = windows
    groups = feature_projectors(task, obs_dim)
    mu_x, sd_x, mu_y, sd_y = (norm[k] for k in ("mu_x", "sd_x", "mu_y", "sd_y"))
    cur, out = s0.copy(), []
    model.eval()
    with torch.no_grad():
        for h in range(1, h_max + 1):
            xa = np.concatenate([cur, acts[:, h - 1]], axis=-1)
            dn = model(torch.tensor((xa - mu_x) / sd_x, dtype=torch.float32))
            cur = cur + (dn.double().numpy() * sd_y + mu_y)
            cur = np.nan_to_num(cur, nan=1e9, posinf=1e9, neginf=-1e9)
            if hold is not None:
                cur[:, hold] = s0[:, hold]
            if clip is not None:
                cur = np.clip(cur, clip[0], clip[1])
            project_features_(cur, groups)
            out.append(cur.copy())
    return np.stack(out, axis=1)                                     # windows x h x features


def score_states(pred, fut, keep, horizons=(1, 2, 4, 8, 16)):
    out = {}
    for h in horizons:
        err = pred[:, h - 1] - fut[:, h - 1]
        per_dim = (err ** 2).mean(0) / np.maximum(fut[:, h - 1].var(0), 1e-8)
        out[str(h)] = {"nmse": float(min(per_dim[keep].mean(), 1e6)), "per_dim": np.minimum(per_dim, 1e6).round(6).tolist()}
    out["score"] = float(np.mean([out[str(h)]["nmse"] for h in horizons]))
    return out


def jacobians(model, z, state_dim, gain):
    """d z'/d z of the one-step map in standardized state coordinates, z' = z + gain * f(z, a): batch x S x S."""
    import torch
    x = torch.tensor(z, dtype=torch.float32, requires_grad=True)
    y = model(x)
    rows = []
    for i in range(state_dim):
        g, = torch.autograd.grad(y[:, i].sum(), x, retain_graph=True)
        rows.append(g[:, :state_dim])
    jm = torch.stack(rows, dim=1).double().numpy()                   # batch x out x in
    return np.eye(state_dim)[None] + gain[None, :, None] * jm


def probe(task, model, cell_obj, n_jac=256) -> dict:
    import torch
    keep, norm, windows = cell_obj["keep_dims"], cell_obj["norm"], cell_obj["val_windows"]
    s0, acts, fut = windows
    S = s0.shape[1]
    out = {}
    pred = rollout_states(task, model, windows, norm, cell_obj["obs_dim"])
    out["val"] = score_states(pred, fut, keep)
    var = np.maximum(fut[:, 15].var(0), 1e-8)
    e_w = (((pred[:, 15] - fut[:, 15]) ** 2) / var)[:, keep].mean(1)                  # per-window normalized error at h = 16
    order = np.sort(e_w)[::-1]
    out["window_h16"] = {"mean": float(e_w.mean()), "median": float(np.median(e_w)), "p90": float(np.quantile(e_w, 0.9)),
                         "p99": float(np.quantile(e_w, 0.99)), "max": float(e_w.max()), "windows": int(len(e_w)),
                         "share_top1pct": float(order[: max(1, len(e_w) // 100)].sum() / e_w.sum()),
                         "windows_above_10": int((e_w > 10).sum()), "mean_without_top1pct": float(order[max(1, len(e_w) // 100):].mean())}
    worst = np.argsort(e_w)[::-1][:8]
    z_pred = (pred[worst] - norm["mu_x"][:S]) / norm["sd_x"][:S]
    out["worst_windows"] = {"index": worst.tolist(), "error": e_w[worst].tolist(),
                            "max_abs_z_by_h": np.abs(z_pred[:, :, keep]).max(2).round(3).tolist(),
                            "argmax_dim_h16": np.abs(np.where(keep, z_pred[:, 15], 0)).argmax(1).tolist()}
    z_true = (fut - norm["mu_x"][:S]) / norm["sd_x"][:S]
    out["true_max_abs_z"] = float(np.abs(z_true[:, :16][:, :, keep]).max())
    X = cell_obj["train"].tensors[0].double().numpy()[:, :S] * norm["sd_x"][:S] + norm["mu_x"][:S]
    lo, hi = X.min(0), X.max(0)
    out["val_clip_train_range"] = score_states(rollout_states(task, model, windows, norm, cell_obj["obs_dim"], clip=(lo, hi)), fut, keep)
    if (~keep).any():
        out["val_hold_uninformative"] = score_states(rollout_states(task, model, windows, norm, cell_obj["obs_dim"], hold=np.where(~keep)[0]), fut, keep)
    # the one-step Jacobian along the TRUE validation trajectories (teacher forced), informative coordinates only
    rng = np.random.default_rng(0)
    idx = rng.choice(len(s0), min(n_jac, len(s0)), replace=False)
    gain = (norm["sd_y"] / norm["sd_x"][:S])
    kk = np.where(keep)[0]
    prod = np.tile(np.eye(len(kk))[None], (len(idx), 1, 1))
    radius, snorm = [], []
    states = np.concatenate([s0[idx, None], fut[idx, :15]], axis=1)                   # the true state before step h
    for h in range(16):
        xa = np.concatenate([states[:, h], acts[idx, h]], axis=-1)
        J = jacobians(model, (xa - norm["mu_x"]) / norm["sd_x"], S, gain)[:, kk][:, :, kk]
        if h == 0:
            radius = np.abs(np.linalg.eigvals(J)).max(1)
            snorm = np.linalg.norm(J, 2, axis=(1, 2))
            fnorm = np.linalg.norm(J - np.eye(len(kk))[None], "fro", axis=(1, 2))
        prod = J @ prod
    p16 = np.linalg.norm(prod, 2, axis=(1, 2))
    q = lambda v: {"median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "max": float(np.max(v))}  # noqa: E731
    out["jacobian"] = {"spectral_radius": q(radius), "spectral_norm": q(snorm), "fro_norm_minus_identity": q(fnorm), "product16_norm": q(p16), "points": int(len(idx))}
    with torch.no_grad():
        ws = [p for n_, p in model.named_parameters() if p.dim() == 2 and "codes" not in n_]
        out["weight_spectral_norms"] = [float(torch.linalg.matrix_norm(w.double(), 2)) for w in ws]
        eff = getattr(model, "linear_layers", None)
        if callable(eff):
            try:
                out["effective_weight_spectral_norms"] = [float(torch.linalg.matrix_norm(l.effective_weight().double(), 2)) for l in model.linear_layers()]
            except Exception:  # noqa: BLE001
                pass
    return out


def rebuild(u: dict):
    """The traced unit's trained model from its saved state (the unit's own builder call, then the saved tensors)."""
    lewibench, _, v05_port = swm16.harness()
    import torch
    torch.set_num_threads(1)
    cell = unit_cell(u)
    cell_obj = swm16.cell_data(lewibench, cell["task"], cell["n"], cell["corpus"])
    hp, seeds = unit_hp_seeds(u, lewibench)
    in_dim, out_dim = cell_obj["train"].tensors[0].shape[1], cell_obj["train"].tensors[1].shape[1]
    torch.manual_seed(seeds["init"])
    model = v05_port.build_model(u["arm"], in_dim, out_dim, dict(v05_port.hp_defaults(u["arm"]) | dict(hp)), seeds["aux"], cell["width"], cell["hidden"])
    model.load_state_dict(torch.load(OUT / f"{u['id']}.pt"))
    return model, cell_obj


def probe2(task, model, cell_obj) -> dict:
    """Second pass on a saved model (validation windows only): (1) magnitude pruning of the effective weights at
    evaluation time (do weak couplings carry the instability?), (2) the coordinates of the growing direction."""
    import copy
    import torch
    lewibench, _, _ = swm16.harness()
    keep, norm, windows = cell_obj["keep_dims"], cell_obj["norm"], cell_obj["val_windows"]
    fut = windows[2]
    out = {"prune": []}
    for frac in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9):
        m = copy.deepcopy(model)
        with torch.no_grad():
            for layer in m.linear_layers():
                eff = layer.effective_weight().abs()
                live = eff[eff > 0]
                if frac > 0 and live.numel():
                    thr = torch.quantile(live, frac)
                    if hasattr(layer, "event_mask"):
                        layer.event_mask.mul_((eff >= thr).to(layer.event_mask.dtype))
                    else:
                        layer.weight.mul_((eff >= thr).to(layer.weight.dtype))
        sc = score_states(rollout_states(task, m, windows, norm, cell_obj["obs_dim"]), fut, keep)
        out["prune"].append({"fraction": frac, "val_one_step": lewibench.one_step_nmse(m, cell_obj["val"], keep)["nmse"], "val_score": sc["score"],
                             "val_h16": sc["16"]["nmse"], "val_h8": sc["8"]["nmse"]})
    # LeWi only: close the marginal edges (gate between 1/2 and tau) at evaluation time; gate statistics of the open edges
    layers = [l for l in model.linear_layers() if hasattr(l, "gate") and hasattr(l, "event_mask")]
    if layers:
        with torch.no_grad():
            g_open = torch.cat([l.gate()[l.event_mask > 0].reshape(-1) for l in layers])
            out["open_gate_quantiles"] = {str(q): float(torch.quantile(g_open, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
            out["open_edges"] = int(g_open.numel())
            out["marginal"] = []
            for tau in (0.52, 0.55, 0.60, 0.65, 0.70):
                m = copy.deepcopy(model)
                for l in m.linear_layers():
                    if hasattr(l, "gate") and hasattr(l, "event_mask"):
                        l.event_mask.mul_((l.gate() > tau).to(l.event_mask.dtype))
                sc = score_states(rollout_states(task, m, windows, norm, cell_obj["obs_dim"]), fut, keep)
                out["marginal"].append({"tau": tau, "closed_share": float((g_open <= tau).float().mean()),
                                        "val_one_step": lewibench.one_step_nmse(m, cell_obj["val"], keep)["nmse"], "val_score": sc["score"], "val_h16": sc["16"]["nmse"]})
    # the growing direction: leading singular vectors of the 16-step Jacobian product along the true trajectories
    s0, acts, fut_ = windows
    S = s0.shape[1]
    kk = np.where(keep)[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(s0), min(256, len(s0)), replace=False)
    gain = norm["sd_y"] / norm["sd_x"][:S]
    prod = np.tile(np.eye(len(kk))[None], (len(idx), 1, 1))
    states = np.concatenate([s0[idx, None], fut_[idx, :15]], axis=1)
    for h in range(16):
        xa = np.concatenate([states[:, h], acts[idx, h]], axis=-1)
        prod = jacobians(model, (xa - norm["mu_x"]) / norm["sd_x"], S, gain)[:, kk][:, :, kk] @ prod
    U, sv, Vt = np.linalg.svd(prod)
    out["product16"] = {"dims": kk.tolist(), "sv1_median": float(np.median(sv[:, 0])), "sv1_max": float(sv[:, 0].max()),
                        "input_direction_weight": (Vt[:, 0, :] ** 2).mean(0).round(4).tolist(),      # which start coordinates the growth reads
                        "output_direction_weight": (U[:, :, 0] ** 2).mean(0).round(4).tolist()}       # which coordinates it lands on
    return out


def probe3(task, model, cell_obj, touch=(24, 25, 26, 27, 28)) -> dict:
    """The one-step Jacobian at true validation states, split by contact (any touch coordinate above zero) and no
    contact: spectral radius of the informative block, of the touch block alone, and of the block without touch."""
    keep, norm = cell_obj["keep_dims"], cell_obj["norm"]
    s0, acts, _ = cell_obj["val_windows"]
    S = s0.shape[1]
    gain = norm["sd_y"] / norm["sd_x"][:S]
    kk = np.where(keep)[0]
    tt = [int(np.where(kk == d)[0][0]) for d in touch]
    rest = [i for i in range(len(kk)) if i not in tt]
    contact = (s0[:, list(touch)] > 0).any(1)
    out = {"contact_share": float(contact.mean())}
    rng = np.random.default_rng(0)
    for name, sel in (("no_contact", np.where(~contact)[0]), ("contact", np.where(contact)[0])):
        idx = rng.choice(sel, min(256, len(sel)), replace=False)
        xa = np.concatenate([s0[idx], acts[idx, 0]], axis=-1)
        J = jacobians(model, (xa - norm["mu_x"]) / norm["sd_x"], S, gain)[:, kk][:, :, kk]
        rad = lambda M: np.abs(np.linalg.eigvals(M)).max(1)  # noqa: E731
        q = lambda v: {"median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "max": float(np.max(v))}  # noqa: E731
        Jt = J[:, tt][:, :, tt]
        out[name] = {"points": int(len(idx)), "radius_full": q(rad(J)), "radius_touch_block": q(rad(Jt)), "radius_without_touch": q(rad(J[:, rest][:, :, rest])),
                     "touch_block_mean": Jt.mean(0).round(3).tolist()}
    return out


def probe3_all(manifest: str, k: int = 0, of: int = 1) -> None:
    for i, line in enumerate(Path(manifest).read_text().splitlines()):
        if i % of != k:
            continue
        u = json.loads(line)
        if not (OUT / f"{u['id']}.pt").exists() or (OUT / f"{u['id']}.probe3.json").exists():
            continue
        model, cell_obj = rebuild(u)
        rec = probe3(u["task"], model, cell_obj)
        tmp = OUT / f"{u['id']}.probe3.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, OUT / f"{u['id']}.probe3.json")
        print(u["id"], {g: (rec[g]["radius_full"]["median"], rec[g]["radius_touch_block"]["median"], rec[g]["radius_without_touch"]["median"]) for g in ("no_contact", "contact")}, flush=True)


def probe2_all(manifest: str, k: int = 0, of: int = 1) -> None:
    for i, line in enumerate(Path(manifest).read_text().splitlines()):
        if i % of != k:
            continue
        u = json.loads(line)
        if not (OUT / f"{u['id']}.pt").exists() or (OUT / f"{u['id']}.probe2.json").exists():
            continue
        model, cell_obj = rebuild(u)
        rec = probe2(u["task"], model, cell_obj)
        tmp = OUT / f"{u['id']}.probe2.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, OUT / f"{u['id']}.probe2.json")
        print(u["id"], [(r["fraction"], round(r["val_one_step"], 4), round(r["val_score"], 3)) for r in rec["prune"]],
              [(r["tau"], round(r["closed_share"], 3), round(r["val_one_step"], 4), round(r["val_score"], 3)) for r in rec.get("marginal", [])], flush=True)


# ---------------------------------------------------------------- the worker

def claim(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = path.with_suffix(".claim")
    try:
        fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.environ.get('SLURM_JOB_ID', '')} {os.getpid()}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - c.stat().st_mtime > 4200:               # a dead worker's claim (a unit takes at most about 55 min)
                c.unlink()
                return claim(path)
        except FileNotFoundError:
            return claim(path)
        return False


def write(path: Path, rec: dict) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rec, default=swm16._json_default))
    os.replace(tmp, path)
    path.with_suffix(".claim").unlink(missing_ok=True)


def work(manifest: str, deadline: float, worker: int) -> None:
    units = [json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip()]
    units = units[worker % max(len(units), 1):] + units[: worker % max(len(units), 1)]
    done = 0
    for u in units:
        path = OUT / f"{u['id']}.json"
        if path.exists():
            continue
        if time.time() > deadline - RESERVE_S.get(u["arm"], RESERVE_DEFAULT_S) * float(u.get("max_steps") or 128000) / 128000:
            continue
        if not claim(path):
            continue
        t_start = time.time()
        rec, model, cell_obj = run_one(u)
        if u.get("trace") and model is not None:
            import torch
            try:
                rec["final_per_input_open"] = per_input_open(model)
                rec["probe"] = probe(u["task"], model, cell_obj)
            except Exception as exc:  # noqa: BLE001
                rec["probe_error"] = f"{type(exc).__name__}: {exc}"
            torch.save(model.state_dict(), path.with_suffix(".pt"))
        rec["wall_total_s"] = time.time() - t_start
        rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write(path, rec)
        done += 1
        print(f"{time.strftime('%H:%M:%S')} {u['id']} val_score={rec.get('val_rollout_score')} {rec['wall_s']:.0f}s {rec.get('error') or ''}", flush=True)
    print(f"worker {worker}: {done} units done", flush=True)


def initcounts(out_csv: str, cells: list[str]) -> None:
    """Open first-transition connections per input at initialization: the HPO seed and the 16 final seeds of a cell."""
    import csv
    lewibench, swm_tasks, v05_port = swm16.harness()
    import torch
    c = swm16.contract()
    rows = []
    for spec in cells:
        task, n = spec.split(":")
        n = int(n)
        meta = json.loads((swm_tasks.DATA / f"{swm16.corpus_of(task) or task}.meta.json").read_text())
        feat = swm_tasks.featurize(task, np.zeros((1, int(meta["obs_dim"])))).shape[1]
        in_dim, out_dim = feat + int(meta["act_dim"]), feat
        hp = dict(v05_port.hp_defaults("lewi-v1"))
        bank = [("hpo", 0, lewibench.trial_seeds(task, n))] + [("final", s, lewibench.final_seeds(task, n, s, "final")) for s in range(16)]
        for stream, s, seeds in bank:
            torch.manual_seed(seeds["init"])
            model = v05_port.build_model("lewi-v1", in_dim, out_dim, hp, seeds["aux"], c["width"], c["hidden"])
            layers = model.structural_layers()
            per_in = layers[0].event_mask.sum(dim=0)
            per_out = layers[-1].event_mask.sum(dim=1)
            rows.append(dict(task=task, n=n, stream=stream, seed=s, inputs=in_dim, width=c["width"], open_first=int(layers[0].event_mask.sum()),
                             min_open_per_input=int(per_in.min()), argmin_input=int(per_in.argmin()), median_open_per_input=float(per_in.median()),
                             inputs_below_5=int((per_in < 5).sum()), inputs_at_0=int((per_in == 0).sum()),
                             min_open_per_output=int(per_out.min()), outputs_at_0=int((per_out == 0).sum()),
                             per_input=" ".join(str(int(v)) for v in per_in.tolist())))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows ->", out_csv)


if __name__ == "__main__":
    verb = sys.argv[1]
    if verb == "work":
        work(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
    elif verb == "probe3":
        probe3_all(sys.argv[2], *(int(a) for a in sys.argv[3:5]))
    elif verb == "probe2":
        probe2_all(sys.argv[2], *(int(a) for a in sys.argv[3:5]))
    elif verb == "initcounts":
        initcounts(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(__doc__)
