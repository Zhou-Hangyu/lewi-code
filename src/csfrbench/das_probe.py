"""Frozen causal-use probe: Boundless DAS on retained campaign winners.

One fit per (winner run dir, layer, restart seed, formulation), under the
frozen metric fb2-freeze-3. Protocol constants are read exclusively from the
hash-verified frozen config (configs/causal_das_frozen.json in the v0.5
bundle, sha256 pinned below); there are no CLI knobs for epochs, learning
rates, pools, cadence, anneal, objective or mask geometry.

Provenance: the computation is the metric-freeze audit's, vendored verbatim
(the frozen Boundless-DAS protocol): ``das_core`` (fit/intervene/evaluate),
``das_teacher_cf`` (roster + counterfactual pools), ``das_go_reference``
(GraphOracle known-block reference). This runner is the campaign adaptation
of that study's ``fb2_unit.py`` with the audit-only surfaces removed
(planted targets, rank sweeps, alternate anneal/objective/cadence/learning
rates) and two additions: the frozen-config guard above and a per-record
resume-fidelity assertion (the resumed forward pass must reproduce the
model's own output on held-back rows).

The SAE recipe for the code-space secondary (K = 128, L1 1e-3, 6 epochs,
Adam 1e-3, batch 1024, seed 1000 + restart) is the audited one from the
freeze report section 10.3; the frozen config scopes both secondaries as
qualitative sensitivity diagnostics.

Random-rotation control seeds hash the runner's ``--target`` string, which
here is a bare run-dir path where the audit study used ``ckpt:<dir>`` -
the control is distributional (Haar), so only bit-level cross-study
reproduction of control draws differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from csfrbench import das_core as dc
from csfrbench import das_teacher_cf as tc
from csfrbench.registry import get_method, get_task, load_builtin_registrations
from csfrbench.utils.seeds import derive

load_builtin_registrations()

FROZEN_SHA256 = "b61366bc97b3b945502f83a748cb219d1c3e855914e550e849054bc6c37fbdbc"
RANDOM_ROTATION_DRAWS = 5
VAL_POOLS = (100, 500, 1000)
TEST_POOLS = (500, 2000, 5000)
SAE_K, SAE_L1, SAE_EPOCHS = 128, 1e-3, 6
RESUME_FIDELITY_ROWS = 256
RESUME_FIDELITY_TOL = 1e-5


def load_frozen_protocol(path: str | Path) -> dict:
    """Load the frozen metric config and verify its self-hash and the pin."""
    cfg = json.loads(Path(path).read_text())
    stored = cfg.pop("metric_hash_sha256")
    payload = json.dumps(cfg, indent=2, sort_keys=True, default=float)
    recomputed = hashlib.sha256(payload.encode()).hexdigest()
    if recomputed != stored:
        raise SystemExit(f"{path}: self-hash mismatch ({recomputed[:16]} != "
                         f"{stored[:16]}); the frozen config was altered")
    if stored != FROZEN_SHA256:
        raise SystemExit(f"{path}: hash {stored[:16]} is not the pinned "
                         f"frozen metric {FROZEN_SHA256[:16]}")
    cfg["metric_hash_sha256"] = stored
    return cfg


def frozen_constants(cfg: dict) -> dict:
    """The operative constants, in ``das_core.fit_das`` terms."""
    assert cfg["temperature_schedule"]["anneal"] == "stretched"
    assert cfg["temperature_schedule"]["endpoints_at_d4096"] == [50.0, 0.1]
    assert cfg["validation_cadence"]["value"] == "every_200_updates"
    assert cfg["score"]["objective"] == "pooled_mse"
    assert cfg["mask_definitions"]["width_clamp"] == [0.0, 1.0]
    return {
        "epochs": int(cfg["update_budget"]["epochs"]),
        "K": int(cfg["restarts"]["K"]),
        "rot_lr": float(cfg["optimizer"]["rotation_lr"]),
        "bnd_lr": float(cfg["optimizer"]["boundary_lr"]),
        "batch_size": int(cfg["optimizer"]["batch_size"]),
        "eval_every": 200,
        "anneal": "stretched",
        "objective": "pooled",
        "train_pairs": int(cfg["counterfactual_data"]["train_pairs_per_variable"]),
        "val_pairs": int(cfg["counterfactual_data"]["val_pairs_per_variable"]),
        "test_pairs": int(cfg["counterfactual_data"]["test_pairs_per_variable"]),
        "split_seed": int(cfg["counterfactual_data"]["split_seed_for_the_frozen_run"]),
        "layers": [int(x) for x in cfg["hidden_layers_probed"]],
        # the four CSFR departures (fb2 section 2.3) are the frozen geometry
        "variant": {"positions": "half", "clamp_lo": 0.0,
                    "simplex": True, "temp_scaled": True},
    }


def _seed_of(*parts) -> int:
    """Deterministic across processes; Python's hash() is salted per run."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16) % (2 ** 32)


def load_frozen_model(run_dir: str | Path, which: str = "best"):
    """Rebuild the run's model and load its retained checkpoint, frozen.

    Identical to the campaign pipeline's model reconstruction (and to the
    metric-freeze study's ``learner_io.load_frozen_model``): the derived
    per-run seeds rebuild any auxiliary state exactly, then the retained
    checkpoint overwrites every learned parameter.
    """
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    target = get_task(spec["task"]).draw(spec["cell"], int(spec["function_seed"]))
    method = get_method(spec["method"])
    pair_key = (target.fingerprint, int(spec["n"]), spec["dataset_seed"])
    if int(spec.get("seed_replicate", 0)):
        pair_key = (*pair_key, int(spec["seed_replicate"]))
    seeds = {name: derive(*pair_key, name)
             for name in ("initialization", "stochastic", "batch_order", "gate_init")}
    cfg = {"width": int(spec["arch"]["width"]), "hidden": int(spec["arch"]["hidden"]),
           "hp": spec["hp"], "auxiliary_init_seed": int(seeds["gate_init"])}
    torch.manual_seed(int(seeds["initialization"]))
    model = method.build(cfg, target)
    state = torch.load(run_dir / f"{which}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.requires_grad_(False)
    return model, target, spec


class SAEWrapped(nn.Module):
    """hidden_layers = real[:L] + [enc, dec] + real[L:]; code = state L+1."""

    def __init__(self, model, enc, dec, layer):
        super().__init__()
        self.hidden_layers = nn.ModuleList(
            list(model.hidden_layers[:layer]) + [enc, dec]
            + list(model.hidden_layers[layer:]))
        self.output_layer = model.output_layer

    def forward(self, x):
        h = x
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        return self.output_layer(h)


def train_sae(H: torch.Tensor, k: int, l1: float, epochs: int, seed: int):
    torch.manual_seed(seed)
    d = H.shape[1]
    enc, dec = nn.Linear(d, k), nn.Linear(k, d)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    n = len(H)
    gen = torch.Generator().manual_seed(seed + 1)
    for ep in range(epochs):
        order = torch.randperm(n, generator=gen)
        for b in range(0, n, 1024):
            x = H[order[b:b + 1024]]
            c = F.relu(enc(x))
            xh = dec(c)
            loss = F.mse_loss(xh, x) + l1 * c.abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        C = F.relu(enc(H))
        R = dec(C)
        nmse = float(((R - H) ** 2).sum() / ((H - H.mean(0)) ** 2).sum())
        dead = float((C.max(0).values <= 0).float().mean())
        l0 = float((C > 0).float().sum(1).mean())
    return enc, dec, {"recon_nmse": nmse, "dead_frac": dead, "mean_l0": l0}


def cap_rows(var_idx: np.ndarray, cap: int | None) -> np.ndarray:
    """First ``cap`` rows of every variable, in dataset order (nested subsets)."""
    if cap is None:
        return np.ones(len(var_idx), bool)
    keep = np.zeros(len(var_idx), bool)
    for v in np.unique(var_idx):
        idx = np.flatnonzero(var_idx == v)[:cap]
        keep[idx] = True
    return keep


def check_dataset(ds: Path, target, fc: dict) -> dict:
    manifest = json.loads((ds / "das_manifest.json").read_text())
    if int(manifest["split_seed"]) != fc["split_seed"]:
        raise SystemExit(f"{ds}: split seed {manifest['split_seed']} != frozen "
                         f"{fc['split_seed']}")
    if not manifest["include_readout"]:
        raise SystemExit(f"{ds}: pools exclude the pre-readout node; the frozen "
                         "roster keeps every local-function intermediate")
    if manifest["fingerprint"] != target.fingerprint:
        raise SystemExit(f"{ds}: fingerprint {manifest['fingerprint']} != "
                         f"target {target.fingerprint}")
    need = {"das_train": fc["train_pairs"], "das_val": fc["val_pairs"],
            "das_test": fc["test_pairs"]}
    for split, per_var in need.items():
        if int(manifest["sizes"][split]) < per_var:
            raise SystemExit(f"{ds}: {split} has {manifest['sizes'][split]} "
                             f"pairs per variable, frozen protocol needs {per_var}")
    roster_now = tc.aligned_variables(target, include_readout=True)
    if [e["var"] for e in roster_now] != [e["var"] for e in manifest["roster"]]:
        raise SystemExit(f"{ds}: stored roster does not match the target's "
                         "aligned variables")
    return manifest


def resume_fidelity(model, X: np.ndarray, layer: int) -> float:
    """Max |resumed - direct| output over held-back rows; must be ~0."""
    with torch.no_grad():
        Xt = torch.from_numpy(X[:RESUME_FIDELITY_ROWS])
        states = dc.layer_states(model, Xt)
        resumed = dc.resume_from(model, states[layer - 1], layer)
        direct = model(Xt)
        direct = direct.squeeze(-1) if direct.ndim > 1 else direct
    return float((resumed - direct).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-config", required=True,
                    help="the bundle's causal_das_frozen.json (hash-verified)")
    ap.add_argument("--target", required=True,
                    help="winner run dir containing spec.json + best.pt")
    ap.add_argument("--dataset", required=True,
                    help="DAS counterfactual dataset dir for the target")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--das-seed", type=int, default=0)
    ap.add_argument("--formulation", default="joint",
                    choices=["joint", "indep", "sae"])
    ap.add_argument("--vars", default=None,
                    help="comma list: a subset (roster sensitivity, indep) or "
                         "a full permutation (order audit)")
    ap.add_argument("--control", default="none",
                    choices=["none", "init", "wrong_variable", "no_edit"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-w", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    out = Path(args.out)
    if out.with_suffix(".json").exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_frozen_protocol(args.frozen_config)
    fc = frozen_constants(cfg)
    if not 0 <= args.das_seed < fc["K"]:
        raise SystemExit(f"--das-seed {args.das_seed} outside the frozen "
                         f"restart range 0..{fc['K'] - 1}")
    if args.layer not in fc["layers"]:
        raise SystemExit(f"--layer {args.layer} not in the frozen layer set "
                         f"{fc['layers']}")
    if args.formulation == "indep" and (not args.vars or "," in args.vars):
        raise SystemExit("--formulation indep requires --vars with exactly "
                         "one variable; without it the fit would silently "
                         "be a joint one under an indep label")

    which = "init" if args.control == "init" else "best"
    model, target, spec = load_frozen_model(args.target, which=which)
    ds = Path(args.dataset)
    manifest = check_dataset(ds, target, fc)

    raw = {s: tc.load_das_split(ds, s) for s in
           ("das_train", "das_val", "das_test")}
    roster0 = raw["das_train"]["roster"]
    keep = None
    if args.vars:
        # keep follows the ORDER of --vars, so a full permuted list assigns
        # the rotated-axis blocks in that order; canonical-order subsets
        # behave exactly as before
        names = args.vars.split(",")
        index_of = {e["var"]: i for i, e in enumerate(roster0)}
        assert all(n in index_of for n in names), f"unknown variable in {args.vars}"
        assert len(set(names)) == len(names), f"duplicate variable in {args.vars}"
        keep = [index_of[n] for n in names]
        remap = {old: new for new, old in enumerate(keep)}
    caps = {"das_train": fc["train_pairs"], "das_val": fc["val_pairs"],
            "das_test": fc["test_pairs"]}
    for s in raw:
        d = raw[s]
        rows = cap_rows(d["var_idx"], caps[s])
        if keep is not None:
            rows &= np.isin(d["var_idx"], keep)
        d["Xb"], d["Xs"] = d["Xb"][rows], d["Xs"][rows]
        y = d["y_cf_wrong"] if args.control == "wrong_variable" else d["y_cf"]
        d["y"] = y[rows]
        vi = d["var_idx"][rows]
        d["var_idx"] = (np.array([remap[v] for v in vi], np.int64)
                        if keep is not None else vi)
        d["roster"] = [roster0[i] for i in keep] if keep is not None else roster0
    m = len(raw["das_train"]["roster"])

    fidelity = resume_fidelity(model, raw["das_val"]["Xb"], args.layer)
    if fidelity > RESUME_FIDELITY_TOL:
        raise SystemExit(f"{args.target}: resumed forward differs from the "
                         f"model output by {fidelity:.3e} at layer "
                         f"{args.layer}; the probe would edit a different "
                         "computation")

    sae_stats, pass_delta, enc, dec = None, None, None, None
    if args.formulation == "sae":
        with torch.no_grad():
            hb_tr, hs_tr = dc.capture_pair(model, raw["das_train"]["Xb"],
                                           raw["das_train"]["Xs"], args.layer)
        enc, dec, sae_stats = train_sae(torch.cat([hb_tr, hs_tr]), SAE_K,
                                        SAE_L1, SAE_EPOCHS,
                                        seed=1000 + args.das_seed)
        wrapped = SAEWrapped(model, enc, dec, args.layer)
        for p in wrapped.parameters():
            p.requires_grad_(False)
        Xv = torch.from_numpy(raw["das_val"]["Xb"][:700])
        with torch.no_grad():
            y_d = model(Xv).squeeze(-1)
            y_w = wrapped(Xv).squeeze(-1)
            # held-out reconstruction: DAS test states, never seen by the SAE
            hb_te, hs_te = dc.capture_pair(model, raw["das_test"]["Xb"],
                                           raw["das_test"]["Xs"], args.layer)
            H = torch.cat([hb_te, hs_te])
            C = torch.relu(enc(H))
            sae_stats["heldout_recon_nmse"] = float(
                ((dec(C) - H) ** 2).sum() / ((H - H.mean(0)) ** 2).sum())
            sae_stats["heldout_mean_l0"] = float((C > 0).float().sum(1).mean())
            sae_stats["heldout_dead_frac"] = float((C.max(0).values <= 0).float().mean())
            sae_stats["state_pool"] = "das_train base and source states only"
        pass_delta = float(((y_w - y_d) ** 2).mean() / y_d.var())
        model, fit_layer = wrapped, args.layer + 1
    else:
        fit_layer = args.layer

    splits = {}
    for s in ("das_train", "das_val", "das_test"):
        hb, hs = dc.capture_pair(model, raw[s]["Xb"], raw[s]["Xs"], fit_layer)
        splits[s] = {"hb": hb, "hs": hs,
                     "var_idx": torch.from_numpy(raw[s]["var_idx"]),
                     "y": torch.from_numpy(np.ascontiguousarray(raw[s]["y"]))}

    def subpool(split: str, cap: int) -> dict:
        rows = torch.from_numpy(cap_rows(raw[split]["var_idx"], cap))
        return {k: v[rows] for k, v in splits[split].items()}

    val_pools = {str(c): subpool("das_val", c) for c in VAL_POOLS
                 if c < fc["val_pairs"]}

    fit = dc.fit_das(model, fit_layer, splits["das_train"], splits["das_val"],
                     args.das_seed, epochs=fc["epochs"],
                     batch_size=fc["batch_size"], eval_every=fc["eval_every"],
                     variant=fc["variant"], fixed_rank=None,
                     anneal=fc["anneal"], objective=fc["objective"],
                     val_pools=val_pools or None, rot_lr=fc["rot_lr"],
                     bnd_lr=fc["bnd_lr"])
    dim = fit["dim"]
    masks = dc.hard_masks_from(fit["lower"], fit["upper"], dim)
    if args.control == "no_edit":
        masks = torch.zeros_like(masks)
    test = splits["das_test"]
    val = splits["das_val"]
    W = fit["W"]
    scores = dc.evaluate_das(model, fit_layer, W, masks, test["hb"],
                             test["hs"], test["var_idx"], test["y"], m)
    val_scores = dc.evaluate_das(model, fit_layer, W, masks, val["hb"],
                                 val["hs"], val["var_idx"], val["y"], m)

    val_by_pool = {}
    for c in VAL_POOLS:
        if c > fc["val_pairs"]:
            continue
        q = subpool("das_val", c)
        s_ = dc.evaluate_das(model, fit_layer, W, masks, q["hb"], q["hs"],
                             q["var_idx"], q["y"], m)
        val_by_pool[str(c)] = {"macro": s_["macro"], "per_var": s_["per_var"]}
    test_by_pool = {}
    for c in TEST_POOLS:
        if c > fc["test_pairs"]:
            continue
        p = subpool("das_test", c)
        s_ = dc.evaluate_das(model, fit_layer, W, masks, p["hb"], p["hs"],
                             p["var_idx"], p["y"], m)
        test_by_pool[str(c)] = {"macro": s_["macro"], "per_var": s_["per_var"]}

    rand = []
    for i in range(RANDOM_ROTATION_DRAWS):
        Q = dc.haar_orthogonal(dim, seed=_seed_of(
            args.target, args.layer, args.das_seed, i) % (2 ** 31))
        rand.append(dc.evaluate_das(model, fit_layer, Q, masks, test["hb"],
                                    test["hs"], test["var_idx"], test["y"], m))
    no_edit = dc.evaluate_das(model, fit_layer, W, torch.zeros_like(masks),
                              test["hb"], test["hs"], test["var_idx"],
                              test["y"], m)

    axis = dc.axis_metrics(W, fit["lower"], fit["upper"])
    assert axis["ortho_max_offdiag"] < 1e-4, "snapped subspaces are not orthogonal"

    oracle = None
    if (spec["method"] == "graph_oracle" and args.formulation != "sae"
            and args.control == "none"):
        from csfrbench import das_go_reference as go_ref
        om, present, info = go_ref.known_block_masks(
            model, target, raw["das_test"]["roster"], args.layer, dim)
        o = dc.evaluate_das(model, fit_layer, torch.eye(dim), om, test["hb"],
                            test["hs"], test["var_idx"], test["y"], m)
        oracle = {"kind": "graph_oracle_known_block", "macro": o["macro"],
                  "per_var": o["per_var"], "live": present,
                  "true_dims": [info["block_units"][e["var"]]
                                for e in raw["das_test"]["roster"]],
                  "stage": info["stage"]}

    import subprocess
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parent)).stdout.strip()
    record = {
        "schema": "csfrbench_das_v1",
        "frozen_metric": {"version": cfg["version"],
                          "sha256": cfg["metric_hash_sha256"]},
        "git_commit": git,
        "target": str(args.target), "dataset": str(ds),
        "formulation": args.formulation, "layer": args.layer,
        "das_seed": args.das_seed, "control": args.control,
        "vars_subset": args.vars, "m": m,
        "roster": raw["das_test"]["roster"],
        "protocol": {
            "epochs": fc["epochs"], "anneal": fc["anneal"],
            "eval_every": fc["eval_every"],
            "rot_lr": fc["rot_lr"], "bnd_lr": fc["bnd_lr"],
            "objective": fc["objective"],
            "train_pairs_per_var": fc["train_pairs"],
            "val_pairs_per_var": fc["val_pairs"],
            "test_pairs_per_var": fc["test_pairs"],
            "split_seed": fc["split_seed"],
            "batch_size": fit["batch_size"], "variant": fc["variant"],
            "sae": ({"k": SAE_K, "l1": SAE_L1, "epochs": SAE_EPOCHS}
                    if args.formulation == "sae" else None),
        },
        "accounting": {
            "train_pairs_total": fit["train_pairs"],
            "batch_size": fit["batch_size"],
            "optimizer_updates": fit["total_steps"],
            "examples_presented": fit["examples"],
            "nominal_epochs": fc["epochs"],
            "steps_per_epoch": fit["steps_per_epoch"],
            "anneal_horizon_steps": fit["anneal_horizon"],
        },
        "dims": fit["dims"], "residual_dim": dim - int(sum(fit["dims"])),
        "ambient_dim": dim,
        "best_step": fit["best_step"], "best_val_macro": fit["best_val_macro"],
        "val_macro": val_scores["macro"], "val_per_var": val_scores["per_var"],
        "test_macro": scores["macro"], "test_per_var": scores["per_var"],
        "test_by_pool": test_by_pool, "val_by_pool": val_by_pool,
        "random_rotation_macros": [r["macro"] for r in rand],
        "random_rotation_per_var": [r["per_var"] for r in rand],
        "no_edit_macro": no_edit["macro"], "no_edit_per_var": no_edit["per_var"],
        "oracle": oracle,
        "axis": {"ortho_max_offdiag": axis["ortho_max_offdiag"],
                 "top_sets": axis["top_sets"],
                 "per_var": [{k: p[k] for k in ("r", "coherence", "n_eff")}
                             for p in axis["per_var"]]},
        "sae": sae_stats, "pass_through_delta_nmse": pass_delta,
        "resume_fidelity_max_abs": fidelity,
        "history": fit["history"],
        "task": spec["task"], "task_preset": spec["task_preset"],
        "method": spec["method"], "family": target.family,
        "fingerprint": target.fingerprint,
        "function_seed": int(spec["function_seed"]),
        "n_train": int(spec["n"]),
        "hpo_trial": int(spec.get("hpo_trial", -1)),
        "checkpoint": which,
    }
    if not args.no_w:
        payload = {"W": W, "widths_raw": fit["widths_raw"],
                   "lower": fit["lower"], "upper": fit["upper"]}
        if dec is not None:
            payload["dec_weight"] = dec.weight.detach().clone()
            payload["enc_weight"] = enc.weight.detach().clone()
        torch.save(payload, out.with_suffix(".pt"))
    out.with_suffix(".json").write_text(json.dumps(record) + "\n")
    print(f"{out.name}: test {scores['macro']:+.3f} dims {fit['dims']} "
          f"updates {fit['total_steps']}")


if __name__ == "__main__":
    main()
