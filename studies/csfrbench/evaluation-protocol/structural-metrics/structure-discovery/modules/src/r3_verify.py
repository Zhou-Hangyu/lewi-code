"""Checkpoint verification: the loaded model carries the distance gate and the hard mask on every
affine transition, the readout included, with one frozen radius; the forward pass equals the
effective-weight pass; the readout mask and gate act on the output.  One record per checkpoint
state, loaded through the campaign's own reconstruction path (``cd_view.load``).

    python r3_verify.py --list ../lists/d32.list --n 32768 --out ../results/verify/d32.json
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SD = HERE.parents[1]
for p in (str(HERE), str(SD / "component-dag" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
import cd_view as cv  # noqa: E402

REPO = Path(__file__).resolve().parents[7]
EXPECTED = {"method": "lewi-v1-rgm", "radius2": 1.5, "layer": "BallMaskedLinear"}
N_PROBE = 512


def verify(run_dir, which):
    model, target, spec = cv.load(REPO / run_dir, which=which)
    layers = list(model.linear_layers())
    out_layer = model.output_layer
    rec = {"checkpoint": str(run_dir), "which": which, "method": spec["method"], "d": int(spec["d"]),
           "n": int(spec["n"]), "function_seed": int(spec["function_seed"]), "hp": spec.get("hp"),
           "width": int(spec["arch"]["width"]), "hidden": int(spec["arch"]["hidden"]),
           "code_dim": int(model.gate_codes[0].shape[1]), "n_positions": len(model.gate_codes),
           "n_transitions": len(layers), "readout_is_last_transition": bool(out_layer is layers[-1]),
           "readout_masked_flag": bool(getattr(model, "lewi_readout_masked", False)),
           "readout_fix": getattr(model, "lewi_readout_fix", None), "floor": int(getattr(model, "lewi_floor", 0)),
           "layer_types": sorted({type(l).__name__ for l in layers}),
           "radius2": sorted({round(float(l.gate_offset), 6) for l in layers}),
           "gate_scale": sorted({round(float(l.gate_scale), 6) for l in layers}),
           "radius_frozen": all(not l.gate_offset.requires_grad and not l.gate_scale.requires_grad for l in layers)}
    with torch.no_grad():
        rec["mask_equals_sign_of_score"] = [int((l.event_mask != (l.score() > 0).to(l.event_mask.dtype)).sum()) for l in layers]
        rec["mask_density"] = [round(float(l.event_mask.mean()), 4) for l in layers]
        rec["readout_open_edges"] = int(out_layer.event_mask.sum())
        rec["readout_gate_mean_open"] = round(float(out_layer.gate()[out_layer.event_mask > 0].mean()), 4) if rec["readout_open_edges"] else None
        rng = np.random.default_rng(int(spec["function_seed"]) * 7 + (which == "init"))
        X = torch.from_numpy(rng.uniform(-1.0, 1.0, size=(N_PROBE, int(spec["d"]))).astype(np.float32))
        y_model = model(X).reshape(-1)
        h = X
        for l in model.hidden_layers:
            h = torch.relu(F.linear(h, l.weight * l.gate() * l.event_mask, l.bias))
        y_eff = F.linear(h, out_layer.weight * out_layer.gate() * out_layer.event_mask, out_layer.bias).reshape(-1)
        y_plain = F.linear(h, out_layer.weight, out_layer.bias).reshape(-1)
        y_gate_only = F.linear(h, out_layer.weight * out_layer.gate(), out_layer.bias).reshape(-1)
        sd = float(y_model.std()) or 1.0
        rec["forward_equals_effective"] = float((y_model - y_eff).abs().max())
        rec["readout_plain_gap"] = round(float((y_model - y_plain).abs().max()) / sd, 4)
        rec["readout_mask_gap"] = round(float((y_model - y_gate_only).abs().max()) / sd, 4)
        rec["readout_closed_nonzero_weight"] = int(((out_layer.event_mask == 0) & (out_layer.weight.abs() > 0)).sum())
    rec["pass"] = (rec["method"] == EXPECTED["method"] and rec["readout_masked_flag"] and rec["readout_is_last_transition"]
                   and rec["layer_types"] == [EXPECTED["layer"]] and rec["radius2"] == [EXPECTED["radius2"]]
                   and rec["gate_scale"] == [1.0] and rec["radius_frozen"] and rec["n_positions"] == rec["n_transitions"] + 1
                   and sum(rec["mask_equals_sign_of_score"]) == 0 and rec["forward_equals_effective"] < 1e-5)
    return rec


def _one(args):
    try:
        return verify(*args)
    except Exception as e:  # noqa: BLE001
        return {"checkpoint": args[0], "which": args[1], "error": repr(e), "pass": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--n", type=int, default=0, help="keep only this sample size (0 = all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    states = []
    for line in Path(a.list).read_text().splitlines():
        if not line.strip():
            continue
        run_dir, cell, fs, n, which = line.split()
        if a.n and int(n) != a.n:
            continue
        states.append((run_dir, which))
    with Pool(a.workers) as p:
        recs = p.map(_one, states)
    summary = {"states": len(recs), "pass": sum(r["pass"] for r in recs), "errors": sum("error" in r for r in recs),
               "methods": sorted({r.get("method") for r in recs if "method" in r}),
               "radius2": sorted({x for r in recs for x in r.get("radius2", [])}),
               "readout_open_edges_min": min((r["readout_open_edges"] for r in recs if "readout_open_edges" in r), default=None),
               "readout_plain_gap_min": min((r["readout_plain_gap"] for r in recs if "readout_plain_gap" in r), default=None),
               "readout_mask_gap_min": min((r["readout_mask_gap"] for r in recs if "readout_mask_gap" in r), default=None),
               "forward_gap_max": max((r["forward_equals_effective"] for r in recs if "forward_equals_effective" in r), default=None)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "records": recs}, indent=0))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
