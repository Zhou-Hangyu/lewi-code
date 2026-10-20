"""Explicit-communication wiring prior: measured scores + per-neuron codes.

Implements the communication-matrix method family (an earlier project of ours; not used by the paper):

    L_total = L_task + lam * R_mod + lam_sup * R_sup

Every neuron at every layer boundary owns a learned code ``phi_i`` in R^code_dim.
For each linear ``W^l`` a measured communication score prices edges from sender
``j`` (activity ``a_j``, the input to the linear) to receiver ``i`` (pre-activation
``z_i``, its output):

    m1   A_ij = E_x |W_ij a_j|              activation-weighted flow
    m2   A_ij = |corr_x(a_j, r_i)|          co-activation, weight-blind
    m3   A_ij = E_x |dy/dz_i * W_ij a_j|    output-Jacobian-weighted
    mag  A_ij = |W_ij|                      magnitude baseline

The module-organization term (v2, causal form) couples layer-mean-normalized
scores ``hatA = A / mean(A)`` with code distances ``d_ij = ||phi_i - phi_j||``:

    R_mod = sum_l sum_ij [ hatA_ij d_ij + c * max(0, m - d_ij) ],    m = 1.

The suppression term prices weak-communication edges with within-layer
max-normalized scores ``tildeA = A / max(A)``:

    R_sup = (1/L) sum_l sum_ij (1 - tildeA_ij) |W_ij|.

Causal variants backpropagate through the scores, so the regularizer also
shapes the weights ("rich-get-richer": strong edges pull codes together, which
cheapens further growth). The detached variant computes the scores under
``no_grad`` each step, so the weights receive no gradient through ``A`` and only
the codes (plus, via ``R_sup``, the raw ``|W|``) feel the regularizer.

Scores m1/m2/m3 read the current minibatch: forward hooks cache each linear's
input and output during training, and ``penalty`` consumes the cache. The codes
are ordinary parameters (in ``state_dict``); the cache is transient.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..registry import method
from .base import Method
from .nns import DenseMLP, boundary_widths

EPS = 1e-12
MARGIN = 1.0
CODE_INIT_SCALE = 0.5
DEFAULT_CODE_DIM = 8


class CommCodes(nn.Module):
    """One learned code vector per neuron, per layer boundary."""

    def __init__(self, widths, *, code_dim: int, init_scale: float = CODE_INIT_SCALE):
        super().__init__()
        self.code_dim = int(code_dim)
        init_scale = float(init_scale)
        if self.code_dim <= 0:
            raise ValueError(f"code_dim must be positive; received {code_dim}.")
        if not init_scale > 0:
            raise ValueError(f"init_scale must be positive; received {init_scale}.")
        self.phi = nn.ParameterDict({
            f"b{b}": nn.Parameter(torch.randn(int(w), self.code_dim) * init_scale)
            for b, w in enumerate(widths)
        })

    def distance(self, boundary: int) -> torch.Tensor:
        """``d_ij = ||phi_recv_i - phi_send_j||`` for one linear, shape (out, in)."""
        recv = self.phi[f"b{boundary + 1}"]
        send = self.phi[f"b{boundary}"]
        difference = recv.unsqueeze(1) - send.unsqueeze(0)
        return difference.square().sum(dim=-1).clamp_min(EPS).sqrt()


def _cache_activations(module, inputs, output):
    """Training-only forward hook: keep this step's input/output for `penalty`."""
    if module.training:
        module._comm_in = inputs[0]
        module._comm_out = output


def _receiver_jacobians(layers, caches) -> list[torch.Tensor]:
    """``dy/dz^{l+1}`` per boundary via a manual top-down sweep, shape (n, out)."""
    grads = [None] * len(layers)
    n = caches[-1][1].shape[0]
    grads[-1] = caches[-1][1].new_ones(n, layers[-1].weight.shape[0])
    for b in range(len(layers) - 2, -1, -1):
        upstream = grads[b + 1] @ layers[b + 1].weight
        grads[b] = upstream * (caches[b][1] > 0).to(upstream.dtype)
    return grads


class _Communication(Method):
    """Shared machinery; concrete variants pin ``score`` and ``detach_scores``."""

    score: str = "m1"
    detach_scores: bool = False

    def build(self, cfg, target):
        hp = cfg.get("hp") or {}
        model = DenseMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
        )
        model.codes = CommCodes(
            boundary_widths(model),
            code_dim=int(hp.get("code_dim", DEFAULT_CODE_DIM)),
            init_scale=float(hp.get("init_scale", CODE_INIT_SCALE)),
        )
        for layer in model.linear_layers():
            layer.register_forward_hook(_cache_activations)
        return model

    def _scores(self, layers, caches) -> list[torch.Tensor]:
        """One nonnegative (out, in) communication matrix per boundary."""
        if self.score == "mag":
            return [layer.weight.abs() for layer in layers]
        if self.score == "m1":
            return [
                layer.weight.abs() * sender.abs().mean(dim=0, keepdim=True)
                for layer, (sender, _z) in zip(layers, caches)
            ]
        if self.score == "m3":
            jacobians = _receiver_jacobians(layers, caches)
            return [
                layer.weight.abs() * (g.abs().t() @ sender.abs()) / sender.shape[0]
                for layer, (sender, _z), g in zip(layers, caches, jacobians)
            ]
        if self.score == "m2":
            scores = []
            for index, (layer, (sender, z)) in enumerate(zip(layers, caches)):
                receiver = z if index == len(layers) - 1 else torch.relu(z)
                s = (sender - sender.mean(dim=0)) / (sender.std(dim=0) + 1e-8)
                r = (receiver - receiver.mean(dim=0)) / (receiver.std(dim=0) + 1e-8)
                scores.append((r.t() @ s).abs() / sender.shape[0])
            return scores
        raise ValueError(f"unknown communication score {self.score!r}")

    def penalty(self, model, hp):
        lam = float(hp["lam"])
        c = float(hp.get("c", 1.0))
        lam_sup = float(hp.get("lam_sup", 0.0))
        for name, value in (("lam", lam), ("c", c), ("lam_sup", lam_sup)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative; received {value}.")

        layers = model.linear_layers()
        caches = []
        for layer in layers:
            cached_in = getattr(layer, "_comm_in", None)
            cached_out = getattr(layer, "_comm_out", None)
            if cached_in is None or cached_out is None:
                raise RuntimeError(
                    "communication penalty requires a training forward pass before "
                    "each penalty call (activation cache is empty)"
                )
            caches.append((cached_in, cached_out))

        if self.detach_scores:
            with torch.no_grad():
                scores = [a.clone() for a in self._scores(layers, caches)]
        else:
            scores = self._scores(layers, caches)

        codes = model.codes
        mod = model.output_layer.weight.new_zeros(())
        sup = mod.new_zeros(())
        for boundary, (layer, A) in enumerate(zip(layers, scores)):
            D = codes.distance(boundary)
            attraction = (A / (A.mean() + EPS)) * D
            repulsion = c * torch.relu(MARGIN - D)
            mod = mod + attraction.sum() + repulsion.sum()
            if lam_sup != 0.0:
                sup = sup + ((1.0 - A / (A.max() + EPS)) * layer.weight.abs()).sum()

        total = lam * mod
        if lam_sup != 0.0:
            total = total + lam_sup * (sup / len(layers))
        return total

    def extra_diagnostics(self, model):
        """Code-geometry summary of the selected checkpoint."""
        codes = model.codes
        distances = []
        with torch.no_grad():
            for boundary in range(len(codes.phi) - 1):
                distances.append(codes.distance(boundary).reshape(-1))
        if not distances:
            return {}
        d = torch.cat(distances)
        return {
            "comm_code_dist_mean": float(d.mean()),
            "comm_code_dist_p10": float(torch.quantile(d, 0.10)),
            "comm_frac_close": float((d < 0.25 * MARGIN).to(torch.float32).mean()),
        }


@method("comm_m1")
class CommM1(_Communication):
    """Causal M1: activation-weighted flow scores, gradients through A."""


@method("comm_m1d")
class CommM1Detached(_Communication):
    """Detached M1: scores measured under no_grad, constant per step."""

    detach_scores = True


@method("comm_m2")
class CommM2(_Communication):
    """Causal M2: weight-blind sender/receiver co-activation scores."""

    score = "m2"


@method("comm_m3")
class CommM3(_Communication):
    """Causal M3: output-Jacobian-weighted, task-aware scores."""

    score = "m3"


@method("comm_mag")
class CommMag(_Communication):
    """Magnitude baseline: A = |W|, no activation dependence."""

    score = "mag"
