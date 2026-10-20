"""RigL dynamic sparse training (Evci et al., 2020) on the shared MLP
backbone (per-workload profile; see configs/profile.yaml).

Sources: arXiv:1911.11134 and the official implementation
google-research/rigl (sparse_optimizers_base.py::SparseRigLOptimizerBase,
sparse_utils.py::get_sparsities_erdos_renyi). Method card:
studies/csfrbench/final-design/versions/v1.0/method_cards/rigl.md

Faithful mechanics, per source:
  * per-layer binary masks over the weight matrices; biases stay dense;
  * layer densities follow the Erdos-Renyi-Kernel rule (kernel dims are 1
    for affine layers): raw probability (n_in + n_out) / (n_in * n_out),
    scaled by a common epsilon to hit the global density, layers whose
    scaled probability exceeds 1 promoted to dense and epsilon re-solved;
  * a mask-update iteration REPLACES the gradient step, exactly as in
    the source (paper Algorithm 1's else-branch; the released code's
    cond(mask_update, apply_gradient), asserted by its own test suite):
    on such an iteration the weight update and the optimizer-state
    advance of that step are undone, and each layer drops the f(t)
    fraction of its active connections with the smallest |w| (tiny
    Gaussian noise breaks ties, as in the source) and grows the same
    count of inactive connections with the largest instantaneous dense
    gradient |dL/dw_eff| from the current batch - which therefore only
    selects grows and never trains, as in the source;
  * update iterations are the first step and then every delta_t = 100
    steps through t_end = 0.75 * total steps inclusive (the source's
    early first update via last_update_step = -frequency; f(t_end) = 0
    makes the final one a pure step-skip, as in the source);
  * f(t) = alpha/2 * (1 + cos(pi * t / t_end)), alpha = 0.3 (cosine
    drop-fraction annealing, the paper's default);
  * connections grown from the inactive state start at weight 0 and have
    their optimizer state zeroed (grow_init='zeros', initial_acc_scale=0);
    a connection dropped and immediately regrown keeps its weight
    (reinit_when_same=False in the source).

Adaptations to the benchmark, disclosed in the method card: the common
contract fixes Adam for every method, so the source's momentum reset
becomes zeroing the Adam exp_avg / exp_avg_sq entries of newly grown
connections; the mask topology is initialized from the benchmark's
auxiliary seed stream at the ERK densities; and one non-emulation of the
update-replaces-step semantics remains - the shared trainer's cosine
learning-rate schedule still advances on update iterations (the source's
global step does not) - and the same iteration-vs-global-step clock also
places update events every 100 data iterations where the source lands
every 100 weight updates (every 101 data iterations) and runs f(t)/t_end
on the iteration clock - together a 0.75% clock skew over the run (event
count and the f-value sequence at matched events are identical).

The replace semantics are implemented by stashing the pre-step parameter
and optimizer state whenever the NEXT step is an update iteration
(``penalty`` captures the initialization for the step-1 update, where the
optimizer state is still empty) and restoring it in the post-step hook
before rewiring. The stash is training-transient model state, never part
of a checkpoint; a hypothetical mid-training resume would therefore skip
the undo at its first post-resume event (no such path exists in the
pipeline), and calling ``penalty`` outside training (analysis code)
plants a harmless transient init stash. Repair record: audit finding
R1/R2, studies/csfrbench/baseline-audit/v05-suite (2026-08-28); round-2
bit-exact verification: report/round2 section 3.1.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..registry import method
from .base import Method, readout_structural
from .nns import DenseLinear, ReLUMLP

DELTA_T = 100          # mask-update interval, source default
ALPHA = 0.3            # initial drop fraction, source default
T_END_FRACTION = 0.75  # updates stop after this fraction of training
TIE_NOISE_STD = 1e-5   # drop-score tie-breaking noise, source default


def erk_densities(shapes: list[tuple[int, int]], density: float) -> list[float]:
    """Per-layer densities under the Erdos-Renyi-Kernel rule.

    Mirrors sparse_utils.get_sparsities_erdos_renyi with include_kernel
    irrelevant (affine layers): raw probability (n_in + n_out)/(n_in*n_out)
    scaled by one epsilon so the total active count matches ``density``;
    any layer whose scaled probability exceeds 1 is promoted to dense and
    epsilon is re-solved over the rest. Count arithmetic rounds where the
    source floors: equivalent up to two connections per layer on the
    envelope (audited).
    """
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must lie in (0, 1]; received {density}")
    n_params = [o * i for (o, i) in shapes]
    dense_layers: set[int] = set()
    while True:
        divisor = 0.0
        rhs = 0.0
        raw = {}
        for idx, (n, (o, i)) in enumerate(zip(n_params, shapes)):
            n_zeros = int(round(n * (1.0 - density)))
            if idx in dense_layers:
                rhs -= n_zeros
            else:
                rhs += n - n_zeros
                raw[idx] = (o + i) / (o * i)
                divisor += raw[idx] * n
        eps = rhs / divisor
        overflow = [idx for idx, p in raw.items() if p * eps > 1.0]
        if not overflow:
            break
        # Promote the largest-probability layers, as the source does.
        max_p = max(raw[idx] for idx in overflow)
        dense_layers.update(idx for idx in overflow
                            if raw[idx] == max_p)
    return [1.0 if idx in dense_layers else raw[idx] * eps
            for idx in range(len(shapes))]


class RigLLinear(DenseLinear):
    """Affine layer with a persistent binary mask and dense-grad capture."""

    def __init__(self, in_features, out_features, *, is_output):
        super().__init__(in_features, out_features, is_output=is_output)
        self.register_buffer("mask", torch.ones_like(self.weight))
        self._dense_grad = None    # training-only, instantaneous |dL/dw_eff|

    def forward(self, x):
        w_eff = self.weight * self.mask
        if w_eff.requires_grad:
            w_eff.register_hook(self._capture)
        return F.linear(x, w_eff, self.bias)

    def _capture(self, grad):
        self._dense_grad = grad.detach()

    def effective_weight(self):
        return self.weight * self.mask


def _is_update_step(step: int, t_end: int) -> bool:
    """The source's update iterations: the first step, then every DELTA_T
    steps through t_end inclusive."""
    return step == 1 or (step % DELTA_T == 0 and step <= t_end)


def _stash_state(model, optimizer) -> dict:
    """Snapshot every parameter and its optimizer state."""
    return {
        "params": [p.detach().clone() for p in model.parameters()],
        "state": [
            ({key: (value.clone() if torch.is_tensor(value) else value)
              for key, value in optimizer.state[p].items()}
             if p in optimizer.state else None)
            for p in model.parameters()
        ],
    }


def _restore_state(model, optimizer, stash) -> None:
    """Undo the step that just ran: parameters and optimizer state back to
    the stashed snapshot. ``state == "initial"`` marks the pre-training
    stash, where the optimizer state was still empty."""
    with torch.no_grad():
        for p, saved in zip(model.parameters(), stash["params"]):
            p.copy_(saved)
    if stash["state"] == "initial":
        for p in model.parameters():
            if p in optimizer.state:
                optimizer.state[p] = {}
        return
    for p, saved in zip(model.parameters(), stash["state"]):
        if saved is None:
            if p in optimizer.state:
                optimizer.state[p] = {}
            continue
        live = optimizer.state[p]
        for key, value in saved.items():
            if torch.is_tensor(value) and torch.is_tensor(live.get(key)):
                live[key].copy_(value)
            else:
                live[key] = value


def _drop_grow(model, optimizer, *, fraction: float) -> None:
    """One official mask update: magnitude drop, gradient grow."""
    for layer in model.structural_layers():
        mask = layer.mask
        n_active = int(mask.sum().item())
        k = int(fraction * n_active)
        if k == 0:
            continue
        grad = layer._dense_grad
        if grad is None:
            continue
        weight = layer.weight.detach()
        # Drop: keep the (n_active - k) largest |w * mask|; inactive
        # entries score ~0 and fall to the bottom, exactly as in the
        # source's whole-tensor sort.
        score_drop = (weight * mask).abs().flatten()
        score_drop += torch.randn_like(score_drop) * TIE_NOISE_STD
        keep_idx = score_drop.topk(n_active - k).indices
        kept = torch.zeros_like(score_drop)
        kept[keep_idx] = 1.0
        # Grow: among connections not kept, take the k largest
        # instantaneous dense gradients.
        score_grow = grad.abs().flatten().clone()
        score_grow[kept.bool()] = float("-inf")
        grow_idx = score_grow.topk(k).indices
        grown = torch.zeros_like(kept)
        grown[grow_idx] = 1.0
        new_mask = (kept + grown).view_as(mask)
        # Connections grown from the inactive state start at zero with
        # zeroed optimizer state; dropped-and-regrown keep their weight.
        fresh = (grown.view_as(mask) > 0) & (mask == 0)
        with torch.no_grad():
            layer.weight[fresh] = 0.0
            state = optimizer.state.get(layer.weight)
            if state:
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        state[key][fresh] = 0.0
        mask.copy_(new_mask)
        layer._dense_grad = None


@method("rigl")
class RigL(Method):
    """RigL: static-sparsity training with magnitude drop / gradient grow."""

    hp_keys = frozenset({"density", "readout"})

    def build(self, cfg, target):
        density = float(cfg["hp"]["density"])
        structural = readout_structural(cfg.get("hp"))
        model = ReLUMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
            # v1.0 rule: the readout stays dense and outside the density
            # accounting; v1.1 ``readout: structural``: the readout is masked
            # and allocated like every other layer (the source's treatment)
            layer_factory=lambda i, o, is_output: (
                DenseLinear(i, o, is_output=True) if (is_output and not structural)
                else RigLLinear(i, o, is_output=is_output)),
            readout_structural=structural,
        )
        layers = list(model.structural_layers())
        shapes = [(la.out_features, la.in_features) for la in layers]
        per_layer = erk_densities(shapes, density)
        generator = torch.Generator().manual_seed(int(cfg["auxiliary_init_seed"]))
        for layer, d in zip(layers, per_layer):
            n = layer.weight.numel()
            n_active = min(n, int(round(d * n)))
            scores = torch.rand(n, generator=generator)
            mask = torch.zeros(n)
            mask[scores.topk(n_active).indices] = 1.0
            layer.mask.copy_(mask.view_as(layer.mask))
            # Inactive connections start exactly at zero so the initial
            # state equals a network that never had them.
            layer.weight.data.mul_(layer.mask)
        model.rigl_density = density
        return model

    def penalty(self, model, hp):
        # No penalty; this hook only captures the pre-training state once,
        # so the source's mask update at the very first step can undo that
        # step's optimizer update (the optimizer state is empty here).
        if (getattr(model, "_rigl_prestep", None) is None
                and not getattr(model, "_rigl_started", False)):
            model._rigl_prestep = {
                "params": [p.detach().clone() for p in model.parameters()],
                "state": "initial",
            }
        return None

    def after_optimizer_step(self, model, optimizer, *, step, total_steps):
        model._rigl_started = True
        t_end = int(T_END_FRACTION * total_steps)
        if _is_update_step(step, t_end):
            stash = getattr(model, "_rigl_prestep", None)
            if stash is not None:
                _restore_state(model, optimizer, stash)
            fraction = 0.5 * ALPHA * (1.0 + math.cos(
                math.pi * min(step, t_end) / t_end))
            _drop_grow(model, optimizer, fraction=fraction)
        model._rigl_prestep = None
        if _is_update_step(step + 1, t_end):
            model._rigl_prestep = _stash_state(model, optimizer)

    def effective_weights(self, model):
        # The masked matrices; the raw tensors would report dropped,
        # momentum-drifted weights into the structure diagnostics.
        return tuple(model.effective_weights())

    def extra_diagnostics(self, model):
        masks = [layer.mask for layer in model.structural_layers()]
        active = sum(int(m.sum().item()) for m in masks)
        total = sum(m.numel() for m in masks)
        return {"active_connections": float(active),
                "density_realized": active / total,
                "density_target": float(model.rigl_density)}
