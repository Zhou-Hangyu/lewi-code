"""LeWi-v05 and LeWi-Reg-v05 - learned-wiring code masks (amendment 3,
section 5).

Migrated from an earlier internal study's ``cmasktt`` / ``cmaskttp`` arms
(its report, section "The method, in full"). Public names: LeWi-v05 (code key ``lewi-v05``) and LeWi-Reg-v05
(``lewi_reg-v05``). The ``-v05`` suffix marks a pre-release version of
the method (design decision 2026-08-28): the version benchmarked here
is a snapshot, not the method's final form. Migration map:
studies/csfrbench/final-design/versions/v0.5/method_cards/lewi-v05.md

The method, exactly as the source defines it:

  * one code phi in R^3 per unit per representation boundary (input, the
    hidden boundaries, output), initialized N(0, 0.5^2) from the
    benchmark auxiliary seed stream; the SAME code serves the unit's
    incoming and outgoing connection scores;
  * each affine layer carries two scalars - a scale (init 1) and an
    offset (init 0) - and forms the soft gate
    G_ij = sigmoid(scale * <phi_recv_i, phi_send_j> + offset) on every
    affine layer;
  * the forward pass of every layer is y = (W * G * M) x + b, where the
    binary mask M = 1[G > 1/2] is re-derived EVERY optimization step at
    the first two layer transitions (boundaries 0 and 1) and stays all
    ones elsewhere. M carries no gradient; commitment is continuous,
    reversible at every step, and active to the last step (no cadence,
    no liveness floor, no horizon cutoff, no prescribed fan-in);
  * LeWi has no penalty; its whole search space is the learning rate.
    LeWi-Reg adds the source's one pressure scalar,
    lam * (1/L) * sum_l mean(G_l), searched alongside the learning rate.

Masking scope is a measured property of the method, not a depth
convention: the source ran the all-boundary variant and rejected it
(uniform-v1), so the 128x8 migration keeps binary masks on exactly the
first two transitions (amendment 3 section 5).

The layer machinery is the package's own
cmask.SmoothCodeMaskedLinear / HardenedCodeMaskedLinear, which are
line-identical to the study's own class definitions (the package carried
them over unchanged from the study; the two trees define them
independently, which is what makes the bit-exact migration regression a
genuine cross-implementation check); the study file stays untouched so
historical runs keep reproducing.

Registered ablations (budget plan tier C; fixed method arguments in the
``hp`` envelope, one knob per arm, defaults reproduce the method above
bit-exactly):

  * ``code_dim``            embedding dimension r (default 3): 1, 2, 8;
  * ``masked_boundaries``   "first2" (default) / "unmasked" (soft-only:
                            every mask stays all-ones) / "all" (every
                            affine layer re-commits, the source's
                            rejected uniform-v1 scope);
  * ``mask_every``          re-derivation cadence in optimizer steps
                            (default 1 = every step); the registered
                            arm uses 500, the common-profile validation
                            cadence, so commitment happens at the same
                            64 per-run events the checkpointer uses;
  * ``score_kernel``        "dot" (default) or "euclidean": the score's
                            bilinear form <phi_r, phi_s> is replaced by
                            -||phi_r - phi_s||^2 (a distance-based
                            compatibility kernel, motivated by
                            position-based wiring; the method-level
                            neuron-positions comparison is the
                            erb_positions baseline). The
                            distance kernel is negative for every pair,
                            so the default offset 0 would start every
                            gate below 1/2, the first two masks all
                            zero, and no gradient could reach the
                            masked layers' codes (measured: density
                            0.000, constant predictor). The offset
                            therefore initializes at the mean initial
                            squared distance, 2 r (0.5)^2 - the
                            distance-kernel analogue of "gate 0.5 at
                            init". Scale and offset stay learned.
"""
from __future__ import annotations

import torch
from torch import nn

from ..registry import method
from .base import Method
from .cmask import CODE_INIT, HardenedCodeMaskedLinear
from .nns import ReLUMLP, boundary_widths

CODE_RANK = 3          # the source's fixed neuron-embedding dimension
MASKED_BOUNDARIES = (0, 1)
_BOUNDARY_CHOICES = ("first2", "unmasked", "all")
_KERNEL_CHOICES = ("dot", "euclidean")


@torch.no_grad()
def _rederive_mask_(layer) -> None:
    """M <- 1[G > 1/2]: the gate's own sign, re-derived per step."""
    layer.event_mask.copy_((layer.gate() > 0.5).to(layer.event_mask.dtype))


class EuclideanCodeMaskedLinear(HardenedCodeMaskedLinear):
    """Score kernel -||phi_r - phi_s||^2 in place of <phi_r, phi_s>."""

    def gate(self) -> torch.Tensor:
        recv, send = self._codes
        return torch.sigmoid(self.gate_scale * (-torch.cdist(recv, send) ** 2)
                             + self.gate_offset)


@method("lewi-v05")
class LeWi(Method):
    """LeWi-v05: per-step midpoint commitment on the first two transitions."""

    code_dim = CODE_RANK
    hp_keys = frozenset({"code_dim", "masked_boundaries", "mask_every",
                         "score_kernel"})

    def build(self, cfg, target):
        hp = cfg.get("hp") or {}
        code_dim = int(hp.get("code_dim", self.code_dim))
        kernel = str(hp.get("score_kernel", "dot"))
        if kernel not in _KERNEL_CHOICES:
            raise ValueError(f"score_kernel must be one of {_KERNEL_CHOICES}")
        scope = str(hp.get("masked_boundaries", "first2"))
        if scope not in _BOUNDARY_CHOICES:
            raise ValueError(
                f"masked_boundaries must be one of {_BOUNDARY_CHOICES}")
        mask_every = int(hp.get("mask_every", 1))
        if mask_every < 1:
            raise ValueError("mask_every must be a positive step count")
        layer_cls = (EuclideanCodeMaskedLinear if kernel == "euclidean"
                     else HardenedCodeMaskedLinear)
        model = ReLUMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
            layer_factory=lambda i, o, is_output: layer_cls(
                i, o, is_output=is_output),
        )
        widths = boundary_widths(model)
        generator = torch.Generator().manual_seed(int(cfg["auxiliary_init_seed"]))
        model.gate_codes = nn.ParameterList([
            nn.Parameter(torch.randn(int(w), code_dim, generator=generator)
                         * CODE_INIT)
            for w in widths
        ])
        for boundary, layer in enumerate(model.linear_layers()):
            layer.bind_codes(model.gate_codes[boundary + 1],
                             model.gate_codes[boundary])
        if kernel == "euclidean":
            with torch.no_grad():
                for layer in model.linear_layers():
                    layer.gate_offset.fill_(2.0 * code_dim * CODE_INIT ** 2)
        n_boundaries = len(list(model.linear_layers()))
        model.lewi_masked = {"first2": MASKED_BOUNDARIES,
                             "unmasked": (),
                             "all": tuple(range(n_boundaries))}[scope]
        model.lewi_mask_every = mask_every
        return model

    def after_optimizer_step(self, model, optimizer, *, step, total_steps):
        masked = getattr(model, "lewi_masked", MASKED_BOUNDARIES)
        if not masked or step % getattr(model, "lewi_mask_every", 1):
            return
        for index, layer in enumerate(model.linear_layers()):
            if index in masked:
                _rederive_mask_(layer)

    def effective_weights(self, model):
        return tuple(layer.effective_weight()
                     for layer in model.linear_layers())

    def extra_diagnostics(self, model):
        masked_set = getattr(model, "lewi_masked", MASKED_BOUNDARIES)
        with torch.no_grad():
            gates = torch.cat([layer.gate().reshape(-1)
                               for layer in model.linear_layers()])
            masked = [layer.event_mask for i, layer
                      in enumerate(model.linear_layers())
                      if i in masked_set]
            active = sum(int(m.sum()) for m in masked)
            total = sum(m.numel() for m in masked)
        return {"gate_open_mean": float(gates.mean()),
                "gate_frac_decided": float(((gates < 0.05) | (gates > 0.95))
                                           .to(torch.float32).mean()),
                "masked_boundary_density": (active / total) if total else 1.0}


@method("lewi_reg-v05")
class LeWiReg(LeWi):
    """LeWi-Reg-v05: LeWi-v05 plus the source's one pressure scalar."""

    hp_keys = LeWi.hp_keys | frozenset({"lam"})

    def penalty(self, model, hp):
        lam = float(hp["lam"])
        if lam < 0:
            raise ValueError(f"lam must be nonnegative, received {lam}")
        gates = [layer.gate() for layer in model.linear_layers()]
        return lam * torch.stack([g.mean() for g in gates]).mean()
