"""The optimized LeWi implementation for the timing validation (decision 2026-09-25): the record's mathematics with the geometry
formed once per step and batched over the transitions, the forward and backward compiled, and the cached K arms with the record's
counter semantics. Built from the record's own model (csfrbench.methods.lewi_v1 through the in5 arm, so the initialization, the input
rule and the readout redraws are the record's, bit for bit) and checkpointed in the record's format (the per-boundary gate_codes and the
event_mask buffers are written back before every state_dict read).

The mathematics (unchanged): s_ij = R^2 - ||e_i - e_j||^2, g = sigmoid(s), M = 1[g > 1/2], y = (W * g * M) x + b; Adam on every weight,
bias and position; the positions move only at refresh steps of the cached arms.

  * `once`: the mask is the gate's own sign taken in the forward pass. The record re-derives M after every optimizer step from the
    updated positions; those are exactly the positions the next forward pass reads, so the two agree bit for bit (step-profile).
  * `batched`: every unit's position in one parameter tensor P; the transitions' receivers and senders are gathered through a padded
    index (a spare zero row) into [L, W, r] each; one cdist, one sigmoid. The record's per-boundary parameters stay in the model, frozen,
    and receive P's rows before any state_dict read, diagnostic or fold (write_back_).
  * cached K > 1: at an optimizer step t = 0, K, 2K, ... (t = step - 1) the forward pass forms the differentiable batched geometry; right
    after that step's update the hook recomputes the gates from the updated positions under no_grad, re-derives the masks, counts the
    edge changes (one tensor reduction per refresh, no host reads) and caches the multiplier g * M per layer; the next K - 1 steps use
    W * cached multiplier (a Python flag per layer, mirrored to the `cache_on` buffer at every refresh). The counters lewi_global_step,
    lewi_refreshes, lewi_cache_passes, lewi_mask_flips / _reopenings / _closures are the record's buffers.
  * K = 1: no hook work at all.

`build_fast(method, model, refresh_every, compile)` wraps a record model; `FastMethod` is the method object for the trainer mirror
(penalty, after_optimizer_step, extra_diagnostics, effective_weights, sync_).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from csfrbench.methods.lewi_v1 import BallMaskedLinear

COUNTERS = ("lewi_global_step", "lewi_refreshes", "lewi_cache_passes", "lewi_mask_flips", "lewi_mask_reopenings", "lewi_mask_closures")


class FastBallLinear(BallMaskedLinear):
    """The record's ball layer with the gate supplied by the batched geometry (`_g`), the mask its sign, the effective weight
    where(M, W g, 0); between refreshes of a cached arm, W * cached_multiplier."""

    _g = None
    _on = False
    _cache_view = None                                 # a view into the geometry's padded multiplier cache (cached arms)

    def forward(self, x):
        if self._on:
            return F.linear(x, self.weight * self._cache_view, self.bias)
        g = self._g if self._g is not None else self.gate()
        return F.linear(x, torch.where(g.detach() > 0.5, self.weight * g, 0.0), self.bias)

    def effective_weight(self) -> torch.Tensor:
        if self._on:
            return self.weight * self._cache_view
        g = self.gate()
        return torch.where(g.detach() > 0.5, self.weight * g, 0.0)

    @torch.no_grad()
    def sync_mask_(self, g: torch.Tensor | None = None) -> None:
        g = self.gate() if g is None else g
        self.event_mask.copy_((g > 0.5).to(self.event_mask.dtype))


class BatchedPositions(nn.Module):
    """P [sum of widths, r]; the padded gathers; one cdist and one sigmoid for every transition's gate."""

    def __init__(self, model, layers, *, batch: bool = True):
        """`batch`: the full-width transitions through one batched cdist (bit-exact with the record's per-transition call on the CPU's
        MKL, not on cuBLAS: the GPU form keeps every transition's own call)."""
        super().__init__()
        self.batch = bool(batch)
        codes = list(model.gate_codes)
        self._codes = codes
        widths = [int(c.shape[0]) for c in codes]
        with torch.no_grad():
            P = torch.cat([c.detach() for c in codes], dim=0)
        self.positions = nn.Parameter(P.clone())
        self.widths, self.layers = widths, layers
        offsets = np.cumsum([0] + widths)
        W = max(widths)
        spare = int(offsets[-1])
        dev = P.device                                  # every buffer on the positions' device (the record's model may already sit on the GPU)
        recv_idx = torch.full((len(layers), W), spare, dtype=torch.long)
        send_idx = torch.full((len(layers), W), spare, dtype=torch.long)
        for l in range(len(layers)):                    # layer l binds recv = codes[l + 1], send = codes[l]
            recv_idx[l, :widths[l + 1]] = torch.arange(offsets[l + 1], offsets[l + 2])
            send_idx[l, :widths[l]] = torch.arange(offsets[l], offsets[l + 1])
        self.register_buffer("recv_idx", recv_idx.to(dev))
        self.register_buffer("send_idx", send_idx.to(dev))
        self.register_buffer("zero_row", torch.zeros(1, P.shape[1], device=dev))
        self.register_buffer("gate_scale", layers[0].gate_scale.detach().clone())
        self.register_buffer("gate_offset", layers[0].gate_offset.detach().clone())
        for layer in layers:
            assert float(layer.gate_scale) == float(self.gate_scale) and float(layer.gate_offset) == float(self.gate_offset)
        for c in codes:                                 # the record's per-boundary parameters step aside; P is the optimized parameter
            c.requires_grad_(False)
        # the transitions the batched call would pad (a width below W on either side: the input transition, the readout) go through the record's
        # own per-transition call - a padded matmul sums in a different order and moves their position gradients by an ulp
        self.single_rows = [(l, (int(offsets[l + 1]), int(offsets[l + 2]), int(offsets[l]), int(offsets[l + 1]))) for l in range(len(layers)) if not self.batch or widths[l + 1] != W or widths[l] != W]
        self.any_batched = len(self.single_rows) < len(layers)
        with torch.no_grad():                           # the padded caches of the cached arms: the multiplier g M and the mask, every transition at once
            G = self.gates()
            self.register_buffer("cache_mask", G > 0.5)
            self.register_buffer("cache_mult", torch.where(G > 0.5, G, 0.0))
        for l, layer in enumerate(layers):
            layer._cache_view = self.cache_mult[l, :layer.out_features, :layer.in_features]

    def gates(self) -> torch.Tensor:
        """[L, W, W]: the batched gates of the full-width transitions; every transition the batch would pad is computed through the record's
        own per-transition call and spliced in (autograd through the cats), so that the gates and the position gradients are the record's
        bit for bit."""
        G = self._batched_gates() if self.any_batched else torch.zeros(len(self.layers), self.recv_idx.shape[1], self.recv_idx.shape[1], device=self.positions.device, dtype=self.positions.dtype)
        for l, (r0, r1, s0, s1) in self.single_rows:
            g = torch.sigmoid(self.gate_scale * (-torch.cdist(self.positions[r0:r1], self.positions[s0:s1]) ** 2) + self.gate_offset)
            row = torch.cat([g, G[l, :r1 - r0, s1 - s0:]], dim=1)
            block = torch.cat([row, G[l, r1 - r0:]], dim=0)
            G = torch.cat([G[:l], block[None], G[l + 1:]], dim=0)
        return G

    def _batched_gates(self) -> torch.Tensor:
        Pz = torch.cat([self.positions, self.zero_row], dim=0)
        recv, send = Pz[self.recv_idx], Pz[self.send_idx]
        return torch.sigmoid(self.gate_scale * (-torch.cdist(recv, send) ** 2) + self.gate_offset)

    def gate_parts(self):
        """The batched gates [L, W, W] of the full-width transitions and, per padded transition, the record's own gate (no splice: the
        per-step path of the forward pass)."""
        G = self._batched_gates() if self.any_batched else None
        own = {l: torch.sigmoid(self.gate_scale * (-torch.cdist(self.positions[r0:r1], self.positions[s0:s1]) ** 2) + self.gate_offset) for l, (r0, r1, s0, s1) in self.single_rows}
        return G, own

    def assign(self) -> None:
        G, own = self.gate_parts()
        for l, layer in enumerate(self.layers):
            layer._g = own[l] if l in own else G[l, :layer.out_features, :layer.in_features]

    @torch.no_grad()
    def write_back_(self) -> None:
        off = 0
        for c, w in zip(self._codes, self.widths):
            c.copy_(self.positions[off:off + w]); off += w


class FastModel(nn.Module):
    """The record's model with the batched geometry assigned before every differentiable forward pass. `state_dict` is the record's
    (the wrapper's own parameters are P and the index buffers, kept under `geometry.`); the record's keys are written back first."""

    def __init__(self, inner, geometry: BatchedPositions, refresh_every: int):
        super().__init__()
        self.inner, self.geometry, self.refresh_every = inner, geometry, int(refresh_every)
        self.layers = geometry.layers
        self._cached_now = False

    def forward(self, x):
        if not self._cached_now:
            self.geometry.assign()
        return self.inner(x)

    def parameters(self, recurse: bool = True):
        return iter([p for p in self.inner.parameters() if p.requires_grad] + [self.geometry.positions])

    def linear_layers(self):
        return self.inner.linear_layers()

    @property
    def hidden_layers(self):
        return self.inner.hidden_layers

    @property
    def output_layer(self):
        return self.inner.output_layer

    @property
    def readout_structural(self):
        return self.inner.readout_structural

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    def state_dict(self, *args, **kwargs):
        """The record's state dict (the inner model's keys, no wrapper prefix), after the write-back."""
        self.sync_()
        return self.inner.state_dict(*args, **kwargs)

    @torch.no_grad()
    def sync_(self) -> None:
        """The record's form: P written to the per-boundary parameters, the masks re-derived, the cache flags mirrored."""
        self.geometry.write_back_()
        if self.refresh_every == 1:
            G = self.geometry.gates()
            for l, layer in enumerate(self.layers):
                layer.sync_mask_(G[l, :layer.out_features, :layer.in_features])
        else:                                           # the record's per-layer buffers from the padded caches (the masks of the last refresh)
            for l, layer in enumerate(self.layers):
                layer.event_mask.copy_(self.geometry.cache_mask[l, :layer.out_features, :layer.in_features].to(layer.event_mask.dtype))
                layer.cached_multiplier.copy_(self.geometry.cache_mult[l, :layer.out_features, :layer.in_features])
                layer.cache_on.fill_(1 if layer._on else 0)

    def load_state_dict(self, state, strict: bool = True):
        out = self.inner.load_state_dict(state, strict=strict)
        with torch.no_grad():                           # the record's positions are the truth after a load; the padded caches from the record's buffers
            self.geometry.positions.copy_(torch.cat([c.detach() for c in self.geometry._codes], dim=0))
            for l, layer in enumerate(self.layers):
                layer._g = None                         # no stale gate view survives a load
                if hasattr(layer, "cache_on"):
                    layer._on = bool(layer.cache_on)
                    self.geometry.cache_mask[l, :layer.out_features, :layer.in_features] = layer.event_mask.bool()
                    self.geometry.cache_mult[l, :layer.out_features, :layer.in_features] = layer.cached_multiplier
            self._cached_now = all(layer._on for layer in self.layers) if hasattr(self.layers[0], "cache_on") else False
        return out


class FastMethod:
    """The method object of the optimized implementation for the trainer mirror; `inner` is the record's method (its penalty, its
    diagnostics on the synced model)."""

    def __init__(self, inner, model: FastModel):
        self.inner, self.model = inner, model
        self.every = model.refresh_every
        self.layers = model.layers

    def penalty(self, model, hp):
        return self.inner.penalty(model.inner if isinstance(model, FastModel) else model, hp)

    @torch.no_grad()
    def after_optimizer_step(self, model, optimizer, *, step, total_steps):
        if self.every == 1:
            return                                      # the mask is formed in the next forward pass
        m = self.model
        t = int(step) - 1
        if t % self.every == 0:                         # a refresh step: the positions moved in this update
            geo = m.geometry
            G = geo.gates_c()
            keep = G > 0.5
            old = geo.cache_mask                        # the padded entries (the spare zero row against itself) never change: they count no flip
            reopen = (~old & keep).sum(); close = (old & ~keep).sum()
            geo.cache_mask.copy_(keep)
            geo.cache_mult.copy_(torch.where(keep, G, 0.0))
            for layer in self.layers:
                layer._on = True
            m.inner.lewi_refreshes += 1; m.inner.lewi_cache_passes += 1
            m.inner.lewi_mask_reopenings += reopen; m.inner.lewi_mask_closures += close; m.inner.lewi_mask_flips += reopen + close
            m._cached_now = True
        if (t + 1) % self.every == 0:                   # the next step is a refresh: fresh differentiable geometry
            for layer in self.layers:
                layer._on = False
            m._cached_now = False
        m.inner.lewi_global_step.fill_(int(step))

    def effective_weights(self, model):
        return tuple(layer.effective_weight() for layer in model.linear_layers())

    def extra_diagnostics(self, model):
        m = model if isinstance(model, FastModel) else self.model
        m.sync_()
        out = dict(self.inner.extra_diagnostics(m.inner))
        if self.every > 1:
            out["refresh_every"] = self.every
            for name in COUNTERS:
                out[name.replace("lewi_", "", 1)] = int(getattr(m.inner, name))
        return out

    def sync_(self):
        self.model.sync_()


def build_fast(method, model, refresh_every: int = 1, *, compile: bool = True, batch: bool | None = None):
    """Wrap a record model (built by the record's method) into the optimized form. For K > 1 the record's cached buffers and counters
    are created as the record creates them (lk_methods); the record's layer class is replaced by FastBallLinear."""
    layers = []
    for layer in model.linear_layers():
        assert isinstance(layer, BallMaskedLinear), f"{type(layer).__name__} is not the ball layer"
        layer.__class__ = FastBallLinear
        layer._g, layer._on = None, False
        layers.append(layer)
    every = int(refresh_every)
    if every > 1:
        for layer in layers:
            if not hasattr(layer, "cached_multiplier"):
                layer.register_buffer("cached_multiplier", torch.zeros_like(layer.weight))
                layer.register_buffer("cache_on", torch.zeros((), dtype=torch.uint8))
        for name in COUNTERS:
            if not hasattr(model, name):
                model.register_buffer(name, torch.zeros((), dtype=torch.int64, device=layers[0].weight.device))
        model.lewi_refresh_every = every
    dev = layers[0].weight.device
    if batch is None:
        batch = dev.type == "cpu"                       # batched where bit-exact with the record (MKL); per transition on cuBLAS
    geometry = BatchedPositions(model, layers, batch=batch).to(dev)
    for l, layer in enumerate(layers):                  # the views again on the device's buffers
        layer._cache_view = geometry.cache_mult[l, :layer.out_features, :layer.in_features]
    fast = FastModel(model, geometry, every)
    geometry.gates_c = torch.compile(geometry.gates, dynamic=False) if compile else geometry.gates   # the refresh hook's gate pass
    if compile:
        fast.inner_forward = fast.forward
        fast.forward = torch.compile(fast.forward, dynamic=False)   # the compiled forward (two graphs for a cached arm: the flag guards)
    return FastMethod(method, fast), fast


class FastDense(nn.Module):
    """Dense with the forward compiled; nothing else changes."""

    def __init__(self, inner, *, compile: bool = True):
        super().__init__()
        self.inner = inner
        self.fn = torch.compile(inner, dynamic=False) if compile else inner

    def forward(self, x):
        return self.fn(x)

    def parameters(self, recurse: bool = True):
        return self.inner.parameters()

    def linear_layers(self):
        return self.inner.linear_layers()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    def state_dict(self, *args, **kwargs):
        return self.inner.state_dict(*args, **kwargs)

    def load_state_dict(self, state, strict: bool = True):
        return self.inner.load_state_dict(state, strict=strict)
