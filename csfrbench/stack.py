"""Many networks per kernel: the optimized training of LeWi and Dense units.

Every unit keeps the protocol of :mod:`csfrbench.training`: its network is built by its method under the unit's own
seeds (the same initial weights, embeddings and initialization checks as the per-network trainer), it draws its
minibatches in the per-network trainer's order, it takes Adam with its own learning rate under the cosine schedule, it is
validated every ``eval_every`` steps on its own validation pool and keeps its best checkpoint, and its test pool is read
once at the end. What changes is the execution:

* the ``B`` networks of a stack (one workload, one method) share every kernel: batched matrix products, one distance
  computation per group of layers, one Adam update over the stacked tensors with one learning rate per network;
* LeWi's gate runs as two fused C++ kernels (:mod:`csfrbench.kernels`); the rest of the step can be compiled;
* Dense takes the same stacked step without gates.

Outcomes are preserved, not digits. A batched kernel sums in another order than a per-network call, so a trajectory
agrees with the per-network trainer's to float rounding until the training dynamics amplify the difference (at high
learning rates, as two runs of the per-network trainer on different processors do too). Trials, winners and learning
curves agree with the records statistically (docs/REPRODUCE.md). For bit-for-bit reproduction of a
record use the per-network trainer: ``csfrbench run --engine exact``.

Two details differ from the per-network trainer and change no outcome: the fused step applies the update in the same
pass that detects a non-finite loss or gradient, so a network that diverges keeps its best checkpoint but records no
validation value when it diverged before its first checkpoint; and the cosine schedule is computed in closed form.
A diverged run has no test value and is never selected in either trainer.
"""
from __future__ import annotations

import math
import os
import shutil
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from . import datasets, search, suite, training

STACKABLE = ("lewi", "dense")
BATCH = training.BATCH_SIZE


# ------------------------------------------------------------------ units


def method_of(unit: dict) -> str:
    arm = suite.ARMS.get(unit["arm"])
    return unit.get("method") or (arm.method if arm else unit["arm"])


def resolve_hp(unit: dict) -> dict:
    """The hyperparameters of a unit, with the fixed arguments of its arm (as :func:`csfrbench.runner.execute`)."""
    w = suite.workload(unit["setting"], unit["workload"])
    arm = suite.ARMS.get(unit["arm"])
    hp = dict(unit["hp"]) if "hp" in unit else search.configurations(w, unit["arm"], unit["n"])[int(unit["trial"])]
    if arm is not None:
        conflict = {k for k, v in arm.fixed.items() if k in hp and hp[k] != v}
        if conflict:
            raise ValueError(f"arm {arm.name!r} has the fixed arguments {arm.fixed}; the unit gives {hp}")
        hp = {**arm.fixed, **hp}
    return hp


def stack_key(unit: dict) -> tuple | None:
    """Units with the same key can share a stack; ``None`` for a unit the stack does not train."""
    method = method_of(unit)
    if method not in STACKABLE:
        return None
    hp = resolve_hp(unit)
    shape = (int(hp.get("code_dim", 3)), float(hp.get("radius", 1.0))) if method == "lewi" else ()
    return (unit["setting"], unit["workload"], method, shape)


@dataclass
class Member:
    """One unit of a stack: the unit, its hyperparameters, its target and its model at the initialization."""
    unit: dict
    hp: dict
    target: Any
    method: Any
    model: torch.nn.Module
    seeds: dict
    data_key: tuple


class Order:
    """The per-network trainer's minibatch order for one network: the order of a ``DataLoader`` with ``shuffle=True``,
    ``drop_last=True`` and the unit's batch-order generator (each epoch draws the iterator's base seed, the epoch's
    permutation and the sampler's remainder permutation, in this order). ``take(k)`` returns the next ``k`` batches of
    indices, ``[k, 32]``. A test checks it against the trainer's loader."""

    def __init__(self, n: int, seed: int, batch: int = BATCH):
        self.n, self.batch = int(n), int(batch)
        self.g = torch.Generator().manual_seed(int(seed))
        self.perm, self.pos = None, self.n

    def take(self, k: int) -> torch.Tensor:
        out = torch.empty(k, self.batch, dtype=torch.long)
        i = 0
        while i < k:
            if self.pos + self.batch > self.n:
                torch.empty((), dtype=torch.int64).random_(generator=self.g)     # the iterator's base seed
                self.perm = torch.randperm(self.n, generator=self.g)             # the epoch's permutation
                torch.randperm(self.n, generator=self.g)                         # the sampler's remainder draw
                self.pos = 0
            m = min(k - i, (self.n - self.pos) // self.batch)
            out[i:i + m] = self.perm[self.pos:self.pos + m * self.batch].view(m, self.batch)
            self.pos += m * self.batch
            i += m
        return out


# ------------------------------------------------------------------ the gated weight


def _sqdist(recv: torch.Tensor, send: torch.Tensor) -> torch.Tensor:
    """``||recv_i - send_j||^2`` as one product of inner dimension ``r + 2`` (the form ``torch.cdist`` uses)."""
    r2 = (recv * recv).sum(-1, keepdim=True)
    s2 = (send * send).sum(-1, keepdim=True)
    a = torch.cat([r2, -2.0 * recv, torch.ones_like(r2)], dim=-1)
    b = torch.cat([torch.ones_like(s2), send, s2], dim=-1)
    return a @ b.transpose(-1, -2)


class GatedWeight(torch.autograd.Function):
    """``W * sigmoid(s) * 1[s > 0]`` with ``s = R^2 - ||e_recv - e_send||^2`` as one autograd function; the backward
    recomputes the scores instead of saving them and reduces the score gradient into the embedding gradients with two
    small products (the same gradients as the autograd chain)."""

    @staticmethod
    def forward(ctx, W, recv, send, R2):
        s = R2 - _sqdist(recv, send)
        g = torch.sigmoid(s)
        ctx.save_for_backward(W, recv, send)
        ctx.R2 = R2
        return torch.where(s > 0, W * g, torch.zeros((), dtype=W.dtype, device=W.device))

    @staticmethod
    def backward(ctx, gW_eff):
        W, recv, send = ctx.saved_tensors
        s = ctx.R2 - _sqdist(recv, send)
        g = torch.sigmoid(s)
        gWe = torch.where(s > 0, gW_eff, torch.zeros((), dtype=W.dtype, device=W.device))
        grad_W = gWe * g
        grad_s = gWe * W * g * (1.0 - g)
        rs = grad_s.sum(-1, keepdim=True)
        cs = grad_s.sum(-2).unsqueeze(-1)
        grad_recv = -2.0 * (rs * recv - grad_s @ send)
        grad_send = -2.0 * (cs * send - grad_s.transpose(-1, -2) @ recv)
        return grad_W, grad_recv, grad_send, None


def _compiler_available() -> bool:
    return any(shutil.which(c) for c in ("c++", "g++", "clang++"))


# ------------------------------------------------------------------ the stack


class Stack:
    """``B`` units of one workload and one method (LeWi or Dense), trained together.

    ``kernels`` selects LeWi's fused C++ gate kernels (``True``; PyTorch is used where they cannot be built), ``compile``
    the compilation of the rest of the step with ``torch.compile`` (``"auto"``: when a C++ compiler is present).
    """

    GATED = ("W0", "b0", "Wh", "bh", "Wo", "bo", "P0", "Ph", "Po")
    PLAIN = ("W0", "b0", "Wh", "bh", "Wo", "bo")

    def __init__(self, units: list[dict], *, device: str = "cpu", kernels: bool = True, compile: bool | str = "auto",
                 eval_chunk: int = 2048, data_cache: dict | None = None):
        if not units:
            raise ValueError("a stack needs at least one unit")
        keys = {stack_key(u) for u in units}
        if None in keys:
            raise ValueError(f"the stack trains only the methods {STACKABLE}")
        if len(keys) != 1:
            raise ValueError("a stack holds units of one setting, workload and method (and one LeWi geometry)")
        self.device = torch.device(device)
        self.eval_chunk = int(eval_chunk)
        u0 = units[0]
        self.workload = suite.workload(u0["setting"], u0["workload"])
        p = self.workload.profile
        self.W, self.L, self.max_steps, self.eval_every = p.width, p.hidden, p.max_steps, p.eval_every
        cache = {} if data_cache is None else data_cache
        self.members = [self._member(u, cache) for u in units]
        self.B = len(self.members)
        self.method_name = method_of(u0)
        self.gated = self.method_name == "lewi"
        self.PARAMS = self.GATED if self.gated else self.PLAIN
        m0 = self.members[0].model
        self.d = int(m0.hidden_layers[0].in_features)
        if self.gated:
            self.R2 = float(m0.hidden_layers[0].gate_offset)
            self.r = int(m0.gate_codes[0].shape[1])
            for m in self.members:
                for layer in m.model.linear_layers():
                    if float(layer.gate_offset) != self.R2 or float(layer.gate_scale) != 1.0:
                        raise ValueError("every layer of a stack carries the same radius and a gate scale of 1")
        self.params = {k: v.clone().to(self.device).requires_grad_(True) for k, v in zip(self.PARAMS, self._stacked())}
        self.init = {k: v.detach().clone() for k, v in self.params.items()}
        keys = sorted({m.data_key for m in self.members})
        pools = [cache[k][1] for k in keys]
        dev = self.device
        self.X_tr = torch.stack([d["X_train"] for d in pools]).to(dev)
        self.y_tr = torch.stack([d["y_train"] for d in pools]).to(dev)
        self.X_va = torch.stack([d["X_val"] for d in pools]).to(dev)
        self.y_va = torch.stack([d["y_val"] for d in pools]).to(dev)
        self.X_te = torch.stack([d["X_test"] for d in pools]).to(dev)
        self.f_te = torch.stack([d["f_test"] for d in pools]).to(dev)
        self.inst = torch.tensor([keys.index(m.data_key) for m in self.members], dtype=torch.long, device=dev)
        self.lr = torch.tensor([float(m.hp["lr"]) for m in self.members], dtype=torch.float32, device=dev)
        self.orders = [Order(int(m.unit["n"]), m.seeds["batch_order"]) for m in self.members]
        self.use_kernels = bool(kernels) and self.gated and self.device.type == "cpu"
        self.compiled = (_compiler_available() and self.device.type == "cpu") if compile == "auto" else bool(compile)

    # ---- construction
    def _member(self, unit: dict, cache: dict) -> Member:
        w = suite.workload(unit["setting"], unit["workload"])
        hp = resolve_hp(unit)
        key = (w.setting, w.name, int(unit["instance"]))
        if key not in cache:
            target = w.target(int(unit["instance"]))
            data = datasets.make_dataset(target, w.profile.n_max, pools=("train", "val", "test"))
            cache[key] = (target, data)
        target = cache[key][0]
        method, model, seeds = training.build_model(method_of(unit), target, width=w.profile.width,
                                                    hidden=w.profile.hidden, hp=hp, n=int(unit["n"]),
                                                    seed_replicate=int(unit.get("seed_replicate", 0)))
        if method.penalty(model, hp) is not None:
            raise ValueError(f"method {method.name!r} adds a penalty; the stack trains penalty-free methods only")
        return Member(unit, hp, target, method, model, seeds, key)

    def _stacked(self) -> list[torch.Tensor]:
        def stack(f):
            return torch.stack([f(m.model) for m in self.members])
        L = self.L
        out = [stack(lambda m: m.hidden_layers[0].weight.detach()), stack(lambda m: m.hidden_layers[0].bias.detach()),
               stack(lambda m: torch.stack([x.weight.detach() for x in m.hidden_layers[1:]]) if L > 1
                     else torch.zeros(0, self.W, self.W)),
               stack(lambda m: torch.stack([x.bias.detach() for x in m.hidden_layers[1:]]) if L > 1
                     else torch.zeros(0, self.W)),
               stack(lambda m: m.output_layer.weight.detach()), stack(lambda m: m.output_layer.bias.detach())]
        if self.gated:
            out += [stack(lambda m: m.gate_codes[0].detach()),
                    stack(lambda m: torch.stack([c.detach() for c in list(m.gate_codes)[1:L + 1]])),
                    stack(lambda m: m.gate_codes[L + 1].detach())]
        return out

    # ---- the forward pass
    def gates(self, p: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The gates of every layer: input ``[B, W, d]``, hidden ``[B, L-1, W, W]``, readout ``[B, 1, W]``."""
        R2 = self.R2
        g0 = torch.sigmoid(R2 - _sqdist(p["Ph"][:, 0], p["P0"]))
        gh = torch.sigmoid(R2 - _sqdist(p["Ph"][:, 1:], p["Ph"][:, :-1])) if self.L > 1 else p["Ph"][:, :0]
        go = torch.sigmoid(R2 - _sqdist(p["Po"], p["Ph"][:, -1]))
        return g0, gh, go

    def effective(self, p: dict):
        if self.gated:
            R2 = self.R2
            W0 = GatedWeight.apply(p["W0"], p["Ph"][:, 0], p["P0"], R2)
            Wo = GatedWeight.apply(p["Wo"], p["Po"], p["Ph"][:, -1], R2)
            Whs = GatedWeight.apply(p["Wh"], p["Ph"][:, 1:], p["Ph"][:, :-1], R2) if self.L > 1 else None
            return W0, [Whs[:, l] for l in range(self.L - 1)], Wo
        return p["W0"], [p["Wh"][:, l] for l in range(self.L - 1)], p["Wo"]

    def apply(self, x: torch.Tensor, p: dict, W0, Wh, Wo) -> torch.Tensor:
        h = torch.relu(torch.baddbmm(p["b0"].unsqueeze(1), x, W0.transpose(1, 2)))
        for l in range(self.L - 1):
            h = torch.relu(torch.baddbmm(p["bh"][:, l].unsqueeze(1), h, Wh[l].transpose(1, 2)))
        return torch.baddbmm(p["bo"].unsqueeze(1), h, Wo.transpose(1, 2)).squeeze(-1)

    def forward(self, x: torch.Tensor, p: dict | None = None) -> torch.Tensor:
        """``x [B, m, d]`` -> predictions ``[B, m]``."""
        p = self.params if p is None else p
        return self.apply(x, p, *self.effective(p))

    def _loss(self, x, y, *flat):
        p = dict(zip(self.PARAMS, flat))
        return (self.forward(x, p) - y).square().mean(dim=1)

    @torch.no_grad()
    def nmse(self, X: torch.Tensor, y: torch.Tensor, p: dict | None = None, *, mode: str = "val") -> torch.Tensor:
        """Per network on its instance's pool: ``"val"`` as :func:`csfrbench.metrics.nmse` (MSE over the variance of the
        targets), ``"test"`` as :func:`csfrbench.metrics.nmse_from_errors`; accumulated in float64."""
        p = self.params if p is None else p
        W0, Wh, Wo = self.effective(p)
        sse = torch.zeros(self.B, dtype=torch.float64, device=self.device)
        s1, s2 = torch.zeros_like(sse), torch.zeros_like(sse)
        m = int(X.shape[1])
        for a in range(0, m, self.eval_chunk):
            yb = y[self.inst, a:a + self.eval_chunk].to(torch.float64)
            pred = self.apply(X[self.inst, a:a + self.eval_chunk], p, W0, Wh, Wo).to(torch.float64)
            sse += (pred - yb).square().sum(dim=1)
            s1 += yb.sum(dim=1)
            s2 += yb.square().sum(dim=1)
        if mode == "val":
            mean = s1 / m
            var = torch.clamp(s2 / m - mean * mean, min=0.0)
            return (sse / m) / (var + 1e-12)
        return sse / (s2 - s1 * s1 / m)

    # ---- the optimizer: torch.optim.Adam's single-tensor arithmetic with one learning rate per network
    def adam_init(self) -> None:
        self.b1, self.b2 = training.ADAM_BETAS
        self.eps = training.ADAM_EPS
        self.m = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    @staticmethod
    def _adam_update(p, g, m, v, step_size, bc2_sqrt, b1: float, b2: float, eps: float) -> None:
        m.lerp_(g, 1.0 - b1)
        v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
        denom = (v.sqrt() / bc2_sqrt).add_(eps)
        p.add_((m / denom) * (-step_size))

    def _adam_all(self, step_size, bc2_sqrt, *tensors):
        k = len(self.PARAMS)
        ps, gs, ms, vs = tensors[:k], tensors[k:2 * k], tensors[2 * k:3 * k], tensors[3 * k:]
        gn2 = torch.zeros(self.B, dtype=torch.float32, device=ps[0].device)
        for p, g, m, v in zip(ps, gs, ms, vs):
            gn2 = gn2 + (g * g).flatten(1).sum(dim=1)
            self._adam_update(p, g, m, v, step_size.view((self.B,) + (1,) * (p.dim() - 1)), bc2_sqrt,
                              self.b1, self.b2, self.eps)
        return gn2

    def lr_at(self, step: int) -> torch.Tensor:
        """The learning rate at optimizer step ``step`` (1-based): the cosine schedule to 1% of the peak."""
        frac = training.LR_FINAL_FRACTION
        return self.lr * (frac + (1.0 - frac) * (1.0 + math.cos(math.pi * (step - 1) / self.max_steps)) / 2.0)

    def _autograd_step(self, x, y, lr_t):
        for p in self.params.values():
            p.grad = None
        flat = [self.params[k] for k in self.PARAMS]
        mse = self._loss_fn(x, y, *flat)
        mse.sum().backward()
        with torch.no_grad():
            self.t += 1
            bc1 = 1.0 - self.b1 ** self.t
            bc2_sqrt = torch.tensor((1.0 - self.b2 ** self.t) ** 0.5, dtype=torch.float32, device=self.device)
            gn2 = self._adam_fn(lr_t / bc1, bc2_sqrt, *flat, *[q.grad for q in flat],
                                *[self.m[k] for k in self.PARAMS], *[self.v[k] for k in self.PARAMS])
        return mse.detach(), gn2

    def _prepare(self) -> None:
        self.adam_init()
        self._loss_fn, self._adam_fn = self._loss, self._adam_all
        self.step_impl = "torch"
        if self.use_kernels:
            from .kernels import ops
            if ops() is not None:
                self._cpp = CppStep(self, compile=self.compiled)
                self.step_impl = "cpp"
        if self.step_impl == "torch" and self.compiled:
            self._loss_fn = torch.compile(self._loss, dynamic=False)
            self._adam_fn = torch.compile(self._adam_all, dynamic=False)

    def _step(self, x, y, lr_t):
        if self.step_impl == "cpp":
            return self._cpp(x, y, lr_t)
        return self._autograd_step(x, y, lr_t)

    # ---- the loop
    def train(self, *, stop_after_evals: int | None = None, log=None) -> dict:
        self._prepare()
        B, dev = self.B, self.device
        best_val = torch.full((B,), float("inf"), dtype=torch.float64, device=dev)
        best_step = torch.zeros(B, dtype=torch.long, device=dev)
        best = {k: v.detach().clone() for k, v in self.params.items()}
        first_bad = torch.zeros(B, dtype=torch.long, device=dev)
        bad_loss = torch.zeros(B, dtype=torch.bool, device=dev)
        grad_norm_max = torch.zeros(B, dtype=torch.float64, device=dev)
        interval = [torch.zeros(B, dtype=torch.float64, device=dev) for _ in range(3)]
        curves: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        chunk = max(1, min(self.eval_every, (64 << 20) // max(1, B * BATCH * 8)))
        if self.eval_every % chunk:
            chunk = math.gcd(self.eval_every, chunk)
        idx_block, at = None, 0
        start = time.perf_counter()
        evals = 0
        steps = self.max_steps
        for step in range(1, self.max_steps + 1):
            if idx_block is None or at >= idx_block.shape[0]:
                k = min(chunk, self.max_steps - step + 1)
                idx_block = torch.stack([o.take(k) for o in self.orders], dim=1).to(dev)     # [k, B, 32]
                at = 0
            idx = idx_block[at]
            at += 1
            inst = self.inst.unsqueeze(1).expand(B, BATCH)
            x, y = self.X_tr[inst, idx], self.y_tr[inst, idx]
            alive_before = first_bad == 0
            mse, gn2 = self._step(x, y, self.lr_at(step))
            with torch.no_grad():
                y64 = y.to(torch.float64)
                interval[0] += mse.to(torch.float64) * BATCH
                interval[1] += y64.sum(dim=1)
                interval[2] += y64.square().sum(dim=1)
                gn = gn2.to(torch.float64).sqrt()
                loss_bad = ~torch.isfinite(mse)
                newly = alive_before & (loss_bad | ~torch.isfinite(gn))
                if bool(newly.any()):
                    first_bad = torch.where(newly, torch.full_like(first_bad, step), first_bad)
                    bad_loss = bad_loss | (newly & loss_bad)
                    self._freeze(newly, best)
                grad_norm_max = torch.where(alive_before & ~newly, torch.maximum(grad_norm_max, gn), grad_norm_max)
            if step % self.eval_every == 0 or step == self.max_steps:
                alive = first_bad == 0
                val = self.nmse(self.X_va, self.y_va, mode="val")
                count = float(BATCH) * (self.eval_every if step % self.eval_every == 0 else step % self.eval_every)
                mean = interval[1] / count
                var = torch.clamp(interval[2] / count - mean * mean, min=0.0)
                train_nmse = (interval[0] / count) / (var + 1e-12)
                interval = [torch.zeros_like(interval[0]) for _ in range(3)]
                improved = (val < best_val) & alive & torch.isfinite(val)
                best_val = torch.where(improved, val, best_val)
                best_step = torch.where(improved, torch.full_like(best_step, step), best_step)
                for key, prm in self.params.items():
                    best[key] = torch.where(improved.view((B,) + (1,) * (prm.dim() - 1)), prm.detach(), best[key])
                curves.append((step, val.cpu().numpy(), train_nmse.cpu().numpy(), alive.cpu().numpy()))
                evals += 1
                if log is not None:
                    log(f"step {step}: median validation NMSE {float(val[alive].median()) if alive.any() else float('nan'):.4f}, "
                        f"diverged {int((~alive).sum())}, {time.perf_counter() - start:.0f} s")
                if stop_after_evals is not None and evals >= int(stop_after_evals):
                    steps = step
                    break
        self.wall = time.perf_counter() - start
        self.best, self.best_val, self.best_step = best, best_val, best_step
        self.first_bad, self.bad_loss, self.grad_norm_max, self.curves, self.steps = (first_bad, bad_loss, grad_norm_max,
                                                                                      curves, steps)
        return {"wall_s": self.wall, "steps": steps, "ms_per_step": 1000.0 * self.wall / max(1, steps),
                "step_impl": self.step_impl, "compiled": bool(self.compiled)}

    @torch.no_grad()
    def _freeze(self, which: torch.Tensor, best: dict) -> None:
        """A diverged network stops: its parameters return to its best checkpoint (or its initialization) and its
        learning rate to 0, so it stays finite and unchanged while the others train on."""
        for key, prm in self.params.items():
            mask = which.view((self.B,) + (1,) * (prm.dim() - 1))
            prm.copy_(torch.where(mask, best[key], prm))
            self.m[key].copy_(torch.where(mask, torch.zeros_like(prm), self.m[key]))
            self.v[key].copy_(torch.where(mask, torch.zeros_like(prm), self.v[key]))
        self.lr = torch.where(which, torch.zeros_like(self.lr), self.lr)

    # ---- the results
    @torch.no_grad()
    def state_dict(self, i: int, which: str = "best") -> dict:
        """Network ``i``'s parameters in its model's ``state_dict`` format (``which``: ``"best"`` or ``"init"``)."""
        p = self.best if which == "best" else self.init
        out = {}
        for l in range(self.L):
            W = p["W0"][i] if l == 0 else p["Wh"][i, l - 1]
            b = p["b0"][i] if l == 0 else p["bh"][i, l - 1]
            out[f"hidden_layers.{l}.weight"] = W.cpu().clone()
            out[f"hidden_layers.{l}.bias"] = b.cpu().clone()
        out["output_layer.weight"] = p["Wo"][i].cpu().clone()
        out["output_layer.bias"] = p["bo"][i].cpu().clone()
        if self.gated:
            g0, gh, go = self.gates({k: v[i:i + 1] for k, v in p.items()})
            for l in range(self.L):
                g = g0[0] if l == 0 else gh[0, l - 1]
                out[f"hidden_layers.{l}.gate_scale"] = torch.tensor(1.0)
                out[f"hidden_layers.{l}.gate_offset"] = torch.tensor(self.R2)
                out[f"hidden_layers.{l}.event_mask"] = (g > 0.5).float().cpu()
            out["output_layer.gate_scale"] = torch.tensor(1.0)
            out["output_layer.gate_offset"] = torch.tensor(self.R2)
            out["output_layer.event_mask"] = (go[0] > 0.5).float().cpu()
            out["gate_codes.0"] = p["P0"][i].cpu().clone()
            for j in range(self.L):
                out[f"gate_codes.{j + 1}"] = p["Ph"][i, j].cpu().clone()
            out[f"gate_codes.{self.L + 1}"] = p["Po"][i].cpu().clone()
        expected = set(self.members[i].model.state_dict())
        if set(out) != expected:
            raise RuntimeError(f"the exported state differs from the model's keys: {sorted(set(out) ^ expected)}")
        return out

    @torch.no_grad()
    def records(self, *, keep_states: bool = False) -> list[dict]:
        """One record per unit in the format of :func:`csfrbench.runner.execute`."""
        from .runner import unit_id
        test = self.nmse(self.X_te, self.f_te, self.best, mode="test").cpu().numpy()
        fb = self.first_bad.cpu().numpy()
        bv, bs = self.best_val.cpu().numpy(), self.best_step.cpu().numpy()
        gm, bl = self.grad_norm_max.cpu().numpy(), self.bad_loss.cpu().numpy()
        env = training.environment()
        out = []
        for i, m in enumerate(self.members):
            u, w = m.unit, self.workload
            diverged = bool(fb[i] > 0)
            steps_run = int(fb[i]) - 1 if diverged else int(self.steps)
            curve = [[int(s), float(v[i]), float(t[i]) if math.isfinite(t[i]) else None]
                     for s, v, t, a in self.curves if a[i]]
            rec = {"unit": unit_id(u), "setting": u["setting"], "workload": u["workload"], "arm": u["arm"],
                   "instance": int(u["instance"]), "n": int(u["n"]), "trial": int(u["trial"]),
                   "target_id": m.target.target_id, "fingerprint": m.target.fingerprint, "method": self.method_name,
                   "hp": dict(m.hp), "d": int(m.target.d), "width": self.W, "hidden": self.L,
                   "max_steps": self.max_steps, "eval_every": self.eval_every,
                   "dataset_seed": int(m.target.function_seed), "seed_replicate": int(u.get("seed_replicate", 0)),
                   "seeds": m.seeds,
                   "best_val_nmse": float(bv[i]) if math.isfinite(bv[i]) else None, "best_step": int(bs[i]),
                   "steps_run": steps_run, "val_curve": curve, "wallclock_s": self.wall / self.B,
                   "diverged": int(diverged), "grad_norm_max": float(gm[i]),
                   "complete": int(steps_run == self.max_steps),
                   "n_params": sum(q.numel() for q in m.model.parameters()),
                   "n_trainable_params": sum(q.numel() for q in m.model.parameters() if q.requires_grad),
                   "unit_wallclock_s": self.wall / self.B, "environment": env,
                   "engine": f"stack ({self.step_impl}{', compiled' if self.compiled else ''}, B = {self.B})"}
            if diverged:
                rec["diverged_step"] = int(fb[i])
                rec["diverged_reason"] = "nonfinite_loss" if bl[i] else "nonfinite_grad_norm"
            else:
                rec["test_nmse"] = float(test[i])
                model = m.model
                model.load_state_dict(self.state_dict(i))
                model.eval()
                rec["diagnostics"] = {k: (v.item() if torch.is_tensor(v) else v)
                                      for k, v in m.method.diagnostics(model).items()}
            if keep_states:
                rec["init_state"] = self.state_dict(i, "init")
                rec["best_state"] = self.state_dict(i, "best")
            out.append(rec)
        return out


class CppStep:
    """LeWi's step with the fused C++ gate kernels: the gates and effective weights of every layer group in one pass
    (``gate_forward``), the MLP's forward and backward as batched products, then one pass per group that forms the
    weight and score gradients, applies Adam to the weights and reduces the score gradient into the embedding gradients
    (``gate_backward_adam``); the biases and embeddings take Adam in PyTorch."""

    def __init__(self, stack: Stack, *, compile: bool = False):
        from .kernels import ops
        self.s = stack
        self.ops = ops()
        s = stack
        self.Wh = s.params["Wh"].detach().view(-1, s.W, s.W) if s.L > 1 else None
        self.mh = s.m["Wh"].view(-1, s.W, s.W) if s.L > 1 else None
        self.vh = s.v["Wh"].view(-1, s.W, s.W) if s.L > 1 else None
        self.mlp = torch.compile(self._mlp, dynamic=False) if compile else self._mlp
        self.tail = torch.compile(self._tail, dynamic=False) if compile else self._tail

    def _mlp(self, x, y, We0, Weh, Weo, b0, bh, bo):
        L = self.s.L
        acts, hs = [], [x]
        a = torch.baddbmm(b0.unsqueeze(1), x, We0.transpose(1, 2))
        acts.append(a)
        hs.append(torch.relu(a))
        for l in range(L - 1):
            a = torch.baddbmm(bh[:, l].unsqueeze(1), hs[-1], Weh[:, l].transpose(1, 2))
            acts.append(a)
            hs.append(torch.relu(a))
        yhat = torch.baddbmm(bo.unsqueeze(1), hs[-1], Weo.transpose(1, 2)).squeeze(-1)
        diff = yhat - y
        mse = diff.square().mean(dim=1)
        delta = (2.0 / x.shape[1]) * diff.unsqueeze(-1)
        gWeo = delta.transpose(1, 2) @ hs[-1]
        gbo = delta.sum(dim=1)
        da = (delta @ Weo) * (acts[-1] > 0)
        gWeh, gbh = [None] * (L - 1), [None] * (L - 1)
        for l in range(L - 2, -1, -1):
            gWeh[l] = da.transpose(1, 2) @ hs[l + 1]
            gbh[l] = da.sum(dim=1)
            da = (da @ Weh[:, l]) * (acts[l] > 0)
        gWe0 = da.transpose(1, 2) @ x
        gb0 = da.sum(dim=1)
        gWeh_t = torch.stack(gWeh, dim=1) if L > 1 else torch.zeros_like(Weh)
        gbh_t = torch.stack(gbh, dim=1) if L > 1 else torch.zeros_like(bh)
        return mse, gWe0, gWeh_t, gWeo, gb0, gbh_t, gbo

    def _tail(self, step_size, bc2_sqrt, gn_w, gb0, gbh, gbo, gsend0, grec0, grech, gsendh, greco, gsendo,
              b0, bh, bo, P0, Ph, Po, mb0, mbh, mbo, mP0, mPh, mPo, vb0, vbh, vbo, vP0, vPh, vPo):
        s = self.s
        B, L = s.B, s.L
        gPh = (torch.cat([grec0.unsqueeze(1), torch.zeros_like(Ph[:, 1:])], dim=1)
               + torch.cat([torch.zeros_like(Ph[:, :-1]), gsendo.unsqueeze(1)], dim=1))
        if L > 1:
            gPh = (gPh + torch.cat([torch.zeros_like(grech[:, :1]), grech], dim=1)
                   + torch.cat([gsendh, torch.zeros_like(gsendh[:, :1])], dim=1))
        gn2 = gn_w
        for p, g, m, v in ((b0, gb0, mb0, vb0), (bh, gbh, mbh, vbh), (bo, gbo, mbo, vbo), (P0, gsend0, mP0, vP0),
                           (Ph, gPh, mPh, vPh), (Po, greco, mPo, vPo)):
            gn2 = gn2 + (g * g).flatten(1).sum(dim=1)
            s._adam_update(p, g, m, v, step_size.view((B,) + (1,) * (p.dim() - 1)), bc2_sqrt, s.b1, s.b2, s.eps)
        return gn2

    @torch.no_grad()
    def __call__(self, x, y, lr_t):
        s = self.s
        s.t += 1
        bc1 = 1.0 - s.b1 ** s.t
        bc2_sqrt = (1.0 - s.b2 ** s.t) ** 0.5
        step_size = (lr_t / bc1).contiguous()
        P = s.params
        B, L, R2, ops = s.B, s.L, s.R2, self.ops
        W0, Wo = P["W0"].detach(), P["Wo"].detach()
        P0, Ph, Po = P["P0"].detach(), P["Ph"].detach(), P["Po"].detach()
        recv0, sendo = Ph[:, 0].contiguous(), Ph[:, -1].contiguous()
        g0, We0 = ops.gate_forward(W0, recv0, P0, R2)
        go, Weo = ops.gate_forward(Wo, Po, sendo, R2)
        if L > 1:
            recvh = Ph[:, 1:].reshape(-1, s.W, s.r).contiguous()
            sendh = Ph[:, :-1].reshape(-1, s.W, s.r).contiguous()
            gh, Weh_flat = ops.gate_forward(self.Wh, recvh, sendh, R2)
            Weh = Weh_flat.view(B, L - 1, s.W, s.W)
        else:
            Weh = P["Wh"].detach()
        mse, gWe0, gWeh, gWeo, gb0, gbh, gbo = self.mlp(x, y, We0, Weh, Weo, P["b0"].detach(), P["bh"].detach(),
                                                        P["bo"].detach())
        grec0, gsend0, gn0 = ops.gate_backward_adam(gWe0.contiguous(), W0, g0, recv0, P0, s.m["W0"], s.v["W0"],
                                                    step_size, bc2_sqrt, s.b1, s.b2, s.eps)
        greco, gsendo, gno = ops.gate_backward_adam(gWeo.contiguous(), Wo, go, Po, sendo, s.m["Wo"], s.v["Wo"],
                                                    step_size, bc2_sqrt, s.b1, s.b2, s.eps)
        gn_w = gn0 + gno
        if L > 1:
            grech, gsendh, gnh = ops.gate_backward_adam(gWeh.reshape(-1, s.W, s.W).contiguous(), self.Wh, gh, recvh,
                                                        sendh, self.mh, self.vh,
                                                        step_size.repeat_interleave(L - 1).contiguous(), bc2_sqrt,
                                                        s.b1, s.b2, s.eps)
            gn_w = gn_w + gnh.view(B, L - 1).sum(dim=1)
            grech, gsendh = grech.view(B, L - 1, s.W, s.r), gsendh.view(B, L - 1, s.W, s.r)
        else:
            grech = gsendh = Ph[:, :0]
        bc2 = torch.tensor(bc2_sqrt, dtype=torch.float32)
        gn2 = self.tail(step_size, bc2, gn_w, gb0, gbh, gbo, gsend0, grec0, grech, gsendh, greco, gsendo,
                        P["b0"].detach(), P["bh"].detach(), P["bo"].detach(), P0, Ph, Po,
                        s.m["b0"], s.m["bh"], s.m["bo"], s.m["P0"], s.m["Ph"], s.m["Po"],
                        s.v["b0"], s.v["bh"], s.v["bo"], s.v["P0"], s.v["Ph"], s.v["Po"])
        return mse, gn2


# ------------------------------------------------------------------ campaigns


def default_stack_size(workload: suite.Workload) -> int:
    """The stack size that gave the lowest cost per network in the timing study: 128 at 32x4, 32 at 64x8."""
    return 128 if workload.profile.width <= 32 else 32


def stacks_of(units: list[dict], stack_size: int | None = None) -> tuple[list[list[dict]], list[dict]]:
    """Split units into stacks (one setting, workload and method each) and the units the stack does not train."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    rest = []
    for u in units:
        key = stack_key(u)
        if key is None:
            rest.append(u)
        else:
            groups[key].append(u)
    stacks = []
    for key, members in groups.items():
        members.sort(key=lambda u: (u["instance"], u["n"], u["trial"]))
        size = int(stack_size or default_stack_size(suite.workload(key[0], key[1])))
        stacks += [members[i:i + size] for i in range(0, len(members), size)]
    return stacks, rest


def train_stack(units: list[dict], *, keep_states: bool = False, stop_after_evals: int | None = None,
                kernels: bool = True, compile: bool | str = "auto", device: str = "cpu") -> list[dict]:
    """Train one stack and return its records."""
    s = Stack(units, kernels=kernels, compile=compile, device=device)
    try:
        s.train(stop_after_evals=stop_after_evals)
    except Exception as exc:  # noqa: BLE001  a compiled step that fails to build: train the stack again without
        if not s.compiled:
            raise
        warnings.warn(f"the compiled step failed ({type(exc).__name__}: {exc}); training the stack without compilation",
                      stacklevel=2)
        s = Stack(units, kernels=kernels, compile=False, device=device)
        s.train(stop_after_evals=stop_after_evals)
    return s.records(keep_states=keep_states)


def _threads_from_env() -> None:
    torch.set_num_threads(int(os.environ.get("CSFRBENCH_THREADS", "1")))
