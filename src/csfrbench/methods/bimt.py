"""BIMT - Brain-Inspired Modular Training (Liu, Gan & Tegmark, 2023).

Sources: arXiv:2305.08746 and the official implementation
github.com/KindXiaoming/BIMT (BioMLP / BioLinear in the released
notebooks), plus an audited earlier implementation of our own. Where the
two disagree the official code wins.

Official mechanics, reproduced exactly:
  * every neuron layer places its neurons at fixed 1-D coordinates
    linspace(1/(2n), 1 - 1/(2n), n) - the official BioLinear coordinates
    in every notebook (the paper's Eq. 1 writes edge-aligned Aj/n; the
    released code wins per the declared hierarchy), normalizing each
    layer to [0, 1] regardless of width;
  * connection cost (get_cc with weight_factor = 1, sum reduction):
        cc = sum_layers sum |w_ij| * (|x_out_i - x_in_j| + l0)
             + sum |b| * l0,           l0 = 0.1
    and the training loss adds lam * cc every step;
  * every S_swap = 200 steps the model relocates its top_k = 5 most
    important neurons per neuron layer (importance = summed |w| of
    incident edges, official get_top_id): each is moved to the position
    whose swap minimizes the connection cost, sequentially, input and
    output layers included - applied only when the swap strictly helps,
    the paper's own rule (the notebooks apply the argmin even on exact
    cost ties);
  * input/output relocations are function-preserving through permutation
    state applied in forward (the official in_perm / out_perm); hidden
    relocations permute rows/columns and biases directly.

The constants (l0 = 0.1, top_k = 5, S_swap = 200, weight_factor = 1,
sum-reduced cost) are the symbolic_formulas_3.1 notebook's - the released
regression experiment, the closest task match to this benchmark - NOT a
universal official set: the other notebooks use l0 in {0.2, 0.5}, top_k
in {10, 20, 30}, swap periods {200, 500, disabled}, weight_factor 2.0,
and a mean-reduced cost. The l0 choice matters (the distance:offset
ratio is not absorbed by the searched lam; l0 = 0.5 measured clearly
worse on the benchmark, audit v05-suite) and the sensitivity is recorded
in the method card.

Differences from the official notebooks, disclosed in the method card:
the notebooks hand-schedule lam (x10 mid-run, with different factors per
notebook) and the symbolic notebook delays the bias term to the last
quarter - per-experiment presentation choices - whereas the benchmark
searches one constant lam and applies the full cost throughout; the
activation is the envelope's ReLU (the notebooks use SiLU); the
relocation argmin evaluates the closed-form cost change of a swap, which
equals the official whole-network brute-force difference (unit-tested,
and confirmed end-to-end against a verbatim BioMLP port in the audit)
because a swap only changes the two neurons' incident edges.
"""
from __future__ import annotations

import torch

from ..registry import method
from .base import Method, readout_structural
from .nns import DenseMLP

L0 = 0.1          # inter-layer cost offset, symbolic_formulas_3.1 self.l0
TOP_K = 5         # neurons relocated per layer, symbolic_formulas_3.1 top_k
S_SWAP = 200      # relocation period in steps, symbolic_formulas_3.1 swap_log


def _coords(n: int, device=None) -> torch.Tensor:
    """Official BioLinear coordinates: n evenly spaced points in [0, 1]."""
    return torch.linspace(1.0 / (2 * n), 1.0 - 1.0 / (2 * n), n,
                          device=device)


def attach_permutations(model, in_dim: int) -> None:
    """Function-preserving input/output relocation state, the official
    in_perm / out_perm, applied in forward. Persistent: the permutation
    is part of the trained function."""
    model.register_buffer("bimt_in_perm",
                          torch.arange(in_dim), persistent=True)
    model.register_buffer("bimt_out_perm",
                          torch.arange(model.out_dim), persistent=True)
    original_forward = type(model).forward

    def forward_with_perms(x):
        x = x[:, model.bimt_in_perm]
        y = original_forward(model, x)
        if model.out_dim > 1:
            inv = torch.empty_like(model.bimt_out_perm)
            inv[model.bimt_out_perm] = torch.arange(
                model.out_dim, device=x.device)
            y = y[:, inv]
        return y

    model.forward = forward_with_perms


@method("bimt")
class BIMT(Method):
    """BIMT: distance-weighted L1 wiring cost with periodic neuron swaps."""

    hp_keys = frozenset({"lam", "readout"})

    def build(self, cfg, target):
        model = DenseMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
            readout_structural=readout_structural(cfg.get("hp")),
        )
        attach_permutations(model, target.d)
        return model

    def penalty(self, model, hp):
        lam = float(hp["lam"])
        if lam < 0:
            raise ValueError(f"lam must be nonnegative, received {lam}")
        cost = torch.zeros((), device=next(model.parameters()).device)
        for lin in model.structural_layers():    # the readout's wiring cost follows the readout rule (v1.1 structural / v1.0 plain)
            x_in = _coords(lin.weight.shape[1], lin.weight.device)
            x_out = _coords(lin.weight.shape[0], lin.weight.device)
            dist = (x_out.unsqueeze(1) - x_in.unsqueeze(0)).abs()
            cost = cost + (lin.weight.abs() * (dist + L0)).sum()
            cost = cost + L0 * lin.bias.abs().sum()
        return lam * cost

    @torch.no_grad()
    def after_optimizer_step(self, model, optimizer, *, step, total_steps):
        if step == 0 or step % S_SWAP:
            return
        linears = list(model.linear_layers())
        L = len(linears)
        for i in range(L + 1):
            w_in = linears[i - 1].weight if i >= 1 else None
            w_out = linears[i].weight if i < L else None
            n = w_in.shape[0] if w_in is not None else w_out.shape[1]
            if n < 2:
                continue
            # Official get_top_id: importance = summed |w| of incident edges.
            score = torch.zeros(n, device=(w_in if w_in is not None
                                           else w_out).device)
            if w_in is not None:
                score += w_in.abs().sum(dim=1)
            if w_out is not None:
                score += w_out.abs().sum(dim=0)
            top = torch.topk(score, min(TOP_K, n)).indices.tolist()
            for j in top:
                # Cost of each neuron at each slot, from current weights
                # (sequential, as the official relocate_ij re-evaluates
                # after every applied swap).
                M = self._slot_cost(linears, i, n)
                diag = M.diagonal()
                delta = (M[j, :] + M[:, j]) - (diag[j] + diag)
                delta[j] = 0.0
                k = int(torch.argmin(delta))
                if k != j and float(delta[k]) < 0.0:
                    self._swap(model, linears, i, j, k)

    @staticmethod
    def _slot_cost(linears, i, n) -> torch.Tensor:
        """M[neuron, slot] = distance-weighted incident |w| cost."""
        L = len(linears)
        w_in = linears[i - 1].weight if i >= 1 else None
        w_out = linears[i].weight if i < L else None
        device = (w_in if w_in is not None else w_out).device
        x_self = _coords(n, device)
        M = torch.zeros(n, n, device=device)
        if w_in is not None:
            x_prev = _coords(w_in.shape[1], device)
            dist = (x_self.unsqueeze(0) - x_prev.unsqueeze(1)).abs()  # (in, slot)
            M += w_in.abs() @ dist                    # (neuron, slot)
        if w_out is not None:
            x_next = _coords(w_out.shape[0], device)
            dist = (x_self.unsqueeze(0) - x_next.unsqueeze(1)).abs()  # (next, slot)
            M += w_out.abs().t() @ dist               # (neuron, slot)
        return M

    @staticmethod
    @torch.no_grad()
    def _swap(model, linears, i, j, k):
        """Official BioMLP.swap: exchange neurons j and k of neuron layer i."""
        L = len(linears)
        if i == 0:
            w = linears[0].weight
            w[:, [j, k]] = w[:, [k, j]].clone()
            p = model.bimt_in_perm
            p[[j, k]] = p[[k, j]].clone()
        elif i == L:
            w = linears[-1].weight
            b = linears[-1].bias
            w[[j, k]] = w[[k, j]].clone()
            b[[j, k]] = b[[k, j]].clone()
            p = model.bimt_out_perm
            p[[j, k]] = p[[k, j]].clone()
        else:
            w_in, w_out = linears[i - 1], linears[i]
            w_in.weight[[j, k]] = w_in.weight[[k, j]].clone()
            w_in.bias[[j, k]] = w_in.bias[[k, j]].clone()
            w_out.weight[:, [j, k]] = w_out.weight[:, [k, j]].clone()

    def effective_weights(self, model):
        """Weight matrices with the io permutations folded back in.

        The first matrix's columns are re-indexed to raw input order and
        the last matrix's rows to raw output order, so downstream
        weight-graph diagnostics see the function's actual wiring.
        """
        mats = [lin.effective_weight() for lin in model.linear_layers()]
        inv_in = torch.empty_like(model.bimt_in_perm)
        inv_in[model.bimt_in_perm] = torch.arange(
            len(model.bimt_in_perm), device=model.bimt_in_perm.device)
        mats[0] = mats[0][:, inv_in]
        if model.out_dim > 1:
            # Rows re-indexed by the INVERSE permutation, mirroring the
            # forward pass (audit finding B5, repaired 2026-08-28).
            inv_out = torch.empty_like(model.bimt_out_perm)
            inv_out[model.bimt_out_perm] = torch.arange(
                len(model.bimt_out_perm), device=model.bimt_out_perm.device)
            mats[-1] = mats[-1][inv_out]
        return tuple(mats)

    def extra_diagnostics(self, model):
        moved = int((model.bimt_in_perm !=
                     torch.arange(len(model.bimt_in_perm),
                                  device=model.bimt_in_perm.device)).sum())
        return {"inputs_relocated": float(moved)}
