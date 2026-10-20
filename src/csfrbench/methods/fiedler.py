"""Fiedler regularization (Tam & Dunson, ICML 2020) on the shared MLP
backbone (per-workload profile; see configs/profile.yaml).

Source: PMLR v119 tam20a (arXiv:2003.00992), Definition 3.2.1 and
Algorithm 1 ("Variational Fiedler Regularization with SGD"); an audited
earlier implementation of our own (agg='sum', the paper-faithful aggregation)
served as a cross-check.

Paper mechanics, reproduced exactly:
  * the network's underlying graph takes one node per neuron and edge
    weights |W_ij| from every affine weight matrix; biases are excluded
    (the paper treats the weights as "the main parameters of interest");
  * the penalty is delta * lambda_2(|W|), the Fiedler value of the
    unnormalized graph Laplacian L = D - A (Definition 3.2.1);
  * Algorithm 1 makes this scalable: keep a cached test vector u
    (initialized to the exact Fiedler vector of the initial network),
    penalize the quadratic form
        u^T L u = sum_{(i,j) in E} |W_ij| (u_i - u_j)^2
    every step - an upper bound on lambda_2 that autograd differentiates
    as a weighted L1 (u detached, d|W|/dW = sign(W)) - and recompute u
    from the current |W| Laplacian every T iterations ("say once every
    100 iterations"), which is the source's stated cadence;
  * the eigendecomposition runs on the exact dense Laplacian; the graph
    has one node per unit (161 at 32x4 up to 1,057 at the 128x8 pilot
    shape), so a refresh costs milliseconds every 100 steps.

The cached vector is a persistent model buffer: the between-refresh
penalty depends on it, so checkpoint restoration must reproduce it.
"""
from __future__ import annotations

import torch

from ..registry import method
from .base import Method, readout_structural
from .nns import DenseMLP

REFRESH_T = 100    # Fiedler-vector refresh period, the paper's cadence


def _boundaries(model) -> list[int]:
    linears = list(model.linear_layers())
    dims = [linears[0].weight.shape[1]] + [l.weight.shape[0] for l in linears]
    offsets = [0]
    for d in dims:
        offsets.append(offsets[-1] + d)
    return offsets


@torch.no_grad()
def _fiedler_vector(model) -> torch.Tensor:
    """Exact Fiedler vector of the |W| graph's unnormalized Laplacian."""
    linears = list(model.linear_layers())
    offsets = _boundaries(model)
    n = offsets[-1]
    adj = torch.zeros(n, n)
    for i, lin in enumerate(linears):
        w = lin.weight.detach().abs().cpu()
        r0, r1 = offsets[i + 1], offsets[i + 2]
        c0, c1 = offsets[i], offsets[i + 1]
        adj[r0:r1, c0:c1] = w
        adj[c0:c1, r0:r1] = w.t()
    lap = -adj
    lap.diagonal().copy_(adj.sum(dim=1))
    # eigh returns ascending eigenvalues; index 0 is the trivial constant
    # eigenvector, index 1 the Fiedler vector (unit norm by convention).
    _, vecs = torch.linalg.eigh(lap)
    return vecs[:, 1].contiguous()


@method("fiedler")
class Fiedler(Method):
    """Variational Fiedler-value penalty with a periodically refreshed u."""

    hp_keys = frozenset({"delta", "readout"})

    def build(self, cfg, target):
        model = DenseMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
            readout_structural=readout_structural(cfg.get("hp")),
        )
        model.register_buffer("fiedler_u", _fiedler_vector(model),
                              persistent=True)
        return model

    def penalty(self, model, hp):
        delta = float(hp["delta"])
        if delta < 0:
            raise ValueError(f"delta must be nonnegative, received {delta}")
        offsets = _boundaries(model)
        u = model.fiedler_u
        device = next(model.parameters()).device
        total = torch.zeros((), device=device)
        # the readout term follows the readout rule (v1.1 structural: every
        # transition; v1.0 plain: the hidden transitions); the Fiedler vector
        # itself is always that of the whole |W| graph
        for i, lin in enumerate(model.structural_layers()):
            u_in = u[offsets[i]:offsets[i + 1]].to(device)
            u_out = u[offsets[i + 1]:offsets[i + 2]].to(device)
            diff_sq = (u_out.unsqueeze(1) - u_in.unsqueeze(0)) ** 2
            total = total + (lin.weight.abs() * diff_sq).sum()
        return delta * total

    def after_optimizer_step(self, model, optimizer, *, step, total_steps):
        if step % REFRESH_T:
            return
        model.fiedler_u.copy_(_fiedler_vector(model))

    def extra_diagnostics(self, model):
        # The quadratic form at the cached u over the structural transitions:
        # the portion of the variational lambda_2 upper bound the run was
        # actually penalizing (over structural_layers(): the readout term is
        # excluded under the plain readout of CSFRBench v1.0 and included under
        # v1.1's structural readout; test_a3_fiedler_bound_is_tight_at_refresh
        # states the exact decomposition).
        offsets = _boundaries(model)
        u = model.fiedler_u
        total = 0.0
        for i, lin in enumerate(model.structural_layers()):
            u_in = u[offsets[i]:offsets[i + 1]]
            u_out = u[offsets[i + 1]:offsets[i + 2]]
            diff_sq = (u_out.unsqueeze(1) - u_in.unsqueeze(0)) ** 2
            total += float((lin.weight.detach().cpu().abs()
                            * diff_sq).sum())
        return {"fiedler_bound": total}
