import math

import torch
import torch.nn as nn

from ..registry import method
from .base import Method
from .nns import DenseMLP, boundary_widths

EPS = 1e-12


class AxonDendriteGeometry(nn.Module):
    """One axon and ``n_dend`` dendrite positions per neuron, per layer boundary.

    Boundary ``b`` is the neuron set feeding linear ``b``. Senders own an axon,
    receivers own dendrites, so the input boundary has no dendrites and the
    output boundary has no axon.
    """

    def __init__(self, widths, *, pos_dim: int, n_dend: int, init_scale: float):
        super().__init__()
        self.pos_dim = int(pos_dim)
        self.n_dend = int(n_dend)
        init_scale = float(init_scale)
        if self.pos_dim <= 0:
            raise ValueError(f"pos_dim must be positive; received {pos_dim}.")
        if self.n_dend <= 0:
            raise ValueError(f"n_dend must be positive; received {n_dend}.")
        if not init_scale > 0:
            raise ValueError(f"init_scale must be positive; received {init_scale}.")

        widths = tuple(int(w) for w in widths)
        last = len(widths) - 1
        self.axon = nn.ParameterDict({
            f"b{b}": nn.Parameter(torch.randn(w, self.pos_dim) * init_scale)
            for b, w in enumerate(widths) if b < last
        })
        self.dend = nn.ParameterDict({
            f"b{b}": nn.Parameter(torch.randn(w, self.n_dend, self.pos_dim) * init_scale)
            for b, w in enumerate(widths) if b >= 1
        })
        # Boundaries that both send and receive; the consistency and density
        # priors act only there.
        self.hidden_boundaries = tuple(range(1, last))

    def edge_cost(self, boundary: int, *, squared: bool, min_tau: float) -> torch.Tensor:
        """Communication cost ``c(j -> i)`` for every (receiver, sender) pair."""
        a_send = self.axon[f"b{boundary}"]
        d_recv = self.dend[f"b{boundary + 1}"]
        difference = a_send[None, :, None, :] - d_recv[:, None, :, :]
        squared_distance = difference.square().sum(dim=-1)
        per_dendrite = (
            squared_distance if squared else squared_distance.clamp_min(EPS).sqrt()
        )
        if min_tau > 0.0:
            return -min_tau * torch.logsumexp(-per_dendrite / min_tau, dim=-1)
        return per_dendrite.min(dim=-1).values

    def chamfer(self, boundary: int, *, mode: str) -> torch.Tensor:
        """Symmetrized dendrite-set discrepancy between every neuron pair."""
        D = self.dend[f"b{boundary}"]
        difference = D[:, None, :, None, :] - D[None, :, None, :, :]
        distance = difference.square().sum(dim=-1).clamp_min(EPS).sqrt()
        nearest_forward = distance.min(dim=3).values
        nearest_backward = distance.min(dim=2).values
        if mode == "mean":
            return nearest_forward.mean(dim=2) + nearest_backward.mean(dim=2)
        return nearest_forward.max(dim=2).values + nearest_backward.max(dim=2).values


@method("axdn")
class AxonDendrite(Method):
    """Asymmetric axon/dendrite communication-cost wiring prior.

    Every neuron carries one axon position a_i in R^p (what it emits) and M
    dendrite positions D_i (its input ports). The directed edge sender j ->
    receiver i is priced by the distance from j's axon to i's nearest dendrite,

        c(j -> i) = min_m ||a_j - D_{i,m}||,

    and the wiring term prices the realized weights by that cost. The asymmetry
    is the point: one axon can sit near unboundedly many receivers, so fan-out
    is free, while a receiver pays a minimum over only M dendrites, so reading
    from an (M+1)-th distinct axon cluster costs. At M = 1 the family degenerates
    to a plain symmetric Euclidean wiring prior.

    Two priors act on the boundaries that both send and receive:

        consistency (``w_consist``) -- neurons claiming the same variable (close
          axons) must read from the same place (similar dendrite sets), gated by
          exp(-||da||/gamma);
        density (``w_dens``) -- Gaussian repulsion among axons.

    The density term is the anti-collapse term: without it the wiring cost is a
    pure alignment objective whose global minimum includes the degenerate
    "all positions coincide" set.

    Positions never enter the forward pass, so the only forces on the geometry
    are the penalty gradient and the optimizer.
    """

    def build(self, cfg, target):
        hp = cfg.get("hp") or {}
        model = DenseMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
        )
        n_dend = hp.get("n_dend", 5)
        if n_dend == "auto":
            raise ValueError(
                "axdn n_dend='auto' reads the target's true maximum fan-in, which "
                "violates the information firewall: competitive arms may condition "
                "only on the public workload interface (d, n, backbone, data). "
                "Set n_dend to a target-independent value or search it."
            )
        model.geometry = AxonDendriteGeometry(
            boundary_widths(model),
            pos_dim=int(hp.get("pos_dim", 2)),
            n_dend=int(round(float(n_dend))),
            init_scale=float(hp.get("init_scale", 0.5)),
        )
        return model

    def penalty(self, model, hp):
        lam = float(hp["lam"])
        pen = str(hp.get("pen", "abs"))
        w_consist = float(hp.get("w_consist", 0.0))
        w_dens = float(hp.get("w_dens", 0.0))
        gamma = float(hp.get("consist_gamma", 1.0))
        sigma = float(hp.get("sigma", 1.0))
        chamfer_mode = str(hp.get("chamfer", "max"))
        min_tau = float(hp.get("min_tau", 0.0))

        if not math.isfinite(lam) or lam < 0:
            raise ValueError(f"lam must be finite and nonnegative; received {lam}.")
        if pen not in ("abs", "sq"):
            raise ValueError(f"pen must be 'abs' or 'sq'; received {pen!r}.")
        if chamfer_mode not in ("max", "mean"):
            raise ValueError(f"chamfer must be 'max' or 'mean'; received {chamfer_mode!r}.")
        if not sigma > 0 or not gamma > 0:
            raise ValueError("sigma and consist_gamma must be positive.")

        geometry = model.geometry
        layers = model.linear_layers()
        squared = pen == "sq"

        wiring = model.output_layer.weight.new_zeros(())
        for boundary, layer in enumerate(layers):
            cost = geometry.edge_cost(boundary, squared=squared, min_tau=min_tau)
            weight = layer.weight.square() if squared else layer.weight.abs()
            wiring = wiring + (weight * cost).mean()
        total = wiring / len(layers)

        if geometry.hidden_boundaries and (w_consist != 0.0 or w_dens != 0.0):
            consistency = total.new_zeros(())
            density = total.new_zeros(())
            for boundary in geometry.hidden_boundaries:
                axons = geometry.axon[f"b{boundary}"]
                count = axons.shape[0]
                if count < 2:
                    continue
                off_diagonal = count * (count - 1)
                axon_sq = torch.cdist(axons, axons).square()
                if w_consist != 0.0:
                    gate = torch.exp(-axon_sq.clamp_min(EPS).sqrt() / gamma)
                    discrepancy = geometry.chamfer(boundary, mode=chamfer_mode)
                    consistency = consistency + (gate * discrepancy).sum() / off_diagonal
                if w_dens != 0.0:
                    kernel = torch.exp(-axon_sq / (2.0 * sigma * sigma))
                    density = density + (kernel.sum() - count) / off_diagonal
            n_hidden = max(len(geometry.hidden_boundaries), 1)
            total = total + w_consist * (consistency / n_hidden) + w_dens * (density / n_hidden)

        return lam * total

    def extra_diagnostics(self, model):
        """Collapse and fan-in diagnostics in the method's own geometry."""
        geometry = model.geometry
        sigma = 1.0
        with torch.no_grad():
            spreads, densities = [], []
            for boundary in geometry.hidden_boundaries:
                axons = geometry.axon[f"b{boundary}"]
                count = axons.shape[0]
                if count < 2:
                    continue
                pairwise = torch.cdist(axons, axons)
                spreads.append(pairwise.sum() / (count * (count - 1)))
                kernel = torch.exp(-pairwise.square() / (2.0 * sigma * sigma))
                densities.append((kernel.sum() - count) / (count * (count - 1)))

            dendrite_spreads = []
            for key, D in geometry.dend.items():
                if D.shape[0] >= 1 and D.shape[1] >= 2:
                    M = D.shape[1]
                    dendrite_spreads.append(
                        torch.cdist(D, D).sum() / (D.shape[0] * M * (M - 1))
                    )

            costs = [
                geometry.edge_cost(boundary, squared=False, min_tau=0.0).mean()
                for boundary in range(len(geometry.axon))
            ]

        out = {}
        if spreads:
            out["axdn_axon_spread"] = float(torch.stack(spreads).mean())
            out["axdn_dens"] = float(torch.stack(densities).mean())
        if dendrite_spreads:
            out["axdn_dend_spread"] = float(torch.stack(dendrite_spreads).mean())
        if costs:
            out["axdn_mean_cost"] = float(torch.stack(costs).mean())
        return out
