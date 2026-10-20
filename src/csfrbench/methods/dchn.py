import math

import torch
import torch.nn as nn

from ..registry import method
from .base import Method
from .nns import DenseMLP, boundary_widths

CONTAINMENT_EPS = 1e-8
UNIFORMITY_TEMPERATURE = 5.0


def bidirectional_containment(recv: torch.Tensor, send: torch.Tensor) -> torch.Tensor:
    """Containment distance between receiver and sender membership codes.

    A code fully contained in the other is distance 0; only the uncovered
    remainder counts, normalized by the source mass. ``min(fwd, bwd)``
    symmetrizes. ``recv`` is ``(n_out, K)``, ``send`` is ``(n_in, K)``.
    """
    a = recv.unsqueeze(1)
    b = send.unsqueeze(0)
    forward = torch.relu(b - a).sum(dim=2) / (send.sum(dim=1).unsqueeze(0) + CONTAINMENT_EPS)
    backward = torch.relu(a - b).sum(dim=2) / (recv.sum(dim=1).unsqueeze(1) + CONTAINMENT_EPS)
    return torch.minimum(forward, backward)


class DualChannelGeometry(nn.Module):
    """An in-channel and an out-channel membership code per neuron, per boundary.

    Pairing the receivers' in-channel with the senders' out-channel lets the
    prior express an arbitrary DAG rather than a block-diagonal partition.
    """

    def __init__(self, widths, *, num_modules: int, spike_init: bool):
        super().__init__()
        self.num_modules = int(num_modules)
        if self.num_modules <= 0:
            raise ValueError(f"K must be positive; received {num_modules}.")

        widths = tuple(int(w) for w in widths)
        self.in_channel = nn.ParameterDict({
            f"b{b}": nn.Parameter(self._draw(w, spike_init)) for b, w in enumerate(widths)
        })
        self.out_channel = nn.ParameterDict({
            f"b{b}": nn.Parameter(self._draw(w, spike_init)) for b, w in enumerate(widths)
        })
        self.hidden_boundaries = tuple(range(1, len(widths) - 1))

    def _draw(self, width: int, spike_init: bool) -> torch.Tensor:
        """Diverse multihot logits: a low baseline plus one preferred module."""
        z = torch.randn(width, self.num_modules) * 0.1 - 2.0
        if width > 0 and spike_init:
            preferred = torch.randint(0, self.num_modules, (width,))
            z[torch.arange(width), preferred] = 2.0
        return z

    def memberships(self, boundary: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sender out-channel and receiver in-channel memberships for one linear."""
        send = torch.sigmoid(self.out_channel[f"b{boundary}"])
        recv = torch.sigmoid(self.in_channel[f"b{boundary + 1}"])
        return recv, send


def pair_distance(recv: torch.Tensor, send: torch.Tensor, *, dist: str, pen: str):
    """Distance between receiver and sender membership codes."""
    if dist == "cont":
        return bidirectional_containment(recv, send)
    difference = recv.unsqueeze(1) - send.unsqueeze(0)
    if dist == "manh":
        return difference.abs().sum(dim=-1)
    squared = difference.square().sum(dim=-1)
    return squared if pen == "sq" else squared.clamp_min(1e-12).sqrt()


@method("dchn")
class Separator(Method):
    """Separator-aware dual-channel membership prior.

    Every layer boundary owns two multihot codes -- an in-channel and an
    out-channel, sigma = sigmoid(z) in [0, 1]^K. For each linear W the receivers'
    in-channel is paired with the senders' out-channel, and the wiring term is

        R = sum_l sum_ij d(sigma_in[i], sigma_out[j]) * pen(W_ij),

    so an edge between neurons whose codes do not contain one another is
    expensive. Optional terms:

        ent -- L1 on marginal membership, keeping modules small;
        unif -- log-mean-exp anti-collapse on the same pair grid. The wiring cost
          alone is a pure alignment objective whose minimum includes the
          degenerate d == 0 set (order collapse: all codes elementwise
          comparable). This term makes that set its own maximum;
        w_dens -- within-boundary Gaussian repulsion on the codes, the
          axon/dendrite density term transferred to membership space.
    """

    def build(self, cfg, target):
        hp = cfg.get("hp") or {}
        model = DenseMLP(
            input_dim=target.d,
            width=cfg["width"],
            num_hidden_layers=cfg["hidden"],
        )
        model.geometry = DualChannelGeometry(
            boundary_widths(model),
            num_modules=int(hp.get("K", 4)),
            spike_init=str(hp.get("pi_init", "spike")) != "uniform",
        )
        return model

    def penalty(self, model, hp):
        lam = float(hp["lam"])
        dist = str(hp.get("dist", "cont"))
        pen = str(hp.get("pen", "abs"))
        sparsity = float(hp.get("ent", 0.0))
        uniformity = float(hp.get("unif", 0.0))
        w_dens = float(hp.get("w_dens", 0.0))
        sigma = float(hp.get("sigma", 1.0))

        if not math.isfinite(lam) or lam < 0:
            raise ValueError(f"lam must be finite and nonnegative; received {lam}.")
        if dist not in ("cont", "euc", "manh"):
            raise ValueError(f"dist must be 'cont', 'euc' or 'manh'; received {dist!r}.")
        if pen not in ("abs", "sq"):
            raise ValueError(f"pen must be 'abs' or 'sq'; received {pen!r}.")
        if not sigma > 0:
            raise ValueError("sigma must be positive.")

        geometry = model.geometry
        layers = model.linear_layers()

        wiring = model.output_layer.weight.new_zeros(())
        uniformity_term = wiring.new_zeros(())
        for boundary, layer in enumerate(layers):
            recv, send = geometry.memberships(boundary)
            distance = pair_distance(recv, send, dist=dist, pen=pen)
            weight = layer.weight.square() if pen == "sq" else layer.weight.abs()
            wiring = wiring + (weight * distance).sum()
            if uniformity != 0.0:
                kernel = torch.exp(-UNIFORMITY_TEMPERATURE * distance)
                uniformity_term = uniformity_term + torch.log(kernel.mean().clamp_min(1e-30))

        total = wiring / len(layers)
        if uniformity != 0.0:
            total = total + uniformity * (uniformity_term / len(layers))

        if w_dens != 0.0 and geometry.hidden_boundaries:
            density = total.new_zeros(())
            num_sets = 0
            for boundary in geometry.hidden_boundaries:
                for store in (geometry.out_channel, geometry.in_channel):
                    codes = torch.sigmoid(store[f"b{boundary}"])
                    count = codes.shape[0]
                    if count < 2:
                        continue
                    kernel = torch.exp(
                        -torch.cdist(codes, codes).square() / (2.0 * sigma * sigma)
                    )
                    density = density + (kernel.sum() - count) / (count * (count - 1))
                    num_sets += 1
            if num_sets:
                total = total + w_dens * (density / num_sets)

        if sparsity != 0.0:
            codes = [
                p for store in (geometry.in_channel, geometry.out_channel)
                for p in store.values() if p.numel()
            ]
            neurons = sum(p.shape[0] for p in codes)
            if neurons:
                mass = torch.stack([torch.sigmoid(p).sum() for p in codes]).sum()
                total = total + sparsity * (mass / neurons)

        return lam * total

    def extra_diagnostics(self, model):
        """Pair-distance statistics separating order collapse from point collapse."""
        geometry = model.geometry
        distances, euclidean = [], []
        with torch.no_grad():
            for boundary in range(len(geometry.in_channel) - 1):
                recv, send = geometry.memberships(boundary)
                if recv.numel() == 0 or send.numel() == 0:
                    continue
                distances.append(
                    bidirectional_containment(recv, send).reshape(-1)
                )
                euclidean.append(torch.cdist(recv, send).reshape(-1))

        if not distances:
            return {}
        d = torch.cat(distances)
        ed = torch.cat(euclidean)
        return {
            "dchn_mean_d": float(d.mean()),
            "dchn_frac_d0": float((d < 1e-3).to(torch.float32).mean()),
            "dchn_max_d": float(d.max()),
            "dchn_mean_eucd": float(ed.mean()),
        }
