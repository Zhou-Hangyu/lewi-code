"""LeWi: Learned Wiring, on the benchmark's backbone.

The mechanism lives in the ``lewi`` package (``lewi/README.md``): every unit has an embedding ``e`` in ``R^r``, the
score of the connection from unit ``j`` to unit ``i`` of the next layer is ``s_ij = R^2 - ||e_i - e_j||^2``, the forward
pass of every affine layer, the readout included, uses ``W * sigmoid(s) * 1[s > 0]``, and the mask is computed again
after every optimizer step. This arm builds the benchmark's ReLU MLP from LeWi layers, draws the embeddings from the
unit's auxiliary seed (``r = 3``, ``sigma_e = 0.5``, ``R^2 = 2 r sigma_e^2 = 1.5``) and applies the two checks at the
initialization (a layer without an open connection; an input with fewer than 5 open connections). LeWi has no penalty
and no target sparsity; the learning rate is its only searched hyperparameter.
"""
from __future__ import annotations

import torch

from lewi import EMBEDDING_DIM, EMBEDDING_STD, MIN_OPEN_INPUT, LeWiLinear, init_embeddings, update_masks
from lewi import connection_density as _density

from ..models import ReLUMLP
from .base import Method, register

__all__ = ["LeWi", "LeWiLinear", "connection_density", "EMBEDDING_DIM", "EMBEDDING_STD", "MIN_OPEN_INPUT"]


def connection_density(model) -> dict[str, float]:
    """The fraction of open connections: overall, of the input layer, of the hidden-to-hidden layers, of the readout."""
    return _density(model.linear_layers())


@register("lewi")
class LeWi(Method):
    hp_keys = frozenset({"code_dim", "radius"})

    def build(self, cfg, target):
        hp = cfg.get("hp") or {}
        r = int(hp.get("code_dim", EMBEDDING_DIM))
        radius = float(hp.get("radius", 1.0))          # multiplier of R^2
        if r < 1:
            raise ValueError("code_dim must be a positive integer")
        if radius <= 0:
            raise ValueError("radius must be positive")
        model = ReLUMLP(target.d, cfg["width"], cfg["hidden"],
                        layer_factory=lambda i, o, is_output: LeWiLinear(i, o, is_output=is_output))
        model.gate_codes, model.lewi_layer_redraws, model.lewi_input_redraws = init_embeddings(
            model.linear_layers(), seed=int(cfg["auxiliary_seed"]), code_dim=r, radius=radius)
        return model

    def after_step(self, model, optimizer, *, step, total_steps):
        update_masks(model.linear_layers())

    def diagnostics(self, model):
        with torch.no_grad():
            hidden = list(model.hidden_layers)
            gates = torch.cat([layer.gate().reshape(-1) for layer in hidden])
            readout_gate = model.output_layer.gate().reshape(-1)
            out = {
                "gate_mean_hidden": float(gates.mean()),
                "gate_mean_readout": float(readout_gate.mean()),
                "input_fan_in_mean": float(hidden[0].event_mask.sum(dim=1).mean()),
                "layer_redraws": int(model.lewi_layer_redraws),
                "input_redraws": int(model.lewi_input_redraws),
            }
            out.update(connection_density(model))
        return out
