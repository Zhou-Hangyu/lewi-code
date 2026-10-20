"""Ground-truth-wired reference architecture.

The network contains one independently parameterized ReLU MLP for every true
computational node. Layer-zero nodes receive their known active raw inputs;
later nodes receive their known parents from the preceding target layer.

The wiring is fixed to the target graph. Only the local node functions are
learned.
"""

from __future__ import annotations

import torch
from torch import nn

from ..registry import method
from .base import Method


def _make_node_mlp(
    input_dim: int,
    width: int,
    depth: int,
) -> nn.Sequential:
    """Construct one scalar-output local MLP.

    ``depth`` is the total number of affine layers:

    - depth=1: Linear(input_dim, 1)
    - depth=2: Linear(input_dim, width), ReLU, Linear(width, 1)
    - depth=3: two hidden affine-ReLU blocks followed by a linear output

    Args:
        input_dim:
            Number of inputs to the target node.
        width:
            Width of every hidden layer. Ignored when ``depth == 1``.
        depth:
            Total number of affine layers, including the output layer.
    """
    input_dim = int(input_dim)
    width = int(width)
    depth = int(depth)

    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if width <= 0:
        raise ValueError("width must be positive.")
    if depth <= 0:
        raise ValueError("depth must be positive.")

    modules: list[nn.Module] = []
    current_dim = input_dim

    for _ in range(depth - 1):
        layer = nn.Linear(current_dim, width)
        nn.init.kaiming_normal_(
            layer.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        nn.init.zeros_(layer.bias)

        modules.extend((layer, nn.ReLU()))
        current_dim = width

    output_layer = nn.Linear(current_dim, 1)
    nn.init.kaiming_normal_(
        output_layer.weight,
        mode="fan_in",
        nonlinearity="linear",
    )
    nn.init.zeros_(output_layer.bias)
    modules.append(output_layer)

    return nn.Sequential(*modules)


class OracleNet(nn.Module):
    """MLP composition with the target graph's exact node-level wiring."""

    def __init__(
        self,
        target,
        node_width: int = 16,
        node_depth: int = 2,
    ):
        super().__init__()

        self.input_dim = int(target.d)
        self.arity = int(target.k)
        self.layer_profile = tuple(
            int(width)
            for width in target.layer_profile
        )

        if self.input_dim <= 0:
            raise ValueError("target.d must be positive.")
        if self.arity <= 0:
            raise ValueError("target.k must be positive.")
        if not self.layer_profile:
            raise ValueError("target.layer_profile must not be empty.")
        if any(width <= 0 for width in self.layer_profile):
            raise ValueError(
                "Every target layer must contain at least one node."
            )
        if self.layer_profile[-1] != 1:
            raise ValueError(
                "The final target layer must contain exactly one output node; "
                f"received layer_profile={self.layer_profile}."
            )

        active_inputs = torch.as_tensor(
            target.active_raw,
            dtype=torch.long,
        ).reshape(-1)

        expected_active = self.layer_profile[0] * self.arity
        if active_inputs.numel() != expected_active:
            raise ValueError(
                "target.active_raw has the wrong number of entries: "
                f"expected {expected_active} = "
                f"{self.layer_profile[0]} first-layer nodes × "
                f"arity {self.arity}, received {active_inputs.numel()}."
            )

        if torch.any(active_inputs < 0) or torch.any(
            active_inputs >= self.input_dim
        ):
            raise ValueError(
                "target.active_raw contains an input index outside "
                f"[0, {self.input_dim})."
            )

        self.register_buffer(
            "active_inputs",
            active_inputs,
            persistent=True,
        )

        parents = tuple(target.parents)
        expected_parent_arrays = len(self.layer_profile) - 1

        if len(parents) != expected_parent_arrays:
            raise ValueError(
                "target.parents has the wrong number of layers: "
                f"expected {expected_parent_arrays}, received {len(parents)}."
            )

        for parent_layer_index, parent_array in enumerate(parents, start=1):
            parent_indices = torch.as_tensor(
                parent_array,
                dtype=torch.long,
            )

            expected_shape = (
                self.layer_profile[parent_layer_index],
                self.arity,
            )
            if tuple(parent_indices.shape) != expected_shape:
                raise ValueError(
                    f"target.parents[{parent_layer_index - 1}] must have "
                    f"shape {expected_shape}; received "
                    f"{tuple(parent_indices.shape)}."
                )

            previous_width = self.layer_profile[parent_layer_index - 1]
            if torch.any(parent_indices < 0) or torch.any(
                parent_indices >= previous_width
            ):
                raise ValueError(
                    f"target.parents[{parent_layer_index - 1}] contains an "
                    f"index outside [0, {previous_width})."
                )

            self.register_buffer(
                f"parent_indices_{parent_layer_index}",
                parent_indices,
                persistent=True,
            )

        num_nodes = sum(self.layer_profile)
        self.node_mlps = nn.ModuleList(
            [
                _make_node_mlp(
                    input_dim=self.arity,
                    width=node_width,
                    depth=node_depth,
                )
                for _ in range(num_nodes)
            ]
        )

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(
                f"OracleNet expects a two-dimensional input tensor; "
                f"received shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}; "
                f"received {x.shape[1]}."
            )

        batch_size = x.shape[0]
        node_offset = 0
        previous_outputs = None

        for layer_index, num_nodes in enumerate(self.layer_profile):
            if layer_index == 0:
                node_inputs = x[:, self.active_inputs].reshape(
                    batch_size,
                    num_nodes,
                    self.arity,
                )
            else:
                parent_indices = getattr(
                    self,
                    f"parent_indices_{layer_index}",
                )

                # previous_outputs: (batch, previous_layer_width)
                # parent_indices:   (current_layer_width, arity)
                # node_inputs:      (batch, current_layer_width, arity)
                node_inputs = previous_outputs[:, parent_indices]

            node_outputs = [
                self.node_mlps[node_offset + node_index](
                    node_inputs[:, node_index, :]
                )
                for node_index in range(num_nodes)
            ]

            previous_outputs = torch.cat(node_outputs, dim=1)
            node_offset += num_nodes

        return previous_outputs[:, 0]


class GraphOracleNet(nn.Module):
    """MLP composition wired exactly by a ``GraphSpec`` target.

    One local MLP per graph node; node ``v`` reads its true parent refs
    (leaves below ``A``, earlier node outputs above), and the output is the
    target's fixed linear readout. Handles per-node arities and additive
    readouts that the layered ``OracleNet`` cannot (flat/dag families).
    """

    def __init__(
        self,
        target,
        node_width: int = 16,
        node_depth: int = 2,
    ):
        super().__init__()

        graph = target.graph
        self.input_dim = int(target.d)
        self.num_leaves = int(graph.A)
        self.node_parents = tuple(tuple(int(r) for r in refs) for refs in graph.node_parents)
        self.readout_indices = tuple(int(i) for i in graph.readout_indices)

        active_inputs = torch.as_tensor(target.active_raw, dtype=torch.long).reshape(-1)
        if active_inputs.numel() != self.num_leaves:
            raise ValueError(
                f"target.active_raw has {active_inputs.numel()} entries; "
                f"graph declares {self.num_leaves} leaves"
            )
        if torch.any(active_inputs < 0) or torch.any(active_inputs >= self.input_dim):
            raise ValueError(
                f"target.active_raw contains an input index outside [0, {self.input_dim})"
            )
        self.register_buffer("active_inputs", active_inputs, persistent=True)
        self.register_buffer(
            "readout_weights",
            torch.as_tensor(graph.readout_weights, dtype=torch.float32),
            persistent=True,
        )

        self.node_mlps = nn.ModuleList(
            _make_node_mlp(
                input_dim=len(refs),
                width=node_width,
                depth=node_depth,
            )
            for refs in self.node_parents
        )

    def node_outputs(self, x) -> torch.Tensor:
        """Every node's scalar output in graph order, shape (batch, S)."""
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"GraphOracleNet expects shape (batch, {self.input_dim}); "
                f"received {tuple(x.shape)}"
            )
        leaves = x[:, self.active_inputs]
        values: list[torch.Tensor] = []
        for refs, mlp in zip(self.node_parents, self.node_mlps):
            columns = [
                leaves[:, r] if r < self.num_leaves else values[r - self.num_leaves]
                for r in refs
            ]
            values.append(mlp(torch.stack(columns, dim=1))[:, 0])
        return torch.stack(values, dim=1)

    def forward(self, x):
        values = self.node_outputs(x)
        indices = torch.as_tensor(self.readout_indices, device=values.device)
        return values[:, indices] @ self.readout_weights


@method("oracle")
class Oracle(Method):
    """Ground-truth-wired reference that learns only local node functions.

    This method knows the target graph's active raw inputs and parent
    relationships. It is an oracle structural reference, but is not guaranteed
    to be a performance ceiling unless the configured local MLPs have sufficient
    capacity and optimization succeeds.
    """

    weight_graph = False
    hp_keys = frozenset({"node_width", "node_depth"})

    def build(self, cfg, target):
        hp = cfg.get("hp", {})
        node_width = int(hp.get("node_width", 16))
        node_depth = int(hp.get("node_depth", 2))

        if getattr(target, "graph", None) is not None:
            return GraphOracleNet(target, node_width=node_width, node_depth=node_depth)
        return OracleNet(target, node_width=node_width, node_depth=node_depth)

@method("factorized_oracle")
class FactorizedOracle(Oracle):
    """Canonical v1 name for the ground-truth-wired reference (spec section 10).

    The frozen configuration governs the name: one single-hidden-layer
    width-256 ReLU module per teacher node (``node_width = 256,
    node_depth = 2``), pinned regardless of hp — the module class is a
    benchmark constant, never a nuisance dimension; the audited space is
    LR-only. Contradictory hp is an error, not an override. Module-capacity
    studies use the legacy ``oracle`` method, which keeps hp-configurable
    modules.

    ``weight_graph`` stays ``False``: the training engine's shared weight
    diagnostics (``metrics.structure`` via ``effective_weights``) apply to
    the masked/regularized single-backbone arms and are routed around for
    the per-node reference; ``effective_weights`` is intentionally not
    implemented (verified by test).
    """

    FROZEN = {"node_width": 256, "node_depth": 2}

    def build(self, cfg, target):
        hp = dict(cfg.get("hp") or {})
        for key, frozen in self.FROZEN.items():
            value = hp.get(key)
            if value is not None and int(value) != frozen:
                raise ValueError(
                    f"factorized_oracle is frozen at {key}={frozen}; received "
                    f"{value}. The module class is a benchmark constant, not a "
                    "nuisance dimension (spec section 10); use the legacy "
                    "'oracle' method for module-capacity studies.")
            hp[key] = frozen
        return super().build({**cfg, "hp": hp}, target)
