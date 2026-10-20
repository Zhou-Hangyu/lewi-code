from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLinear(nn.Module):
    """Benchmark affine layer with explicitly controlled initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        is_output: bool,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.is_output = bool(is_output)

        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")

        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )
        self.bias = nn.Parameter(torch.zeros(self.out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nonlinearity = "linear" if self.is_output else "relu"
        nn.init.kaiming_normal_(
            self.weight,
            mode="fan_in",
            nonlinearity=nonlinearity,
        )
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def effective_weight(self) -> torch.Tensor:
        """Return the deterministic weight matrix used at evaluation."""
        return self.weight


class HardConcreteLinear(DenseLinear):
    """Affine layer with an independent hard-concrete gate per scalar weight."""

    DEFAULT_BETA = 2.0 / 3.0
    DEFAULT_GAMMA = -0.1
    DEFAULT_ZETA = 1.1
    SAMPLE_EPS = 1e-6

    # Bounds used by the released implementation.
    LOG_ALPHA_MIN = math.log(1e-2)
    LOG_ALPHA_MAX = math.log(1e2)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        is_output: bool,
        p_open_init: float = 0.5,
        gate_init_generator: torch.Generator,
        gate_init_std: float = 1e-2,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        zeta: float = DEFAULT_ZETA,
        stochastic: bool = True,
    ):
        # False makes training use the same deterministic gate as evaluation, which
        # separates the multiplicative-noise effect from the reparameterization.
        self.stochastic = bool(stochastic)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)

        if self.beta <= 0:
            raise ValueError("beta must be positive.")
        if not self.gamma < 0:
            raise ValueError("gamma must be negative.")
        if not self.zeta > 1:
            raise ValueError("zeta must be greater than 1.")
        if not 0.0 < float(p_open_init) < 1.0:
            raise ValueError("p_open_init must lie strictly between 0 and 1.")
        if float(gate_init_std) < 0:
            raise ValueError("gate_init_std must be nonnegative.")

        # Initializes the raw weight using the same global RNG sequence as the
        # corresponding DenseLinear layer.
        super().__init__(
            in_features,
            out_features,
            is_output=is_output,
        )

        # kappa = beta * log(-gamma / zeta)
        self._open_logit_shift = (
            self.beta * math.log(-self.gamma / self.zeta)
        )

        p = float(p_open_init)
        initial_log_alpha = (
            math.log(p) - math.log1p(-p) + self._open_logit_shift
        )

        # Use a separate generator so gate initialization does not consume from
        # the global stream used for paired backbone initialization.
        noise = torch.randn(
            self.weight.shape,
            generator=gate_init_generator,
            dtype=self.weight.dtype,
            device="cpu",
        )

        self.log_alpha = nn.Parameter(
            torch.full_like(self.weight, initial_log_alpha)
            + float(gate_init_std) * noise
        )

    def sampled_gate(self) -> torch.Tensor:
        """Draw one edge-wise gate matrix for the current minibatch."""
        uniform = torch.empty_like(self.log_alpha).uniform_(
            self.SAMPLE_EPS,
            1.0 - self.SAMPLE_EPS,
        )

        logistic_noise = (
            torch.log(uniform)
            - torch.log1p(-uniform)
        )

        concrete = torch.sigmoid(
            (logistic_noise + self.log_alpha) / self.beta
        )

        stretched = (
            concrete * (self.zeta - self.gamma)
            + self.gamma
        )

        return stretched.clamp(0.0, 1.0)

    def deterministic_gate(self) -> torch.Tensor:
        """Equation 13 test-time hard-concrete gate estimator."""
        stretched_mean = (
            torch.sigmoid(self.log_alpha)
            * (self.zeta - self.gamma)
            + self.gamma
        )
        return stretched_mean.clamp(0.0, 1.0)

    def expected_open_probability(self) -> torch.Tensor:
        """Equation 12 probability that each gate is nonzero."""
        return torch.sigmoid(
            self.log_alpha - self._open_logit_shift
        )

    def expected_l0(self) -> torch.Tensor:
        """Expected number of nonzero gated scalar weights."""
        return self.expected_open_probability().sum()

    def effective_weight(self) -> torch.Tensor:
        """Deterministic evaluation-time effective weight matrix."""
        return self.weight * self.deterministic_gate()

    @torch.no_grad()
    def clamp_log_alpha_(self) -> None:
        """Apply the gate-logit bounds used by the released implementation."""
        self.log_alpha.clamp_(
            min=self.LOG_ALPHA_MIN,
            max=self.LOG_ALPHA_MAX,
        )

    def forward(self, x):
        gate = (
            self.sampled_gate()
            if self.training and self.stochastic
            else self.deterministic_gate()
        )
        return F.linear(x, self.weight * gate, self.bias)


LayerFactory = Callable[[int, int, bool], nn.Module]


def boundary_widths(model) -> tuple[int, ...]:
    """Neuron counts at each layer boundary, input boundary through output."""
    layers = model.linear_layers()
    return (int(layers[0].in_features), *(int(layer.out_features) for layer in layers))


class ReLUMLP(nn.Module):
    """Shared ReLU MLP shell for dense and gated affine layers."""

    def __init__(
        self,
        input_dim: int,
        width: int,
        num_hidden_layers: int,
        *,
        out_dim: int = 1,
        layer_factory: LayerFactory,
        readout_structural: bool = False,
    ):
        super().__init__()

        input_dim = int(input_dim)
        width = int(width)
        num_hidden_layers = int(num_hidden_layers)
        out_dim = int(out_dim)

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if width <= 0:
            raise ValueError("width must be positive.")
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be nonnegative.")
        if out_dim <= 0:
            raise ValueError("out_dim must be positive.")

        self.out_dim = out_dim
        # CSFRBench v1.1: the readout is a structural transition when the
        # method's fixed argument ``readout`` is "structural"; the v1.0 form
        # (False) leaves it outside every penalty, gate, mask and pruning.
        self.readout_structural = bool(readout_structural)

        hidden_dimensions = (
            [input_dim]
            + [width] * num_hidden_layers
        )

        self.hidden_layers = nn.ModuleList(
            layer_factory(
                hidden_dimensions[i],
                hidden_dimensions[i + 1],
                False,
            )
            for i in range(num_hidden_layers)
        )

        output_input_dim = hidden_dimensions[-1]
        self.output_layer = layer_factory(
            output_input_dim,
            out_dim,
            True,
        )

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        output = self.output_layer(x)
        return output.squeeze(-1) if self.out_dim == 1 else output

    def linear_layers(self) -> tuple[nn.Module, ...]:
        """Return affine layers in forward order."""
        return (*self.hidden_layers, self.output_layer)

    def structural_layers(self) -> tuple[nn.Module, ...]:
        """Affine layers a method may penalize, gate, mask or prune: every
        hidden transition, plus the readout when the model was built with
        ``readout_structural`` (CSFRBench v1.1: the readout is treated like
        every other transition; v1.0 left it plain)."""
        if self.readout_structural:
            return (*self.hidden_layers, self.output_layer)
        return tuple(self.hidden_layers)

    def regularized_weights(self) -> tuple[torch.Tensor, ...]:
        """Return raw weight tensors included in sparsity penalties."""
        return tuple(
            layer.weight
            for layer in self.structural_layers()
        )

    def effective_weights(self) -> tuple[torch.Tensor, ...]:
        """Return deterministic effective weights in forward order."""
        return tuple(
            layer.effective_weight()
            for layer in self.linear_layers()
        )


class DenseMLP(ReLUMLP):
    """Standard fully connected ReLU MLP."""

    def __init__(
        self,
        input_dim: int,
        width: int,
        num_hidden_layers: int,
        out_dim: int = 1,
        readout_structural: bool = False,
    ):
        def make_layer(
            in_features: int,
            out_features: int,
            is_output: bool,
        ):
            return DenseLinear(
                in_features,
                out_features,
                is_output=is_output,
            )

        super().__init__(
            input_dim=input_dim,
            width=width,
            num_hidden_layers=num_hidden_layers,
            out_dim=out_dim,
            layer_factory=make_layer,
            readout_structural=readout_structural,
        )


class HardConcreteMLP(ReLUMLP):
    """ReLU MLP with one independent hard-concrete gate per scalar weight."""

    def __init__(
        self,
        input_dim: int,
        width: int,
        num_hidden_layers: int,
        *,
        p_open_init: float,
        gate_init_seed: int,
        out_dim: int = 1,
        stochastic: bool = True,
        readout_structural: bool = False,
    ):
        gate_generator = torch.Generator(device="cpu")
        gate_generator.manual_seed(int(gate_init_seed))

        def make_layer(
            in_features: int,
            out_features: int,
            is_output: bool,
        ):
            # the readout is gated only under the v1.1 structural rule; the
            # hidden gates draw first, so they equal the plain form's
            if is_output and not readout_structural:
                return DenseLinear(in_features, out_features, is_output=True)
            return HardConcreteLinear(
                in_features,
                out_features,
                is_output=is_output,
                p_open_init=p_open_init,
                gate_init_generator=gate_generator,
                stochastic=stochastic,
            )

        super().__init__(
            input_dim=input_dim,
            width=width,
            num_hidden_layers=num_hidden_layers,
            out_dim=out_dim,
            layer_factory=make_layer,
            readout_structural=readout_structural,
        )

    def expected_l0(self) -> torch.Tensor:
        return torch.stack(
            [
                layer.expected_l0()
                for layer in self.structural_layers()
            ]
        ).sum()

    @torch.no_grad()
    def clamp_log_alpha_(self) -> None:
        for layer in self.structural_layers():
            layer.clamp_log_alpha_()
