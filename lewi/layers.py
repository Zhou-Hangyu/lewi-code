"""The LeWi layer: an affine layer whose connections open and close with the distance of its units' embeddings."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LeWiLinear(nn.Module):
    """An affine layer ``y = (W * G * M) x + b`` with a distance gate ``G`` and its binary mask ``M``.

    Every sending unit ``j`` and every receiving unit ``i`` has an embedding (``e_j``, ``e_i``). The score of the
    connection is ``s_ij = R^2 - ||e_i - e_j||^2``, the gate is ``G_ij = sigmoid(s_ij)`` and the mask is
    ``M_ij = 1[G_ij > 1/2]``, that is ``1[s_ij > 0]``. The embeddings belong to the network, not to the layer: the
    receivers of one layer are the senders of the next. :func:`lewi.init_embeddings` draws them and binds them.

    The mask is a buffer without a gradient. Call :meth:`update_mask` after every optimizer step (or
    :func:`lewi.update_masks` on the whole network) so that a connection opens or closes when its units move.

    The weight is initialized as in the paper's backbone: Kaiming normal in fan-in mode with the gain of a ReLU
    layer (``is_output=False``) or of a linear layer (``is_output=True``); the bias starts at zero.
    """

    def __init__(self, in_features: int, out_features: int, *, is_output: bool = False):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.is_output = bool(is_output)
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        self.reset_parameters()
        self.gate_scale = nn.Parameter(torch.tensor(1.0))       # fixed at 1 (not trained)
        self.gate_offset = nn.Parameter(torch.tensor(0.0))      # fixed at R^2 by init_embeddings (not trained)
        self.register_buffer("event_mask", torch.ones(self.out_features, self.in_features))
        self._embeddings = None

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear" if self.is_output else "relu")
        nn.init.zeros_(self.bias)

    def bind(self, receivers: nn.Parameter, senders: nn.Parameter) -> None:
        """Use ``receivers`` (``out_features x r``) and ``senders`` (``in_features x r``) as the units' embeddings."""
        if receivers.shape[0] != self.out_features or senders.shape[0] != self.in_features:
            raise ValueError(f"embeddings of {receivers.shape[0]} receivers and {senders.shape[0]} senders do not fit a "
                             f"layer of {self.in_features} -> {self.out_features} units")
        self._embeddings = (receivers, senders)

    def score(self) -> torch.Tensor:
        """``s_ij = R^2 - ||e_i - e_j||^2`` for every receiver ``i`` and sender ``j``."""
        if self._embeddings is None:
            raise RuntimeError("the layer has no embeddings: call lewi.init_embeddings on the network first")
        receivers, senders = self._embeddings
        return self.gate_scale * (-torch.cdist(receivers, senders) ** 2) + self.gate_offset

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.score())

    @torch.no_grad()
    def update_mask(self) -> None:
        self.event_mask.copy_((self.gate() > 0.5).to(self.event_mask.dtype))

    def effective_weight(self) -> torch.Tensor:
        """The weight that acts in the forward pass, ``W * G * M``."""
        return self.weight * self.gate() * self.event_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.gate() * self.event_mask, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, is_output={self.is_output}"
