# LeWi: Learned Wiring

LeWi gives every neuron a learned position in a small space and connects two neurons of adjacent layers only when
they are close. The positions and the weights train together, so the network learns its own wiring: when gradients
move two neurons across the connection radius, the connection between them opens or closes.

## The rule

Every unit (input, hidden and output) has an embedding $e \in \mathbb{R}^r$. For a sending unit $j$ and a receiving
unit $i$ of the next layer:

$$s_{ij} = R^2 - \lVert e_i - e_j \rVert^2, \qquad \widetilde{W}_{ij} = W_{ij}\,\sigma(s_{ij})\,\mathbf{1}[s_{ij} > 0].$$

The forward pass of every affine layer, the readout included, uses $\widetilde{W}$. The soft gate $\sigma(s_{ij})$
carries the gradient to the embeddings; the binary mask $\mathbf{1}[s_{ij} > 0]$ has no gradient and is recomputed
after every optimizer step.

| Constant | Value |
|---|---|
| embedding dimension $r$ | 3 |
| initial embeddings | $\mathcal{N}(0, \sigma_e^2 I)$, $\sigma_e = 0.5$ |
| connection radius | $R^2 = 2 r \sigma_e^2 = 1.5$, the same for every layer, not trained |
| open connections at initialization | about 61% |

Two checks at the initialization use no information about the task and treat every input alike: a layer without an
open connection has its receiving units' embeddings drawn again, and an input with fewer than 5 open outgoing
connections has its embedding drawn again. LeWi has no penalty and no target sparsity. In the paper the learning rate
is its only searched hyperparameter.

## Use

```python
import torch
from lewi import LeWiMLP

model = LeWiMLP(in_features=8, width=32, depth=4, seed=0)        # 4 hidden ReLU layers, one output
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)        # weights and embeddings together

x, y = torch.randn(512, 8), torch.randn(512)
for step in range(100):
    loss = torch.nn.functional.mse_loss(model(x).squeeze(-1), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.update_masks()                                         # after every optimizer step

print(model.connection_density())     # fraction of open connections per layer group
print(model.positions()[1].shape)     # embeddings of the first hidden layer: (32, 3)
```

For a network of your own, build it from `LeWiLinear` layers, draw and bind the embeddings once with
`init_embeddings`, register them on the network, and call `update_masks(model)` after each optimizer step:

```python
from torch import nn
from lewi import LeWiLinear, init_embeddings, update_masks

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([LeWiLinear(16, 64), LeWiLinear(64, 64), LeWiLinear(64, 4, is_output=True)])
        self.embeddings, _, _ = init_embeddings(self.layers, seed=0)   # one embedding per unit of every layer boundary

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)
```

`init_embeddings` expects a chain: layer $k$ reads the units that layer $k-1$ writes, so consecutive layers share
the embeddings of the units between them.

## Files

| File | Content |
|---|---|
| `layers.py` | `LeWiLinear`: the gated and masked affine layer |
| `network.py` | `init_embeddings` (the embeddings and the two checks), `update_masks`, `connection_density`, `LeWiMLP` |

CSFRBench's `lewi` method (`csfrbench/methods/lewi.py`) is this module on the benchmark's backbone; the paper's runs
are reproduced with it bit for bit (see the top-level README).

## Cost

This implementation evaluates every candidate connection densely: each step computes the pairwise distances, gates
and masks of all pairs of units in adjacent layers. A step therefore costs more than a dense layer of the same shape,
and the learned sparsity does not reduce the training computation. At inference, the gates and masks can be folded
into the weights (`layer.effective_weight()`). Appendix D.7 of the paper reports the cost.
