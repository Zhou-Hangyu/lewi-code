"""LeWi (Learned Wiring): every neuron has a learned position, and only nearby neurons connect.

    import torch
    from lewi import LeWiMLP

    model = LeWiMLP(in_features=8, width=32, depth=4)            # out_features=1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for x, y in loader:
        loss = torch.nn.functional.mse_loss(model(x).squeeze(-1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_masks()                                     # after every optimizer step

The weights and the neuron embeddings train together; a connection opens or closes when its two neurons move
across the connection radius. See README.md beside this file.
"""
from .layers import LeWiLinear
from .network import (EMBEDDING_DIM, EMBEDDING_STD, MIN_OPEN_INPUT, LeWiMLP, connection_density, derive_seed,
                      init_embeddings, update_masks)

__version__ = "1.6.1"
__all__ = ["LeWiLinear", "LeWiMLP", "init_embeddings", "update_masks", "connection_density", "derive_seed",
           "EMBEDDING_DIM", "EMBEDDING_STD", "MIN_OPEN_INPUT"]
