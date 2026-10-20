"""CSFRBench in five minutes: benchmark your own learner on the depth-4 tree.

    python examples/quickstart.py            # under a minute on one CPU core

1. A task hands the learner the training and validation examples of one target instance. The target's computation
   graph stays hidden.
2. Fit any model (here a small PyTorch MLP) and score it on the noise-free test pool: the test NMSE of the paper.
3. A learning curve repeats this at every training-set size of the workload. Its AULC compares directly with the
   paper's methods on the same target instance.
4. After training, the hidden graph is available for structural diagnostics.

The learner below trains for 2,000 steps with one fixed configuration. The recorded methods trained for 64,000 steps
and selected the best of 32 configurations, so treat the comparison as a demonstration of the interface, not as a
result. examples/new_method.py runs the paper's protocol.
"""
import copy

import numpy as np
import torch

import csfrbench

torch.set_num_threads(1)


def fit_mlp(X_train, y_train, X_val, y_val, *, steps=2000, width=64, lr=3e-3, seed=0):
    """A ReLU MLP trained with Adam on minibatches of 32; returns the checkpoint with the lowest validation error."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(X_train.shape[1], width), torch.nn.ReLU(),
                                torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, 1),
                                torch.nn.Flatten(0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X, y = torch.from_numpy(X_train), torch.from_numpy(y_train)
    Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val)
    best, best_state = float("inf"), None
    for step in range(1, steps + 1):
        batch = torch.randint(len(y), (32,))
        loss = torch.nn.functional.mse_loss(model(X[batch]), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 250 == 0:
            with torch.no_grad():
                val = torch.nn.functional.mse_loss(model(Xv), yv).item()
            if val < best:
                best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model                                     # a torch module, a callable or an object with .predict


# 1. one task: a target instance of the depth-4 tree and 1,024 training examples
task = csfrbench.task("depth4_tree", instance=6000, n=1024)
print(task)

# 2. fit on the examples, score on the noise-free test pool
model = fit_mlp(task.X_train, task.y_train, task.X_val, task.y_val)
print(f"test NMSE at n = {task.n}: {task.evaluate(model):.4f}\n")

# 3. a learning curve over the workload's grid, n = 512 ... 32,768, and the comparison with the recorded methods
curve = csfrbench.learning_curve(fit_mlp, "depth4_tree", instance=6000, name="quick MLP", verbose=False)
print(curve, "\n")
csfrbench.compare([curve])

# 4. after training: the hidden computation, for diagnostics
target = task.reveal()
print("\nthe hidden graph, the parents of every local function (0-7 are inputs, 8 and up local functions):")
print(target.graph.node_parents)
z = target.intermediates(task.X_val[:5])
print("the value of every local function on five validation inputs:\n", np.round(z, 3))
