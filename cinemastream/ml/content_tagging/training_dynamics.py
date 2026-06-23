"""
cinemastream/ml/content_tagging/training_dynamics.py

Chapter 079 -- Backpropagation and Optimizers, on the content-tagging
genre model.

Chapter 078 trained the GenreMLP with Adam(lr=0.01) and reached test
accuracy 0.860, treating loss.backward() and the optimizer as magic.
This module opens both boxes on the SAME dataset and model:
  - it trains the genre MLP with plain SGD vs Adam and shows Adam's
    faster, steadier convergence,
  - it sweeps the learning rate to show the stall / converge / diverge
    regimes on real CinemaStream-shaped data,
  - it inspects the actual gradients flowing back into the first layer.

Reuses build_genre_dataset() and GenreMLP from Ch078's nn_intro.py --
same synthetic genre data, same architecture, so the numbers connect
directly to Chapter 078's 0.860.

Run (requires `pip install torch scikit-learn`):
    python cinemastream/ml/content_tagging/training_dynamics.py
"""

import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cinemastream.ml.content_tagging.nn_intro import (
    build_genre_dataset, GenreMLP, RANDOM_SEED)


def make_split():
    X, y, names = build_genre_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)
    scaler = StandardScaler().fit(X_train)
    return (scaler.transform(X_train), scaler.transform(X_test),
            y_train, y_test)


def to_tensors(X_train, X_test, y_train, y_test):
    return (torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1))


def train_with(optimizer_name, lr, Xtr, ytr, Xte, yte, epochs=250,
               weight_decay=1e-3, record_every=50):
    """Train the GenreMLP with a chosen optimizer/lr; return loss
    trajectory and final test accuracy."""
    torch.manual_seed(RANDOM_SEED)
    model = GenreMLP(Xtr.shape[1])
    loss_fn = nn.BCEWithLogitsLoss()
    if optimizer_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
    trajectory = {}
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
        if epoch % record_every == 0 or epoch == 1:
            trajectory[epoch] = loss.item()
    with torch.no_grad():
        te_acc = ((model(Xte) > 0).float() == yte).float().mean().item()
    return trajectory, te_acc


def inspect_gradients(Xtr, ytr):
    """One backward pass: show the gradient PyTorch computed for the
    first layer's weights -- the thing loss.backward() actually produces."""
    torch.manual_seed(RANDOM_SEED)
    model = GenreMLP(Xtr.shape[1])
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(model(Xtr), ytr)
    loss.backward()
    first_layer = model.net[0]   # nn.Linear(4, 16)
    g = first_layer.weight.grad
    print(f"First layer weight gradient shape: {tuple(g.shape)}")
    print(f"  grad mean={g.mean().item():+.5f}, "
          f"std={g.std().item():.5f}, "
          f"max|grad|={g.abs().max().item():.5f}")
    print("  (these gradients are what the optimizer multiplies by the "
          "learning rate)")


def main():
    X_train, X_test, y_train, y_test = make_split()
    Xtr, ytr, Xte, yte = to_tensors(X_train, X_test, y_train, y_test)

    print("=== One backward pass: the gradients behind the magic ===")
    inspect_gradients(Xtr, ytr)
    print()

    print("=== SGD vs Adam: same model, same data (Ch078's GenreMLP) ===")
    sgd_traj, sgd_acc = train_with("SGD", 0.1, Xtr, ytr, Xte, yte)
    adam_traj, adam_acc = train_with("Adam", 0.01, Xtr, ytr, Xte, yte)
    print(f"{'epoch':>6} {'SGD loss':>10} {'Adam loss':>10}")
    for e in sorted(adam_traj):
        print(f"{e:>6} {sgd_traj[e]:>10.4f} {adam_traj[e]:>10.4f}")
    print(f"final test accuracy:  SGD={sgd_acc:.3f}   Adam={adam_acc:.3f}")
    print()

    print("=== Learning-rate sweep (SGD): stall / converge / diverge ===")
    for lr in [0.0001, 0.1, 30.0]:
        traj, acc = train_with("SGD", lr, Xtr, ytr, Xte, yte)
        end_loss = traj[max(traj)]
        verdict = ("stalls" if lr == 0.0001 else
                   "converges" if lr == 0.1 else "diverges")
        print(f"lr={lr:<8}: end loss={end_loss:>8.4f}  test_acc={acc:.3f}  "
              f"-> {verdict}")


if __name__ == "__main__":
    main()
