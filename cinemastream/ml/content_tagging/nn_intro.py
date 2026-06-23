"""
cinemastream/ml/content_tagging/nn_intro.py

Chapter 078 -- Neural Network Foundations: the first file in the
content-tagging portfolio thread (Part 8).

Part 8's goal (per the bible) is a model that reads a movie's
description and predicts its genre tags. The real text work -- raw
tokens, embeddings, CNNs on posters, transformers on descriptions --
arrives in Ch080-085. This chapter establishes the neural-network
*mechanics* on a small, synthetic, fully-reproducible dataset of
description-derived numeric features, and shows where a multi-layer
network earns its keep over a linear model: when the signal lives in a
NON-LINEAR INTERACTION between features (Section 2's XOR lesson, applied
to movie metadata).

This is deliberately NOT the churn thread (cinemastream/ml/churn/) --
it's a new model, a new folder, a new problem.

Run (requires `pip install torch scikit-learn`):
    python cinemastream/ml/content_tagging/nn_intro.py
"""

import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def build_genre_dataset(n=2000, random_seed=RANDOM_SEED):
    """Synthetic movies with description-derived numeric features and a
    binary genre label (1 = Action, 0 = Drama).

    The label is built with a deliberate NON-LINEAR rule: a movie reads
    as "Action" when its pace and runtime AGREE at an extreme -- either
    short-and-fast (a tight thriller) or long-and-fast (an epic) -- but a
    fast pace with a middling runtime, or a slow pace at any runtime,
    reads as Drama. A single linear boundary cannot capture "extreme at
    both ends," which is exactly why this needs more than logistic
    regression.
    """
    rng = np.random.default_rng(random_seed)
    runtime_min = rng.uniform(80, 170, n)
    pace_score = rng.uniform(0, 1, n)          # density of action keywords
    emotion_score = rng.uniform(0, 1, n)       # density of emotional keywords
    dialogue_ratio = rng.uniform(0.2, 0.8, n)  # share of description that is dialogue

    runtime_norm = (runtime_min - 80) / 90     # -> [0, 1]
    # "extremeness" of runtime: high near both ends, low in the middle
    runtime_extreme = np.abs(runtime_norm - 0.5) * 2
    action_signal = pace_score * runtime_extreme - emotion_score * 0.6
    prob = 1 / (1 + np.exp(-10 * (action_signal - 0.15)))
    genre = (rng.uniform(0, 1, n) < prob).astype(int)

    X = np.column_stack([runtime_min, pace_score, emotion_score, dialogue_ratio])
    feature_names = ["runtime_min", "pace_score", "emotion_score", "dialogue_ratio"]
    return X, genre, feature_names


class GenreMLP(nn.Module):
    """A small multi-layer perceptron: 4 features -> 16 -> 8 -> 1 logit.
    ReLU hidden activations give it the non-linearity a linear model
    lacks; we output a raw logit and use BCEWithLogitsLoss (numerically
    stabler than a Sigmoid layer + BCELoss)."""

    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_test, y_test, epochs=250, lr=0.01,
              weight_decay=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(RANDOM_SEED)

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    Xte = torch.tensor(X_test, dtype=torch.float32).to(device)
    yte = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)

    model = GenreMLP(X_train.shape[1]).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    # weight_decay is L2 regularisation -- it discourages large weights,
    # the neural-net analogue of Ch073's min_samples_leaf tree constraint.
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            print(f"  epoch {epoch:>3}: train loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        tr_acc = ((model(Xtr) > 0).float() == ytr).float().mean().item()
        te_acc = ((model(Xte) > 0).float() == yte).float().mean().item()
    return model, tr_acc, te_acc, device


def main():
    X, y, names = build_genre_dataset()
    print(f"Synthetic genre dataset: {len(y)} movies, "
          f"{y.mean():.1%} Action / {1 - y.mean():.1%} Drama")
    print(f"Features: {names}")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)

    # Always scale before a neural net (Ch069 lesson; NNs are even more
    # sensitive to feature scale than tree-free linear models).
    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

    print("=== Linear baseline: Logistic Regression ===")
    logreg = LogisticRegression().fit(Xtr_s, y_train)
    print(f"LogReg test accuracy: {logreg.score(Xte_s, y_test):.3f}")
    print()

    print("=== Multi-layer perceptron (PyTorch) ===")
    model, tr_acc, te_acc, device = train_mlp(Xtr_s, y_train, Xte_s, y_test)
    print(f"device: {device}")
    print(f"MLP train accuracy: {tr_acc:.3f}")
    print(f"MLP test accuracy:  {te_acc:.3f}")
    print()
    print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()
