"""
cinemastream/ml/content_tagging/poster_cnn.py

Chapter 080 -- Convolutional Neural Networks, on CinemaStream movie posters.

Chapters 078-079 predicted genre from numeric *description* features.
This chapter predicts genre from the *poster image* -- a different
modality, the same Action-vs-Drama question. The real posters arrive
in Chapter 082 (transfer learning on a sample of the actual catalog);
here we use a SYNTHETIC poster generator so the chapter is fully
reproducible and the lesson is clean.

The synthetic posters are designed so the genre signal lives in
POSITION-INVARIANT SPATIAL TEXTURE, not average colour or any fixed pixel:
  - Action posters carry sharp, high-contrast bursts (an explosion, a
    backlit hero) at RANDOM positions -- high spatial frequency, localized.
  - Drama posters carry a smooth, low-frequency wash (a calm graded
    composition) at a random orientation -- low spatial frequency.
The added pattern is zero-mean and equal-power in both genres, applied
equally to all channels, so the two classes share the same average
colour AND the same brightness -- a model that looks only at mean colour
is at chance. And because the bursts sit at RANDOM positions, no fixed
pixel reliably signals the genre, so a flattened linear model (fixed
per-pixel weights) is also near chance. Only a translation-invariant
feature detector -- a convolution -- can learn "sharp local contrast
*somewhere*," which is exactly the inductive bias a CNN provides.

Run (requires `pip install torch scikit-learn`):
    python cinemastream/ml/content_tagging/poster_cnn.py
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

RANDOM_SEED = 42
IMG_SIZE = 32          # posters are 3 x 32 x 32 RGB tensors
N_POSTERS = 1200


def _zero_mean_unit_power(field, target_std):
    """Force a 2D field to zero mean and a fixed standard deviation, so
    Action and Drama patterns are indistinguishable by brightness/power --
    only their spatial frequency differs."""
    field = field - field.mean()
    s = field.std()
    return field * (target_std / s) if s > 1e-6 else field


def build_poster_dataset(n=N_POSTERS, size=IMG_SIZE, random_seed=RANDOM_SEED):
    """Synthetic RGB posters. label 1 = Action (sharp random bursts =
    high spatial frequency), 0 = Drama (smooth wash = low frequency).
    Both share base colour, mean, and power -- only TEXTURE differs."""
    rng = np.random.default_rng(random_seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    images = np.zeros((n, 3, size, size), dtype=np.float32)
    labels = rng.integers(0, 2, n)
    TARGET_STD = 0.11   # identical pattern power for both genres

    for i in range(n):
        base = rng.uniform(0.42, 0.48, 3)              # same muted base both genres
        canvas = np.ones((3, size, size), dtype=np.float32) * base[:, None, None]
        if labels[i] == 1:  # Action: sharp bursts at RANDOM positions (high-freq)
            field = np.zeros((size, size), dtype=np.float32)
            for _ in range(rng.integers(2, 5)):
                cx, cy = rng.uniform(4, size - 4, 2)
                sigma = rng.uniform(1.3, 2.3)          # small -> sharp
                field += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        else:               # Drama: smooth low-frequency wash, random orientation
            kx, ky = rng.uniform(0.5, 1.5, 2) * rng.choice([-1, 1], 2)
            phase = rng.uniform(0, 2 * np.pi)
            field = np.sin(2 * np.pi * (kx * xx + ky * yy) / size + phase)
        field = _zero_mean_unit_power(field, TARGET_STD)   # equalize mean & power
        canvas += field[None, :, :]                        # same modulation all channels
        canvas += rng.normal(0, 0.05, canvas.shape)        # sensor noise
        images[i] = np.clip(canvas, 0, 1)

    return images, labels


class PosterCNN(nn.Module):
    """3x32x32 -> conv/pool stack -> 2-class logits. Small on purpose."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32->16
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16->8
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 8 * 8, 2))

    def forward(self, x):
        return self.head(self.features(x))


def train_cnn(Xtr, ytr, Xte, yte, epochs=12, lr=0.002):
    torch.manual_seed(RANDOM_SEED)
    model = PosterCNN()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte); yte_t = torch.tensor(yte, dtype=torch.long)
    n = len(Xtr_t)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(epoch))
        for start in range(0, n, 64):
            idx = perm[start:start + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        if epoch % 4 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xte_t).argmax(1) == yte_t).float().mean().item()
            print(f"  epoch {epoch:>2}: train loss={loss.item():.4f}, test acc={acc:.3f}")
    model.eval()
    with torch.no_grad():
        te_acc = (model(Xte_t).argmax(1) == yte_t).float().mean().item()
    return model, te_acc


def main():
    X, y = build_poster_dataset()
    print(f"{len(y)} synthetic posters, 3x{IMG_SIZE}x{IMG_SIZE} RGB, "
          f"{y.mean():.1%} Action / {1 - y.mean():.1%} Drama")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)

    # Baseline 1: logistic regression on MEAN colour (3 features). The
    # genres share base colour, mean, and power -> this should be at
    # chance, proving the signal is NOT in average colour/brightness.
    mean_tr = Xtr.mean(axis=(2, 3))
    mean_te = Xte.mean(axis=(2, 3))
    color_acc = LogisticRegression().fit(mean_tr, ytr).score(mean_te, yte)
    print(f"\nMean-colour logistic (3 features):     {color_acc:.3f}  "
          f"-> at chance: colour/brightness can't tell the genres apart")

    # Baseline 2: logistic regression on FLATTENED pixels (3072 features,
    # fixed per-pixel weights, no spatial/translation awareness).
    flat_model = LogisticRegression(max_iter=300).fit(
        Xtr.reshape(len(Xtr), -1), ytr)
    flat_acc = flat_model.score(Xte.reshape(len(Xte), -1), yte)
    flat_params = flat_model.coef_.size + flat_model.intercept_.size
    print(f"Flattened-pixel logistic (3072 feat):  {flat_acc:.3f}  "
          f"({flat_params:,} params; partly catches texture, but no "
          f"translation invariance)")

    print("\n=== PosterCNN (convolutions = translation-invariant feature detectors) ===")
    model, cnn_acc = train_cnn(Xtr, ytr, Xte, yte)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PosterCNN test accuracy: {cnn_acc:.3f}  ({n_params:,} parameters)")
    print(f"-> beats the flattened model because a burst at ANY position "
          f"fires the same learned kernel (translation invariance)")


if __name__ == "__main__":
    main()
