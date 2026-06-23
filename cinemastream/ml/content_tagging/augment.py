"""
cinemastream/ml/content_tagging/augment.py

Chapter 081 -- Image Preprocessing and Augmentation, on CinemaStream posters.

Chapter 080's PosterCNN reached 0.987 on clean synthetic posters. But
real posters are not clean: the subject is off-center, the crop varies,
the image is sometimes mirrored in a thumbnail. A model trained only on
clean, centered renders learns position-locked features and shatters on
that variation -- even though Ch080's CNN is translation-invariant by
architecture, it has only ever SEEN centered bursts, so it never learned
to expect them elsewhere.

Data augmentation fixes this: during training, each image is randomly
flipped / shifted / rotated on the fly, so the model sees its training
examples in many poses and learns features that survive real-world
variation. Augmentation is applied to TRAINING ONLY -- never at test time.

This module demonstrates the payoff: it trains the Ch080 PosterCNN with
and without augmentation on the SAME clean training posters, then
evaluates both on a "field" test set whose posters have been perturbed
(shift / rotate / flip) to mimic real-world variation. The augmented
model should hold up where the un-augmented one collapses.

Run (requires `pip install torch torchvision scikit-learn`):
    python cinemastream/ml/content_tagging/augment.py
"""

import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from sklearn.model_selection import train_test_split

from cinemastream.ml.content_tagging.poster_cnn import (
    build_poster_dataset, PosterCNN, RANDOM_SEED)

# The kinds of variation real posters have that clean renders don't:
# mirror flips, small shifts, small rotations.
FIELD_PERTURB = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=15, translate=(0.12, 0.12)),
])
# Training augmentation: the SAME family of transforms, applied on the fly.
TRAIN_AUG = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=15, translate=(0.12, 0.12)),
])


def perturb_to_field(X, seed=0):
    """Apply a fixed random perturbation once to make a 'field' test set --
    posters as they'd actually arrive, not as cleanly rendered."""
    torch.manual_seed(seed)
    Xt = torch.tensor(X)
    return torch.stack([FIELD_PERTURB(img) for img in Xt]).numpy()


def train_poster_cnn(Xtr, ytr, use_aug, epochs=15, lr=0.002):
    """Train the Ch080 PosterCNN, optionally augmenting each batch."""
    torch.manual_seed(RANDOM_SEED)
    model = PosterCNN()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(Xtr)
    yt = torch.tensor(ytr, dtype=torch.long)
    n = len(Xt)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(epoch))
        for start in range(0, n, 64):
            idx = perm[start:start + 64]
            xb = Xt[idx]
            if use_aug:
                xb = TRAIN_AUG(xb)          # on-the-fly, different every epoch
            opt.zero_grad()
            loss_fn(model(xb), yt[idx]).backward()
            opt.step()
    return model


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X)).argmax(1)
    return (pred == torch.tensor(y, dtype=torch.long)).float().mean().item()


def main():
    X, y = build_poster_dataset()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)

    # The "field" test set: the same test posters, perturbed to mimic
    # real-world variation the clean training set never showed.
    Xte_field = perturb_to_field(Xte, seed=0)

    print(f"{len(ytr)} clean training posters, {len(yte)} test posters")
    print(f"{'model':22s} {'clean test':>12s} {'field test':>12s}")
    for use_aug in [False, True]:
        model = train_poster_cnn(Xtr, ytr, use_aug=use_aug)
        clean = accuracy(model, Xte, yte)
        field = accuracy(model, Xte_field, yte)
        label = "with augmentation" if use_aug else "no augmentation"
        print(f"{label:22s} {clean:>12.3f} {field:>12.3f}")


if __name__ == "__main__":
    main()
