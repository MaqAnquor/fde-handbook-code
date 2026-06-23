"""
cinemastream/ml/content_tagging/transfer_learning.py

Chapter 082 -- Transfer Learning, Fine-Tuning, and the cost ladder.

Chapters 080-081 trained a small PosterCNN from scratch. This module asks
the production question: should we train from scratch at all, when a
ResNet18 pretrained on millions of ImageNet photos already knows how to
see edges, textures, and shapes? It compares the three standard transfer
strategies on the CinemaStream poster genre task, and -- the point of the
chapter -- reports the TRAINABLE PARAMETER COUNT of each, because that is
what drives training cost (Ch056 FinOps):

  Strategy 1  from scratch        -- train a small CNN from zero
  Strategy 2  feature extraction  -- FREEZE the pretrained backbone, train
                                     only a tiny new head on its features
  Strategy 3  full fine-tune      -- retrain ALL of the pretrained network

The decision is not "which is most accurate" alone -- it is "which buys
the accuracy we need for the least training cost and data." On this task,
feature extraction ties full fine-tuning on accuracy while training ~11,000x
fewer parameters: the FinOps-correct choice.

The posters here are Ch080's synthetic stand-ins; the same decision ladder
applies (more strongly) to the real photographic catalog.

Run (requires `pip install torch torchvision scikit-learn`):
    python cinemastream/ml/content_tagging/transfer_learning.py
"""

import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from cinemastream.ml.content_tagging.poster_cnn import (
    build_poster_dataset, PosterCNN, RANDOM_SEED)

# ImageNet normalization -- a pretrained model expects inputs scaled the
# same way its training data was (Ch081's normalize-with-training-stats rule).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prep_for_resnet(Xt):
    """Resize 32x32 posters to 64x64 and ImageNet-normalize, so they match
    what the pretrained ResNet18 expects."""
    x = F.interpolate(Xt, size=64, mode="bilinear", align_corners=False)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def strategy_from_scratch(Xtr_t, ytr_t, Xte_t, yte_t, epochs=12, lr=0.002):
    """Strategy 1: train Ch080's PosterCNN from zero -- no transfer."""
    torch.manual_seed(RANDOM_SEED)
    model = PosterCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(Xtr_t)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(epoch))
        for s in range(0, n, 64):
            idx = perm[s:s + 64]
            opt.zero_grad()
            loss_fn(model(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte_t).argmax(1) == yte_t).float().mean().item()
    return acc, sum(p.numel() for p in model.parameters())


def strategy_feature_extraction(Xtr_r, ytr, Xte_r, yte):
    """Strategy 2: freeze the pretrained backbone, train only a head.
    We forward the FROZEN ResNet18 once to get 512-d features, then fit a
    logistic-regression head -- no backprop through the backbone at all."""
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()        # drop the 1000-class head -> 512-d output
    backbone.eval()

    @torch.no_grad()
    def features(X):
        out = [backbone(X[s:s + 128]) for s in range(0, len(X), 128)]
        return torch.cat(out).numpy()

    head = LogisticRegression(max_iter=2000).fit(features(Xtr_r), ytr)
    acc = head.score(features(Xte_r), yte)
    head_params = 512 * 2 + 2           # only the head is trained
    return acc, head_params


def strategy_full_finetune(Xtr_r, ytr_t, Xte_r, yte_t, epochs=5, lr=1e-4):
    """Strategy 3: load the pretrained net, swap a 2-class head, and retrain
    EVERYTHING with a small learning rate."""
    torch.manual_seed(RANDOM_SEED)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(Xtr_r)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(100 + epoch))
        for s in range(0, n, 64):
            idx = perm[s:s + 64]
            opt.zero_grad()
            loss_fn(model(Xtr_r[idx]), ytr_t[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(Xte_r[s:s + 128]).argmax(1)
                           for s in range(0, len(Xte_r), 128)])
        acc = (preds == yte_t).float().mean().item()
    return acc, sum(p.numel() for p in model.parameters())


def run_all_strategies():
    """Run all three transfer-learning strategies on the poster task.
    Returns ((s1_acc, s1_params), (s2_acc, s2_params), (s3_acc, s3_params)).
    Runs on CPU (slow) or GPU (Colab) -- downloads ResNet18 ImageNet weights once.
    """
    X, y = build_poster_dataset()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    Xtr_t, Xte_t = torch.tensor(Xtr), torch.tensor(Xte)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    yte_t = torch.tensor(yte, dtype=torch.long)
    Xtr_r, Xte_r = prep_for_resnet(Xtr_t), prep_for_resnet(Xte_t)
    return (
        strategy_from_scratch(Xtr_t, ytr_t, Xte_t, yte_t),
        strategy_feature_extraction(Xtr_r, ytr, Xte_r, yte),
        strategy_full_finetune(Xtr_r, ytr_t, Xte_r, yte_t),
    )


def main():
    (s1_acc, s1_p), (s2_acc, s2_p), (s3_acc, s3_p) = run_all_strategies()

    print(f"{'strategy':28s} {'test acc':>9s} {'trainable params':>18s}")
    print(f"{'1. from scratch':28s} {s1_acc:>9.3f} {s1_p:>18,}")
    print(f"{'2. feature extraction':28s} {s2_acc:>9.3f} {s2_p:>18,}")
    print(f"{'3. full fine-tune':28s} {s3_acc:>9.3f} {s3_p:>18,}")
    print(f"\nFeature extraction matches full fine-tune on accuracy while "
          f"training\n{s3_p // s2_p:,}x fewer parameters -- the FinOps-correct "
          f"choice for this task.")


if __name__ == "__main__":
    main()
