"""Spatially weighted Dice + cross-entropy (Sec. 3.8.2, Eqs. 3-7).

With ``weight=None`` the loss reduces to the standard (unweighted) Dice + CE used for
Stage I and for the "Baseline" / "+Feat" ablation variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedDiceCELoss(nn.Module):
    def __init__(self, eps: float = 1e-5, dice_weight: float = 1.0, ce_weight: float = 1.0):
        super().__init__()
        self.eps = eps
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        logits: (B, 2, X, Y, Z) raw network output
        target: (B, 1, X, Y, Z) binary ground truth
        weight: (B, 1, X, Y, Z) voxel weights W(p) or None
        """
        logits = logits.float()
        y = (target > 0.5).float()
        w = torch.ones_like(y) if weight is None else weight.float()
        prob = torch.softmax(logits, dim=1)[:, 1:2]  # foreground probability  \hat{y}_p

        dims = tuple(range(2, logits.ndim))
        inter = (w * prob * y).sum(dims)              # Eq. 3
        union = (w * (prob + y)).sum(dims)            # Eq. 4
        dice = 1.0 - (2.0 * inter + self.eps) / (union + self.eps)  # Eq. 5
        dice = dice.mean()

        ce = F.cross_entropy(logits, y[:, 0].long(), reduction="none")  # binary CE via 2-class softmax
        ce = (w[:, 0] * ce).mean()                                      # Eq. 6 (1/|Omega| sum)
        return self.dice_weight * dice + self.ce_weight * ce             # Eq. 7
