"""Combined Cross-Entropy + Dice loss for semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss averaged over classes (ignores ``ignore_index`` pixels)."""

    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # mask out ignored pixels
        valid = targets != self.ignore_index
        targets_clean = targets.clone()
        targets_clean[~valid] = 0

        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        one_hot = F.one_hot(targets_clean, self.num_classes).permute(0, 3, 1, 2).float()  # (B,C,H,W)

        # zero out ignored pixels
        mask = valid.unsqueeze(1).float()
        probs = probs * mask
        one_hot = one_hot * mask

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)
        dice = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return dice.mean()


class SegmentationLoss(nn.Module):
    """
    Weighted sum of Cross-Entropy and Dice losses.

    Args:
        num_classes: number of segmentation classes.
        ignore_index: label value to ignore (default 255).
        ce_weight: scalar weight for CE term.
        dice_weight: scalar weight for Dice term.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index)
        self.ce_w = ce_weight
        self.dice_w = dice_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total = self.ce_w * ce_loss + self.dice_w * dice_loss
        return total, {"ce": ce_loss, "dice": dice_loss}
