"""Segmentation metrics: mIoU, per-class IoU, pixel accuracy."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchmetrics import Metric
from torchmetrics.segmentation import MeanIoU


class SegmentationMetrics(nn.Module):
    """
    Wraps torchmetrics for distributed-safe accumulation.

    Returns a dict with ``mIoU``, ``pixel_acc``, and per-class IoU values.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.miou = MeanIoU(num_classes=num_classes, per_class=True)
        # confusion matrix for pixel accuracy
        self.register_buffer("_tp", torch.zeros((), dtype=torch.long))
        self.register_buffer("_total", torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------------
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Args:
            preds: (B, H, W) int64 predicted class indices.
            targets: (B, H, W) int64 ground-truth class indices.
        """
        valid = targets != self.ignore_index
        self.miou.update(preds[valid].unsqueeze(0), targets[valid].unsqueeze(0))
        self._tp += (preds[valid] == targets[valid]).sum()
        self._total += valid.sum()

    def compute(self) -> dict[str, torch.Tensor]:
        per_class_iou = self.miou.compute()  # (num_classes,)
        pixel_acc = self._tp.float() / self._total.float().clamp(min=1)
        result = {
            "mIoU": per_class_iou.mean(),
            "pixel_acc": pixel_acc,
        }
        for i, v in enumerate(per_class_iou):
            result[f"iou_cls{i:02d}"] = v
        return result

    def reset(self) -> None:
        self.miou.reset()
        self._tp.zero_()
        self._total.zero_()
