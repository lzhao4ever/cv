"""Tests for loss functions and metrics."""

import torch
import pytest

from urban_seg.training.losses import SegmentationLoss, DiceLoss
from urban_seg.training.metrics import SegmentationMetrics


def test_segmentation_loss_perfect_prediction():
    """Zero loss when logits are perfectly confident and correct."""
    num_classes = 4
    B, H, W = 2, 16, 16
    targets = torch.randint(0, num_classes, (B, H, W))
    logits = torch.zeros(B, num_classes, H, W)
    logits.scatter_(1, targets.unsqueeze(1), 10.0)  # very confident correct prediction

    criterion = SegmentationLoss(num_classes=num_classes)
    loss, parts = criterion(logits, targets)
    assert loss.item() < 0.1


def test_segmentation_loss_ignore_index():
    """Loss should not blow up when all targets are ignore_index."""
    num_classes = 4
    B, H, W = 2, 16, 16
    targets = torch.full((B, H, W), 255)
    logits = torch.rand(B, num_classes, H, W)

    criterion = SegmentationLoss(num_classes=num_classes, ignore_index=255)
    loss, _ = criterion(logits, targets)
    assert not torch.isnan(loss)


def test_metrics_miou_perfect():
    num_classes = 4
    B, H, W = 2, 16, 16
    preds = torch.randint(0, num_classes, (B, H, W))
    metrics = SegmentationMetrics(num_classes=num_classes)
    metrics.update(preds, preds)  # perfect prediction
    result = metrics.compute()
    assert abs(result["mIoU"].item() - 1.0) < 1e-4
    assert abs(result["pixel_acc"].item() - 1.0) < 1e-4


def test_metrics_ignore_index_excluded():
    num_classes = 4
    preds = torch.zeros(1, 16, 16, dtype=torch.long)
    targets = torch.full((1, 16, 16), 255, dtype=torch.long)  # all ignored
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=255)
    metrics.update(preds, targets)
    result = metrics.compute()
    # With zero valid pixels, pixel_acc should be 0 (clamped denominator)
    assert result["pixel_acc"].item() == 0.0
