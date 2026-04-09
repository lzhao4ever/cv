"""Tests for detection losses."""

from __future__ import annotations

import pytest
import torch

from urban_det.training.losses import (
    DETRCriterion,
    HungarianMatcher,
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)


def test_box_conversion():
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    xyxy = box_cxcywh_to_xyxy(boxes)
    assert torch.allclose(xyxy, torch.tensor([[0.3, 0.3, 0.7, 0.7]]), atol=1e-5)


def test_giou_self():
    boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.3, 0.3, 0.8, 0.8]])
    giou = generalized_box_iou(boxes, boxes)
    # GIoU with itself should be 1.0
    assert torch.allclose(giou.diag(), torch.ones(2), atol=1e-5)


def test_hungarian_matcher():
    matcher = HungarianMatcher()
    B, Q, C = 2, 10, 80
    logits = torch.zeros(B, Q, C)
    boxes = torch.rand(B, Q, 4).clamp(0.05, 0.95)
    targets = [
        {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "labels": torch.tensor([3])},
        {"boxes": torch.tensor([[0.3, 0.4, 0.1, 0.15], [0.7, 0.6, 0.3, 0.2]]),
         "labels": torch.tensor([0, 5])},
    ]
    indices = matcher(logits, boxes, targets)
    assert len(indices) == B
    # Each matched prediction index should be within [0, Q)
    for pred_idx, tgt_idx in indices:
        assert (pred_idx < Q).all()


def test_criterion_forward(fake_batch):
    criterion = DETRCriterion(num_classes=80)
    B = 2
    Q = 10
    outputs = {
        "pred_logits": torch.zeros(B, Q, 80),
        "pred_boxes": torch.rand(B, Q, 4).clamp(0.05, 0.95),
    }
    targets = fake_batch["targets"]
    losses = criterion(outputs, targets)
    assert "total" in losses
    assert losses["total"].item() >= 0.0
    assert "loss_ce" in losses
    assert "loss_bbox" in losses
    assert "loss_giou" in losses


def test_criterion_backward(fake_batch):
    criterion = DETRCriterion(num_classes=80)
    B = 2
    Q = 10
    logits = torch.zeros(B, Q, 80, requires_grad=True)
    boxes = torch.rand(B, Q, 4).clamp(0.05, 0.95).requires_grad_(True)
    outputs = {"pred_logits": logits, "pred_boxes": boxes}
    losses = criterion(outputs, fake_batch["targets"])
    losses["total"].backward()
    assert logits.grad is not None
