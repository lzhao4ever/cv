"""Tests for model architecture components."""

from __future__ import annotations

import pytest
import torch

from urban_det.models import DetectionModel
from urban_det.models.backbones import build_backbone
from urban_det.models.heads import HybridEncoder


def test_backbone_output_channels(minimal_cfg):
    bb = build_backbone(minimal_cfg.model.backbone)
    assert len(bb.out_channels) == 3  # out_indices=[1,2,3]
    assert bb.out_channels == [512, 1024, 2048]


def test_backbone_forward_shapes(minimal_cfg, device):
    bb = build_backbone(minimal_cfg.model.backbone).to(device)
    x = torch.rand(1, 3, 640, 640, device=device)
    feats = bb(x)
    assert len(feats) == 3
    # Strides 8, 16, 32 → spatial dims 80, 40, 20
    expected_hw = [(80, 80), (40, 40), (20, 20)]
    for f, (h, w) in zip(feats, expected_hw):
        assert f.shape[2] == h and f.shape[3] == w, f"Got {f.shape}"


def test_hybrid_encoder_output(minimal_cfg, device):
    enc_cfg = minimal_cfg.model.encoder
    in_channels = [512, 1024, 2048]
    encoder = HybridEncoder(
        in_channels=in_channels,
        hidden_dim=enc_cfg.hidden_dim,
        use_encoder_idx=list(enc_cfg.use_encoder_idx),
        num_encoder_layers=1,
        nhead=enc_cfg.nhead,
        dim_feedforward=enc_cfg.dim_feedforward,
        expansion=enc_cfg.expansion,
        depth_mult=enc_cfg.depth_mult,
    ).to(device)

    feats = [
        torch.rand(1, 512, 80, 80, device=device),
        torch.rand(1, 1024, 40, 40, device=device),
        torch.rand(1, 2048, 20, 20, device=device),
    ]
    outs = encoder(feats)
    assert len(outs) == 3
    for out in outs:
        assert out.shape[1] == enc_cfg.hidden_dim


def test_detection_model_forward(minimal_cfg, device):
    model = DetectionModel(minimal_cfg.model, num_classes=80).to(device)
    model.eval()
    x = torch.rand(2, 3, 640, 640, device=device)
    with torch.no_grad():
        out = model(x)
    assert "pred_logits" in out
    assert "pred_boxes" in out
    B, Q, C = out["pred_logits"].shape
    assert B == 2
    assert C == 80
    assert out["pred_boxes"].shape == (B, Q, 4)
    # Boxes should be in [0, 1]
    assert out["pred_boxes"].min() >= 0.0
    assert out["pred_boxes"].max() <= 1.0


def test_detection_model_train_forward(minimal_cfg, fake_batch, device):
    model = DetectionModel(minimal_cfg.model, num_classes=80).to(device)
    model.train()
    images = fake_batch["images"].to(device)
    targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)}
               for t in fake_batch["targets"]]
    out = model(images, targets=targets)
    assert "pred_logits" in out
    assert "pred_boxes" in out
