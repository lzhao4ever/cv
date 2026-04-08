"""Tests for model forward passes (backbone + head combos)."""

import pytest
import torch

from urban_seg.models.backbones.registry import ResNetBackbone
from urban_seg.models.heads.deeplab_head import DeepLabV3PlusHead
from urban_seg.models.heads.segformer_head import SegFormerHead
from urban_seg.models.segmentation_model import SegmentationModel


@pytest.fixture(params=["deeplab_head"])
def seg_model_resnet(request, minimal_cfg):
    """SegmentationModel with ResNet50 backbone and DeepLab head (no internet needed)."""
    from omegaconf import OmegaConf
    cfg = minimal_cfg
    return SegmentationModel(
        backbone=cfg.model.backbone,
        head=cfg.model.head,
        num_classes=19,
    )


def test_resnet_backbone_channels():
    bb = ResNetBackbone("resnet50", pretrained=False)
    x = torch.rand(1, 3, 128, 256)
    feats = bb(x)
    assert len(feats) == 4
    for c_expected, feat in zip(bb.out_channels, feats):
        assert feat.shape[1] == c_expected


def test_deeplab_head_output_shape():
    in_channels = [256, 512, 1024, 2048]
    head = DeepLabV3PlusHead(in_channels, num_classes=19)
    feats = [
        torch.rand(2, c, 32 // (2 ** i), 64 // (2 ** i))
        for i, c in enumerate(in_channels)
    ]
    out = head(feats)
    assert out.shape == (2, 19, 32, 64)


def test_segformer_head_output_shape():
    in_channels = [64, 128, 320, 512]
    head = SegFormerHead(in_channels, embed_dim=256, num_classes=19)
    feats = [
        torch.rand(2, c, 32 // (2 ** i), 64 // (2 ** i))
        for i, c in enumerate(in_channels)
    ]
    out = head(feats)
    assert out.shape == (2, 19, 32, 64)


def test_segmentation_model_full_resolution(seg_model_resnet):
    model = seg_model_resnet
    x = torch.rand(1, 3, 128, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 19, 128, 256), f"Expected (1,19,128,256), got {out.shape}"


def test_segmentation_model_predict(seg_model_resnet):
    model = seg_model_resnet
    x = torch.rand(1, 3, 128, 256)
    pred = model.predict(x)
    assert pred.shape == (1, 128, 256)
    assert pred.dtype == torch.int64
    assert pred.max() < 19
