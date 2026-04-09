"""Backbone registry.

Each backbone exposes:
  .out_channels: list[int]  — channels at each requested output stage
  .forward(x) -> list[Tensor]  — feature maps, coarse to fine

Supported:
  resnet50, resnet101 (timm)
  convnextv2_base, convnextv2_large (timm)
  internimage_t, internimage_b (optional, requires dcnv3)
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig

# Channels at each stage for standard ResNet family
_RESNET_CHANNELS = {
    "resnet50":  [256, 512, 1024, 2048],
    "resnet101": [256, 512, 1024, 2048],
    "resnet50d": [256, 512, 1024, 2048],
}

_CONVNEXT_CHANNELS = {
    "convnextv2_base":  [128, 256, 512, 1024],
    "convnextv2_large": [192, 384, 768, 1536],
    "convnext_base":    [128, 256, 512, 1024],
    "convnext_large":   [192, 384, 768, 1536],
}

_ALL_CHANNELS = {**_RESNET_CHANNELS, **_CONVNEXT_CHANNELS}


class TimmBackbone(nn.Module):
    """Thin wrapper around a timm feature extractor."""

    def __init__(self, name: str, out_indices: tuple[int, ...], pretrained: bool, freeze_at: int):
        super().__init__()
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        all_ch = _ALL_CHANNELS[name]
        self.out_channels: list[int] = [all_ch[i] for i in out_indices]
        self._freeze_stages(freeze_at)

    def _freeze_stages(self, freeze_at: int) -> None:
        """Freeze stem + first `freeze_at` stages."""
        if freeze_at <= 0:
            return
        # timm feature extractors expose .feature_info
        for i, stage in enumerate(self.model.children()):
            if i <= freeze_at:
                for p in stage.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.model(x)


def build_backbone(cfg: DictConfig) -> TimmBackbone:
    name = cfg.type
    if name not in _ALL_CHANNELS:
        raise ValueError(f"Unknown backbone '{name}'. Choose from: {sorted(_ALL_CHANNELS)}")
    out_indices = tuple(cfg.get("out_indices", [1, 2, 3]))
    return TimmBackbone(
        name=name,
        out_indices=out_indices,
        pretrained=cfg.get("pretrained", True),
        freeze_at=cfg.get("freeze_at", 0),
    )
