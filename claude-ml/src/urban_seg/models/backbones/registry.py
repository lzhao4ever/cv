"""Backbone registry — thin wrapper around timm & HuggingFace transformers."""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerModel

# Maps backbone name → number of feature channels (per stage, low→high resolution)
_CHANNEL_MAP: dict[str, list[int]] = {
    "mit_b0": [32, 64, 160, 256],
    "mit_b1": [64, 128, 320, 512],
    "mit_b2": [64, 128, 320, 512],
    "mit_b3": [64, 128, 320, 512],
    "mit_b4": [64, 128, 320, 512],
    "mit_b5": [64, 128, 320, 512],
    "resnet50": [256, 512, 1024, 2048],
    "resnet101": [256, 512, 1024, 2048],
}

# HuggingFace model IDs for MixTransformer variants
_HF_SEGFORMER_IDS: dict[str, str] = {
    "mit_b0": "nvidia/mit-b0",
    "mit_b1": "nvidia/mit-b1",
    "mit_b2": "nvidia/mit-b2",
    "mit_b3": "nvidia/mit-b3",
    "mit_b4": "nvidia/mit-b4",
    "mit_b5": "nvidia/mit-b5",
}


class MixTransformerBackbone(nn.Module):
    """SegFormer MixTransformer backbone, returns 4 multi-scale feature maps."""

    def __init__(self, name: str, pretrained: bool = True) -> None:
        super().__init__()
        hf_id = _HF_SEGFORMER_IDS[name]
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(hf_id)
        else:
            cfg = SegformerConfig.from_pretrained(hf_id)
            self.encoder = SegformerModel(cfg)
        self.out_channels = _CHANNEL_MAP[name]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = self.encoder(pixel_values=x, output_hidden_states=True)
        # hidden_states: tuple of (B, C, H/4, W/4) … (B, C, H/32, W/32)
        return list(out.hidden_states)


class ResNetBackbone(nn.Module):
    """timm ResNet backbone returning 4 stage feature maps."""

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        output_stride: int = 32,
        replace_stride_with_dilation: list[bool] | None = None,
    ) -> None:
        super().__init__()
        extra: dict[str, Any] = {}
        if replace_stride_with_dilation:
            extra["replace_stride_with_dilation"] = replace_stride_with_dilation
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
            **extra,
        )
        self.out_channels = _CHANNEL_MAP.get(name, self.model.feature_info.channels())

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.model(x)


def build_backbone(name: str, **kwargs) -> nn.Module:
    if name in _HF_SEGFORMER_IDS:
        return MixTransformerBackbone(name, pretrained=kwargs.get("pretrained", True))
    if name.startswith("resnet"):
        return ResNetBackbone(name, **kwargs)
    raise ValueError(f"Unknown backbone: {name!r}. Available: {list(_CHANNEL_MAP)}")


def list_backbones() -> list[str]:
    return list(_CHANNEL_MAP)
