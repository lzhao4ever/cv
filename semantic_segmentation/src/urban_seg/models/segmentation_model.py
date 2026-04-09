"""Full segmentation model: backbone + decode head + output upsampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .backbones import build_backbone
from .heads import DeepLabV3PlusHead, SegFormerHead, UNetHead

_HEAD_REGISTRY = {
    "segformer_head": SegFormerHead,
    "deeplab_head": DeepLabV3PlusHead,
    "unet_head": UNetHead,
}


class SegmentationModel(nn.Module):
    """
    Composes a backbone + decode head into a single segmentation model.

    The forward pass returns logits at 1/4 input resolution; the training
    :class:`~urban_seg.training.lit_module.SegLitModule` upsamples them to
    full resolution for loss/metric computation.

    Args:
        backbone: config for :func:`~urban_seg.models.backbones.build_backbone`.
        head: config specifying head ``name`` plus constructor kwargs.
        num_classes: number of segmentation classes.
    """

    def __init__(self, backbone: DictConfig, head: DictConfig, num_classes: int) -> None:
        super().__init__()

        # ---- backbone ------------------------------------------------
        backbone_kwargs = {k: v for k, v in backbone.items() if k != "name"}
        self.backbone = build_backbone(backbone.name, **backbone_kwargs)

        # ---- decode head ---------------------------------------------
        head_name = head.name
        head_cls = _HEAD_REGISTRY[head_name]
        head_kwargs = {k: v for k, v in head.items() if k != "name"}
        self.head = head_cls(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            **head_kwargs,
        )

        self.num_classes = num_classes

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised input image.
        Returns:
            logits: (B, num_classes, H, W) — full resolution.
        """
        h, w = x.shape[-2:]
        features = self.backbone(x)
        logits = self.head(features)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns argmax class map (B, H, W)."""
        return self.forward(x).argmax(dim=1)
