"""SegFormer decode head (All-MLP decoder from Xie et al., 2021)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, H*W, C) → linear → (B, H*W, out) → (B, out, H, W)
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x


class SegFormerHead(nn.Module):
    """
    Lightweight All-MLP segmentation head.

    Args:
        in_channels: list of channel counts from each backbone stage (4 stages).
        embed_dim: projection dimension shared by all MLP branches.
        num_classes: number of output segmentation classes.
        dropout: dropout probability before the final conv.
    """

    def __init__(
        self,
        in_channels: list[int],
        embed_dim: int = 768,
        num_classes: int = 19,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear_layers = nn.ModuleList(
            [MLP(c, embed_dim) for c in in_channels]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 4 feature maps of shape (B, C_i, H_i, W_i).
                      H_i/W_i are in decreasing resolution order (stage 0 = finest).
        Returns:
            logits of shape (B, num_classes, H_0, W_0).
        """
        target_h, target_w = features[0].shape[-2:]

        projected = []
        for feat, mlp in zip(features, self.linear_layers):
            out = mlp(feat)
            if out.shape[-2:] != (target_h, target_w):
                out = F.interpolate(out, size=(target_h, target_w), mode="bilinear", align_corners=False)
            projected.append(out)

        x = self.fuse(torch.cat(projected, dim=1))
        x = self.dropout(x)
        return self.classifier(x)
