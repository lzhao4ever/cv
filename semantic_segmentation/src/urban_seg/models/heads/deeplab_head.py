"""DeepLabV3+ decode head (Chen et al., 2018)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.conv(self.pool(x))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 256, dilations: tuple[int, ...] = (6, 12, 18)) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ),
                *[ASPPConv(in_ch, out_ch, d) for d in dilations],
                ASPPPooling(in_ch, out_ch),
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(dilations) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([c(x) for c in self.convs], dim=1))


class DeepLabV3PlusHead(nn.Module):
    """
    DeepLabV3+ decoder combining ASPP on high-level features with low-level detail.

    Args:
        in_channels: list[int] of length 4 (backbone stage channels).
            Stage 0 is the low-level (finest) feature; stage 3 is the coarsest.
        aspp_dilations: dilation rates for ASPP module.
        low_level_channels: projected channels for low-level skip connection.
        num_classes: output classes.
        dropout: dropout before final conv.
    """

    def __init__(
        self,
        in_channels: list[int],
        aspp_dilations: tuple[int, ...] = (6, 12, 18),
        low_level_channels: int = 48,
        num_classes: int = 19,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        high_ch = in_channels[-1]  # coarsest stage
        low_ch = in_channels[0]    # finest stage

        self.aspp = ASPP(high_ch, 256, aspp_dilations)

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_ch, low_level_channels, 1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )

        self.decode = nn.Sequential(
            nn.Conv2d(256 + low_level_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        low_feat = features[0]   # (B, C0, H/4,  W/4)
        high_feat = features[-1]  # (B, C3, H/32, W/32)

        aspp_out = self.aspp(high_feat)
        aspp_up = F.interpolate(
            aspp_out, size=low_feat.shape[-2:], mode="bilinear", align_corners=False
        )
        low_proj = self.low_level_proj(low_feat)

        x = self.decode(torch.cat([aspp_up, low_proj], dim=1))
        return self.classifier(x)
