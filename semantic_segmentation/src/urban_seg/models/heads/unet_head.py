"""UNet decode head — progressive upsampling with skip connections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _DecodeBlock(nn.Module):
    """Bilinear upsample + skip concatenation + two 3×3 conv blocks."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBnRelu(in_ch + skip_ch, out_ch),
            _ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetHead(nn.Module):
    """
    UNet-style decode head with progressive upsampling and skip connections.

    Fuses all four backbone stages (coarsest → finest) via skip connections,
    outputting logits at 1/4 input resolution (matching the finest backbone
    feature). SegmentationModel upsamples to full resolution.

    Args:
        in_channels: list[int] of length 4 from backbone (finest → coarsest).
        decoder_channels: output channels for each of the 4 decode blocks.
        num_classes: number of segmentation output classes.
        dropout: spatial dropout applied before the classifier.
    """

    def __init__(
        self,
        in_channels: list[int],
        decoder_channels: list[int] = (256, 128, 64, 32),
        num_classes: int = 19,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d0, d1, d2, d3 = decoder_channels

        self.bottleneck = nn.Sequential(
            _ConvBnRelu(in_channels[3], d0),
            _ConvBnRelu(d0, d0),
        )
        self.decode2 = _DecodeBlock(d0, in_channels[2], d1)
        self.decode1 = _DecodeBlock(d1, in_channels[1], d2)
        self.decode0 = _DecodeBlock(d2, in_channels[0], d3)

        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(d3, num_classes, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # features[0]: H/4 (finest) … features[3]: H/32 (coarsest)
        x = self.bottleneck(features[3])         # H/32, d0 ch
        x = self.decode2(x, features[2])          # H/16, d1 ch
        x = self.decode1(x, features[1])          # H/8,  d2 ch
        x = self.decode0(x, features[0])          # H/4,  d3 ch
        return self.classifier(self.dropout(x))   # H/4,  num_classes
