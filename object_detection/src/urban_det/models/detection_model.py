"""Top-level DetectionModel: composes backbone + encoder + decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from urban_det.models.backbones import build_backbone
from urban_det.models.heads import (
    ChannelMapper,
    DINOTransformer,
    HybridEncoder,
    RTDETRDecoder,
)


def build_encoder(cfg: DictConfig, in_channels: list[int]) -> nn.Module:
    t = cfg.type
    if t == "HybridEncoder":
        return HybridEncoder(
            in_channels=in_channels,
            hidden_dim=cfg.hidden_dim,
            use_encoder_idx=list(cfg.get("use_encoder_idx", [len(in_channels) - 1])),
            num_encoder_layers=cfg.get("num_encoder_layers", 1),
            nhead=cfg.get("nhead", 8),
            dim_feedforward=cfg.get("dim_feedforward", 1024),
            dropout=cfg.get("dropout", 0.0),
            enc_act=cfg.get("enc_act", "gelu"),
            expansion=cfg.get("expansion", 1.0),
            depth_mult=cfg.get("depth_mult", 1.0),
        )
    if t == "ChannelMapper":
        return ChannelMapper(
            in_channels=in_channels,
            out_channels=cfg.out_channels,
            kernel_size=cfg.get("kernel_size", 1),
            num_outs=cfg.get("num_outs", None),
        )
    raise ValueError(f"Unknown encoder type '{t}'")


def build_decoder(cfg: DictConfig, num_classes: int, encoder_channels: list[int]) -> nn.Module:
    t = cfg.type
    hidden = cfg.hidden_dim
    if t == "RTDETRDecoder":
        return RTDETRDecoder(
            num_classes=num_classes,
            hidden_dim=hidden,
            num_queries=cfg.get("num_queries", 300),
            num_decoder_layers=cfg.get("num_decoder_layers", 6),
            nhead=cfg.get("nhead", 8),
            dim_feedforward=cfg.get("dim_feedforward", 1024),
            dropout=cfg.get("dropout", 0.0),
            num_denoising=cfg.get("num_denoising", 100),
            label_noise_ratio=cfg.get("label_noise_ratio", 0.5),
            box_noise_scale=cfg.get("box_noise_scale", 1.0),
            eval_spatial_size=cfg.get("eval_spatial_size", None),
            eval_idx=cfg.get("eval_idx", -1),
        )
    if t == "DINOTransformer":
        return DINOTransformer(
            num_classes=num_classes,
            hidden_dim=hidden,
            num_queries=cfg.get("num_queries", 900),
            num_encoder_layers=cfg.get("num_encoder_layers", 6),
            num_decoder_layers=cfg.get("num_decoder_layers", 6),
            nhead=cfg.get("nhead", 8),
            dim_feedforward=cfg.get("dim_feedforward", 2048),
            dropout=cfg.get("dropout", 0.0),
            num_denoising=cfg.get("num_denoising", 100),
            label_noise_ratio=cfg.get("label_noise_ratio", 0.5),
            box_noise_scale=cfg.get("box_noise_scale", 1.0),
            num_select=cfg.get("num_select", 300),
        )
    raise ValueError(f"Unknown decoder type '{t}'")


class DetectionModel(nn.Module):
    """
    Compose backbone → encoder → decoder into a single detection model.

    Forward:
      images: (B, 3, H, W)
      targets: list[dict] with keys "labels", "boxes"  (training only)
    Returns dict from decoder (see rtdetr_head.py / dino_head.py).
    """

    def __init__(self, cfg: DictConfig, num_classes: int):
        super().__init__()
        self.backbone = build_backbone(cfg.backbone)

        # Some model configs use encoder (RT-DETR), others use neck (DINO)
        enc_cfg = cfg.get("encoder") or cfg.get("neck")
        self.encoder = build_encoder(enc_cfg, self.backbone.out_channels)

        dec_cfg = cfg.get("decoder") or cfg.get("transformer")
        enc_out_channels = getattr(self.encoder, "out_channels", [enc_cfg.get("out_channels", 256)])
        self.decoder = build_decoder(dec_cfg, num_classes, enc_out_channels)

    def forward(
        self,
        images: Tensor,
        targets: list[dict] | None = None,
    ) -> dict[str, Tensor]:
        feats = self.backbone(images)
        enc_feats = self.encoder(feats)
        # encoder may return list or single tensor
        if isinstance(enc_feats, Tensor):
            enc_feats = [enc_feats]
        return self.decoder(enc_feats, targets=targets)
