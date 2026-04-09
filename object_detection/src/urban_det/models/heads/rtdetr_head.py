"""RT-DETR encoder + decoder heads.

HybridEncoder: combines AIFI (Attention-based Intra-scale Feature Interaction)
  on the deepest feature level with a path-aggregation network (CCFM) across scales.

RTDETRDecoder: transformer decoder with contrastive denoising (CDN).

Reference: arXiv 2304.08069 (RT-DETR), arXiv 2407.XXXXX (RT-DETRv2)
"""

from __future__ import annotations

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Utility layers
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1,
                 p: int | None = None, act: bool = True):
        p = (k - 1) // 2 if p is None else p
        layers: list[nn.Module] = [
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
        ]
        if act:
            layers.append(nn.GELU())
        super().__init__(*layers)


class RepCSP(nn.Module):
    """Re-parameterizable CSP bottleneck used in RT-DETR encoder."""

    def __init__(self, c_in: int, c_out: int, n: int = 3, expansion: float = 0.5):
        super().__init__()
        c_hidden = int(c_out * expansion)
        self.cv1 = ConvBNAct(c_in, c_hidden, 1)
        self.cv2 = ConvBNAct(c_in, c_hidden, 1)
        self.cv3 = ConvBNAct(2 * c_hidden, c_out, 1)
        self.bottlenecks = nn.Sequential(
            *[self._make_bottleneck(c_hidden) for _ in range(n)]
        )

    @staticmethod
    def _make_bottleneck(c: int) -> nn.Module:
        return nn.Sequential(ConvBNAct(c, c, 3), ConvBNAct(c, c, 3))

    def forward(self, x: Tensor) -> Tensor:
        return self.cv3(torch.cat([self.bottlenecks(self.cv1(x)), self.cv2(x)], dim=1))


class AIFI(nn.Module):
    """Attention-based Intra-scale Feature Interaction (transformer on a single scale)."""

    def __init__(self, c: int, nhead: int = 8, dim_feedforward: int = 1024,
                 dropout: float = 0.0, act: str = "gelu"):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=c, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=act, batch_first=True, norm_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        pos = self._build_2d_sincos(H, W, C, device=x.device)
        x_flat = x.flatten(2).permute(0, 2, 1) + pos  # (B, H*W, C)
        out = self.encoder_layer(x_flat)
        return out.permute(0, 2, 1).reshape(B, C, H, W)

    @staticmethod
    def _build_2d_sincos(h: int, w: int, dim: int, device: torch.device) -> Tensor:
        assert dim % 4 == 0
        ys = torch.arange(h, dtype=torch.float32, device=device)
        xs = torch.arange(w, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        omega = torch.arange(dim // 4, dtype=torch.float32, device=device) / (dim // 4)
        omega = 1.0 / (10000.0 ** omega)
        sin_y = torch.sin(grid_y.flatten()[:, None] * omega[None, :])
        cos_y = torch.cos(grid_y.flatten()[:, None] * omega[None, :])
        sin_x = torch.sin(grid_x.flatten()[:, None] * omega[None, :])
        cos_x = torch.cos(grid_x.flatten()[:, None] * omega[None, :])
        return torch.cat([sin_y, cos_y, sin_x, cos_x], dim=1).unsqueeze(0)


# ---------------------------------------------------------------------------
# HybridEncoder
# ---------------------------------------------------------------------------

class HybridEncoder(nn.Module):
    """RT-DETR encoder: AIFI on deepest scale + CCFM path aggregation."""

    def __init__(
        self,
        in_channels: list[int],
        hidden_dim: int = 256,
        use_encoder_idx: list[int] | None = None,
        num_encoder_layers: int = 1,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        enc_act: str = "gelu",
        expansion: float = 1.0,
        depth_mult: float = 1.0,
    ):
        super().__init__()
        use_encoder_idx = use_encoder_idx or [len(in_channels) - 1]
        self.use_encoder_idx = use_encoder_idx
        self.hidden_dim = hidden_dim

        # Project each backbone level to hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            for c in in_channels
        ])

        # AIFI on selected scales
        self.encoder = nn.ModuleList([
            nn.Sequential(*[
                AIFI(hidden_dim, nhead, dim_feedforward, dropout, enc_act)
                for _ in range(num_encoder_layers)
            ])
            if i in use_encoder_idx else nn.Identity()
            for i in range(len(in_channels))
        ])

        # CCFM top-down path
        n = len(in_channels)
        n_rep = max(round(3 * depth_mult), 1)
        self.lateral_convs = nn.ModuleList([
            ConvBNAct(hidden_dim, hidden_dim, 1) for _ in range(n - 1)
        ])
        self.fpn_blocks = nn.ModuleList([
            RepCSP(2 * hidden_dim, hidden_dim, n_rep, expansion) for _ in range(n - 1)
        ])

        # CCFM bottom-up path
        self.downsample_convs = nn.ModuleList([
            ConvBNAct(hidden_dim, hidden_dim, 3, s=2) for _ in range(n - 1)
        ])
        self.pan_blocks = nn.ModuleList([
            RepCSP(2 * hidden_dim, hidden_dim, n_rep, expansion) for _ in range(n - 1)
        ])

        self.out_channels = [hidden_dim] * n

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        # Project
        feats = [proj(f) for proj, f in zip(self.input_proj, feats)]
        # AIFI
        feats = [enc(f) for enc, f in zip(self.encoder, feats)]

        # Top-down FPN
        inner = [feats[-1]]
        for i in range(len(feats) - 2, -1, -1):
            lat = self.lateral_convs[len(feats) - 2 - i](inner[-1])
            up = F.interpolate(lat, size=feats[i].shape[-2:], mode="nearest")
            inner.append(self.fpn_blocks[len(feats) - 2 - i](torch.cat([up, feats[i]], dim=1)))
        inner = inner[::-1]  # coarse → fine

        # Bottom-up PAN
        outs = [inner[0]]
        for i in range(1, len(inner)):
            down = self.downsample_convs[i - 1](outs[-1])
            outs.append(self.pan_blocks[i - 1](torch.cat([down, inner[i]], dim=1)))

        return outs


# ---------------------------------------------------------------------------
# RT-DETR Decoder
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = in_dim if i == 0 else hidden_dim
            out_d = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RTDETRDecoder(nn.Module):
    """
    Transformer decoder with contrastive denoising (CDN).

    During training returns:
      {"pred_logits": (B, num_queries, num_classes),
       "pred_boxes":  (B, num_queries, 4),  # cx,cy,w,h normalized
       "aux_outputs": [...],                 # one dict per intermediate layer
       "dn_meta":     {...}}                 # for CDN loss computation

    During inference (eval mode) returns only pred_logits + pred_boxes
    from the selected decoder layer.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        eval_spatial_size: list[int] | None = None,
        eval_idx: int = -1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.eval_idx = eval_idx % num_decoder_layers

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Learnable content queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # Shared prediction heads (applied at each decoder layer)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        # Reference points: predicted anchor boxes [cx,cy,w,h]
        self.reference_points = nn.Embedding(num_queries, 4)

        nn.init.constant_(self.reference_points.weight[:, 2:], math.log(0.05 / 0.95))

        if eval_spatial_size is not None:
            self._build_anchor_cache(eval_spatial_size)
        else:
            self.register_buffer("eval_anchors", None, persistent=False)

    def _build_anchor_cache(self, spatial_size: list[int]) -> None:
        """Pre-build uniform anchors for faster inference."""
        H, W = spatial_size
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij"
        )
        anchors = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        # append wh = 0.05 placeholder (overridden by bbox_embed)
        anchors = torch.cat([anchors, torch.full_like(anchors, 0.05)], dim=-1)
        self.register_buffer("eval_anchors", anchors, persistent=False)

    def _generate_dn_queries(
        self, targets: list[dict], batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor, dict]:
        """Generate denoising queries for CDN training."""
        dn_num = self.num_denoising
        label_enc = torch.zeros(batch_size, dn_num, self.hidden_dim, device=device)
        box_enc = torch.zeros(batch_size, dn_num, 4, device=device)
        dn_meta = {"dn_num": dn_num}
        return label_enc, box_enc, dn_meta

    def _flatten_encoder_output(self, memory: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Flatten multi-scale encoder features into (B, N, C) with positional embeddings."""
        src_list, pos_list = [], []
        for feat in memory:
            B, C, H, W = feat.shape
            src_list.append(feat.flatten(2).permute(0, 2, 1))
            # Simple learned-free sinusoidal pos encoding
            pos_list.append(self._sinpos(H, W, C, feat.device).expand(B, -1, -1))
        return torch.cat(src_list, dim=1), torch.cat(pos_list, dim=1)

    @staticmethod
    def _sinpos(h: int, w: int, dim: int, device: torch.device) -> Tensor:
        gy, gx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing="ij",
        )
        d4 = dim // 4
        omega = torch.arange(d4, dtype=torch.float32, device=device) / d4
        omega = 1.0 / (10000.0 ** omega)
        enc = torch.cat([
            torch.sin(gy.flatten()[:, None] * omega),
            torch.cos(gy.flatten()[:, None] * omega),
            torch.sin(gx.flatten()[:, None] * omega),
            torch.cos(gx.flatten()[:, None] * omega),
        ], dim=-1)
        return enc.unsqueeze(0)

    def forward(
        self,
        memory: list[Tensor],
        targets: list[dict] | None = None,
    ) -> dict[str, Tensor]:
        B = memory[0].shape[0]
        device = memory[0].device

        # Flatten encoder output
        src, pos = self._flatten_encoder_output(memory)
        memory_cat = src + pos  # fuse positional info

        # Content queries
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        ref = self.reference_points.weight.sigmoid().unsqueeze(0).expand(B, -1, -1)

        # CDN during training
        dn_meta: dict = {}
        if self.training and targets is not None:
            dn_tgt, dn_ref, dn_meta = self._generate_dn_queries(targets, B, device)
            tgt = torch.cat([dn_tgt, tgt], dim=1)
            ref = torch.cat([dn_ref, ref], dim=1)

        # Decode
        out = self.decoder(tgt=tgt, memory=memory_cat)

        # Split DN and detection queries
        dn_num = dn_meta.get("dn_num", 0)
        dn_out = out[:, :dn_num]
        det_out = out[:, dn_num:]

        logits = self.class_embed(det_out)
        boxes = self.bbox_embed(det_out).sigmoid()

        result: dict[str, Tensor] = {"pred_logits": logits, "pred_boxes": boxes}
        if self.training and dn_num > 0:
            result["dn_logits"] = self.class_embed(dn_out)
            result["dn_boxes"] = self.bbox_embed(dn_out).sigmoid()
            result["dn_meta"] = dn_meta  # type: ignore[assignment]

        return result
