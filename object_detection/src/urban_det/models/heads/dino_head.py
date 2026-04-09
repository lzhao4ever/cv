"""DINO detector head components.

ChannelMapper: lateral projection + optional extra down-sample scale.
DINOTransformer: encoder-decoder with DN-DETR-style denoising and mixed query selection.

Reference: arXiv 2203.03605 (DINO)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ChannelMapper(nn.Module):
    """Project multi-scale backbone features to a uniform channel width.

    Optionally appends an extra down-sampled feature level (P6 = stride 64)
    by applying a strided convolution on the last input feature.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        kernel_size: int = 1,
        num_outs: int | None = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, kernel_size,
                          padding=(kernel_size - 1) // 2, bias=False),
                nn.GroupNorm(32, out_channels),
            )
            for c in in_channels
        ])
        num_outs = num_outs or len(in_channels)
        self.extra_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(),
            )
            for _ in range(max(0, num_outs - len(in_channels)))
        ])

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        outs = [lat(f) for lat, f in zip(self.lateral_convs, feats)]
        for extra in self.extra_convs:
            outs.append(extra(outs[-1]))
        return outs


# ---------------------------------------------------------------------------
# DAC / DN query generation
# ---------------------------------------------------------------------------

def build_dn_queries(
    targets: list[dict],
    num_denoising: int,
    num_classes: int,
    hidden_dim: int,
    label_noise_ratio: float,
    box_noise_scale: float,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, dict]:
    """Build denoising content queries, reference boxes, and attention mask."""
    B = len(targets)
    max_gt = max((len(t["labels"]) for t in targets), default=0)
    if max_gt == 0 or num_denoising == 0:
        empty = torch.zeros(B, 0, hidden_dim, device=device)
        return empty, empty, torch.zeros(0, 0, device=device, dtype=torch.bool), {}

    num_groups = max(1, num_denoising // max_gt)
    dn_num = num_groups * max_gt

    # Noised class embeddings (one-hot → perturbed)
    known_labels = torch.zeros(B, max_gt, num_classes, device=device)
    known_boxes = torch.zeros(B, max_gt, 4, device=device)
    for b, t in enumerate(targets):
        n = len(t["labels"])
        if n == 0:
            continue
        known_labels[b, :n].scatter_(1, t["labels"][:n].unsqueeze(1), 1.0)
        known_boxes[b, :n] = t["boxes"][:n]

    # Replicate for num_groups
    known_labels = known_labels.unsqueeze(1).repeat(1, num_groups, 1, 1).reshape(B, dn_num, -1)
    known_boxes = known_boxes.unsqueeze(1).repeat(1, num_groups, 1, 1).reshape(B, dn_num, 4)

    # Label noise: randomly flip some labels
    if label_noise_ratio > 0:
        noise_mask = torch.rand_like(known_labels[..., 0]) < label_noise_ratio
        random_labels = torch.randint(0, num_classes, (B, dn_num), device=device)
        rand_one_hot = torch.zeros_like(known_labels)
        rand_one_hot.scatter_(2, random_labels.unsqueeze(2), 1.0)
        known_labels = torch.where(noise_mask.unsqueeze(-1), rand_one_hot, known_labels)

    # Box noise: random offset
    if box_noise_scale > 0:
        diff = known_boxes[..., 2:4] / 2 * box_noise_scale
        noise = (torch.rand_like(known_boxes[..., :2]) * 2 - 1) * diff
        known_boxes = torch.cat([known_boxes[..., :2] + noise, known_boxes[..., 2:]], dim=-1)
        known_boxes = known_boxes.clamp(0.0, 1.0)

    # Build attn mask: DN queries cannot attend to matching queries (and vice versa)
    total = dn_num  # will be concatenated with detection queries outside
    attn_mask = torch.zeros(total, total, device=device, dtype=torch.bool)

    dn_meta = {"dn_num": dn_num, "num_groups": num_groups, "max_gt": max_gt}
    return known_labels, known_boxes, attn_mask, dn_meta


# ---------------------------------------------------------------------------
# DINO Transformer
# ---------------------------------------------------------------------------

class DINOTransformer(nn.Module):
    """
    DINO transformer encoder-decoder.

    forward() returns:
      {"pred_logits": (B, num_select, C),
       "pred_boxes":  (B, num_select, 4),
       "aux_outputs": [...],
       "dn_outputs":  [...]}   # only during training
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_queries: int = 900,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        num_select: int = 300,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_select = num_select
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        # Mixed query selection: learned content + predicted reference
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Level embedding (added to each scale's tokens)
        self.level_embed = nn.Parameter(torch.zeros(4, hidden_dim))
        nn.init.normal_(self.level_embed)

    def _encode(self, feats: list[Tensor]) -> tuple[Tensor, list[tuple[int, int]]]:
        tokens, shapes = [], []
        for i, f in enumerate(feats):
            B, C, H, W = f.shape
            shapes.append((H, W))
            t = f.flatten(2).permute(0, 2, 1)
            t = t + self.level_embed[i].unsqueeze(0).unsqueeze(0)
            tokens.append(t)
        src = torch.cat(tokens, dim=1)
        return self.encoder(src), shapes

    def _mixed_query_selection(
        self, enc_out: Tensor, shapes: list[tuple[int, int]]
    ) -> tuple[Tensor, Tensor]:
        """Select top-k encoder tokens as initial reference points."""
        scores = self.class_embed(enc_out).max(dim=-1).values  # (B, N)
        topk_idx = scores.topk(self.num_select, dim=1).indices  # (B, k)
        # Gather selected tokens as content queries
        sel = torch.gather(enc_out, 1, topk_idx.unsqueeze(-1).expand(-1, -1, enc_out.shape[-1]))
        # Predict reference boxes from selected tokens
        ref = self.bbox_embed(sel).sigmoid()
        return sel, ref

    def forward(
        self,
        feats: list[Tensor],
        targets: list[dict] | None = None,
    ) -> dict[str, Tensor]:
        B, device = feats[0].shape[0], feats[0].device

        enc_out, shapes = self._encode(feats)

        # Mixed query selection
        content_queries, ref_boxes = self._mixed_query_selection(enc_out, shapes)

        # Optionally prepend DN queries during training
        dn_meta: dict = {}
        if self.training and targets is not None and self.num_denoising > 0:
            dn_labels, dn_boxes, _, dn_meta = build_dn_queries(
                targets, self.num_denoising, self.num_classes, self.hidden_dim,
                self.label_noise_ratio, self.box_noise_scale, device,
            )
            # Use label embeddings as content queries
            dn_content = dn_labels @ self.class_embed.weight  # rough projection
            content_queries = torch.cat([dn_content, content_queries], dim=1)
            ref_boxes = torch.cat([dn_boxes, ref_boxes], dim=1)

        tgt = content_queries
        dec_out = self.decoder(tgt=tgt, memory=enc_out)

        dn_num = dn_meta.get("dn_num", 0)
        dn_dec = dec_out[:, :dn_num]
        det_dec = dec_out[:, dn_num:]

        logits = self.class_embed(det_dec[:, :self.num_select])
        boxes = self.bbox_embed(det_dec[:, :self.num_select]).sigmoid()

        out: dict[str, Tensor] = {"pred_logits": logits, "pred_boxes": boxes}
        if self.training and dn_num > 0:
            out["dn_logits"] = self.class_embed(dn_dec)
            out["dn_boxes"] = self.bbox_embed(dn_dec).sigmoid()
            out["dn_meta"] = dn_meta  # type: ignore[assignment]
        return out
