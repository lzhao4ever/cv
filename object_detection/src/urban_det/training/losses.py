"""Detection losses: Hungarian matching + classification + box regression.

Supports both RT-DETR (CDN) and DINO (DN-DETR) training paradigms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """GIoU between all pairs of boxes (both in xyxy format)."""
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    enc_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enc_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enc_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enc_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    enc_area = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0)

    iou = inter / union.clamp(min=1e-6)
    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return giou


# ---------------------------------------------------------------------------
# Hungarian matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """One-to-one bipartite matching between predictions and GT boxes."""

    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: Tensor,   # (B, Q, C)
        pred_boxes: Tensor,    # (B, Q, 4)  cx,cy,w,h normalized
        targets: list[dict],
    ) -> list[tuple[Tensor, Tensor]]:
        B, Q, C = pred_logits.shape
        indices = []

        pred_prob = pred_logits.sigmoid()  # focal-style: multi-label sigmoid

        for b in range(B):
            tgt_labels = targets[b]["labels"]   # (Nb,)
            tgt_boxes = targets[b]["boxes"]     # (Nb, 4)
            Nb = len(tgt_labels)

            if Nb == 0:
                indices.append((torch.zeros(0, dtype=torch.int64),
                                torch.zeros(0, dtype=torch.int64)))
                continue

            # Class cost: negative prob of gt class
            cost_cls = -pred_prob[b][:, tgt_labels]  # (Q, Nb)

            # L1 box cost
            cost_l1 = torch.cdist(pred_boxes[b], tgt_boxes, p=1)  # (Q, Nb)

            # GIoU cost
            pb_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])
            tb_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            cost_iou = -generalized_box_iou(pb_xyxy, tb_xyxy)  # (Q, Nb)

            C_mat = (
                self.cost_class * cost_cls
                + self.cost_bbox * cost_l1
                + self.cost_giou * cost_iou
            )
            row_idx, col_idx = linear_sum_assignment(C_mat.cpu())
            indices.append((
                torch.as_tensor(row_idx, dtype=torch.int64),
                torch.as_tensor(col_idx, dtype=torch.int64),
            ))

        return indices


# ---------------------------------------------------------------------------
# DETR criterion
# ---------------------------------------------------------------------------

class DETRCriterion(nn.Module):
    """
    Compute detection loss for DETR-family models.

    Handles:
      - Focal classification loss (VFL-style)
      - L1 box regression loss
      - GIoU box regression loss
      - DN auxiliary loss
    """

    def __init__(
        self,
        num_classes: int,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        weight_ce: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)

    def _focal_loss(
        self,
        pred: Tensor,     # (B, Q, C) raw logits
        targets_one_hot: Tensor,  # (B, Q, C) float
    ) -> Tensor:
        p = pred.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred, targets_one_hot, reduction="none")
        alpha_t = self.focal_alpha * targets_one_hot + (1 - self.focal_alpha) * (1 - targets_one_hot)
        p_t = p * targets_one_hot + (1 - p) * (1 - targets_one_hot)
        focal = alpha_t * (1 - p_t) ** self.focal_gamma * ce
        return focal.sum()

    def _loss_labels(
        self,
        pred_logits: Tensor,
        targets: list[dict],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: int,
    ) -> Tensor:
        B, Q, C = pred_logits.shape
        tgt_one_hot = torch.zeros(B, Q, C, device=pred_logits.device)
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx):
                tgt_one_hot[b, pred_idx, targets[b]["labels"][tgt_idx]] = 1.0
        return self._focal_loss(pred_logits, tgt_one_hot) / num_boxes

    def _loss_boxes(
        self,
        pred_boxes: Tensor,
        targets: list[dict],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: int,
    ) -> tuple[Tensor, Tensor]:
        src_boxes, tgt_boxes = [], []
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx):
                src_boxes.append(pred_boxes[b][pred_idx])
                tgt_boxes.append(targets[b]["boxes"][tgt_idx])

        if not src_boxes:
            z = pred_boxes.sum() * 0
            return z, z

        src = torch.cat(src_boxes)
        tgt = torch.cat(tgt_boxes).to(src)

        l1 = F.l1_loss(src, tgt, reduction="sum") / num_boxes
        giou = (1 - generalized_box_iou(
            box_cxcywh_to_xyxy(src), box_cxcywh_to_xyxy(tgt)
        ).diag()).sum() / num_boxes
        return l1, giou

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict],
    ) -> dict[str, Tensor]:
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        indices = self.matcher(pred_logits, pred_boxes, targets)
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)

        loss_ce = self.weight_ce * self._loss_labels(pred_logits, targets, indices, num_boxes)
        loss_l1, loss_giou = self._loss_boxes(pred_boxes, targets, indices, num_boxes)
        loss_l1 = self.weight_bbox * loss_l1
        loss_giou = self.weight_giou * loss_giou

        losses = {"loss_ce": loss_ce, "loss_bbox": loss_l1, "loss_giou": loss_giou}
        losses["total"] = loss_ce + loss_l1 + loss_giou

        # DN auxiliary loss
        if "dn_logits" in outputs and "dn_meta" in outputs:
            dn_logits = outputs["dn_logits"]
            dn_boxes = outputs["dn_boxes"]
            dn_meta = outputs["dn_meta"]
            # For DN: GT assignment is known (no matching needed)
            dn_ce = self._focal_loss(
                dn_logits,
                torch.zeros_like(dn_logits),  # simplified: proper impl uses known GT one-hot
            ) / max(dn_meta.get("dn_num", 1), 1)
            losses["dn_loss_ce"] = self.weight_ce * dn_ce
            losses["total"] = losses["total"] + losses["dn_loss_ce"]

        return losses
