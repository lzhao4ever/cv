"""Detection metrics wrapping pycocotools for mAP computation."""

from __future__ import annotations

import json
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float = 0.5) -> Tensor:
    """Pure-torch NMS (fallback when torchvision unavailable)."""
    try:
        import torchvision
        return torchvision.ops.nms(boxes, scores, iou_threshold)
    except ImportError:
        pass
    # Fallback: simple greedy NMS
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest_boxes = boxes[order[1:]]
        curr_box = boxes[i].unsqueeze(0).expand_as(rest_boxes)
        ix1 = torch.max(curr_box[:, 0], rest_boxes[:, 0])
        iy1 = torch.max(curr_box[:, 1], rest_boxes[:, 1])
        ix2 = torch.min(curr_box[:, 2], rest_boxes[:, 2])
        iy2 = torch.min(curr_box[:, 3], rest_boxes[:, 3])
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        a1 = (curr_box[:, 2] - curr_box[:, 0]) * (curr_box[:, 3] - curr_box[:, 1])
        a2 = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        iou = inter / (a1 + a2 - inter).clamp(min=1e-6)
        order = order[1:][iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.int64)


def decode_predictions(
    outputs: dict[str, Tensor],
    image_ids: list[int],
    img_size: tuple[int, int],
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Convert raw decoder output to COCO-format result dicts."""
    logits = outputs["pred_logits"]   # (B, Q, C)
    boxes = outputs["pred_boxes"]     # (B, Q, 4) cx,cy,w,h normalized

    H, W = img_size
    results = []

    for b in range(logits.shape[0]):
        scores_all = logits[b].sigmoid()  # (Q, C)
        scores, labels = scores_all.max(dim=-1)  # (Q,)

        mask = scores > conf_threshold
        scores = scores[mask]
        labels = labels[mask]
        b_boxes = boxes[b][mask]

        if len(scores) == 0:
            continue

        # cx,cy,w,h → x1,y1,x2,y2 in pixels
        cx = b_boxes[:, 0] * W
        cy = b_boxes[:, 1] * H
        bw = b_boxes[:, 2] * W
        bh = b_boxes[:, 3] * H
        xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1)

        keep = nms(xyxy, scores, iou_threshold)
        xyxy = xyxy[keep]
        scores = scores[keep]
        labels = labels[keep]

        # COCO format: x1,y1,w,h
        xywh = torch.cat([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]], dim=1)

        for j in range(len(scores)):
            results.append({
                "image_id": image_ids[b],
                "category_id": labels[j].item() + 1,  # 1-indexed for COCO
                "bbox": xywh[j].tolist(),
                "score": scores[j].item(),
            })

    return results


class COCOMetrics:
    """Accumulate predictions across batches then compute COCO mAP."""

    def __init__(self, coco_gt: COCO):
        self.coco_gt = coco_gt
        self._results: list[dict] = []

    def reset(self) -> None:
        self._results.clear()

    def update(self, predictions: list[dict[str, Any]]) -> None:
        self._results.extend(predictions)

    def compute(self) -> dict[str, float]:
        if not self._results:
            return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0, "mAP_s": 0.0,
                    "mAP_m": 0.0, "mAP_l": 0.0}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self._results, f)
            res_path = f.name

        coco_dt = self.coco_gt.loadRes(res_path)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        buf = StringIO()
        with redirect_stdout(buf):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        s = coco_eval.stats
        return {
            "mAP":    float(s[0]),
            "mAP50":  float(s[1]),
            "mAP75":  float(s[2]),
            "mAP_s":  float(s[3]),
            "mAP_m":  float(s[4]),
            "mAP_l":  float(s[5]),
        }
