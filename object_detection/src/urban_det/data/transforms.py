"""Detection augmentation pipeline.

Training uses: Mosaic → MixUp → RandomAffine → HSV → Flip
Validation uses: LetterBox only (preserves aspect ratio).

Boxes are stored as [cx, cy, w, h] normalized to [0, 1].
"""

from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    stride: int = 32,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize + pad image to new_shape while preserving aspect ratio."""
    h, w = img.shape[:2]
    nh, nw = new_shape
    ratio = min(nh / h, nw / w)
    new_unpad = (round(w * ratio), round(h * ratio))
    dw = (nw - new_unpad[0]) / 2
    dh = (nh - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def random_hsv(img: np.ndarray, h: float, s: float, v: float) -> np.ndarray:
    r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    lut_h = np.arange(256, dtype=np.uint8)
    lut_s = np.clip(np.arange(256) * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(np.arange(256) * r[2], 0, 255).astype(np.uint8)
    hue = cv2.LUT(hue, np.roll(lut_h, int(r[0] * 180)))
    img_hsv = cv2.merge((hue, cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def xywhn2xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    """Normalized [cx,cy,w,h] → pixel [x1,y1,x2,y2]."""
    xy = boxes.copy()
    xy[:, 0] = w * (boxes[:, 0] - boxes[:, 2] / 2)
    xy[:, 1] = h * (boxes[:, 1] - boxes[:, 3] / 2)
    xy[:, 2] = w * (boxes[:, 0] + boxes[:, 2] / 2)
    xy[:, 3] = h * (boxes[:, 1] + boxes[:, 3] / 2)
    return xy


def xyxy2xywhn(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    """Pixel [x1,y1,x2,y2] → normalized [cx,cy,w,h]."""
    xy = boxes.copy().astype(float)
    xy[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / w
    xy[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / h
    xy[:, 2] = (boxes[:, 2] - boxes[:, 0]) / w
    xy[:, 3] = (boxes[:, 3] - boxes[:, 1]) / h
    return xy


class MosaicTransform:
    """4-image mosaic: tiles 4 samples into a 2×2 grid."""

    def __init__(self, img_size: int, border: int = -320):
        self.img_size = img_size
        self.border = border  # negative = crop from center

    def __call__(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        assert len(samples) == 4
        s = self.img_size
        yc = random.randint(s // 4, 3 * s // 4)
        xc = random.randint(s // 4, 3 * s // 4)
        canvas = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
        all_labels, all_boxes = [], []

        placements = [
            (0, 0, xc, yc),        # top-left
            (xc, 0, 2 * s, yc),    # top-right
            (0, yc, xc, 2 * s),    # bottom-left
            (xc, yc, 2 * s, 2 * s),# bottom-right
        ]
        for i, (sample, (x1, y1, x2, y2)) in enumerate(zip(samples, placements)):
            img = sample["image"]
            h0, w0 = img.shape[:2]
            pw, ph = x2 - x1, y2 - y1
            img_resized = cv2.resize(img, (pw, ph))
            canvas[y1:y2, x1:x2] = img_resized

            if len(sample["boxes"]):
                boxes = sample["boxes"].copy()
                # scale normalized → canvas pixel coords
                bx = boxes[:, 0] * pw + x1
                by = boxes[:, 1] * ph + y1
                bw = boxes[:, 2] * pw
                bh = boxes[:, 3] * ph
                all_boxes.append(np.stack([bx, by, bw, bh], axis=1))
                all_labels.append(sample["labels"])

        # Crop to [s, s]
        canvas = canvas[s // 2: s // 2 + s, s // 2: s // 2 + s]
        offset_x, offset_y = s // 2, s // 2

        boxes_out = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4))
        labels_out = np.concatenate(all_labels) if all_labels else np.zeros(0, dtype=np.int64)

        if len(boxes_out):
            boxes_out[:, 0] -= offset_x
            boxes_out[:, 1] -= offset_y
            # clamp to canvas
            x1c = np.clip(boxes_out[:, 0] - boxes_out[:, 2] / 2, 0, s)
            y1c = np.clip(boxes_out[:, 1] - boxes_out[:, 3] / 2, 0, s)
            x2c = np.clip(boxes_out[:, 0] + boxes_out[:, 2] / 2, 0, s)
            y2c = np.clip(boxes_out[:, 1] + boxes_out[:, 3] / 2, 0, s)
            keep = (x2c - x1c > 2) & (y2c - y1c > 2)
            boxes_out = np.stack([
                (x1c + x2c) / 2 / s,
                (y1c + y2c) / 2 / s,
                (x2c - x1c) / s,
                (y2c - y1c) / s,
            ], axis=1)[keep]
            labels_out = labels_out[keep]

        return {"image": canvas, "boxes": boxes_out, "labels": labels_out}


class DetectionTransform:
    """Composable per-sample transform for train/val."""

    def __init__(self, img_size: int, augment: bool, aug_cfg: dict):
        self.img_size = img_size
        self.augment = augment
        self.aug_cfg = aug_cfg

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        img = sample["image"]
        boxes = sample.get("boxes", np.zeros((0, 4)))
        labels = sample.get("labels", np.zeros(0, dtype=np.int64))

        img, ratio, pad = letterbox(img, (self.img_size, self.img_size))

        if self.augment:
            cfg = self.aug_cfg
            img = random_hsv(img, cfg.get("hsv_h", 0.015),
                             cfg.get("hsv_s", 0.7), cfg.get("hsv_v", 0.4))
            if random.random() < cfg.get("fliplr", 0.5):
                img = np.fliplr(img)
                if len(boxes):
                    boxes[:, 0] = 1 - boxes[:, 0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB, HWC → CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0

        return {"image": img, "boxes": boxes.astype(np.float32), "labels": labels}
