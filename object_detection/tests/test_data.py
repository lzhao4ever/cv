"""Tests for data transforms and collate."""

from __future__ import annotations

import numpy as np
import pytest

from urban_det.data.coco import detection_collate
from urban_det.data.transforms import (
    DetectionTransform,
    MosaicTransform,
    letterbox,
    xyxy2xywhn,
    xywhn2xyxy,
)


def make_sample(h: int = 480, w: int = 640, n_boxes: int = 3) -> dict:
    return {
        "image": np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
        "boxes": np.random.uniform(0.1, 0.5, (n_boxes, 4)).astype(np.float32),
        "labels": np.array([0, 1, 2][:n_boxes], dtype=np.int64),
    }


def test_letterbox_shape():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out, ratio, pad = letterbox(img, (640, 640))
    assert out.shape == (640, 640, 3)


def test_box_roundtrip():
    boxes_n = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
    xyxy = xywhn2xyxy(boxes_n, w=640, h=480)
    back = xyxy2xywhn(xyxy, w=640, h=480)
    assert np.allclose(boxes_n, back, atol=1e-5)


def test_detection_transform_output():
    tf = DetectionTransform(img_size=640, augment=True, aug_cfg={"fliplr": 0.0, "hsv_h": 0.0,
                                                                   "hsv_s": 0.0, "hsv_v": 0.0})
    sample = make_sample()
    out = tf(sample)
    assert out["image"].shape == (3, 640, 640)
    assert out["image"].dtype == np.float32
    assert out["image"].max() <= 1.0
    assert out["boxes"].shape == (3, 4)


def test_mosaic_transform():
    mt = MosaicTransform(img_size=640)
    samples = [make_sample(480, 640, 3) for _ in range(4)]
    out = mt(samples)
    assert out["image"].shape == (640, 640, 3)
    assert out["boxes"].ndim == 2


def test_collate_fn():
    import torch
    samples = [
        {"image": np.zeros((3, 640, 640), dtype=np.float32),
         "boxes": np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
         "labels": np.array([0], dtype=np.int64),
         "image_id": 1},
        {"image": np.zeros((3, 640, 640), dtype=np.float32),
         "boxes": np.zeros((0, 4), dtype=np.float32),
         "labels": np.array([], dtype=np.int64),
         "image_id": 2},
    ]
    batch = detection_collate(samples)
    assert batch["images"].shape == (2, 3, 640, 640)
    assert len(batch["targets"]) == 2
    assert batch["targets"][1]["boxes"].shape[0] == 0
