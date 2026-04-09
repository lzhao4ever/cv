"""COCO detection dataset."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class COCODetection(Dataset):
    """
    COCO-format detection dataset.

    Each item:
      image: np.ndarray (H, W, 3) BGR
      boxes: np.ndarray (N, 4) normalized [cx, cy, w, h]
      labels: np.ndarray (N,) int64  class ids (0-indexed)
      image_id: int
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable | None = None,
        mosaic_transform: Callable | None = None,
        mosaic_prob: float = 0.0,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.mosaic_transform = mosaic_transform
        self.mosaic_prob = mosaic_prob

        ann_file = self.root / "annotations" / f"instances_{split}.json"
        self.coco = COCO(str(ann_file))
        self.img_ids = sorted(self.coco.imgs.keys())

        # Filter images with no annotations
        self.img_ids = [
            i for i in self.img_ids
            if len(self.coco.getAnnIds(imgIds=i, iscrowd=False)) > 0
        ]

        # Build continuous label mapping: COCO cat_id → 0-indexed
        cats = sorted(self.coco.getCatIds())
        self.cat_to_idx = {c: i for i, c in enumerate(cats)}
        self.classes = [self.coco.cats[c]["name"] for c in cats]

    def __len__(self) -> int:
        return len(self.img_ids)

    def _load_sample(self, idx: int) -> dict[str, Any]:
        img_id = self.img_ids[idx]
        info = self.coco.imgs[img_id]
        img_path = self.root / self.split / info["file_name"]
        img = cv2.imread(str(img_path))  # BGR
        h, w = img.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]  # COCO: x1,y1,w,h
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            if nw > 0 and nh > 0:
                boxes.append([cx, cy, nw, nh])
                labels.append(self.cat_to_idx[ann["category_id"]])

        return {
            "image": img,
            "boxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "image_id": img_id,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import random
        if self.mosaic_transform is not None and random.random() < self.mosaic_prob:
            indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
            samples = [self._load_sample(i) for i in indices]
            sample = self.mosaic_transform(samples)
        else:
            sample = self._load_sample(idx)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def detection_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate fn that pads boxes to the max count in the batch."""
    import torch

    images = torch.from_numpy(np.stack([s["image"] for s in batch]))
    max_n = max(len(s["boxes"]) for s in batch)
    targets = []
    for s in batch:
        n = len(s["boxes"])
        boxes = torch.from_numpy(s["boxes"])
        labels = torch.from_numpy(s["labels"])
        targets.append({"boxes": boxes, "labels": labels,
                        "image_id": s.get("image_id", -1)})
    return {"images": images, "targets": targets}
