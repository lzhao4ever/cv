"""Offline evaluator: produces per-class IoU table and qualitative overlays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.cityscapes import TRAINID_TO_NAME
from ..models import SegmentationModel

# Cityscapes colour palette (19 classes)
_PALETTE: list[tuple[int, int, int]] = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32),
]


class Evaluator:
    """
    Runs a model over a DataLoader and computes segmentation metrics.

    Usage::

        evaluator = Evaluator(model, loader, num_classes=19, device="cuda")
        results = evaluator.run()
        evaluator.print_table(results)
        evaluator.save(results, output_dir="outputs/eval")
    """

    def __init__(
        self,
        model: SegmentationModel,
        loader: DataLoader,
        num_classes: int = 19,
        ignore_index: int = 255,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device).eval()
        self.loader = loader
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run(self) -> dict:
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        for batch in tqdm(self.loader, desc="Evaluating", unit="batch"):
            images = batch["image"].to(self.device)
            targets = batch["mask"].numpy()  # (B, H, W)

            logits = self.model(images)
            preds = logits.argmax(dim=1).cpu().numpy()  # (B, H, W)

            for pred, gt in zip(preds, targets):
                valid = gt != self.ignore_index
                np.add.at(
                    confusion,
                    (gt[valid], pred[valid]),
                    1,
                )

        return self._compute_metrics(confusion)

    # ------------------------------------------------------------------
    def _compute_metrics(self, confusion: np.ndarray) -> dict:
        tp = np.diag(confusion)
        fn = confusion.sum(axis=1) - tp
        fp = confusion.sum(axis=0) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / denom, np.nan)
        pixel_acc = tp.sum() / confusion.sum()

        return {
            "mIoU": float(np.nanmean(iou)),
            "pixel_acc": float(pixel_acc),
            "per_class_iou": {TRAINID_TO_NAME.get(i, str(i)): float(v) for i, v in enumerate(iou)},
            "confusion_matrix": confusion.tolist(),
        }

    # ------------------------------------------------------------------
    def print_table(self, results: dict) -> None:
        console = Console()
        table = Table(title=f"Evaluation — mIoU: {results['mIoU']:.4f}  |  Pixel Acc: {results['pixel_acc']:.4f}")
        table.add_column("Class", style="cyan")
        table.add_column("IoU", justify="right")
        for cls, iou in results["per_class_iou"].items():
            val = f"{iou:.4f}" if not np.isnan(iou) else "—"
            table.add_row(cls, val)
        console.print(table)

    # ------------------------------------------------------------------
    def save(self, results: dict, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics = {k: v for k, v in results.items() if k != "confusion_matrix"}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        np.save(out / "confusion_matrix.npy", np.array(results["confusion_matrix"]))

    # ------------------------------------------------------------------
    @staticmethod
    def colorize(mask: np.ndarray) -> Image.Image:
        """Convert (H, W) label map to RGB visualisation."""
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cls_id, color in enumerate(_PALETTE):
            rgb[mask == cls_id] = color
        return Image.fromarray(rgb)
