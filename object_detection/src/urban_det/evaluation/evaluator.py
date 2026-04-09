"""Offline evaluator: runs a checkpoint on a COCO val split and prints COCO metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from pycocotools.coco import COCO
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from tqdm import tqdm

from urban_det.data.coco import COCODetection, detection_collate
from urban_det.data.transforms import DetectionTransform
from urban_det.models import DetectionModel
from urban_det.training.metrics import COCOMetrics, decode_predictions

console = Console()


class DetectionEvaluator:
    def __init__(
        self,
        checkpoint_path: str | Path,
        data_root: str | Path,
        split: str = "val2017",
        img_size: int = 640,
        batch_size: int = 16,
        num_workers: int = 4,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        device: str = "cuda",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._load_checkpoint()
        self._build_dataset()

    def _load_checkpoint(self) -> None:
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        cfg = ckpt["hyper_parameters"]
        num_classes = cfg.data.num_classes

        self.model = DetectionModel(cfg.model, num_classes=num_classes)
        # Strip "model." prefix added by Lightning
        state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
                 if k.startswith("model.")}
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.num_classes = num_classes
        console.print(f"[green]Loaded checkpoint:[/] {self.checkpoint_path}")

    def _build_dataset(self) -> None:
        ann_file = self.data_root / "annotations" / f"instances_{self.split}.json"
        self.coco_gt = COCO(str(ann_file))
        tf = DetectionTransform(self.img_size, augment=False, aug_cfg={})
        self.dataset = COCODetection(self.data_root, self.split, transform=tf)
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=detection_collate,
        )
        self.metrics = COCOMetrics(self.coco_gt)

    @torch.inference_mode()
    def run(self, output_dir: str | Path | None = None) -> dict[str, float]:
        self.metrics.reset()

        for batch in tqdm(self.loader, desc="Evaluating"):
            images = batch["images"].to(self.device)
            targets = batch["targets"]
            image_ids = [t.get("image_id", -1) for t in targets]

            outputs = self.model(images)
            preds = decode_predictions(
                outputs, image_ids, (self.img_size, self.img_size),
                self.conf_threshold, self.iou_threshold,
            )
            self.metrics.update(preds)

        results = self.metrics.compute()
        self._print_results(results)

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "metrics.json", "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]Saved metrics to[/] {out / 'metrics.json'}")

        return results

    @staticmethod
    def _print_results(results: dict[str, float]) -> None:
        table = Table(title="COCO Detection Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="magenta")
        for k, v in results.items():
            table.add_row(k, f"{v:.4f}")
        console.print(table)
