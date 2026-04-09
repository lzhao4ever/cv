"""PyTorch Lightning DataModule for detection datasets."""

from __future__ import annotations

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from urban_det.data.coco import COCODetection, detection_collate
from urban_det.data.transforms import DetectionTransform, MosaicTransform


class DetectionDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.current_epoch = 0

    # called by Lightning before fit/test
    def setup(self, stage: str | None = None) -> None:
        cfg = self.cfg
        img_size = cfg.image_size[0]
        aug_cfg = dict(cfg.get("augmentation", {}))
        mosaic_prob = aug_cfg.get("mosaic_prob", 0.0)
        close_mosaic = aug_cfg.get("close_mosaic_epochs", 10)

        # Mosaic is disabled for the last `close_mosaic_epochs` epochs.
        # Lightning doesn't pass epoch here, so mosaic_prob is set to 0
        # if the trainer has progressed past the threshold (updated externally).
        effective_mosaic = mosaic_prob if self.current_epoch < close_mosaic else 0.0

        mosaic_tf = MosaicTransform(img_size) if effective_mosaic > 0 else None
        train_tf = DetectionTransform(img_size, augment=True, aug_cfg=aug_cfg)
        val_tf = DetectionTransform(img_size, augment=False, aug_cfg={})

        self.train_ds = COCODetection(
            root=cfg.root,
            split=cfg.train_split,
            transform=train_tf,
            mosaic_transform=mosaic_tf,
            mosaic_prob=effective_mosaic,
        )
        self.val_ds = COCODetection(
            root=cfg.root,
            split=cfg.val_split,
            transform=val_tf,
        )
        if hasattr(cfg, "test_split"):
            self.test_ds = COCODetection(
                root=cfg.root,
                split=cfg.test_split,
                transform=val_tf,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.get("pin_memory", True),
            persistent_workers=self.cfg.get("persistent_workers", True),
            collate_fn=detection_collate,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.get("pin_memory", True),
            persistent_workers=self.cfg.get("persistent_workers", True),
            collate_fn=detection_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=detection_collate,
        )
