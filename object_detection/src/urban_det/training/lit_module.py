"""PyTorch Lightning module for DETR-family object detection."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch import Tensor

from urban_det.models import DetectionModel
from urban_det.training.losses import DETRCriterion
from urban_det.training.metrics import COCOMetrics, decode_predictions


class DetectionLitModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, num_classes: int, coco_gt=None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = DetectionModel(cfg.model, num_classes=num_classes)
        self.criterion = DETRCriterion(num_classes=num_classes)
        self.val_metrics = COCOMetrics(coco_gt) if coco_gt is not None else None
        self._img_size = tuple(cfg.data.image_size)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        images = batch["images"].to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items() if isinstance(v, Tensor)}
                   for t in batch["targets"]]

        outputs = self.model(images, targets=targets)
        losses = self.criterion(outputs, targets)

        self.log("train/loss", losses["total"], prog_bar=True, sync_dist=True)
        self.log("train/loss_ce", losses["loss_ce"], sync_dist=True)
        self.log("train/loss_bbox", losses["loss_bbox"], sync_dist=True)
        self.log("train/loss_giou", losses["loss_giou"], sync_dist=True)
        if "dn_loss_ce" in losses:
            self.log("train/dn_loss_ce", losses["dn_loss_ce"], sync_dist=True)

        return losses["total"]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        if self.val_metrics:
            self.val_metrics.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images = batch["images"].to(self.device)
        targets = batch["targets"]

        outputs = self.model(images)

        image_ids = [t.get("image_id", -1) for t in targets]
        if isinstance(image_ids[0], Tensor):
            image_ids = [i.item() for i in image_ids]

        if self.val_metrics:
            preds = decode_predictions(outputs, image_ids, self._img_size)
            self.val_metrics.update(preds)

    def on_validation_epoch_end(self) -> None:
        if self.val_metrics:
            metrics = self.val_metrics.compute()
            for k, v in metrics.items():
                self.log(f"val/{k}", v, prog_bar=(k == "mAP"), sync_dist=True)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        train_cfg = self.cfg.training
        opt_cfg = train_cfg.optimizer

        # Separate backbone params for lower LR
        backbone_params, other_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        base_lr = opt_cfg.lr
        # Linear LR scaling across world size
        world_size = self.trainer.world_size if self.trainer else 1
        scaled_lr = base_lr * world_size

        param_groups = [
            {"params": backbone_params, "lr": scaled_lr * train_cfg.get("backbone_lr_multiplier", 0.1)},
            {"params": other_params, "lr": scaled_lr},
        ]

        optimizer = optim.AdamW(
            param_groups,
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        sched_cfg = train_cfg.lr_scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(sched_cfg.milestones),
            gamma=sched_cfg.get("gamma", 0.1),
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    # ------------------------------------------------------------------
    # Epoch hooks for mosaic close
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        # Notify datamodule so it can disable mosaic in the last N epochs
        dm = self.trainer.datamodule
        if hasattr(dm, "current_epoch"):
            dm.current_epoch = self.current_epoch
