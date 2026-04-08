"""PyTorch Lightning module — handles training, validation, and test steps."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from ..models import SegmentationModel
from .losses import SegmentationLoss
from .metrics import SegmentationMetrics


class SegLitModule(pl.LightningModule):
    """
    Lightning module for semantic segmentation.

    All hyperparameters come from the merged Hydra config so the entire
    experiment is reproducible from a single config snapshot.

    Args:
        cfg: full resolved Hydra config (``OmegaConf.DictConfig``).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # Model
        self.model = SegmentationModel(
            backbone=cfg.model.backbone,
            head=cfg.model.head,
            num_classes=cfg.data.num_classes,
        )

        # Loss
        loss_cfg = cfg.training.loss
        self.criterion = SegmentationLoss(
            num_classes=cfg.data.num_classes,
            ignore_index=cfg.data.ignore_index,
            ce_weight=loss_cfg.ce_weight,
            dice_weight=loss_cfg.dice_weight,
        )

        # Metrics (separate instances avoid state leakage between phases)
        self.train_metrics = SegmentationMetrics(cfg.data.num_classes, cfg.data.ignore_index)
        self.val_metrics = SegmentationMetrics(cfg.data.num_classes, cfg.data.ignore_index)
        self.test_metrics = SegmentationMetrics(cfg.data.num_classes, cfg.data.ignore_index)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def _shared_step(self, batch: dict, metrics: SegmentationMetrics) -> torch.Tensor:
        images, targets = batch["image"], batch["mask"]
        logits = self(images)
        loss, loss_parts = self.criterion(logits, targets)
        preds = logits.argmax(dim=1)
        metrics.update(preds.detach(), targets.detach())
        return loss, loss_parts

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, parts = self._shared_step(batch, self.train_metrics)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in parts.items():
            self.log(f"train/loss_{k}", v, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self._log_metrics(self.train_metrics, "train")
        self.train_metrics.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, parts = self._shared_step(batch, self.val_metrics)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in parts.items():
            self.log(f"val/loss_{k}", v, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics(self.val_metrics, "val")
        self.val_metrics.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, self.test_metrics)

    def on_test_epoch_end(self) -> None:
        self._log_metrics(self.test_metrics, "test")
        self.test_metrics.reset()

    # ------------------------------------------------------------------
    # Optimiser / scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt_cfg = self.cfg.training.optimizer
        sched_cfg = self.cfg.training.scheduler

        # Linear LR scaling: base_lr * (effective_batch / 8)
        world = max(self.trainer.num_devices * self.trainer.accumulate_grad_batches, 1)
        lr = opt_cfg.lr * world / 8

        optimizer = instantiate(opt_cfg, params=self._param_groups(lr))
        scheduler = instantiate(sched_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def _param_groups(self, lr: float) -> list[dict]:
        """Separate backbone (lower LR) from head (full LR)."""
        backbone_params = list(self.model.backbone.parameters())
        head_params = list(self.model.head.parameters())
        return [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params, "lr": lr},
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log_metrics(self, metrics: SegmentationMetrics, phase: str) -> None:
        result = metrics.compute()
        for k, v in result.items():
            prog_bar = k in ("mIoU", "pixel_acc")
            self.log(f"{phase}/{k}", v, on_epoch=True, prog_bar=prog_bar, sync_dist=True)
