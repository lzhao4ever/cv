#!/usr/bin/env python3
"""Training entry point.

Single GPU:
  python scripts/train.py

Multi-GPU (DDP):
  python scripts/train.py training=distributed

Override examples:
  python scripts/train.py model=rtdetr_r101 data.batch_size=8 training.max_epochs=36
"""

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger

from urban_det.data import DetectionDataModule
from urban_det.training import DetectionLitModule


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.project.seed, workers=True)

    dm = DetectionDataModule(cfg)
    dm.setup("fit")
    coco_gt = dm.val_ds.coco

    model = DetectionLitModule(cfg, num_classes=cfg.data.num_classes, coco_gt=coco_gt)

    # Callbacks
    ckpt_cfg = cfg.training.callbacks.model_checkpoint
    callbacks = [
        ModelCheckpoint(
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.mode,
            save_top_k=ckpt_cfg.save_top_k,
            save_last=ckpt_cfg.save_last,
            filename=ckpt_cfg.filename,
        ),
        EarlyStopping(
            monitor=cfg.training.callbacks.early_stopping.monitor,
            patience=cfg.training.callbacks.early_stopping.patience,
            mode=cfg.training.callbacks.early_stopping.mode,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = MLFlowLogger(
        experiment_name=cfg.project.experiment_name,
        tracking_uri="http://mlflow:5000",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.training.get("devices", 1),
        num_nodes=cfg.training.get("num_nodes", 1),
        strategy=cfg.training.get("strategy", "auto"),
        precision=cfg.training.get("precision", "16-mixed"),
        gradient_clip_val=cfg.training.get("gradient_clip_val", 0.1),
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        sync_batchnorm=cfg.training.get("sync_batchnorm", True),
        log_every_n_steps=cfg.training.get("log_every_n_steps", 50),
        val_check_interval=cfg.training.get("val_check_interval", 1.0),
        callbacks=callbacks,
        logger=logger,
        benchmark=cfg.training.get("benchmark", True),
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
