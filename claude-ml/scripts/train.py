"""
Main training entry point — driven entirely by Hydra config.

Single-GPU::

    python scripts/train.py

Multi-GPU (DDP)::

    python scripts/train.py +training=distributed

Override anything inline::

    python scripts/train.py model=deeplabv3plus training.max_epochs=80 data.batch_size=16
"""

from __future__ import annotations

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import MLFlowLogger

from urban_seg.data import CityscapesDataModule
from urban_seg.training import SegLitModule


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    # ---- DataModule -------------------------------------------------------
    datamodule = CityscapesDataModule(**cfg.data)

    # ---- LightningModule --------------------------------------------------
    module = SegLitModule(cfg)

    # ---- Callbacks --------------------------------------------------------
    cb_cfg = cfg.training.callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoint_dir,
            **cb_cfg.checkpoint,
        ),
        EarlyStopping(**cb_cfg.early_stopping),
        LearningRateMonitor(**cb_cfg.lr_monitor),
        RichProgressBar(),
    ]

    # ---- Logger -----------------------------------------------------------
    logger = MLFlowLogger(
        experiment_name=cfg.training.logger.mlflow.experiment_name,
        tracking_uri=cfg.training.logger.mlflow.tracking_uri,
        tags={"model": cfg.model.backbone.name, **dict(cfg.project.tags)},
    )

    # ---- Trainer ----------------------------------------------------------
    trainer_kwargs: dict = {
        "max_epochs": cfg.training.max_epochs,
        "val_check_interval": cfg.training.val_check_interval,
        "log_every_n_steps": cfg.training.log_every_n_steps,
        "gradient_clip_val": cfg.training.gradient_clip_val,
        "precision": cfg.training.precision,
        "callbacks": callbacks,
        "logger": logger,
        "enable_model_summary": True,
    }

    # Distributed overrides
    if hasattr(cfg.training, "trainer"):
        trainer_kwargs.update(OmegaConf.to_container(cfg.training.trainer, resolve=True))

    trainer = pl.Trainer(**trainer_kwargs)

    # Log full config as artifact
    with open("config_resolved.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    logger.experiment.log_artifact(logger.run_id, "config_resolved.yaml")

    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
