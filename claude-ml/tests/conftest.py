"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dummy_batch():
    """Small batch of images + masks (no real data needed)."""
    B, H, W = 2, 128, 256
    return {
        "image": torch.rand(B, 3, H, W),
        "mask": torch.randint(0, 19, (B, H, W)),
        "path": ["/fake/path_0.png", "/fake/path_1.png"],
    }


@pytest.fixture
def minimal_cfg():
    """Minimal Hydra-compatible config for unit tests."""
    return OmegaConf.create(
        {
            "data": {
                "num_classes": 19,
                "ignore_index": 255,
                "image_size": [128, 256],
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "mean": [0.2869, 0.3251, 0.2839],
                "std": [0.1870, 0.1902, 0.1872],
            },
            "model": {
                "backbone": {"name": "resnet50", "pretrained": False},
                "head": {"name": "deeplab_head", "aspp_dilations": [6, 12, 18], "dropout": 0.1},
            },
            "training": {
                "max_epochs": 2,
                "val_check_interval": 1.0,
                "log_every_n_steps": 1,
                "gradient_clip_val": 1.0,
                "precision": "32",
                "loss": {"ce_weight": 1.0, "dice_weight": 1.0},
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "lr": 1e-4,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                },
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.PolynomialLR",
                    "power": 1.0,
                    "total_iters": 2,
                },
                "callbacks": {
                    "checkpoint": {"monitor": "val/mIoU", "mode": "max", "save_top_k": 1},
                    "early_stopping": {"monitor": "val/mIoU", "mode": "max", "patience": 5},
                    "lr_monitor": {"logging_interval": "step"},
                },
                "logger": {
                    "mlflow": {
                        "experiment_name": "test",
                        "tracking_uri": "file:///tmp/mlruns",
                    }
                },
            },
            "paths": {"checkpoint_dir": "/tmp/test_ckpts"},
            "project": {"name": "test", "tags": []},
            "seed": 0,
        }
    )
