"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def minimal_cfg():
    return OmegaConf.create({
        "project": {
            "name": "test",
            "seed": 0,
            "output_dir": "/tmp/test_outputs",
            "experiment_name": "test-rtdetr",
        },
        "data": {
            "name": "coco",
            "root": "/tmp/fake_coco",
            "num_classes": 80,
            "image_size": [640, 640],
            "batch_size": 2,
            "num_workers": 0,
            "augmentation": {},
        },
        "model": {
            "name": "rtdetr_r50",
            "backbone": {
                "type": "resnet50",
                "pretrained": False,
                "out_indices": [1, 2, 3],
                "freeze_at": 0,
            },
            "encoder": {
                "type": "HybridEncoder",
                "in_channels": [512, 1024, 2048],
                "hidden_dim": 64,
                "use_encoder_idx": [2],
                "num_encoder_layers": 1,
                "nhead": 4,
                "dim_feedforward": 128,
                "dropout": 0.0,
                "enc_act": "gelu",
                "expansion": 0.5,
                "depth_mult": 0.33,
            },
            "decoder": {
                "type": "RTDETRDecoder",
                "hidden_dim": 64,
                "num_queries": 10,
                "num_decoder_layers": 2,
                "nhead": 4,
                "dim_feedforward": 128,
                "dropout": 0.0,
                "num_denoising": 5,
                "label_noise_ratio": 0.5,
                "box_noise_scale": 1.0,
                "eval_spatial_size": [640, 640],
                "eval_idx": -1,
            },
        },
        "training": {
            "max_epochs": 1,
            "optimizer": {"lr": 1e-4, "weight_decay": 1e-4},
            "lr_scheduler": {"milestones": [1], "gamma": 0.1},
            "backbone_lr_multiplier": 0.1,
            "gradient_clip_val": 0.1,
            "callbacks": {
                "model_checkpoint": {
                    "monitor": "val/mAP", "mode": "max",
                    "save_top_k": 1, "save_last": False,
                    "filename": "best",
                },
                "early_stopping": {"monitor": "val/mAP", "patience": 5, "mode": "max"},
            },
        },
        "deployment": {
            "format": "onnx",
            "opset": 17,
            "simplify": False,
            "half": False,
        },
    })


@pytest.fixture
def fake_batch() -> dict:
    """A minimal batch of 2 images with random boxes."""
    images = torch.rand(2, 3, 640, 640)
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.4, 0.1, 0.15]]),
            "labels": torch.tensor([0, 3]),
            "image_id": 1,
        },
        {
            "boxes": torch.tensor([[0.7, 0.6, 0.3, 0.2]]),
            "labels": torch.tensor([5]),
            "image_id": 2,
        },
    ]
    return {"images": images, "targets": targets}
