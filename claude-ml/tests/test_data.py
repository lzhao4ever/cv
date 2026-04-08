"""Tests for data pipeline (transforms and datamodule)."""

import numpy as np
import pytest
import torch

from urban_seg.data.transforms import build_train_transforms, build_val_transforms


def test_train_transforms_output_shape():
    tf = build_train_transforms(image_size=(128, 256))
    img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
    mask = np.random.randint(0, 19, (512, 1024), dtype=np.uint8)
    out = tf(image=img, mask=mask)
    assert out["image"].shape == (3, 128, 256)
    assert out["mask"].shape == (128, 256)


def test_val_transforms_deterministic():
    tf = build_val_transforms(image_size=(128, 256))
    img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
    mask = np.zeros((512, 1024), dtype=np.uint8)
    out1 = tf(image=img, mask=mask)
    out2 = tf(image=img, mask=mask)
    assert torch.allclose(out1["image"], out2["image"])


def test_train_transforms_ignore_index_preserved():
    tf = build_train_transforms(image_size=(128, 256))
    img = np.zeros((512, 1024, 3), dtype=np.uint8)
    mask = np.full((512, 1024), 255, dtype=np.uint8)
    out = tf(image=img, mask=mask)
    # After crop/flip, 255 pixels should still exist somewhere
    assert (out["mask"] == 255).any()
