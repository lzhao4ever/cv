"""Albumentations-based augmentation pipelines for semantic segmentation."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transforms(
    image_size: tuple[int, int] = (512, 1024),
    mean: tuple[float, ...] = (0.2869, 0.3251, 0.2839),
    std: tuple[float, ...] = (0.1870, 0.1902, 0.1872),
    scale: tuple[float, float] = (0.5, 2.0),
    hflip_p: float = 0.5,
) -> A.Compose:
    """Heavy augmentation for training."""
    h, w = image_size
    return A.Compose(
        [
            # 1. Scale jitter then crop
            A.RandomScale(scale_limit=(scale[0] - 1, scale[1] - 1), p=1.0),
            A.PadIfNeeded(min_height=h, min_width=w, border_mode=0, value=0, mask_value=255),
            A.RandomCrop(height=h, width=w),
            # 2. Flips
            A.HorizontalFlip(p=hflip_p),
            # 3. Color augmentation
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.5),
            A.ToGray(p=0.1),
            # 4. Blur / noise
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.2,
            ),
            A.GaussNoise(p=0.1),
            # 5. Normalize & tensorize
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def build_val_transforms(
    image_size: tuple[int, int] = (512, 1024),
    mean: tuple[float, ...] = (0.2869, 0.3251, 0.2839),
    std: tuple[float, ...] = (0.1870, 0.1902, 0.1872),
) -> A.Compose:
    """Deterministic resize + normalize for validation/test."""
    h, w = image_size
    return A.Compose(
        [
            A.Resize(height=h, width=w),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
