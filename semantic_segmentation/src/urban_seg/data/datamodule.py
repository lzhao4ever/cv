"""PyTorch Lightning DataModule for Cityscapes."""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .cityscapes import CityscapesDataset
from .transforms import build_train_transforms, build_val_transforms


class CityscapesDataModule(pl.LightningDataModule):
    """
    LightningDataModule wrapping :class:`CityscapesDataset`.

    All constructor arguments map 1-to-1 to the ``data:`` Hydra config block.
    """

    def __init__(
        self,
        root: str,
        num_classes: int = 19,
        ignore_index: int = 255,
        image_size: list[int] = (512, 1024),
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        mean: list[float] = (0.2869, 0.3251, 0.2839),
        std: list[float] = (0.1870, 0.1902, 0.1872),
        augmentation: dict | None = None,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.image_size = tuple(image_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.aug_cfg = augmentation or {}

        self._train: CityscapesDataset | None = None
        self._val: CityscapesDataset | None = None
        self._test: CityscapesDataset | None = None

    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        scale = self.aug_cfg.get("random_resize_crop", {}).get("scale", [0.5, 2.0])
        hflip_p = self.aug_cfg.get("horizontal_flip_p", 0.5)

        train_tf = build_train_transforms(self.image_size, self.mean, self.std, scale, hflip_p)
        val_tf = build_val_transforms(self.image_size, self.mean, self.std)

        if stage in ("fit", None):
            self._train = CityscapesDataset(self.root, "train", train_tf)
            self._val = CityscapesDataset(self.root, "val", val_tf)
        if stage in ("test", None):
            self._test = CityscapesDataset(self.root, "val", val_tf)

    # ------------------------------------------------------------------
    def _loader(self, dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test, shuffle=False)
