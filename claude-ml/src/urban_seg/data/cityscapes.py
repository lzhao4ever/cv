"""Cityscapes dataset — 19-class evaluation protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Maps Cityscapes trainId (0-18 + 255 ignore) labels
TRAINID_TO_NAME: dict[int, str] = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

# Raw Cityscapes label id → trainId mapping (official)
_LABEL_TO_TRAINID: dict[int, int] = {
    -1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255,
    6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3,
    13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6,
    20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
    27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18,
}
_LUT = np.full(256, 255, dtype=np.uint8)
for k, v in _LABEL_TO_TRAINID.items():
    if 0 <= k <= 255:
        _LUT[k] = v


class CityscapesDataset(Dataset):
    """
    Cityscapes semantic segmentation dataset.

    Expected directory layout::

        root/
          leftImg8bit/{train,val,test}/city/city_000000_000000_leftImg8bit.png
          gtFine/{train,val,test}/city/city_000000_000000_gtFine_labelIds.png

    Args:
        root: path to the Cityscapes root directory.
        split: one of ``"train"``, ``"val"``, ``"test"``.
        transform: joint image+mask callable (albumentations Compose).
        use_trainids: if True convert raw labelIds → 19-class trainIds.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        use_trainids: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.use_trainids = use_trainids

        self.images, self.masks = self._collect_files()

    # ------------------------------------------------------------------
    def _collect_files(self) -> tuple[list[Path], list[Path]]:
        img_dir = self.root / "leftImg8bit" / self.split
        mask_dir = self.root / "gtFine" / self.split

        if not img_dir.exists():
            raise FileNotFoundError(
                f"Cityscapes image dir not found: {img_dir}\n"
                "Download from https://www.cityscapes-dataset.com/"
            )

        images, masks = [], []
        for city in sorted(img_dir.iterdir()):
            for img_path in sorted(city.glob("*_leftImg8bit.png")):
                stem = img_path.name.replace("_leftImg8bit.png", "")
                mask_path = mask_dir / city.name / f"{stem}_gtFine_labelIds.png"
                if mask_path.exists() or self.split == "test":
                    images.append(img_path)
                    masks.append(mask_path)

        return images, masks

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image = np.array(Image.open(self.images[idx]).convert("RGB"))

        if self.split != "test" and self.masks[idx].exists():
            mask = np.array(Image.open(self.masks[idx]))
            if self.use_trainids:
                mask = _LUT[mask]
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        return {"image": image, "mask": mask.long(), "path": str(self.images[idx])}
