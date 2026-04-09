#!/usr/bin/env python3
"""Data preparation helpers.

Commands:
  download-coco   Download COCO 2017 train/val + annotations
  verify-coco     Verify annotation integrity
  download-nuscenes  Print nuScenes download instructions (manual license required)
  stats           Print dataset statistics
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
from pathlib import Path


COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017":   "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url} → {dest}")
    urllib.request.urlretrieve(url, dest)


def download_coco(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, url in COCO_URLS.items():
        zip_path = root / f"{name}.zip"
        if not zip_path.exists():
            _download(url, zip_path)
        print(f"Extracting {zip_path}…")
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(root)], check=True)
    print("COCO download complete.")


def verify_coco(root: Path) -> None:
    from pycocotools.coco import COCO
    for split in ["train2017", "val2017"]:
        ann = root / "annotations" / f"instances_{split}.json"
        img_dir = root / split
        coco = COCO(str(ann))
        missing = []
        for img_id, info in coco.imgs.items():
            p = img_dir / info["file_name"]
            if not p.exists():
                missing.append(str(p))
        if missing:
            print(f"[{split}] {len(missing)} missing images — first 5: {missing[:5]}")
        else:
            print(f"[{split}] All {len(coco.imgs)} images present. ✓")


def print_nuscenes_instructions() -> None:
    print("""
nuScenes requires accepting a license at https://www.nuscenes.org/nuscenes#download
After download, extract so the structure is:
  /data/nuscenes/
    v1.0-trainval/
      samples/CAM_FRONT/...
      annotations/...
    maps/
    v1.0-trainval_meta.tgz
""")


def dataset_stats(root: Path) -> None:
    for split in ["train2017", "val2017"]:
        ann = root / "annotations" / f"instances_{split}.json"
        if not ann.exists():
            print(f"{split}: annotations not found")
            continue
        with open(ann) as f:
            data = json.load(f)
        n_imgs = len(data["images"])
        n_anns = len(data["annotations"])
        n_cats = len(data["categories"])
        print(f"{split}: {n_imgs:,} images, {n_anns:,} annotations, {n_cats} categories")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_dl = sub.add_parser("download-coco")
    p_dl.add_argument("--root", default="/data/coco")

    p_ver = sub.add_parser("verify-coco")
    p_ver.add_argument("--root", default="/data/coco")

    sub.add_parser("download-nuscenes")

    p_stats = sub.add_parser("stats")
    p_stats.add_argument("--root", default="/data/coco")

    args = parser.parse_args()
    if args.cmd == "download-coco":
        download_coco(Path(args.root))
    elif args.cmd == "verify-coco":
        verify_coco(Path(args.root))
    elif args.cmd == "download-nuscenes":
        print_nuscenes_instructions()
    elif args.cmd == "stats":
        dataset_stats(Path(args.root))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
