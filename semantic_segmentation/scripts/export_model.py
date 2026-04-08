"""
Export a trained checkpoint to ONNX and TorchScript.

Usage::

    python scripts/export_model.py checkpoint=path/to/best.ckpt [export_dir=exports]
"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from urban_seg.deployment import export_onnx, export_torchscript
from urban_seg.training import SegLitModule


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("Pass checkpoint=path/to/model.ckpt")

    export_dir = cfg.get("export_dir", "exports")
    image_size = tuple(cfg.data.image_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = SegLitModule.load_from_checkpoint(checkpoint, cfg=cfg)
    model = module.model.to(device).eval()

    export_onnx(model, f"{export_dir}/model.onnx", image_size=image_size)
    export_torchscript(model, f"{export_dir}/model.pt", image_size=image_size)
    print("Export complete.")


if __name__ == "__main__":
    main()
