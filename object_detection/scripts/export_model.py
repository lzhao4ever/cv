#!/usr/bin/env python3
"""Model export entry point.

Usage:
  # ONNX (default)
  python scripts/export_model.py checkpoint=outputs/best.ckpt

  # TensorRT FP16
  python scripts/export_model.py checkpoint=outputs/best.ckpt deployment=tensorrt

  # TensorRT INT8 with DLA
  python scripts/export_model.py checkpoint=outputs/best.ckpt deployment=tensorrt \
    deployment.precision=int8 deployment.dla_core=0
"""

import hydra
from omegaconf import DictConfig

from urban_det.deployment import export_model


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if checkpoint is None:
        raise ValueError("Pass checkpoint=<path/to/best.ckpt>")
    export_model(cfg, checkpoint_path=checkpoint)


if __name__ == "__main__":
    main()
