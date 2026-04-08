"""
Offline evaluation: load a checkpoint, run on val split, save metrics + overlays.

Usage::

    python scripts/evaluate.py checkpoint=path/to/best.ckpt [output_dir=outputs/eval]
"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from urban_seg.data import CityscapesDataModule
from urban_seg.evaluation import Evaluator
from urban_seg.training import SegLitModule


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("Pass checkpoint=path/to/model.ckpt")

    output_dir = cfg.get("output_dir", "outputs/eval")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Data -------------------------------------------------------------
    dm = CityscapesDataModule(**cfg.data)
    dm.setup(stage="test")
    loader = dm.test_dataloader()

    # ---- Model ------------------------------------------------------------
    module = SegLitModule.load_from_checkpoint(checkpoint, cfg=cfg)
    model = module.model

    # ---- Evaluate ---------------------------------------------------------
    evaluator = Evaluator(model, loader, num_classes=cfg.data.num_classes, device=device)
    results = evaluator.run()
    evaluator.print_table(results)
    evaluator.save(results, output_dir=output_dir)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
