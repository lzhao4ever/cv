#!/usr/bin/env python3
"""Offline evaluation entry point.

Usage:
  python scripts/evaluate.py \
    --checkpoint outputs/best.ckpt \
    --data-root /data/coco \
    --split val2017
"""

import argparse

from urban_det.evaluation import DetectionEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    evaluator = DetectionEvaluator(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )
    evaluator.run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
