# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```bash
# Install for development
make install-dev          # pip install -e ".[dev,notebook]"

# Linting / formatting
make lint                 # ruff check + mypy
make format               # ruff format

# Tests
make test                 # full suite with coverage
pytest tests/test_models.py::test_deeplab_head_output_shape   # single test
pytest tests/ -m "not slow"   # skip slow tests

# Data validation
python scripts/prepare_data.py --root /data/cityscapes

# Train (single GPU)
make train DATA_ROOT=/data/cityscapes
# Train (multi-GPU DDP)
make train-dist DATA_ROOT=/data/cityscapes
# Override any config inline
python scripts/train.py model=deeplabv3plus training.max_epochs=80

# Evaluate
make evaluate CHECKPOINT=outputs/best.ckpt

# Export to ONNX + TorchScript
make export CHECKPOINT=outputs/best.ckpt

# Run inference server
make serve                # requires exports/model.onnx

# Docker Compose (MLflow + Prometheus + Grafana + serve)
make compose-up
```

## Architecture

### Config system
Everything is driven by **Hydra 1.3** (`configs/`). The root `configs/config.yaml` composes four sub-configs (`data`, `model`, `training`, `deployment`) via `defaults`. All script entry points accept inline overrides (`key=value`). The full resolved config is logged as an MLflow artifact on each run.

### Model composition (`src/urban_seg/models/`)
`SegmentationModel` wires a backbone + decode head together. Backbones are resolved by name via `backbones/registry.py` — MixTransformer variants come from HuggingFace Transformers (`nvidia/mit-b*`), ResNets from timm. Both expose `.out_channels: list[int]` (4 values, coarsest to finest) which decode heads consume. Two heads are implemented: `SegFormerHead` (all-MLP) and `DeepLabV3PlusHead` (ASPP + low-level skip). The model always upsamples output logits to full input resolution before returning.

### Training (`src/urban_seg/training/`)
`SegLitModule` is the PyTorch Lightning module. It reads the full Hydra config at construction. LR is scaled linearly with `num_gpus × accumulate_grad_batches / 8` (linear scaling rule). Backbone parameters use 0.1× the head LR. Loss is CE + Dice (both `ignore_index`-masked). Metrics (`SegmentationMetrics`) accumulate a confusion matrix across batches for correct distributed mIoU.

### Distributed training
Enable with `+training=distributed` which overlays `configs/training/distributed.yaml`. This sets `strategy: ddp_find_unused_parameters_false` and `devices: -1`. For multi-node K8s training, set `NUM_NODES` env var and use `k8s/training-job.yaml` (Indexed Job).

### Serving pipeline
`scripts/export_model.py` → ONNX (via `deployment/export.py`, optionally simplified with onnxsim) → `deployment/server.py` (FastAPI + ONNX Runtime). The server exposes `/predict/mask` (PNG), `/predict/classes` (JSON), `/health`, and `/metrics` (Prometheus). `MODEL_PATH` env var points to the `.onnx` file at startup.

### Experiment tracking
`monitoring/tracker.py` wraps MLflow (primary) with an optional W&B mirror. The Lightning module uses `MLFlowLogger` directly; `ExperimentTracker` is for standalone evaluation scripts. Set `MLFLOW_TRACKING_URI` env var (default: `http://localhost:5000`).

### Adding a new backbone
1. Add its stage channel list to `_CHANNEL_MAP` in `backbones/registry.py`.
2. Implement a class with `.out_channels: list[int]` and `forward() → list[Tensor]` (4 tensors, low→high resolution order).
3. Register it in `build_backbone()`.

### Adding a new decode head
1. Implement a class accepting `in_channels: list[int]` and `num_classes: int`, returning `(B, num_classes, H, W)`.
2. Add it to `_HEAD_REGISTRY` in `models/segmentation_model.py`.
3. Add a config file in `configs/model/`.
