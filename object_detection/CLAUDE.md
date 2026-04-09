# urban-det — Developer Guide

SOTA object detection for urban traffic scenes (self-driving car perception).

## Architecture overview

```
DetectionModel
  └─ TimmBackbone (ResNet-50/101, ConvNeXtV2)
       ↓ [C3, C4, C5]
  └─ HybridEncoder  (AIFI + CCFM)   ← RT-DETR
   ─ ChannelMapper + DINOTransformer ← DINO
       ↓ multi-scale features
  └─ RTDETRDecoder / DINOTransformer
       ↓ {pred_logits (B,Q,C), pred_boxes (B,Q,4)}
```

Boxes are **cx, cy, w, h normalized to [0, 1]** throughout.

## Key design decisions

- **RT-DETR (default)**: HybridEncoder (AIFI on C5 + CCFM FPN) + RTDETRDecoder with CDN
  denoising. Achieves real-time speed at high accuracy (better accuracy/speed trade-off than YOLOv8).
- **DINO (optional)**: `model=dino_r50`. Heavier; use for highest mAP when latency budget allows.
- **No NMS at train time**: DETR-family models use bipartite matching — no NMS needed.
  NMS is applied only in `decode_predictions()` as a post-processing safety net.
- **Backbone LR × 0.1**: Standard for DETR fine-tuning. Configured via `training.backbone_lr_multiplier`.
- **Mosaic disabled for nuScenes**: AV cameras have fixed geometry; mosaic destroys spatial relationships.
  Set via `data.augmentation.mosaic_prob: 0.0` in `configs/data/nuscenes.yaml`.

## Adding a new backbone

1. Add its stage channels to `_ALL_CHANNELS` in `src/urban_det/models/backbones/registry.py`.
2. Verify `timm.create_model(name, features_only=True, out_indices=...)` works.
3. Add a model config in `configs/model/` referencing the new backbone.
4. No other changes needed.

## Adding a new decoder head

1. Implement `forward(memory: list[Tensor], targets) → dict[str, Tensor]`
   with at minimum `pred_logits` and `pred_boxes` keys.
2. Register in `build_decoder()` in `src/urban_det/models/detection_model.py`.
3. Add a YAML config under `configs/model/`.

## Training

```bash
# Install
make install

# Download COCO
make data-download DATA_ROOT=/data/coco

# Single GPU
make train

# 4-node × 8-GPU distributed
make train-distributed

# DINO model
make train-dino
```

Training uses PyTorch Lightning + Hydra. All hyper-parameters are in `configs/`.
MLflow is the primary experiment tracker (runs at `http://localhost:5000`).

## Evaluation

```bash
make eval CHECKPOINT=outputs/best.ckpt DATA_ROOT=/data/coco
```

Outputs COCO metrics (mAP, mAP50, mAP75, mAP_s/m/l) to stdout and `outputs/eval/metrics.json`.

## Export & deployment

```bash
# ONNX (default)
make export-onnx CHECKPOINT=outputs/best.ckpt

# TensorRT FP16 for Jetson / NVIDIA Drive
make export-trt CHECKPOINT=outputs/best.ckpt
```

The TensorRT export requires CUDA 12 + TensorRT 10 installed (pre-installed on Jetson Orin / NVIDIA Drive images).

## Inference server

```bash
# Local
make serve

# Docker stack (MLflow + Prometheus + Grafana + serve)
make docker-up
```

Endpoints:
- `POST /predict`        — JSON with detections, scores, labels, latency_ms
- `POST /predict/image`  — annotated PNG
- `GET  /health`         — readiness
- `GET  /metrics`        — Prometheus metrics

## On-vehicle deployment (ROS2)

```bash
ros2 run urban_det detection_node \
  --ros-args -p engine_path:=/opt/model/rtdetr.engine -p conf_threshold:=0.35
```

Subscribes to `/camera/image_raw`, publishes `vision_msgs/Detection2DArray`
on `/perception/detections`.

## Kubernetes

```bash
# Deploy full stack
make k8s-apply

# Launch distributed training job (4 nodes × 8 GPUs)
make k8s-train

# Check status
make k8s-status
```

Update `k8s/training-job.yaml` → `image:` with your registry before applying.

## Testing

```bash
make test       # full suite with coverage
make test-fast  # stop on first failure
```

Tests use CPU-only and do not require real data (fake tensors / mocked ONNX sessions).

## Monitoring

Grafana dashboard at `http://localhost:3000` (admin/admin).
Key metrics:
- `det_inference_latency_seconds` — p50/p95/p99 latency
- `det_inference_requests_total{status="error"}` — error rate
- `det_detected_objects_per_image` — distribution of object counts
- `det_avg_confidence` — mean detection confidence (drift proxy)
- `det_data_drift_score` — feature-level distribution drift (requires drift monitor)
