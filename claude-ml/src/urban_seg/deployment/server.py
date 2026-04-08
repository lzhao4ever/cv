"""FastAPI inference server with ONNX Runtime backend and Prometheus metrics."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter("segmentation_requests_total", "Total inference requests")
REQUEST_LATENCY = Histogram(
    "segmentation_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
ERROR_COUNT = Counter("segmentation_errors_total", "Total inference errors")

# ---------------------------------------------------------------------------
# ONNX session (loaded at startup via env var MODEL_PATH)
# ---------------------------------------------------------------------------
_session: "ort.InferenceSession | None" = None

# Cityscapes normalisation constants
_MEAN = np.array([0.2869, 0.3251, 0.2839], dtype=np.float32)
_STD = np.array([0.1870, 0.1902, 0.1872], dtype=np.float32)

# Cityscapes colour palette
_PALETTE: list[tuple[int, int, int]] = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32),
]

# ---------------------------------------------------------------------------
app = FastAPI(
    title="Urban Segmentation API",
    description="Semantic segmentation of urban traffic scenes.",
    version="0.1.0",
)


@app.on_event("startup")
def load_model() -> None:
    global _session
    model_path = os.environ.get("MODEL_PATH", "model.onnx")
    if not Path(model_path).exists():
        raise RuntimeError(f"MODEL_PATH={model_path!r} not found. Set env var before starting.")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _session = ort.InferenceSession(model_path, providers=providers)


def _preprocess(image: Image.Image, size: tuple[int, int] = (512, 1024)) -> np.ndarray:
    img = image.convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[None]  # (1, 3, H, W)


def _colorize(mask: np.ndarray) -> bytes:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(_PALETTE):
        rgb[mask == cls_id] = color
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _session is not None}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict/mask")
async def predict_mask(file: UploadFile = File(...)) -> Response:
    """Return a colourised PNG segmentation mask."""
    REQUEST_COUNT.inc()
    t0 = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(await file.read()))
        inp = _preprocess(img)
        [logits] = _session.run(None, {"input": inp})     # (1, C, H, W)
        pred = logits[0].argmax(axis=0).astype(np.uint8)  # (H, W)
        png_bytes = _colorize(pred)
        REQUEST_LATENCY.observe(time.perf_counter() - t0)
        return Response(png_bytes, media_type="image/png")
    except Exception as exc:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/classes")
async def predict_classes(file: UploadFile = File(...)) -> JSONResponse:
    """Return class-index array as nested JSON list (for downstream processing)."""
    REQUEST_COUNT.inc()
    t0 = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(await file.read()))
        inp = _preprocess(img)
        [logits] = _session.run(None, {"input": inp})
        pred = logits[0].argmax(axis=0).astype(int).tolist()
        REQUEST_LATENCY.observe(time.perf_counter() - t0)
        return JSONResponse({"mask": pred, "height": len(pred), "width": len(pred[0])})
    except Exception as exc:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
