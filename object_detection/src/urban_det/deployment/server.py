"""FastAPI inference server with ONNX Runtime backend.

Endpoints:
  POST /predict          → {boxes, scores, labels} JSON
  POST /predict/image    → annotated PNG
  GET  /health
  GET  /metrics          (Prometheus)
"""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# Prometheus metrics (imported lazily so server starts even without prometheus)
try:
    from urban_det.monitoring.tracker import (
        CONFIDENCE_GAUGE,
        DETECTED_OBJECTS,
        INFERENCE_LATENCY,
        INFERENCE_REQUESTS,
    )
    _PROM = True
except Exception:
    _PROM = False

# Runtime state
_state: dict = {}

MODEL_NAME = "rtdetr"
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    import onnxruntime as ort

    model_path = Path(_state.get("model_path", "model.onnx"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _state["session"] = ort.InferenceSession(str(model_path), sess_opts, providers=providers)
    _state["input_name"] = _state["session"].get_inputs()[0].name
    yield
    _state.clear()


app = FastAPI(title="Urban Detection API", version="0.1.0", lifespan=lifespan)


def _preprocess(img_bytes: bytes, img_size: int = 640) -> tuple[np.ndarray, float, tuple]:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    h0, w0 = img.shape[:2]
    ratio = img_size / max(h0, w0)
    new_h, new_w = int(h0 * ratio), int(w0 * ratio)
    img_resized = cv2.resize(img, (new_w, new_h))

    # Letterbox pad
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    pad_y = (img_size - new_h) // 2
    pad_x = (img_size - new_w) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized

    inp = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return inp, ratio, (pad_x, pad_y, w0, h0)


def _postprocess(
    logits: np.ndarray,
    boxes: np.ndarray,
    meta: tuple,
    conf_thr: float,
) -> dict:
    """Decode raw decoder output → pixel boxes for the original image."""
    _, pad_x, pad_y, w0, h0 = meta[0], meta[1][0], meta[1][1], meta[1][2], meta[1][3]

    IMG = 640
    scores_all = 1 / (1 + np.exp(-logits[0]))  # sigmoid
    scores = scores_all.max(axis=-1)
    labels = scores_all.argmax(axis=-1)
    mask = scores > conf_thr

    boxes_f = boxes[0][mask]
    scores_f = scores[mask]
    labels_f = labels[mask]

    results = []
    for box, score, label in zip(boxes_f, scores_f, labels_f):
        cx, cy, bw, bh = box
        x1 = (cx - bw / 2) * IMG - pad_x
        y1 = (cy - bh / 2) * IMG - pad_y
        x2 = (cx + bw / 2) * IMG - pad_x
        y2 = (cy + bh / 2) * IMG - pad_y
        x1 = max(0.0, x1 / meta[0])
        y1 = max(0.0, y1 / meta[0])
        x2 = min(float(w0), x2 / meta[0])
        y2 = min(float(h0), y2 / meta[0])
        results.append({
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "score": round(float(score), 4),
            "label": int(label),
        })
    return {"detections": results, "count": len(results)}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": "session" in _state}


@app.get("/metrics")
async def metrics() -> Response:
    if not _PROM:
        raise HTTPException(status_code=503, detail="Prometheus not available")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]) -> JSONResponse:
    t0 = time.perf_counter()
    try:
        inp, ratio, pad_meta = _preprocess(await file.read())
        logits, boxes = _state["session"].run(None, {_state["input_name"]: inp})
        result = _postprocess(logits, boxes, (ratio, pad_meta[:2], pad_meta[2], pad_meta[3]),
                               CONF_THRESHOLD)
        latency = time.perf_counter() - t0

        if _PROM:
            INFERENCE_REQUESTS.labels(model=MODEL_NAME, status="ok").inc()
            INFERENCE_LATENCY.labels(model=MODEL_NAME).observe(latency)
            DETECTED_OBJECTS.labels(model=MODEL_NAME).observe(result["count"])
            if result["count"]:
                avg_conf = sum(d["score"] for d in result["detections"]) / result["count"]
                CONFIDENCE_GAUGE.labels(model=MODEL_NAME).set(avg_conf)

        result["latency_ms"] = round(latency * 1000, 2)
        return JSONResponse(result)

    except HTTPException:
        if _PROM:
            INFERENCE_REQUESTS.labels(model=MODEL_NAME, status="error").inc()
        raise
    except Exception as e:
        if _PROM:
            INFERENCE_REQUESTS.labels(model=MODEL_NAME, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/image")
async def predict_image(file: Annotated[UploadFile, File()]) -> Response:
    """Return annotated image as PNG."""
    raw = await file.read()
    inp, ratio, pad_meta = _preprocess(raw)
    logits, boxes = _state["session"].run(None, {_state["input_name"]: inp})
    result = _postprocess(logits, boxes, (ratio, pad_meta[:2], pad_meta[2], pad_meta[3]),
                           CONF_THRESHOLD)

    nparr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    for det in result["detections"]:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"{det['label']} {det['score']:.2f}",
                    (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    _, buf = cv2.imencode(".png", img)
    return Response(buf.tobytes(), media_type="image/png")


def serve(model_path: str, host: str = "0.0.0.0", port: int = 8080) -> None:
    _state["model_path"] = model_path
    uvicorn.run(app, host=host, port=port)
