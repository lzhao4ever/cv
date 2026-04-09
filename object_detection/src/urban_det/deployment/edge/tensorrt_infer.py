"""TensorRT inference engine wrapper for on-vehicle deployment.

Designed for NVIDIA Jetson Orin / NVIDIA Drive Orin (Ampere GPU).
Supports FP16 and INT8 engines built by export.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


class TRTDetector:
    """Run a serialized TensorRT engine for real-time detection.

    Args:
        engine_path: Path to .engine file produced by export_model().
        input_shape: (H, W) expected by the engine.
        conf_threshold: Minimum sigmoid score for a detection.

    Usage:
        detector = TRTDetector("model.engine")
        detections = detector.infer(bgr_image_np)  # → list of dicts
    """

    def __init__(
        self,
        engine_path: str | Path,
        input_shape: tuple[int, int] = (640, 640),
        conf_threshold: float = 0.3,
    ):
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(engine_path)

        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa: F401 — initializes CUDA context
            import pycuda.driver as cuda
        except ImportError as e:
            raise ImportError(
                "TensorRT + pycuda required. Install CUDA 12 + TensorRT 10 on Jetson."
            ) from e

        self._trt = trt
        self._cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_shape = input_shape
        self.conf_threshold = conf_threshold
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        H, W = self.input_shape
        cuda = self._cuda
        self.h_input = np.zeros((1, 3, H, W), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)

        # Outputs: logits (1, Q, C) and boxes (1, Q, 4)
        self.h_logits = np.zeros((1, 300, 80), dtype=np.float32)
        self.h_boxes = np.zeros((1, 300, 4), dtype=np.float32)
        self.d_logits = cuda.mem_alloc(self.h_logits.nbytes)
        self.d_boxes = cuda.mem_alloc(self.h_boxes.nbytes)

        self.stream = cuda.Stream()

    def _preprocess(self, bgr: np.ndarray) -> None:
        import cv2
        H, W = self.input_shape
        h0, w0 = bgr.shape[:2]
        ratio = min(H / h0, W / w0)
        nh, nw = int(h0 * ratio), int(w0 * ratio)
        resized = cv2.resize(bgr, (nw, nh))
        canvas = np.full((H, W, 3), 114, dtype=np.uint8)
        py, px = (H - nh) // 2, (W - nw) // 2
        canvas[py:py + nh, px:px + nw] = resized
        rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
        self.h_input[0] = rgb.transpose(2, 0, 1)
        self._ratio = ratio
        self._pad = (px, py)
        self._orig = (w0, h0)

    def infer(self, bgr: np.ndarray) -> list[dict]:
        cuda = self._cuda
        self._preprocess(bgr)

        t0 = time.perf_counter()
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_logits), int(self.d_boxes)],
            stream_handle=self.stream.handle,
        )
        cuda.memcpy_dtoh_async(self.h_logits, self.d_logits, self.stream)
        cuda.memcpy_dtoh_async(self.h_boxes, self.d_boxes, self.stream)
        self.stream.synchronize()
        self._last_latency_ms = (time.perf_counter() - t0) * 1000

        return self._decode()

    def _decode(self) -> list[dict]:
        H, W = self.input_shape
        px, py = self._pad
        w0, h0 = self._orig
        ratio = self._ratio

        scores = (1 / (1 + np.exp(-self.h_logits[0]))).max(axis=-1)
        labels = self.h_logits[0].argmax(axis=-1)
        mask = scores > self.conf_threshold

        results = []
        for box, score, label in zip(self.h_boxes[0][mask], scores[mask], labels[mask]):
            cx, cy, bw, bh = box
            x1 = max(0, (cx - bw / 2) * W - px) / ratio
            y1 = max(0, (cy - bh / 2) * H - py) / ratio
            x2 = min(w0, (cx + bw / 2) * W - px) / ratio
            y2 = min(h0, (cy + bh / 2) * H - py) / ratio
            results.append({"box": [x1, y1, x2, y2], "score": float(score), "label": int(label)})
        return results

    @property
    def latency_ms(self) -> float:
        return self._last_latency_ms
