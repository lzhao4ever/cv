"""Tests for the FastAPI inference server."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_session():
    """ONNX Runtime session mock returning zeros."""
    sess = MagicMock()
    # pred_logits: (1, 10, 80), pred_boxes: (1, 10, 4)
    sess.run.return_value = [
        np.zeros((1, 10, 80), dtype=np.float32),
        np.full((1, 10, 4), 0.5, dtype=np.float32),
    ]
    sess.get_inputs.return_value = [MagicMock(name="images")]
    return sess


@pytest.fixture
def test_client(mock_session, tmp_path):
    from urban_det.deployment.server import _state, app

    # Write a dummy model file so lifespan won't fail on path check
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dummy")
    _state["model_path"] = str(model_path)

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        with TestClient(app) as client:
            yield client

    _state.clear()


def _png_bytes() -> bytes:
    """Return a minimal valid PNG (1×1 white pixel)."""
    import cv2
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_health(test_client):
    resp = test_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_json(test_client):
    resp = test_client.post(
        "/predict",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "detections" in data
    assert "count" in data
    assert "latency_ms" in data


def test_predict_image_returns_png(test_client):
    resp = test_client.post(
        "/predict/image",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"


def test_predict_bad_file(test_client):
    resp = test_client.post(
        "/predict",
        files={"file": ("bad.bin", b"notanimage", "application/octet-stream")},
    )
    assert resp.status_code == 400
