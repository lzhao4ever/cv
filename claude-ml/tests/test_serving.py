"""Integration tests for the FastAPI inference server."""

from __future__ import annotations

import io
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def mock_ort_session():
    """Return a mock ONNX Runtime session that outputs zero logits."""
    session = MagicMock()
    # Simulate (1, 19, 512, 1024) output
    session.run.return_value = [np.zeros((1, 19, 512, 1024), dtype=np.float32)]
    return session


@pytest.fixture
def client(mock_ort_session):
    """FastAPI test client with a mocked ONNX session."""
    with patch.dict(os.environ, {"MODEL_PATH": "fake_model.onnx"}):
        with patch("urban_seg.deployment.server._session", mock_ort_session):
            from urban_seg.deployment.server import app
            with TestClient(app) as c:
                yield c


def _make_upload(width: int = 1024, height: int = 512) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_mask_returns_png(client):
    resp = client.post(
        "/predict/mask",
        files={"file": ("test.jpg", _make_upload(), "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"


def test_predict_classes_returns_json(client):
    resp = client.post(
        "/predict/classes",
        files={"file": ("test.jpg", _make_upload(), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "mask" in data
    assert data["height"] == 512
    assert data["width"] == 1024
