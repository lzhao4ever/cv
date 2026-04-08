"""Export trained models to ONNX and TorchScript for production serving."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

try:
    import onnx
    import onnxsim

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    image_size: tuple[int, int] = (512, 1024),
    opset_version: int = 17,
    simplify: bool = True,
    dynamic_batch: bool = True,
) -> Path:
    """
    Export model to ONNX.

    Args:
        model: trained segmentation model (on CPU or CUDA).
        output_path: destination ``.onnx`` file path.
        image_size: (H, W) of the export dummy input.
        opset_version: ONNX opset to target.
        simplify: run onnx-simplifier after export.
        dynamic_batch: register batch dimension as dynamic.

    Returns:
        resolved path of the exported file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.zeros(1, 3, *image_size, device=next(model.parameters()).device)

    dynamic_axes: dict | None = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    if simplify and _ONNX_AVAILABLE:
        onnx_model = onnx.load(str(output_path))
        simplified, ok = onnxsim.simplify(onnx_model)
        if ok:
            onnx.save(simplified, str(output_path))

    print(f"ONNX model saved → {output_path}")
    return output_path


def export_torchscript(
    model: nn.Module,
    output_path: str | Path,
    image_size: tuple[int, int] = (512, 1024),
    method: str = "trace",
) -> Path:
    """
    Export model to TorchScript via tracing or scripting.

    Args:
        model: trained segmentation model.
        output_path: destination ``.pt`` file path.
        image_size: (H, W) for the tracing dummy input.
        method: ``"trace"`` or ``"script"``.

    Returns:
        resolved path of the exported file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    if method == "trace":
        dummy = torch.zeros(1, 3, *image_size, device=next(model.parameters()).device)
        with torch.no_grad():
            scripted = torch.jit.trace(model, dummy)
    elif method == "script":
        scripted = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown export method: {method!r}")

    scripted.save(str(output_path))
    print(f"TorchScript model saved → {output_path}")
    return output_path
