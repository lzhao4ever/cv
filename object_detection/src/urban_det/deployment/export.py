"""Export trained detection model to ONNX or TensorRT.

Usage:
  python scripts/export_model.py checkpoint=best.ckpt deployment=default    # ONNX
  python scripts/export_model.py checkpoint=best.ckpt deployment=tensorrt   # TRT
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from rich.console import Console

from urban_det.models import DetectionModel
from urban_det.training.lit_module import DetectionLitModule

console = Console()


def load_model_from_checkpoint(checkpoint_path: str | Path) -> tuple[DetectionModel, DictConfig]:
    ckpt_path = Path(checkpoint_path)
    lit = DetectionLitModule.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    lit.eval()
    return lit.model, lit.cfg


def export_onnx(
    model: DetectionModel,
    output_path: Path,
    img_size: tuple[int, int],
    opset: int = 17,
    simplify: bool = True,
    half: bool = False,
    dynamic_axes: dict | None = None,
) -> Path:
    model.eval()
    device = next(model.parameters()).device

    if half:
        model = model.half()

    dummy = torch.zeros(1, 3, *img_size, device=device)
    if half:
        dummy = dummy.half()

    dynamic_axes = dynamic_axes or {"images": {0: "batch_size"}}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
        dynamic_axes=dynamic_axes,
    )
    console.print(f"[green]ONNX exported:[/] {output_path}")

    if simplify:
        try:
            import onnx
            import onnxsim
            model_onnx = onnx.load(str(output_path))
            model_simplified, ok = onnxsim.simplify(model_onnx)
            if ok:
                onnx.save(model_simplified, str(output_path))
                console.print("[green]ONNX simplified successfully[/]")
            else:
                console.print("[yellow]ONNX simplification failed — keeping original[/]")
        except ImportError:
            console.print("[yellow]onnxsim not installed — skipping simplification[/]")

    return output_path


def export_tensorrt(
    onnx_path: Path,
    output_path: Path,
    precision: str = "fp16",
    workspace_gb: int = 4,
    calibration_cfg: DictConfig | None = None,
    dla_core: int = -1,
) -> Path:
    """Convert ONNX → TensorRT engine for edge deployment (Jetson / NVIDIA Drive)."""
    try:
        import tensorrt as trt
    except ImportError:
        raise RuntimeError(
            "TensorRT not installed. Install with: pip install tensorrt. "
            "See configs/deployment/tensorrt.yaml for DLA/precision options."
        )

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                console.print(f"[red]TRT parse error:[/] {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX for TensorRT")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # TODO: attach calibrator from calibration_cfg

    if dla_core >= 0:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = builder.build_serialized_network(network, config)
    with open(output_path, "wb") as f:
        f.write(serialized)

    console.print(f"[green]TensorRT engine saved:[/] {output_path}  ({precision})")
    return output_path


def export_model(cfg: DictConfig, checkpoint_path: str | Path) -> Path:
    """Main export entry point — dispatches to ONNX or TensorRT based on config."""
    model, model_cfg = load_model_from_checkpoint(checkpoint_path)
    img_size = tuple(cfg.data.image_size)
    deploy_cfg = cfg.deployment
    stem = Path(checkpoint_path).stem

    if deploy_cfg.format == "onnx":
        out = Path(cfg.project.output_dir) / f"{stem}.onnx"
        return export_onnx(
            model, out, img_size,
            opset=deploy_cfg.get("opset", 17),
            simplify=deploy_cfg.get("simplify", True),
            half=deploy_cfg.get("half", False),
        )

    if deploy_cfg.format == "tensorrt":
        onnx_path = Path(cfg.project.output_dir) / f"{stem}.onnx"
        if not onnx_path.exists():
            export_onnx(model, onnx_path, img_size)
        out = Path(cfg.project.output_dir) / f"{stem}.engine"
        return export_tensorrt(
            onnx_path, out,
            precision=deploy_cfg.get("precision", "fp16"),
            workspace_gb=deploy_cfg.get("workspace_gb", 4),
            dla_core=deploy_cfg.get("dla_core", -1),
        )

    raise ValueError(f"Unknown export format: {deploy_cfg.format}")
