"""Unified experiment tracker: MLflow primary, W&B optional mirror.

Also exposes Prometheus metrics for the inference server.
"""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
from omegaconf import DictConfig, OmegaConf

# Prometheus (optional)
with suppress(ImportError):
    from prometheus_client import Counter, Gauge, Histogram

    INFERENCE_REQUESTS = Counter("det_inference_requests_total", "Total inference requests",
                                 ["model", "status"])
    INFERENCE_LATENCY = Histogram("det_inference_latency_seconds", "Inference latency",
                                  ["model"], buckets=[.005, .01, .025, .05, .1, .25, .5, 1.0])
    DETECTED_OBJECTS = Histogram("det_detected_objects_per_image", "Objects per image",
                                 ["model"], buckets=[0, 1, 2, 5, 10, 20, 50, 100])
    CONFIDENCE_GAUGE = Gauge("det_avg_confidence", "Average detection confidence", ["model"])
    DATA_DRIFT_SCORE = Gauge("det_data_drift_score", "Feature distribution drift score", ["model"])


class ExperimentTracker:
    """Wraps MLflow with optional W&B mirroring."""

    def __init__(
        self,
        cfg: DictConfig,
        run_name: str | None = None,
        use_wandb: bool = False,
    ):
        self.cfg = cfg
        self.use_wandb = use_wandb
        self._wandb_run = None

        # MLflow
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(cfg.project.experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # Log full config as artifact
        cfg_path = Path(cfg.project.output_dir) / "config.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, cfg_path)
        mlflow.log_artifact(str(cfg_path))

        # W&B
        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=cfg.project.name,
                    name=run_name,
                    config=OmegaConf.to_container(cfg, resolve=True),
                )
            except ImportError:
                self.use_wandb = False

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)
        if self.use_wandb and self._wandb_run:
            self._wandb_run.log(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        mlflow.pytorch.log_model(model, artifact_path)

    def log_artifact(self, path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    def finish(self) -> None:
        mlflow.end_run()
        if self.use_wandb and self._wandb_run:
            self._wandb_run.finish()

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *args: Any) -> None:
        self.finish()
