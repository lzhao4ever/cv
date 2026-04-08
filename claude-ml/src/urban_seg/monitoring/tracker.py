"""Unified experiment tracker: MLflow + optional W&B side-car."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch


class ExperimentTracker:
    """
    Thin wrapper providing a consistent interface over MLflow (primary) and
    optionally W&B (secondary mirror).

    Usage::

        tracker = ExperimentTracker(experiment_name="urban-seg", tracking_uri="http://localhost:5000")
        with tracker.start_run(run_name="segformer-b2-cityscapes"):
            tracker.log_params({"backbone": "mit_b2", "lr": 6e-5})
            tracker.log_metrics({"val/mIoU": 0.782}, step=10)
            tracker.log_model(model, "segmentation_model")
    """

    def __init__(
        self,
        experiment_name: str = "urban-seg",
        tracking_uri: str = "http://localhost:5000",
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._wandb_enabled = wandb_project is not None and wandb_project != ""
        self._wandb_project = wandb_project
        self._wandb_entity = wandb_entity
        self._wandb_run = None

    # ------------------------------------------------------------------
    def start_run(self, run_name: str | None = None, tags: dict | None = None):
        """Context manager — use as ``with tracker.start_run(): ...``"""
        mlflow.start_run(run_name=run_name, tags=tags)
        if self._wandb_enabled:
            import wandb
            self._wandb_run = wandb.init(
                project=self._wandb_project,
                entity=self._wandb_entity,
                name=run_name,
                tags=list((tags or {}).values()),
                reinit=True,
            )
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end_run()

    def end_run(self) -> None:
        mlflow.end_run()
        if self._wandb_run is not None:
            self._wandb_run.finish()

    # ------------------------------------------------------------------
    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)
        if self._wandb_run:
            self._wandb_run.config.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)
        if self._wandb_run:
            self._wandb_run.log({**metrics, **({"_step": step} if step is not None else {})})

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(local_path), artifact_path)
        if self._wandb_run:
            import wandb
            self._wandb_run.save(str(local_path))

    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log a PyTorch model to MLflow model registry."""
        mlflow.pytorch.log_model(model, artifact_path)

    # ------------------------------------------------------------------
    @staticmethod
    def get_best_run(experiment_name: str, metric: str = "val/mIoU") -> dict:
        """Retrieve the run with the highest value of ``metric``."""
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return {}
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if not runs:
            return {}
        r = runs[0]
        return {
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "metrics": r.data.metrics,
            "params": r.data.params,
        }
