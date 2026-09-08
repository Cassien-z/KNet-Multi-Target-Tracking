"""Model discovery and atomic activation for the prototype backend."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tracking_system.backend.runtime import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT,
    TransformerEngine,
)


MODEL_CATALOG: dict[str, dict[str, str | Path]] = {
    "v5": {
        "name": "V5 标准数据集最佳模型",
        "run_id": "run_20260715_194636",
        "training_dataset": "train_data_v5_registered",
        "validation_dataset": "val_data_v5_registered",
        "path": PROJECT_ROOT
        / "results"
        / "runs_standard(v5)"
        / "run_20260715_194636"
        / "best_model.pt",
    },
    "v6": {
        "name": "V6 降噪数据集最佳模型",
        "run_id": "run_20260717_152359",
        "training_dataset": "train_data_v6_registered",
        "validation_dataset": "val_data_v6_registered",
        "path": PROJECT_ROOT
        / "results"
        / "runs_30off(v6)"
        / "run_20260717_152359"
        / "best_model.pt",
    },
    "v7": {
        "name": "V7 低噪数据集最佳模型",
        "run_id": "run_20260728_103728",
        "training_dataset": "train_data_v7_registered",
        "validation_dataset": "val_data_v7_registered",
        "path": PROJECT_ROOT
        / "results"
        / "runs_40off(v7)"
        / "run_20260728_103728"
        / "best_model.pt",
    },
}


class ModelManager:
    """Maintain a registry of model weights and one active inference engine."""

    def __init__(
        self,
        default_model_path: str | Path | None = None,
        data_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        configured_model = os.getenv("TRACKER_MODEL_PATH")
        self.default_model_path = Path(
            default_model_path or configured_model or DEFAULT_MODEL_PATH
        ).expanduser().resolve()
        self.data_dir = Path(
            data_dir or os.getenv("TRACKER_DATA_DIR") or DEFAULT_DATA_DIR
        ).expanduser().resolve()
        self.requested_device = device or os.getenv("TRACKER_DEVICE", "auto")

        self._state_lock = threading.RLock()
        self._activation_lock = threading.Lock()
        self._models: dict[str, Path] = {}
        self._engine: TransformerEngine | None = None
        self._active_model_id: str | None = None
        self._activated_at: str | None = None
        self._last_load_ms: float | None = None

        self.refresh()
        default_id = self._id_for_path(self.default_model_path)
        if default_id not in self._models:
            raise FileNotFoundError(
                f"Default model is outside the registry or missing: {self.default_model_path}"
            )
        self.activate(default_id, self.requested_device)

    @staticmethod
    def _id_for_path(path: Path) -> str:
        resolved = path.resolve()
        for model_id, metadata in MODEL_CATALOG.items():
            if Path(metadata["path"]).resolve() == resolved:
                return model_id
        return ""

    def refresh(self) -> list[dict[str, Any]]:
        discovered: dict[str, Path] = {}
        for model_id, metadata in MODEL_CATALOG.items():
            path = Path(metadata["path"]).resolve()
            if path.is_file():
                discovered[model_id] = path
        with self._state_lock:
            self._models = discovered
        return self.list_models()

    @staticmethod
    def _read_summary(path: Path) -> dict[str, Any] | None:
        summary_path = path.with_name("metrics_summary.json")
        if not summary_path.is_file():
            return None
        try:
            raw = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        accuracy = raw.get("metrics_wo_warmup") or raw.get("metrics_all") or {}
        full_latency = (raw.get("mode_latency_ms") or {}).get("full") or {}
        return {
            "rmse_m": accuracy.get("rmse"),
            "mae_m": accuracy.get("mae"),
            "p95_m": accuracy.get("p95"),
            "max_error_m": accuracy.get("max"),
            "mean_latency_ms": full_latency.get("mean"),
            "p95_latency_ms": full_latency.get("p95"),
            "epochs": raw.get("epochs"),
            "d_model": raw.get("d_model"),
        }

    def _describe(self, model_id: str, path: Path) -> dict[str, Any]:
        stat = path.stat()
        catalog = MODEL_CATALOG[model_id]
        with self._state_lock:
            active = model_id == self._active_model_id
        return {
            "id": model_id,
            "name": catalog["name"],
            "version": model_id.upper(),
            "run_id": catalog["run_id"],
            "training_dataset": catalog["training_dataset"],
            "validation_dataset": catalog["validation_dataset"],
            "file_name": path.name,
            "path": str(path),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 3),
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "active": active,
            "metrics": self._read_summary(path),
        }

    def list_models(self) -> list[dict[str, Any]]:
        with self._state_lock:
            items = list(self._models.items())
        return [self._describe(model_id, path) for model_id, path in items]

    def get_model_path(self, model_id: str) -> Path:
        with self._state_lock:
            path = self._models.get(model_id)
        if path is None:
            raise KeyError(f"Unknown model id: {model_id}")
        return path

    def available_devices(self) -> list[dict[str, Any]]:
        devices: list[dict[str, Any]] = [
            {"id": "cpu", "name": "CPU", "available": True}
        ]
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "id": f"cuda:{index}",
                        "name": props.name,
                        "available": True,
                        "total_memory_bytes": props.total_memory,
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
        else:
            devices.append({"id": "cuda:0", "name": "CUDA", "available": False})
        return devices

    def get_engine(self) -> TransformerEngine:
        with self._state_lock:
            if self._engine is None:
                raise RuntimeError("No model is currently active")
            return self._engine

    def active_info(self) -> dict[str, Any]:
        engine = self.get_engine()
        with self._state_lock:
            model_id = self._active_model_id
            activated_at = self._activated_at
            load_ms = self._last_load_ms
        return {
            "model_id": model_id,
            "activated_at": activated_at,
            "load_time_ms": load_ms,
            **engine.info(),
        }

    def activate(self, model_id: str, device: str = "auto") -> dict[str, Any]:
        with self._activation_lock:
            with self._state_lock:
                path = self._models.get(model_id)
                current_engine = self._engine
                current_id = self._active_model_id
            if path is None:
                raise KeyError(f"Unknown model id: {model_id}")

            resolved_device = TransformerEngine._select_device(device)
            if (
                current_engine is not None
                and current_id == model_id
                and current_engine.device == resolved_device
            ):
                return {"changed": False, **self.active_info()}

            started = time.perf_counter()
            candidate = TransformerEngine(
                model_path=path,
                data_dir=self.data_dir,
                device=str(resolved_device),
            )
            load_ms = (time.perf_counter() - started) * 1000.0

            with self._state_lock:
                self._engine = candidate
                self._active_model_id = model_id
                self._activated_at = datetime.now(timezone.utc).isoformat()
                self._last_load_ms = float(load_ms)
            return {"changed": True, **self.active_info()}
