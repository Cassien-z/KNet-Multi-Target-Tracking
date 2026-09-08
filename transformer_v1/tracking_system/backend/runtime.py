"""Inference runtime for the deployed end-to-end cooperative tracker."""

from __future__ import annotations

import os
import pickle
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import _parse_registered_pkl
from src.model import CoopTrackerModel


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "runs_40off(v7)"
    / "run_20260728_103728"
    / "best_model.pt"
)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "val_data_v7_registered"


def _configured_path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    return Path(value).expanduser().resolve() if value else default.resolve()


@dataclass(frozen=True)
class TrackingSequence:
    scenario_id: str
    scenario_name: str
    target_index: int
    times: np.ndarray
    truth_states: np.ndarray
    predicted_states: np.ndarray
    position_errors: np.ndarray
    sequence_inference_ms: float
    per_frame_inference_ms: float

    @property
    def frame_count(self) -> int:
        return int(self.times.size)

    @property
    def max_time(self) -> float:
        return float(self.times[-1]) if self.frame_count else 0.0


class TransformerEngine:
    """Load the trained Transformer once and execute complete PKL sequences."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        data_dir: str | Path | None = None,
        device: str | None = None,
        seq_len: int = 100,
        num_uavs: int = 5,
    ) -> None:
        self.model_path = (
            Path(model_path).expanduser().resolve()
            if model_path
            else _configured_path("TRACKER_MODEL_PATH", DEFAULT_MODEL_PATH)
        )
        self.data_dir = (
            Path(data_dir).expanduser().resolve()
            if data_dir
            else _configured_path("TRACKER_DATA_DIR", DEFAULT_DATA_DIR)
        )
        self.seq_len = int(os.getenv("TRACKER_SEQ_LEN", str(seq_len)))
        self.num_uavs = num_uavs
        self.device = self._select_device(device or os.getenv("TRACKER_DEVICE", "auto"))
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, int], TrackingSequence] = {}

        self._validate_paths()
        self.model = self._load_model()

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        normalized = requested.strip().lower()
        if normalized == "auto":
            normalized = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif normalized == "cuda":
            normalized = "cuda:0"
        selected = torch.device(normalized)
        if selected.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("TRACKER_DEVICE requests CUDA, but CUDA is unavailable")
            index = selected.index if selected.index is not None else 0
            if index >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device index is unavailable: {index}")
            selected = torch.device(f"cuda:{index}")
        return selected

    def _validate_paths(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Registered dataset not found: {self.data_dir}")

    def _load_model(self) -> CoopTrackerModel:
        model = CoopTrackerModel(d_model=96).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        return model

    def scenario_files(self) -> list[Path]:
        return sorted(self.data_dir.glob("*.pkl"))

    def list_scenarios(self) -> list[dict[str, str | int]]:
        return [
            {
                "id": str(100 + index),
                "name": path.stem,
                "file": path.name,
                "size_bytes": path.stat().st_size,
            }
            for index, path in enumerate(self.scenario_files())
        ]

    def resolve_scenario(self, scenario_id: str) -> tuple[str, Path]:
        files = self.scenario_files()
        token = str(scenario_id).strip()
        if token.isdecimal():
            index = int(token) - 100
            if 0 <= index < len(files):
                return token, files[index]
            raise ValueError(f"Unknown scenario id: {scenario_id}")

        safe_name = Path(token).name
        if safe_name != token:
            raise ValueError("Scenario must be an id or a PKL file name")
        wanted = safe_name[:-4] if safe_name.lower().endswith(".pkl") else safe_name
        for index, path in enumerate(files):
            if path.stem == wanted:
                return str(100 + index), path
        raise ValueError(f"Unknown scenario: {scenario_id}")

    def _prepare_sample(self, data: dict, path: Path, target_index: int) -> dict:
        samples = _parse_registered_pkl(data, self.seq_len, self.num_uavs, path.name)
        if not samples:
            raise ValueError(f"No registered targets in {path.name}")
        if not 0 <= target_index < len(samples):
            raise ValueError(
                f"Target index {target_index} is out of range for {path.name}; "
                f"available targets: 0..{len(samples) - 1}"
            )

        sample = samples[target_index]
        steps = int(sample["radar"].shape[0])
        agents = int(sample["radar"].shape[1])
        zeros = torch.zeros
        defaults = {
            "imu": zeros((steps, agents, 6), dtype=torch.float32),
            "fast_link": zeros((steps, agents, 3), dtype=torch.float32),
            "slow_link": zeros((steps, agents, 3), dtype=torch.float32),
            "fast_age": zeros((steps, agents), dtype=torch.float32),
            "slow_age": zeros((steps, agents), dtype=torch.float32),
            "fast_update": zeros((steps, agents), dtype=torch.float32),
            "slow_update": zeros((steps, agents), dtype=torch.float32),
            "esm_mode": zeros((steps, agents), dtype=torch.float32),
        }
        for name, value in defaults.items():
            sample.setdefault(name, value)
        return sample

    def _forward(self, sample: dict) -> tuple[np.ndarray, float]:
        def batch(name: str) -> torch.Tensor:
            value = sample[name]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            return value.to(self.device).unsqueeze(0)

        dt = sample["dt"]
        if not isinstance(dt, torch.Tensor):
            dt = torch.tensor(dt, dtype=torch.float32)
        dt = dt.to(self.device).reshape(1)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = self.model(
                radar=batch("radar"),
                eo=batch("eo"),
                esm=batch("esm"),
                radar_mask=batch("radar_mask"),
                eo_mask=batch("eo_mask"),
                esm_mask=batch("esm_mask"),
                ref_pos=batch("ref_pos"),
                rotmat_b2e=batch("rotmat_b2e"),
                uav_pos=batch("uav_pos"),
                fast_link=batch("fast_link"),
                slow_link=batch("slow_link"),
                fast_update=batch("fast_update"),
                slow_update=batch("slow_update"),
                imu=batch("imu"),
                fast_age=batch("fast_age"),
                slow_age=batch("slow_age"),
                esm_mode=batch("esm_mode"),
                dt=dt,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return prediction[0].detach().cpu().numpy(), elapsed_ms

    def infer_scenario(self, scenario_id: str, target_index: int = 0) -> TrackingSequence:
        resolved_id, path = self.resolve_scenario(scenario_id)
        cache_key = (str(path), int(target_index))
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

            with path.open("rb") as handle:
                data = pickle.load(handle)
            sample = self._prepare_sample(data, path, target_index)
            predicted, elapsed_ms = self._forward(sample)

            truth = sample["label_abs"].detach().cpu().numpy()
            valid = sample["target_valid_mask"].detach().cpu().numpy().astype(bool)
            frame_indices = np.flatnonzero(valid)
            if frame_indices.size == 0:
                raise ValueError(f"No valid ground-truth frames in {path.name}")

            raw_times = np.asarray(data.get("t_grid", []), dtype=np.float64).reshape(-1)
            if raw_times.size < self.seq_len:
                dt_value = float(sample["dt"].item())
                raw_times = np.arange(self.seq_len, dtype=np.float64) * dt_value
            times = raw_times[: self.seq_len][frame_indices]
            truth = truth[frame_indices]
            predicted = predicted[frame_indices]
            errors = np.linalg.norm(
                predicted[:, [0, 2, 4]] - truth[:, [0, 2, 4]], axis=1
            )
            if not np.all(np.isfinite(errors)):
                raise RuntimeError(f"Model produced non-finite values for {path.name}")

            result = TrackingSequence(
                scenario_id=resolved_id,
                scenario_name=path.stem,
                target_index=target_index,
                times=times,
                truth_states=truth,
                predicted_states=predicted,
                position_errors=errors,
                sequence_inference_ms=float(elapsed_ms),
                per_frame_inference_ms=float(elapsed_ms / frame_indices.size),
            )
            self._cache[cache_key] = result
            return result

    def info(self) -> dict[str, str | int | bool]:
        return {
            "model": "CoopTrackerModel",
            "architecture": "Learned-K + dual-stream Transformer + collaborative GAT",
            "weights": str(self.model_path),
            "data_dir": str(self.data_dir),
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "sequence_length": self.seq_len,
            "num_uavs": self.num_uavs,
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "scenario_count": len(self.scenario_files()),
        }
