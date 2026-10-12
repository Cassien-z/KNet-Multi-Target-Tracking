"""Build a deterministic 10 Hz compatibility view from registered V8 PKLs.

The high-fidelity 40 Hz source files are never modified. Every array whose
leading dimension matches ``t_grid`` is sampled with the same time indices so
truth, platform pose, measurements, masks, and cooperative links remain
aligned.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "val_data_v8_interference_registered"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "val_data_v8_interference_registered_10hz"


def select_resample_indices(
    t_grid: np.ndarray, target_period_s: float = 0.1
) -> np.ndarray:
    """Return source indices nearest to a regular grid anchored at the first frame."""
    times = np.asarray(t_grid, dtype=np.float64).reshape(-1)
    if times.size < 2:
        raise ValueError("t_grid must contain at least two frames")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("t_grid must be finite and strictly increasing")
    if not np.isfinite(target_period_s) or target_period_s <= 0.0:
        raise ValueError("target_period_s must be a positive finite value")

    source_period_s = float(np.median(np.diff(times)))
    if target_period_s + 1e-12 < source_period_s:
        raise ValueError("target period cannot be shorter than the source period")

    frame_count = int(
        np.floor((times[-1] - times[0] + source_period_s * 0.5) / target_period_s)
    ) + 1
    wanted = times[0] + np.arange(frame_count, dtype=np.float64) * target_period_s
    right = np.searchsorted(times, wanted, side="left")
    right = np.clip(right, 0, times.size - 1)
    left = np.clip(right - 1, 0, times.size - 1)
    choose_left = np.abs(times[left] - wanted) <= np.abs(times[right] - wanted)
    indices = np.where(choose_left, left, right).astype(np.int64)

    tolerance = max(source_period_s * 0.51, 1e-9)
    if np.any(np.abs(times[indices] - wanted) > tolerance):
        raise ValueError("source t_grid does not cover the requested regular grid")
    if np.unique(indices).size != indices.size:
        raise ValueError("resampling selected duplicate source frames")
    return indices


def _slice_time_axes(value: Any, source_frames: int, indices: np.ndarray) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim and value.shape[0] == source_frames:
            return value[indices].copy()
        return value.copy()
    if isinstance(value, dict):
        return {
            key: _slice_time_axes(item, source_frames, indices)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_slice_time_axes(item, source_frames, indices) for item in value]
    if isinstance(value, tuple):
        return tuple(_slice_time_axes(item, source_frames, indices) for item in value)
    return copy.deepcopy(value)


def resample_payload(
    payload: dict[str, Any], target_period_s: float = 0.1
) -> tuple[dict[str, Any], np.ndarray]:
    """Resample every time-aligned ndarray in one registered PKL payload."""
    if "t_grid" not in payload:
        raise ValueError("registered payload is missing t_grid")
    source_times = np.asarray(payload["t_grid"], dtype=np.float64).reshape(-1)
    indices = select_resample_indices(source_times, target_period_s)
    result = _slice_time_axes(payload, source_times.size, indices)
    result["t_grid"] = source_times[indices].copy()
    result["CompatibilityView"] = {
        "method": "nearest_frame_on_regular_grid",
        "source_frame_period_s": float(np.median(np.diff(source_times))),
        "target_frame_period_s": float(target_period_s),
        "source_frame_count": int(source_times.size),
        "target_frame_count": int(indices.size),
        "source_indices": indices.copy(),
    }
    validate_compatibility_payload(result, target_period_s)
    return result, indices


def validate_compatibility_payload(
    payload: dict[str, Any], target_period_s: float = 0.1
) -> None:
    """Validate the fields consumed by the deployed tracker."""
    times = np.asarray(payload.get("t_grid", []), dtype=np.float64).reshape(-1)
    if times.size < 2 or not np.allclose(
        np.diff(times), target_period_s, rtol=0.0, atol=1e-8
    ):
        raise ValueError("output t_grid is not a regular target-period grid")
    frame_count = times.size
    for name in ("UAV_Pos", "UAV_Rotmat"):
        value = np.asarray(payload.get(name))
        if not value.ndim or value.shape[0] != frame_count:
            raise ValueError(f"{name} is not aligned with output t_grid")

    tracks = payload.get("RegisteredTrack")
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("RegisteredTrack must be a non-empty list")
    for track_index, track in enumerate(tracks):
        for name in ("target_pos", "target_vel"):
            value = np.asarray(track.get(name))
            if not value.ndim or value.shape[0] != frame_count:
                raise ValueError(
                    f"RegisteredTrack[{track_index}].{name} is not time aligned"
                )
        for sensor_name in ("Radar", "EO", "ESM", "FastLink"):
            for sensor_index, sensor in enumerate(track.get(sensor_name, [])):
                for field_name, value in sensor.items():
                    if isinstance(value, np.ndarray) and value.ndim:
                        if value.shape[0] != frame_count:
                            raise ValueError(
                                f"RegisteredTrack[{track_index}].{sensor_name}"
                                f"[{sensor_index}].{field_name} is not time aligned"
                            )


def build_compatibility_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    target_period_s: float = 0.1,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("source and output directories must be different")
    source_files = sorted(source_dir.glob("*.pkl"))
    if not source_files:
        raise FileNotFoundError(f"no registered PKL files found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, Any]] = []
    for source_path in source_files:
        output_path = output_dir / source_path.name
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"output already exists: {output_path}; pass --overwrite to replace it"
            )
        with source_path.open("rb") as handle:
            payload = pickle.load(handle)
        converted, indices = resample_payload(payload, target_period_s)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(converted, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(output_path)
        files.append(
            {
                "file": source_path.name,
                "source_frames": int(np.asarray(payload["t_grid"]).size),
                "output_frames": int(indices.size),
                "first_source_index": int(indices[0]),
                "last_source_index": int(indices[-1]),
            }
        )

    manifest = {
        "source_directory": str(source_dir),
        "output_directory": str(output_dir),
        "method": "nearest frame at 0.1 s intervals; no measurement averaging",
        "target_frame_period_s": float(target_period_s),
        "files": files,
    }
    manifest_path = output_dir / "compatibility_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a 10 Hz compatibility view from registered V8 PKLs"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-period", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build_compatibility_dataset(
        args.source,
        args.output,
        target_period_s=args.target_period,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
