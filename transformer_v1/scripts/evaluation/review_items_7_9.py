"""Generate auditable evidence for midterm-review items 7 through 9.

The audit does not retrain or modify the model. It fixes the stable-stage RMSE
boundary, benchmarks one-frame model-forward latency with a recorded runtime
environment, and reports the 8 m RMSE and 3-second maneuver-distance criteria
side by side without treating them as interchangeable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import CoopTrackerModel
from tracking_system.backend.batch_testing import (
    _BatchRegisteredPklDataset,
    _predict_batch,
    position_rmse_definition,
    stable_stage_mask,
)
from tracking_system.backend.scenario_catalog import (
    class_counts,
    selected_dataset_files,
)


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "results"
    / "runs_40off(v7)"
    / "run_20260728_103728"
    / "best_model.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "midterm_review"
    / "items_7_9_evidence.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_position_errors(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
    warmup_frames: int,
) -> torch.Tensor:
    """Return 3-D position errors after the initial transition stage."""
    errors = torch.linalg.vector_norm(
        prediction[..., [0, 2, 4]] - truth[..., [0, 2, 4]], dim=-1
    )
    return errors[stable_stage_mask(valid_mask, warmup_frames)]


def three_second_maneuver_comparison(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
    dt: torch.Tensor,
    warmup_frames: int,
    horizon_seconds: float = 3.0,
) -> dict[str, Any]:
    """Compare current position error with target displacement over the next 3 s."""
    current_errors = torch.linalg.vector_norm(
        prediction[..., [0, 2, 4]] - truth[..., [0, 2, 4]], dim=-1
    )
    positions = truth[..., [0, 2, 4]]
    valid = stable_stage_mask(valid_mask, warmup_frames)
    dt_values = dt.detach().cpu().reshape(-1).tolist()
    error_parts: list[torch.Tensor] = []
    distance_parts: list[torch.Tensor] = []
    horizons: set[int] = set()

    for row, dt_seconds in enumerate(dt_values):
        if not np.isfinite(dt_seconds) or dt_seconds <= 0:
            continue
        horizon_frames = int(round(horizon_seconds / float(dt_seconds)))
        if horizon_frames <= 0 or horizon_frames >= prediction.shape[1]:
            continue
        horizons.add(horizon_frames)
        eligible = valid[row, :-horizon_frames] & valid_mask[
            row, horizon_frames:
        ].bool()
        if not eligible.any():
            continue
        error_parts.append(current_errors[row, :-horizon_frames][eligible])
        displacement = torch.linalg.vector_norm(
            positions[row, horizon_frames:] - positions[row, :-horizon_frames],
            dim=-1,
        )
        distance_parts.append(displacement[eligible])

    if not error_parts:
        return {
            "eligible_points": 0,
            "horizon_frames": sorted(horizons),
            "compliant_points": 0,
            "compliance_rate": None,
            "position_error_rmse_m": None,
            "maneuver_distance_mean_m": None,
            "maneuver_distance_p05_m": None,
            "error_to_distance_p95_ratio": None,
        }

    errors = torch.cat(error_parts).detach().cpu().numpy().astype(np.float64)
    distances = torch.cat(distance_parts).detach().cpu().numpy().astype(np.float64)
    compliant = errors <= distances
    ratios = np.divide(
        errors,
        distances,
        out=np.full_like(errors, np.inf),
        where=distances > 0,
    )
    return {
        "eligible_points": int(errors.size),
        "horizon_frames": sorted(horizons),
        "compliant_points": int(np.count_nonzero(compliant)),
        "compliance_rate": float(np.mean(compliant)),
        "position_error_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "maneuver_distance_mean_m": float(np.mean(distances)),
        "maneuver_distance_p05_m": float(np.percentile(distances, 5)),
        "error_to_distance_p95_ratio": float(np.percentile(ratios, 95)),
    }


def single_frame_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Slice a registered sequence into the one-frame model input under test."""
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            result[key] = value
        elif key == "ref_pos" or value.ndim < 2:
            result[key] = value.clone()
        else:
            result[key] = value[:, :1, ...].clone()
    return result


def benchmark_single_frame(
    model: CoopTrackerModel,
    batch: dict[str, Any],
    device: torch.device,
    warmup_iterations: int,
    test_iterations: int,
) -> dict[str, Any]:
    sample = single_frame_batch(batch)
    latencies: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _predict_batch(model, sample, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(test_iterations):
                start_event.record()
                _predict_batch(model, sample, device)
                end_event.record()
                torch.cuda.synchronize(device)
                latencies.append(float(start_event.elapsed_time(end_event)))
        else:
            for _ in range(test_iterations):
                started = time.perf_counter()
                _predict_batch(model, sample, device)
                latencies.append((time.perf_counter() - started) * 1000.0)

    values = np.asarray(latencies, dtype=np.float64)
    return {
        "warmup_iterations": warmup_iterations,
        "test_iterations": test_iterations,
        "batch_size": 1,
        "sequence_length": 1,
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
        "threshold_ms": 100.0,
        "pass_rule": "p95_ms < 100 ms",
        "passed": bool(np.percentile(values, 95) < 100.0),
        "timer": "torch.cuda.Event with synchronize"
        if device.type == "cuda"
        else "time.perf_counter",
    }


def _windows_registry_value(path: str, name: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            return str(winreg.QueryValueEx(key, name)[0]).strip()
    except (OSError, ImportError):
        return None


def _gpu_driver() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _total_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except (ImportError, OSError):
        return None


def runtime_environment(device: torch.device) -> dict[str, Any]:
    manufacturer = _windows_registry_value(
        r"HARDWARE\DESCRIPTION\System\BIOS", "SystemManufacturer"
    )
    model = _windows_registry_value(
        r"HARDWARE\DESCRIPTION\System\BIOS", "SystemProductName"
    )
    cpu = _windows_registry_value(
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        "ProcessorNameString",
    ) or platform.processor()
    os_name = _windows_registry_value(
        r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", "ProductName"
    )
    os_version = _windows_registry_value(
        r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", "DisplayVersion"
    )
    os_build = _windows_registry_value(
        r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", "CurrentBuildNumber"
    )
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "driver_version": _gpu_driver(),
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    return {
        "platform_category": "笔记本式开发工作站",
        "manufacturer": manufacturer,
        "model": model,
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "total_memory_bytes": _total_memory_bytes(),
        "gpu": gpu,
        "operating_system": {
            "name": os_name or platform.system(),
            "display_version": os_version,
            "build": os_build,
            "architecture": platform.machine(),
        },
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "is_server": False,
        "is_industrial_pc": False,
        "is_actual_deployment_platform": False,
        "applicability_note": "开发工作站测试结果；实际部署平台须按同一脚本复测。",
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    selected_paths = selected_dataset_files(
        args.dataset_id,
        args.max_scenarios,
        strategy="stratified_by_scenario_type",
    )
    if not selected_paths:
        raise FileNotFoundError(f"No registered scenarios for {args.dataset_id}")

    dataset = _BatchRegisteredPklDataset(selected_paths, args.sequence_length, 5)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model_path = Path(args.model).resolve()
    model = CoopTrackerModel(d_model=96).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    stable_parts: list[np.ndarray] = []
    guide_totals: list[dict[str, Any]] = []
    benchmark_batch: dict[str, Any] | None = None
    dt_values: set[float] = set()
    with torch.inference_mode():
        for batch in loader:
            if benchmark_batch is None:
                benchmark_batch = {
                    key: value[:1] if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
            prediction = _predict_batch(model, batch, device)
            truth = batch["label_abs"].to(device)
            valid = batch["target_valid_mask"].to(device)
            errors = stable_position_errors(
                prediction, truth, valid, args.warmup_frames
            )
            stable_parts.append(errors.detach().cpu().numpy())
            dt_tensor = batch["dt"].to(device).reshape(-1)
            dt_values.update(float(value) for value in dt_tensor.detach().cpu())
            guide_totals.append(
                three_second_maneuver_comparison(
                    prediction,
                    truth,
                    valid,
                    dt_tensor,
                    args.warmup_frames,
                )
            )

    merged_errors = np.concatenate(stable_parts).astype(np.float64)
    eligible = sum(item["eligible_points"] for item in guide_totals)
    compliant = sum(item["compliant_points"] for item in guide_totals)
    weighted_distance_mean = sum(
        (item["maneuver_distance_mean_m"] or 0.0) * item["eligible_points"]
        for item in guide_totals
    ) / max(eligible, 1)
    weighted_ratio_p95 = max(
        (item["error_to_distance_p95_ratio"] or 0.0) for item in guide_totals
    )
    if benchmark_batch is None:
        raise RuntimeError("No batch available for latency benchmark")
    latency = benchmark_single_frame(
        model,
        benchmark_batch,
        device,
        args.latency_warmups,
        args.latency_iterations,
    )

    stable_rmse = float(np.sqrt(np.mean(np.square(merged_errors))))
    output = Path(args.output).resolve()
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "dataset_id": args.dataset_id,
            "sampling_strategy": "stratified_by_scenario_type",
            "scenario_count": len(selected_paths),
            "track_count": len(dataset),
            "scenario_class_counts": class_counts(selected_paths),
            "sequence_length": args.sequence_length,
            "warmup_frames_excluded": args.warmup_frames,
            "sample_period_seconds": sorted(dt_values),
            "model_weights_modified": False,
        },
        "review_item_7": {
            "opinion": "限定为跟踪稳定阶段的位置均方根误差，排除冷启动和初始过渡阶段。",
            "definition": position_rmse_definition(args.warmup_frames),
            "position_rmse_m": stable_rmse,
            "evaluated_points": int(merged_errors.size),
            "threshold_m": 8.0,
            "passed": stable_rmse <= 8.0,
        },
        "review_item_8": {
            "opinion": "补充小于100 ms指标的硬件、软件、运行环境和平台类型。",
            "metric_clarification": "项目输出跟踪状态而非飞行控制指令；本指标对应单帧模型前向的跟踪结果生成时间。",
            "measurement_scope": {
                "starts_at": "aligned and associated multisource tensors",
                "ends_at": "target position and velocity state output",
                "excluded": [
                    "file I/O",
                    "raw detection extraction",
                    "target association",
                    "GUI rendering",
                    "flight-control law and actuator/communication delay",
                ],
            },
            "environment": runtime_environment(device),
            "latency": latency,
        },
        "review_item_9": {
            "opinion": "明确3秒机动距离与8 m RMSE的定义依据、关系和确认状态。",
            "conversion_basis": None,
            "relationship": "independent metrics; no mathematical or empirical conversion is asserted",
            "contract_metric": {
                "name": "跟踪稳定阶段位置RMSE",
                "threshold_m": 8.0,
                "value_m": stable_rmse,
                "passed": stable_rmse <= 8.0,
            },
            "guide_supplementary_metric": {
                "definition": "e(t)=||pred(t)-truth(t)||; D3(t)=||truth(t+3s)-truth(t)||; pass when e(t)<=D3(t)",
                "eligible_points": eligible,
                "compliant_points": compliant,
                "compliance_rate": compliant / eligible if eligible else None,
                "mean_three_second_maneuver_distance_m": weighted_distance_mean,
                "conservative_batch_p95_error_to_distance_ratio": weighted_ratio_p95,
            },
            "repository_status": "definitions fixed and both metrics tested",
            "external_confirmation_status": "pending owner/customer written confirmation; not represented as approved",
        },
    }
    _write_json_atomic(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--dataset-id", default="v7")
    parser.add_argument("--max-scenarios", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--latency-warmups", type=int, default=50)
    parser.add_argument("--latency-iterations", type=int, default=500)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
