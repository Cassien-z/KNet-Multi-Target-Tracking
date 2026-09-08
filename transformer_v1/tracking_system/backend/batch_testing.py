"""Persistent background jobs for batch evaluation of tracking models."""

from __future__ import annotations

import copy
import json
import os
import pickle
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset_pkl import RegisteredPklDataset
from tracking_system.backend.model_manager import ModelManager
from tracking_system.backend.runtime import PROJECT_ROOT, TransformerEngine
from tracking_system.backend.scenario_catalog import (
    DATASET_CATALOG,
    class_counts,
    dataset_files,
    selected_dataset_files,
)

REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports" / "batch_tests"
FINAL_STATUSES = {"completed", "failed", "cancelled"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _predict_batch(
    model: torch.nn.Module, batch: dict[str, Any], device: torch.device
) -> torch.Tensor:
    return model(
        radar=batch["radar"].to(device),
        eo=batch["eo"].to(device),
        esm=batch["esm"].to(device),
        radar_mask=batch["radar_mask"].to(device),
        eo_mask=batch["eo_mask"].to(device),
        esm_mask=batch["esm_mask"].to(device),
        ref_pos=batch["ref_pos"].to(device),
        rotmat_b2e=batch["rotmat_b2e"].to(device),
        uav_pos=batch["uav_pos"].to(device),
        fast_link=batch["fast_link"].to(device),
        slow_link=batch["slow_link"].to(device),
        fast_update=batch["fast_update"].to(device),
        slow_update=batch["slow_update"].to(device),
        imu=batch["imu"].to(device),
        fast_age=batch["fast_age"].to(device),
        slow_age=batch["slow_age"].to(device),
        esm_mode=batch["esm_mode"].to(device),
        dt=batch["dt"].to(device).view(-1),
        use_gat=True,
        use_temporal=True,
    )


def _forward_batch(
    model: torch.nn.Module, batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return _predict_batch(model, batch, device), batch["label_abs"].to(device)


def _association_summary(paths: list[Path]) -> dict[str, Any] | None:
    correct = 0
    total = 0
    audited_files = 0
    per_sensor: dict[str, dict[str, int | float | None]] = {}
    for path in paths:
        try:
            with path.open("rb") as handle:
                stats = pickle.load(handle).get("AssociationStats")
        except (OSError, pickle.UnpicklingError, EOFError):
            continue
        if not stats:
            continue
        audited_files += 1
        correct += int(stats.get("correct", 0))
        total += int(stats.get("total", 0))
        for sensor, values in stats.get("per_sensor", {}).items():
            target = per_sensor.setdefault(sensor, {"correct": 0, "total": 0})
            target["correct"] = int(target["correct"]) + int(values.get("correct", 0))
            target["total"] = int(target["total"]) + int(values.get("total", 0))
    if audited_files == 0:
        return None
    for values in per_sensor.values():
        sensor_total = int(values["total"])
        values["accuracy"] = (
            int(values["correct"]) / sensor_total if sensor_total else None
        )
    return {
        "audited_files": audited_files,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "per_sensor": per_sensor,
    }


class _JobCancelled(Exception):
    pass


class _BatchRegisteredPklDataset(RegisteredPklDataset):
    """Registered dataset that indexes an explicit reproducible scenario set."""

    def __init__(
        self,
        paths: list[Path],
        seq_len: int,
        num_uavs: int,
    ) -> None:
        self.data_dir = str(paths[0].parent) if paths else ""
        self.seq_len = seq_len
        self.num_uavs = num_uavs
        self.return_extra = True
        self.file_list = [str(path) for path in paths]
        self.index_map: list[tuple[int, int]] = []
        for file_index, path in enumerate(paths):
            try:
                with path.open("rb") as handle:
                    target_count = len(pickle.load(handle).get("RegisteredTrack", []))
            except (OSError, pickle.UnpicklingError, EOFError):
                continue
            self.index_map.extend(
                (file_index, target_index) for target_index in range(target_count)
            )
        self._cache_file_idx = -1
        self._cache_samples = None


class BatchTestManager:
    """Run one GPU-heavy batch test at a time and persist job reports."""

    def __init__(
        self,
        model_manager: ModelManager,
        report_dir: str | Path | None = None,
    ) -> None:
        self.model_manager = model_manager
        self.report_dir = Path(report_dir or REPORT_DIR).resolve()
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._futures: dict[str, Future] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tracking-batch-test"
        )
        self._load_history()

    def status(self) -> dict[str, Any]:
        """Return a compact lifecycle summary for health/operations endpoints."""
        with self._lock:
            counts = {status: 0 for status in (*FINAL_STATUSES, "queued", "running")}
            for job in self._jobs.values():
                status = str(job.get("status", "unknown"))
                counts[status] = counts.get(status, 0) + 1
            active_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.get("status") in {"queued", "running"}
            ]
        return {
            "worker_count": 1,
            "report_dir": str(self.report_dir),
            "jobs": counts,
            "active_job_ids": active_ids,
        }

    def shutdown(self) -> None:
        """Request cancellation and release the background executor on API shutdown."""
        with self._lock:
            pending = [
                job_id
                for job_id, job in self._jobs.items()
                if job.get("status") not in FINAL_STATUSES
            ]
        for job_id in pending:
            self.cancel_job(job_id)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def list_datasets(self) -> list[dict[str, Any]]:
        items = []
        for dataset_id, metadata in DATASET_CATALOG.items():
            path = Path(metadata["path"]).resolve()
            paths = dataset_files(dataset_id)
            items.append(
                {
                    "id": dataset_id,
                    "name": metadata["name"],
                    "description": metadata["description"],
                    "path": str(path),
                    "available": path.is_dir(),
                    "scenario_count": len(paths),
                    "kind": metadata.get("kind", "technical_dataset"),
                    "scenario_classes": list(metadata.get("scenario_classes", ())),
                    "sensor_modalities": list(metadata.get("sensor_modalities", ())),
                    "interference_modes": list(metadata.get("interference_modes", ())),
                    "scenario_note": metadata.get("scenario_note"),
                }
            )
        return items

    def _report_path(self, job_id: str) -> Path:
        return self.report_dir / f"{job_id}.json"

    def _persist(self, job_id: str) -> None:
        with self._lock:
            job = copy.deepcopy(self._jobs[job_id])
        destination = self._report_path(job_id)
        temporary = destination.with_suffix(
            f".json.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)

    def _load_history(self) -> None:
        for path in sorted(self.report_dir.glob("*.json")):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(job, dict) or not job.get("id"):
                continue
            if job.get("status") not in FINAL_STATUSES:
                job["status"] = "failed"
                job["finished_at"] = _utc_now()
                job["error"] = "Backend restarted before the batch test finished"
                job.setdefault("progress", {})["message"] = "任务因后端重启而中断"
            self._jobs[job["id"]] = job

    @staticmethod
    def _summary(job: dict[str, Any]) -> dict[str, Any]:
        summary = copy.deepcopy(job)
        result = summary.get("result")
        if result:
            summary["result"] = {
                "metrics": result.get("metrics"),
                "latency": result.get("latency"),
                "association": result.get("association"),
                "report_path": result.get("report_path"),
            }
        return summary

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(), key=lambda item: item["created_at"], reverse=True
            )
            return [self._summary(job) for job in jobs[:limit]]

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown batch test id: {job_id}")
            return copy.deepcopy(job)

    def create_job(
        self,
        *,
        name: str | None,
        model_id: str | None,
        dataset_id: str,
        device: str,
        warmup_frames: int,
        batch_size: int,
        workers: int,
        max_scenarios: int | None,
    ) -> dict[str, Any]:
        chosen_model = model_id or str(self.model_manager.active_info()["model_id"])
        self.model_manager.get_model_path(chosen_model)
        dataset_metadata = DATASET_CATALOG.get(dataset_id)
        if dataset_metadata is None:
            raise KeyError(f"Unknown dataset id: {dataset_id}")
        dataset_path = Path(dataset_metadata["path"]).resolve()
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        resolved_device = TransformerEngine._select_device(device)

        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "name": name or f"{chosen_model.upper()} × {dataset_id.upper()} 批量测试",
            "status": "queued",
            "created_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            "config": {
                "model_id": chosen_model,
                "dataset_id": dataset_id,
                "device": str(resolved_device),
                "warmup_frames": warmup_frames,
                "batch_size": batch_size,
                "workers": workers,
                "max_scenarios": max_scenarios,
                "sequence_length": 100,
                "sampling_strategy": "stratified_by_scenario_type",
            },
            "progress": {
                "stage": "queued",
                "message": "等待执行",
                "completed_batches": 0,
                "total_batches": 0,
                "processed_tracks": 0,
                "total_tracks": 0,
                "percent": 0.0,
            },
            "result": None,
            "error": None,
        }
        cancel_event = threading.Event()
        with self._lock:
            self._jobs[job_id] = job
            self._cancel_events[job_id] = cancel_event
        self._persist(job_id)
        future = self._executor.submit(self._run_job, job_id)
        with self._lock:
            self._futures[job_id] = future
        return self.get_job(job_id)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown batch test id: {job_id}")
            if job["status"] in FINAL_STATUSES:
                return copy.deepcopy(job)
            event = self._cancel_events[job_id]
            future = self._futures.get(job_id)
            event.set()
            job["progress"]["message"] = "正在取消"
            if future is not None and future.cancel():
                job["status"] = "cancelled"
                job["finished_at"] = _utc_now()
                job["progress"]["stage"] = "cancelled"
                job["progress"]["message"] = "任务已取消"
        self._persist(job_id)
        return self.get_job(job_id)

    def _update_progress(
        self,
        job_id: str,
        *,
        stage: str,
        message: str,
        completed_batches: int | None = None,
        total_batches: int | None = None,
        processed_tracks: int | None = None,
        total_tracks: int | None = None,
    ) -> None:
        with self._lock:
            progress = self._jobs[job_id]["progress"]
            progress["stage"] = stage
            progress["message"] = message
            if completed_batches is not None:
                progress["completed_batches"] = completed_batches
            if total_batches is not None:
                progress["total_batches"] = total_batches
            if processed_tracks is not None:
                progress["processed_tracks"] = processed_tracks
            if total_tracks is not None:
                progress["total_tracks"] = total_tracks
            if progress["total_batches"]:
                progress["percent"] = round(
                    100.0
                    * progress["completed_batches"]
                    / progress["total_batches"],
                    2,
                )

    def _run_job(self, job_id: str) -> None:
        model: torch.nn.Module | None = None
        try:
            with self._lock:
                job = self._jobs[job_id]
                config = copy.deepcopy(job["config"])
                event = self._cancel_events[job_id]
                job["status"] = "running"
                job["started_at"] = _utc_now()
            self._update_progress(
                job_id, stage="preparing", message="正在读取数据集索引"
            )
            self._persist(job_id)
            if event.is_set():
                raise _JobCancelled()

            dataset_path = Path(DATASET_CATALOG[config["dataset_id"]]["path"]).resolve()
            selected_paths = selected_dataset_files(
                config["dataset_id"],
                config["max_scenarios"],
                strategy=config.get("sampling_strategy", "leading_sorted"),
            )
            dataset = _BatchRegisteredPklDataset(
                paths=selected_paths,
                seq_len=config["sequence_length"],
                num_uavs=5,
            )
            selected_file_count = len(dataset.file_list)
            if not dataset:
                raise RuntimeError("No registered targets found in the selected dataset")

            loader = DataLoader(
                dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["workers"],
                pin_memory=config["device"].startswith("cuda"),
            )
            total_batches = len(loader)
            total_tracks = len(dataset)
            self._update_progress(
                job_id,
                stage="loading_model",
                message="正在加载批测模型",
                total_batches=total_batches,
                total_tracks=total_tracks,
            )
            self._persist(job_id)
            if event.is_set():
                raise _JobCancelled()

            model_path = self.model_manager.get_model_path(config["model_id"])
            device = torch.device(config["device"])
            engine = TransformerEngine(
                model_path=model_path,
                data_dir=dataset_path,
                device=str(device),
                seq_len=config["sequence_length"],
            )
            model = engine.model

            point_errors: list[np.ndarray] = []
            track_details: list[dict[str, Any]] = []
            batch_latencies: list[float] = []
            benchmark_batch: dict[str, Any] | None = None
            processed_tracks = 0
            valid_points = 0
            self._update_progress(
                job_id, stage="running", message="正在执行批量推理"
            )

            with torch.inference_mode():
                for batch_index, batch in enumerate(loader, start=1):
                    if event.is_set():
                        raise _JobCancelled()
                    if benchmark_batch is None:
                        benchmark_batch = {
                            key: value[:1] if isinstance(value, torch.Tensor) else value
                            for key, value in batch.items()
                        }
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    prediction, truth = _forward_batch(model, batch, device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    batch_latencies.append(elapsed_ms)

                    distances = torch.linalg.vector_norm(
                        prediction[..., [0, 2, 4]] - truth[..., [0, 2, 4]],
                        dim=-1,
                    )
                    mask = batch["target_valid_mask"].to(device).bool()
                    if config["warmup_frames"]:
                        mask[:, : config["warmup_frames"]] = False
                    selected = distances[mask]
                    if selected.numel():
                        point_errors.append(selected.detach().cpu().numpy())
                        valid_points += int(selected.numel())

                    names = list(batch.get("file_name", []))
                    for row in range(distances.shape[0]):
                        row_values = distances[row][mask[row]]
                        if not row_values.numel():
                            continue
                        values = row_values.detach().cpu().numpy().astype(np.float64)
                        track_details.append(
                            {
                                "track": names[row]
                                if row < len(names)
                                else f"track_{processed_tracks + row}",
                                "valid_frames": int(values.size),
                                "rmse_m": float(np.sqrt(np.mean(np.square(values)))),
                                "mae_m": float(np.mean(values)),
                                "p95_m": float(np.percentile(values, 95)),
                                "max_error_m": float(np.max(values)),
                            }
                        )

                    processed_tracks += int(distances.shape[0])
                    self._update_progress(
                        job_id,
                        stage="running",
                        message=f"已完成 {batch_index}/{total_batches} 个批次",
                        completed_batches=batch_index,
                        processed_tracks=processed_tracks,
                    )
                    if batch_index % 10 == 0 or batch_index == total_batches:
                        self._persist(job_id)

            if not point_errors:
                raise RuntimeError("No valid target points remained after masking")
            errors = np.concatenate(point_errors).astype(np.float64)
            track_rmses = np.asarray(
                [item["rmse_m"] for item in track_details], dtype=np.float64
            )
            steady_latencies = np.asarray(
                batch_latencies[1:] or batch_latencies, dtype=np.float64
            )
            total_inference_ms = float(np.sum(batch_latencies))
            single_sequence_latencies: list[float] = []
            if benchmark_batch is not None:
                with torch.inference_mode():
                    for _ in range(3):
                        _predict_batch(model, benchmark_batch, device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    for _ in range(10):
                        started = time.perf_counter()
                        _predict_batch(model, benchmark_batch, device)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        single_sequence_latencies.append(
                            (time.perf_counter() - started) * 1000.0
                        )
            sequence_latencies = np.asarray(single_sequence_latencies, dtype=np.float64)
            metrics = {
                "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
                "position_mae_m": float(np.mean(errors)),
                "position_median_m": float(np.median(errors)),
                "position_p95_m": float(np.percentile(errors, 95)),
                "position_max_m": float(np.max(errors)),
                "mean_track_rmse_m": float(np.mean(track_rmses)),
                "evaluated_scenarios": selected_file_count,
                "evaluated_tracks": int(track_rmses.size),
                "evaluated_points": valid_points,
            }
            latency = {
                "total_inference_ms": total_inference_ms,
                "mean_batch_ms": float(np.mean(steady_latencies)),
                "p95_batch_ms": float(np.percentile(steady_latencies, 95)),
                "mean_track_ms": total_inference_ms / max(len(track_details), 1),
                "estimated_frame_ms": total_inference_ms / max(valid_points, 1),
                "measured_frame_ms": float(np.mean(sequence_latencies))
                / config["sequence_length"],
                "p95_measured_frame_ms": float(np.percentile(sequence_latencies, 95))
                / config["sequence_length"],
                "mean_single_sequence_ms": float(np.mean(sequence_latencies)),
                "latency_method": "single_track_sequence_10_runs_after_3_warmups",
                "throughput_tracks_per_second": 1000.0
                * len(track_details)
                / max(total_inference_ms, 1e-9),
                "cold_start_batch_excluded": len(batch_latencies) > 1,
            }
            association = _association_summary(selected_paths)
            report_path = str(self._report_path(job_id))
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "completed"
                job["finished_at"] = _utc_now()
                job["progress"].update(
                    {
                        "stage": "completed",
                        "message": "批量测试完成",
                        "completed_batches": total_batches,
                        "processed_tracks": processed_tracks,
                        "percent": 100.0,
                    }
                )
                job["result"] = {
                    "model_id": config["model_id"],
                    "model_path": str(model_path),
                    "dataset_id": config["dataset_id"],
                    "dataset_path": str(dataset_path),
                    "metrics": metrics,
                    "latency": latency,
                    "association": association,
                    "scenario_manifest": {
                        "sampling_strategy": config["sampling_strategy"],
                        "files": [path.stem for path in selected_paths],
                        "class_counts": class_counts(selected_paths),
                    },
                    "tracks": track_details,
                    "report_path": report_path,
                }
            self._persist(job_id)
        except _JobCancelled:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "cancelled"
                job["finished_at"] = _utc_now()
                job["progress"]["stage"] = "cancelled"
                job["progress"]["message"] = "任务已取消"
            self._persist(job_id)
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["finished_at"] = _utc_now()
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["progress"]["stage"] = "failed"
                job["progress"]["message"] = "批量测试失败"
            self._persist(job_id)
        finally:
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
