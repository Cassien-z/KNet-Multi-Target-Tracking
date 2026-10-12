"""Shared helpers for task-book acceptance evidence scripts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tracking_system.backend.scenario_catalog import DATASET_CATALOG, selected_dataset_files


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts" / "metrics" / "acceptance"
RAW_DATASETS = {
    dataset_id: Path(metadata["raw_path"]).resolve()
    for dataset_id, metadata in DATASET_CATALOG.items()
}
REGISTERED_DATASETS = {
    dataset_id: Path(metadata["path"]).resolve()
    for dataset_id, metadata in DATASET_CATALOG.items()
}


def load_batch_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read batch report: {report_path}") from exc
    if not isinstance(report, dict) or not report.get("id"):
        raise ValueError("The batch report does not contain a job id")
    if report.get("status") != "completed" or not report.get("result"):
        raise ValueError("Acceptance evidence requires a completed batch report")
    config = report.get("config") or {}
    dataset_id = config.get("dataset_id")
    if dataset_id not in RAW_DATASETS:
        raise ValueError(f"Unsupported dataset id in batch report: {dataset_id!r}")
    return report


def job_signature(report: dict[str, Any]) -> str:
    config = report.get("config") or {}
    result = report.get("result") or {}
    metrics = result.get("metrics") or {}
    identity = {
        "job_id": report.get("id"),
        "model_id": config.get("model_id"),
        "dataset_id": config.get("dataset_id"),
        "device": config.get("device"),
        "batch_size": config.get("batch_size"),
        "workers": config.get("workers"),
        "max_scenarios": config.get("max_scenarios"),
        "warmup_frames": config.get("warmup_frames"),
        "sequence_length": config.get("sequence_length"),
        "evaluated_scenarios": metrics.get("evaluated_scenarios"),
        "position_rmse_m": metrics.get("position_rmse_m"),
        "model_path": result.get("model_path"),
        "dataset_path": result.get("dataset_path"),
        "provenance": result.get("provenance"),
    }
    scenario_files = ((report.get("result") or {}).get("scenario_manifest") or {}).get("files")
    if config.get("sampling_strategy") is not None:
        identity["sampling_strategy"] = config["sampling_strategy"]
    if scenario_files is not None:
        identity["scenario_files"] = scenario_files
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def selected_files(report: dict[str, Any], suffix: str) -> list[Path]:
    config = report["config"]
    dataset_id = str(config["dataset_id"])
    root = RAW_DATASETS[dataset_id] if suffix == ".mat" else REGISTERED_DATASETS[dataset_id]
    manifest = ((report.get("result") or {}).get("scenario_manifest") or {}).get("files")
    if manifest:
        files = [root / f"{stem}{suffix}" for stem in manifest]
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise ValueError(f"Scenario manifest file is missing: {missing[0]}")
    else:
        files = selected_dataset_files(
            dataset_id,
            config.get("max_scenarios"),
            suffix=suffix,
            strategy=config.get("sampling_strategy", "leading_sorted"),
        )
    expected = int((report["result"].get("metrics") or {}).get("evaluated_scenarios") or len(files))
    if len(files) != expected:
        raise ValueError(
            f"Selected {len(files)} {suffix} files from {root}, but the batch report evaluated {expected} scenarios"
        )
    return files


def evidence_path(report: dict[str, Any], metric_id: str) -> Path:
    return EVIDENCE_ROOT / str(report["id"]) / f"{metric_id}.json"


def evidence_context(report: dict[str, Any]) -> dict[str, Any]:
    config = report["config"]
    metrics = report["result"]["metrics"]
    return {
        "job_id": report["id"],
        "job_signature": job_signature(report),
        "model_id": config["model_id"],
        "dataset_id": config["dataset_id"],
        "max_scenarios": config.get("max_scenarios"),
        "warmup_frames": config["warmup_frames"],
        "sequence_length": config["sequence_length"],
        "evaluated_scenarios": metrics["evaluated_scenarios"],
    }


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination
