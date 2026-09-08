"""Multidimensional assessment and reference results for batch tests."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tracking_system.backend.batch_testing import BatchTestManager
from tracking_system.backend.runtime import PROJECT_ROOT


ACCEPTANCE_CRITERIA: dict[str, dict[str, Any]] = {
    "position_rmse_m": {
        "name": "位置均方根误差",
        "operator": "<=",
        "threshold": 8.0,
        "unit": "m",
        "source": "任务书",
    },
    "association_accuracy": {
        "name": "数据关联正确率",
        "operator": ">=",
        "threshold": 0.85,
        "display_threshold": "85%",
        "unit": "ratio",
        "source": "任务书",
    },
    "rmse_improvement_pct": {
        "name": "相对 IMM-JPDA 的 RMSE 降幅",
        "operator": ">=",
        "threshold": 15.0,
        "unit": "%",
        "source": "任务书",
    },
    "inference_time_ms": {
        "name": "单帧推理耗时",
        "operator": "<=",
        "threshold": 100.0,
        "unit": "ms",
        "source": "任务书",
    },
}

REFERENCE_FILES = {
    "cross_dataset": PROJECT_ROOT / "artifacts" / "metrics" / "cross_dataset_rmse.json",
    "imm_comparison": PROJECT_ROOT
    / "artifacts"
    / "reports"
    / "evaluation"
    / "eval_summary.json",
    "association_audit": PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "v5_standard_on_associated.json",
}
ACCEPTANCE_EVIDENCE_DIR = PROJECT_ROOT / "artifacts" / "metrics" / "acceptance"

_MANEUVER_PATTERN = re.compile(r"(?:^|[\\/_-])track[_-](ca|cv|ct|cssw)(?:[_-]|$)", re.I)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _job_signature(job: dict[str, Any]) -> str:
    """Fingerprint the exact batch scope accepted by metric evidence scripts."""
    config = job.get("config") or {}
    metrics = (job.get("result") or {}).get("metrics") or {}
    identity = {
        "job_id": job.get("id"),
        "model_id": config.get("model_id"),
        "dataset_id": config.get("dataset_id"),
        "max_scenarios": config.get("max_scenarios"),
        "warmup_frames": config.get("warmup_frames"),
        "sequence_length": config.get("sequence_length"),
        "evaluated_scenarios": metrics.get("evaluated_scenarios"),
        "position_rmse_m": metrics.get("position_rmse_m"),
    }
    scenario_files = ((job.get("result") or {}).get("scenario_manifest") or {}).get("files")
    if config.get("sampling_strategy") is not None:
        identity["sampling_strategy"] = config["sampling_strategy"]
    if scenario_files is not None:
        identity["scenario_files"] = scenario_files
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _criterion(
    criterion_id: str,
    value: float | None,
    *,
    note: str | None = None,
) -> dict[str, Any]:
    definition = ACCEPTANCE_CRITERIA[criterion_id]
    if value is None:
        status = "unavailable"
        passed = None
    elif definition["operator"] == "<=":
        passed = value <= float(definition["threshold"])
        status = "passed" if passed else "failed"
    else:
        passed = value >= float(definition["threshold"])
        status = "passed" if passed else "failed"
    return {
        "id": criterion_id,
        **definition,
        "value": value,
        "status": status,
        "passed": passed,
        "note": note,
    }


def _maneuver_name(track_name: str) -> str:
    match = _MANEUVER_PATTERN.search(track_name.replace("\\", "/"))
    return match.group(1).upper() if match else "OTHER"


def _summarize_tracks(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    if not tracks:
        return {"distribution": None, "by_maneuver": []}

    rmses = np.asarray([float(item["rmse_m"]) for item in tracks], dtype=np.float64)
    distribution = {
        "minimum_rmse_m": float(np.min(rmses)),
        "median_rmse_m": float(np.median(rmses)),
        "p75_rmse_m": float(np.percentile(rmses, 75)),
        "p90_rmse_m": float(np.percentile(rmses, 90)),
        "p95_rmse_m": float(np.percentile(rmses, 95)),
        "maximum_rmse_m": float(np.max(rmses)),
        "tracks_within_8m": int(np.count_nonzero(rmses <= 8.0)),
        "tracks_within_8m_ratio": float(np.mean(rmses <= 8.0)),
    }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in tracks:
        grouped.setdefault(_maneuver_name(str(item.get("track", ""))), []).append(item)
    by_maneuver = []
    preferred_order = {name: index for index, name in enumerate(("CV", "CA", "CT", "CSSW", "OTHER"))}
    for maneuver, items in sorted(grouped.items(), key=lambda pair: preferred_order.get(pair[0], 99)):
        group_rmses = np.asarray([float(item["rmse_m"]) for item in items], dtype=np.float64)
        by_maneuver.append(
            {
                "maneuver": maneuver,
                "tracks": len(items),
                "mean_track_rmse_m": float(np.mean(group_rmses)),
                "median_track_rmse_m": float(np.median(group_rmses)),
                "p95_track_rmse_m": float(np.percentile(group_rmses, 95)),
                "max_track_rmse_m": float(np.max(group_rmses)),
            }
        )
    return {"distribution": distribution, "by_maneuver": by_maneuver}


class EvaluationService:
    """Build task-book assessments without changing persisted raw test results."""

    def __init__(
        self,
        batch_manager: BatchTestManager,
        evidence_root: Path | None = None,
    ) -> None:
        self.batch_manager = batch_manager
        self.evidence_root = evidence_root or ACCEPTANCE_EVIDENCE_DIR
        self._evidence_lock = threading.Lock()

    def _metric_evidence(
        self, job: dict[str, Any], metric_id: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        job_id = str(job["id"])
        path = self.evidence_root / job_id / f"{metric_id}.json"
        script = {
            "association_accuracy": "test_association_accuracy.py",
            "rmse_improvement_pct": "test_rmse_improvement.py",
        }[metric_id]
        report_path = self.batch_manager.report_dir / f"{job_id}.json"
        command = (
            f'conda run -n pytorch python scripts/evaluation/{script} '
            f'--batch-report "{report_path}"'
        )
        status = {
            "metric_id": metric_id,
            "available": path.is_file(),
            "accepted": False,
            "path": str(path),
            "command": command,
        }
        payload = _read_json(path)
        if not isinstance(payload, dict):
            status["message"] = "尚未生成测试证据" if not path.is_file() else "测试证据不是有效 JSON"
            return None, status
        if payload.get("metric_id") != metric_id or payload.get("job_id") != job_id:
            status["message"] = "测试证据的指标或批测任务不匹配"
            return None, status
        if payload.get("job_signature") != _job_signature(job):
            status["message"] = "批测配置或结果已变化，请重新运行测试脚本"
            return None, status
        if payload.get("valid_for_assessment") is not True:
            status["message"] = "测试证据未标记为可用于验收"
            return None, status
        try:
            value = float(payload["value"])
        except (KeyError, TypeError, ValueError):
            status["message"] = "测试证据缺少有效指标值"
            return None, status
        if not np.isfinite(value):
            status["message"] = "测试证据指标值不是有限数"
            return None, status
        if metric_id == "rmse_improvement_pct":
            try:
                baseline_rmse = float(payload["imm_jpda_rmse_m"])
            except (KeyError, TypeError, ValueError):
                status["message"] = "基线测试证据缺少有效 IMM-JPDA RMSE"
                return None, status
            if not np.isfinite(baseline_rmse) or baseline_rmse <= 0:
                status["message"] = "基线测试证据中的 IMM-JPDA RMSE 无效"
                return None, status
        status.update({"accepted": True, "message": "已加载与当前批测严格匹配的测试证据"})
        return payload, status

    @staticmethod
    def criteria() -> dict[str, Any]:
        return {
            "items": [
                {"id": criterion_id, **copy.deepcopy(definition)}
                for criterion_id, definition in ACCEPTANCE_CRITERIA.items()
            ]
        }

    def generate_evidence(self, job_id: str, metric_id: str) -> dict[str, Any]:
        """Run a whitelisted acceptance script with the backend's Python runtime."""
        scripts = {
            "association_accuracy": "test_association_accuracy.py",
            "rmse_improvement_pct": "test_rmse_improvement.py",
        }
        script_name = scripts.get(metric_id)
        if script_name is None:
            raise KeyError(f"Unknown evidence metric: {metric_id}")
        job = self.batch_manager.get_job(job_id)
        if job.get("status") != "completed" or not job.get("result"):
            raise ValueError("Only completed batch tests can generate evidence")
        report_path = self.batch_manager.report_dir / f"{job_id}.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Batch report not found: {report_path}")

        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluation" / script_name),
            "--batch-report",
            str(report_path),
            "--output",
            str(self.evidence_root / job_id / f"{metric_id}.json"),
        ]
        with self._evidence_lock:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3600,
                check=False,
            )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "测试脚本执行失败").strip()
            raise RuntimeError(detail[-2000:])
        refreshed_job = self.batch_manager.get_job(job_id)
        payload, status = self._metric_evidence(refreshed_job, metric_id)
        if payload is None:
            raise RuntimeError(str(status.get("message") or "测试证据未通过完整性校验"))
        return {
            "metric_id": metric_id,
            "status": status,
            "assessment": self.assess_job(job_id),
        }

    def assess_job(
        self, job_id: str, baseline_rmse_m: float | None = None
    ) -> dict[str, Any]:
        job = self.batch_manager.get_job(job_id)
        if job.get("status") != "completed" or not job.get("result"):
            raise ValueError("Only completed batch tests can be assessed")

        result = job["result"]
        metrics = result.get("metrics") or {}
        latency = result.get("latency") or {}
        association = result.get("association") or {}
        rmse = metrics.get("position_rmse_m")
        association_evidence, association_status = self._metric_evidence(
            job, "association_accuracy"
        )
        rmse_evidence, rmse_status = self._metric_evidence(job, "rmse_improvement_pct")
        association_accuracy = association.get("accuracy")
        association_note = None
        if association_evidence is not None:
            association_accuracy = float(association_evidence["value"])
            association_details = association_evidence.get("association")
            if not isinstance(association_details, dict):
                association_details = association_evidence
            association_note = (
                "独立关联测试："
                f'{int(association_details.get("correct", 0))}/'
                f'{int(association_details.get("total", 0))} 次关联正确'
            )
        elif association_accuracy is None:
            association_note = f'尚无关联测试证据；运行：{association_status["command"]}'
        else:
            association_note = "使用注册数据中的 AssociationStats"
        frame_latency = latency.get("measured_frame_ms", latency.get("estimated_frame_ms"))
        latency_note = (
            "单航迹序列重复实测并按序列长度折算"
            if latency.get("measured_frame_ms") is not None
            else "按有效目标点折算的历史单帧推理耗时"
        )
        improvement = None
        improvement_note = f'尚无同条件基线证据；运行：{rmse_status["command"]}'
        if baseline_rmse_m is not None:
            if baseline_rmse_m <= 0:
                raise ValueError("baseline_rmse_m must be greater than zero")
            improvement = (float(baseline_rmse_m) - float(rmse)) / float(baseline_rmse_m) * 100.0
            improvement_note = f"基线 RMSE={baseline_rmse_m:.6g} m（手动提供）"
        elif rmse_evidence is not None:
            improvement = float(rmse_evidence["value"])
            baseline_rmse = float(rmse_evidence["imm_jpda_rmse_m"])
            improvement_note = f"同条件 IMM-JPDA 基线 RMSE={baseline_rmse:.6g} m"

        criteria = [
            _criterion("position_rmse_m", float(rmse) if rmse is not None else None),
            _criterion(
                "association_accuracy",
                float(association_accuracy) if association_accuracy is not None else None,
                note=association_note,
            ),
            _criterion("rmse_improvement_pct", improvement, note=improvement_note),
            _criterion(
                "inference_time_ms",
                float(frame_latency) if frame_latency is not None else None,
                note=latency_note,
            ),
        ]
        passed = sum(item["status"] == "passed" for item in criteria)
        failed = sum(item["status"] == "failed" for item in criteria)
        unavailable = sum(item["status"] == "unavailable" for item in criteria)
        overall_status = "failed" if failed else ("incomplete" if unavailable else "passed")

        return {
            "job_id": job_id,
            "job_name": job.get("name"),
            "model_id": result.get("model_id"),
            "dataset_id": result.get("dataset_id"),
            "overall_status": overall_status,
            "summary": {
                "total": len(criteria),
                "passed": passed,
                "failed": failed,
                "unavailable": unavailable,
            },
            "criteria": criteria,
            "evidence": {
                "association_accuracy": association_status,
                "rmse_improvement_pct": rmse_status,
            },
            "dimensions": {
                "accuracy": copy.deepcopy(metrics),
                "latency": copy.deepcopy(latency),
                "association": copy.deepcopy(
                    (association_evidence or {}).get("association")
                    or association_evidence
                    or result.get("association")
                ),
                **_summarize_tracks(result.get("tracks") or []),
            },
        }

    def compare_jobs(self, job_ids: list[str] | None = None) -> dict[str, Any]:
        """Return normalized rows suitable for a frontend comparison table."""
        if job_ids:
            jobs = [self.batch_manager.get_job(job_id) for job_id in job_ids]
        else:
            jobs = self.batch_manager.list_jobs(limit=200)
        rows = []
        for job in jobs:
            if job.get("status") != "completed" or not job.get("result"):
                continue
            result = job["result"]
            config = job.get("config") or {}
            metrics = result.get("metrics") or {}
            latency = result.get("latency") or {}
            association = result.get("association") or {}
            rows.append(
                {
                    "job_id": job.get("id"),
                    "name": job.get("name"),
                    # list_jobs() deliberately returns a compact result object, so
                    # keep the task configuration as the canonical fallback here.
                    "model_id": result.get("model_id") or config.get("model_id"),
                    "dataset_id": result.get("dataset_id") or config.get("dataset_id"),
                    "created_at": job.get("created_at"),
                    "finished_at": job.get("finished_at"),
                    "position_rmse_m": metrics.get("position_rmse_m"),
                    "position_mae_m": metrics.get("position_mae_m"),
                    "position_p95_m": metrics.get("position_p95_m"),
                    "estimated_frame_ms": latency.get("measured_frame_ms", latency.get("estimated_frame_ms")),
                    "throughput_tracks_per_second": latency.get("throughput_tracks_per_second"),
                    "association_accuracy": association.get("accuracy"),
                    "evaluated_scenarios": metrics.get("evaluated_scenarios"),
                    "evaluated_tracks": metrics.get("evaluated_tracks"),
                }
            )
        rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {"total": len(rows), "items": rows}

    def report_catalog(self, limit: int = 200) -> dict[str, Any]:
        """Return persisted run artifacts and their assessment readiness."""
        jobs = self.batch_manager.list_jobs(limit=limit)
        items: list[dict[str, Any]] = []
        total_bytes = 0
        for job in jobs:
            job_id = str(job.get("id"))
            status = str(job.get("status"))
            config = job.get("config") or {}
            result = job.get("result") or {}
            metrics = result.get("metrics") or {}
            latency = result.get("latency") or {}
            report_path = self.batch_manager.report_dir / f"{job_id}.json"
            report_size = report_path.stat().st_size if report_path.is_file() else 0
            total_bytes += report_size
            assessment = None
            if status == "completed" and result:
                try:
                    assessed = self.assess_job(job_id)
                    assessment = {
                        "overall_status": assessed["overall_status"],
                        "summary": assessed["summary"],
                    }
                except (KeyError, ValueError):
                    assessment = None
            items.append(
                {
                    "id": job_id,
                    "name": job.get("name"),
                    "status": status,
                    "created_at": job.get("created_at"),
                    "finished_at": job.get("finished_at"),
                    "model_id": config.get("model_id") or result.get("model_id"),
                    "dataset_id": config.get("dataset_id") or result.get("dataset_id"),
                    "device": config.get("device"),
                    "scope": config.get("max_scenarios"),
                    "metrics": {
                        "position_rmse_m": metrics.get("position_rmse_m"),
                        "position_mae_m": metrics.get("position_mae_m"),
                        "position_p95_m": metrics.get("position_p95_m"),
                        "estimated_frame_ms": latency.get("measured_frame_ms", latency.get("estimated_frame_ms")),
                        "evaluated_scenarios": metrics.get("evaluated_scenarios"),
                        "evaluated_tracks": metrics.get("evaluated_tracks"),
                    }
                    if result
                    else None,
                    "assessment": assessment,
                    "error": job.get("error"),
                    "artifacts": {
                        "raw_json": report_path.is_file(),
                        "tracks_csv": status == "completed"
                        and bool(metrics.get("evaluated_tracks")),
                        "assessment_json": status == "completed" and bool(result),
                        "report_size_bytes": report_size,
                    },
                }
            )
        completed = sum(item["status"] == "completed" for item in items)
        failed = sum(item["status"] in {"failed", "cancelled"} for item in items)
        return {
            "total": len(items),
            "summary": {
                "completed": completed,
                "failed_or_cancelled": failed,
                "assessable": sum(item["assessment"] is not None for item in items),
                "total_size_bytes": total_bytes,
            },
            "items": items,
        }

    def assessment_bundle(
        self, job_id: str, baseline_rmse_m: float | None = None
    ) -> dict[str, Any]:
        """Build a self-contained, downloadable assessment artifact."""
        job = self.batch_manager.get_job(job_id)
        assessment = self.assess_job(job_id, baseline_rmse_m)
        return {
            "schema_version": "1.0",
            "kind": "tracking_batch_assessment",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "job": job,
            "assessment": assessment,
            "provenance": {
                "criteria": self.criteria()["items"],
                "baseline_rmse_m": baseline_rmse_m,
                "baseline_requirement": "IMM-JPDA baseline must use the same test dataset and settings",
            },
        }

    @staticmethod
    def references() -> dict[str, Any]:
        imm = _read_json(REFERENCE_FILES["imm_comparison"])
        association = _read_json(REFERENCE_FILES["association_audit"])
        association_audit = association.get("association_audit") if isinstance(association, dict) else None
        return {
            "kind": "historical_reference",
            "notice": "历史结果仅用于展示；正式判定应使用与当前批测相同数据集和设置的基线。",
            "imm_jpda_comparison": imm,
            "association_audit": association_audit,
            "sources": {key: str(path) for key, path in REFERENCE_FILES.items()},
        }

    @staticmethod
    def cross_dataset() -> dict[str, Any]:
        raw = _read_json(REFERENCE_FILES["cross_dataset"])
        if not isinstance(raw, dict):
            return {"available": False, "metric": "position_rmse_m", "matrix": []}
        model_ids = {"Model-v5 (Standard)": "v5", "Model-v6 (-30% Noise)": "v6", "Model-v7 (-40% Noise)": "v7"}
        dataset_ids = {"Val-v5 (Standard)": "v5", "Val-v6 (-30% Noise)": "v6", "Val-v7 (-40% Noise)": "v7", "Val-v8 (Interference)": "v8_interference"}
        matrix = []
        for model_name, values in raw.items():
            for dataset_name, rmse in values.items():
                matrix.append(
                    {
                        "model_id": model_ids.get(model_name, model_name),
                        "dataset_id": dataset_ids.get(dataset_name, dataset_name),
                        "position_rmse_m": float(rmse),
                    }
                )
        return {
            "available": True,
            "metric": "position_rmse_m",
            "source": str(REFERENCE_FILES["cross_dataset"]),
            "matrix": matrix,
        }
