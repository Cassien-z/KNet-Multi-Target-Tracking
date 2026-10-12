"""Generate auditable evidence for midterm-review items 10 through 12."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking_system.backend.scenario_catalog import (
    V7_QUALITY_PROFILE,
    V8_QUALITY_PROFILE,
    class_counts,
    scenario_class,
    selected_dataset_files,
)


DEFAULT_AGREEMENT = Path(
    r"D:\workstation\项目对接\V20250916公开_基于多源异构数据融合的端对端适变跟踪方法研究技术协议书.docx"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "midterm_review"
    / "items_10_12_evidence.json"
)
SENSORS = ("Radar", "EO", "ESM", "FastLink")
GRAVITY_MPS2 = 9.80665


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stats(values: list[np.ndarray]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "rmse": None, "mean": None, "p95": None, "max": None}
    merged = np.concatenate(values).astype(np.float64)
    return {
        "samples": int(merged.size),
        "rmse": float(np.sqrt(np.mean(np.square(merged)))),
        "mean": float(np.mean(merged)),
        "p95": float(np.percentile(merged, 95)),
        "max": float(np.max(merged)),
    }


def audit_observation_quality(paths: Iterable[Path]) -> dict[str, Any]:
    totals = {sensor: {"slots": 0, "valid": 0, "errors": []} for sensor in SENSORS}
    scenario_count = 0
    track_count = 0
    frame_counts: set[int] = set()
    target_counts: set[int] = set()
    platform_counts: set[int] = set()
    sample_periods: set[float] = set()

    for path in paths:
        with path.open("rb") as handle:
            data = pickle.load(handle)
        scenario_count += 1
        tracks = data.get("RegisteredTrack", [])
        track_count += len(tracks)
        target_counts.add(len(tracks))
        t_grid = np.asarray(data.get("t_grid", []), dtype=np.float64)
        frame_counts.add(int(t_grid.size))
        if t_grid.size > 1:
            sample_periods.add(round(float(np.median(np.diff(t_grid))), 6))
        platform_counts.add(int(np.asarray(data.get("UAV_Pos", [])).shape[1]))

        for track in tracks:
            truth = np.asarray(track["target_pos"], dtype=np.float64)
            for sensor in SENSORS:
                for platform_record in track.get(sensor, []):
                    valid = np.asarray(platform_record.get("valid", []), dtype=bool)
                    totals[sensor]["slots"] += int(valid.size)
                    totals[sensor]["valid"] += int(np.count_nonzero(valid))
                    measured = np.asarray(platform_record.get("Meas_ECEF", []), dtype=np.float64)
                    if measured.shape != truth.shape or valid.size != truth.shape[0]:
                        continue
                    eligible = valid & np.isfinite(measured).all(axis=1) & np.isfinite(truth).all(axis=1)
                    if np.any(eligible):
                        totals[sensor]["errors"].append(
                            np.linalg.norm(measured[eligible] - truth[eligible], axis=1)
                        )

    sensors: dict[str, Any] = {}
    for sensor, values in totals.items():
        slots = int(values["slots"])
        valid = int(values["valid"])
        error_stats = _safe_stats(values["errors"])
        sensors[sensor] = {
            "total_input_slots": slots,
            "valid_input_slots": valid,
            "availability_rate": valid / slots if slots else None,
            "missing_rate": 1.0 - valid / slots if slots else None,
            "ecef_equivalent_observation_error_m": error_stats,
            "error_note": "EO/ESM ECEF values are registered equivalents and do not imply direct range observation."
            if sensor in {"EO", "ESM"}
            else None,
        }
    return {
        "scenario_count": scenario_count,
        "track_count": track_count,
        "targets_per_scenario": sorted(target_counts),
        "frames_per_scenario": sorted(frame_counts),
        "platforms": sorted(platform_counts),
        "sample_period_seconds": sorted(sample_periods),
        "sensors": sensors,
    }


def kinematic_samples(
    velocity_xyz: np.ndarray, dt_seconds: float
) -> dict[str, np.ndarray]:
    velocity = np.asarray(velocity_xyz, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[0] < 2 or velocity.shape[1] != 3:
        raise ValueError("velocity_xyz must have shape [T, 3] with T >= 2")
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")
    acceleration = np.diff(velocity, axis=0) / dt_seconds
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    speed_change_rate = np.abs(np.diff(speed)) / dt_seconds
    left = velocity[:-1]
    right = velocity[1:]
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 0,
    )
    turn_rate_deg_s = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) / dt_seconds
    return {
        "acceleration_magnitude_mps2": acceleration_magnitude,
        "turn_rate_deg_s": turn_rate_deg_s,
        "speed_change_rate_mps2": speed_change_rate,
        "equivalent_kinematic_load_factor_g": acceleration_magnitude / GRAVITY_MPS2,
    }


def audit_maneuver_intensity(paths: Iterable[Path]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    track_counts: dict[str, int] = defaultdict(int)
    for path in paths:
        with path.open("rb") as handle:
            data = pickle.load(handle)
        t_grid = np.asarray(data["t_grid"], dtype=np.float64)
        dt_seconds = float(np.median(np.diff(t_grid)))
        class_name = scenario_class(path)
        for track in data.get("RegisteredTrack", []):
            samples = kinematic_samples(track["target_vel"], dt_seconds)
            track_counts[class_name] += 1
            for metric, values in samples.items():
                finite = values[np.isfinite(values)]
                if finite.size:
                    grouped[class_name][metric].append(finite)

    rows = []
    for class_name in sorted(grouped):
        metrics = {
            metric: _safe_stats(values)
            for metric, values in sorted(grouped[class_name].items())
        }
        rows.append(
            {
                "scenario_class": class_name,
                "track_count": track_counts[class_name],
                "metrics": metrics,
            }
        )
    return {
        "method": "finite differences of registered truth velocity; 3-D turn angle between adjacent velocity vectors",
        "load_factor_definition": "equivalent kinematic load factor = |dv/dt| / 9.80665; excludes gravitational/aerodynamic load modeling",
        "classes": rows,
    }


def _docx_table_rows(path: Path) -> list[list[str]]:
    """Extract DOCX table cells using only the standard library."""
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    rows: list[list[str]] = []
    for row in root.findall(".//w:tbl/w:tr", namespace):
        cells: list[str] = []
        for cell in row.findall("./w:tc", namespace):
            paragraphs: list[str] = []
            for paragraph in cell.findall(".//w:p", namespace):
                text = "".join(
                    node.text or "" for node in paragraph.findall(".//w:t", namespace)
                ).strip()
                if text:
                    paragraphs.append(text)
            cells.append("\n".join(paragraphs).strip())
        rows.append(cells)
    return rows


def audit_financial_agreement(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"agreement_available": False, "agreement_path": str(path)}
    rows: list[dict[str, Any]] = []
    for cells in _docx_table_rows(path):
        if len(cells) < 3:
            continue
        category = cells[-2]
        amount_text = cells[-1]
        if category and (amount_text or category == "设计费"):
            try:
                amount = float(amount_text)
            except ValueError:
                amount = None
            rows.append({"category": category, "amount_10k_cny": amount})
    design_rows = [row for row in rows if row["category"] == "设计费"]
    total_rows = [row for row in rows if row["category"] == "支出合计"]
    return {
        "agreement_available": True,
        "agreement_path": str(path.resolve()),
        "agreement_sha256": _sha256(path),
        "design_fee_entries": design_rows,
        "reported_total_10k_cny": total_rows[0]["amount_10k_cny"] if total_rows else None,
        "finding": "The agreement contains design fee 11 (10k CNY); the review cites Document No. 18 as not providing that category.",
        "document_no_18_available_in_provided_materials": False,
        "safe_change_rule": "Do not invent a replacement category. Finance and both contracting parties must confirm the compliant category and sign a written change while keeping the 300k CNY total unchanged.",
        "change_checklist": [
            "obtain the applicable Document No. 18 and finance interpretation",
            "confirm the compliant replacement category and whether the 110k CNY amount is redistributed",
            "verify all category amounts still total 300k CNY",
            "complete finance review and written approval by both parties",
            "archive the approved change with the original agreement",
        ],
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    v7_paths = selected_dataset_files(
        "v7", args.max_v7_scenarios, strategy="stratified_by_scenario_type"
    )
    v8_paths = selected_dataset_files(
        "v8_interference", None, strategy="stratified_by_scenario_type"
    )
    maneuver_paths = list(v7_paths) + list(v8_paths)
    v7_quality = audit_observation_quality(v7_paths)
    v8_quality = audit_observation_quality(v8_paths)
    maneuver = audit_maneuver_intensity(maneuver_paths)
    financial = audit_financial_agreement(Path(args.agreement))
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_item_10": {
            "opinion": "补充传感器噪声、观测误差、数据缺失、干扰强度和场景复杂度等测试数据质量前提。",
            "acceptance_profile": V7_QUALITY_PROFILE,
            "acceptance_dataset_empirical_audit": v7_quality,
            "robustness_profile": V8_QUALITY_PROFILE,
            "robustness_dataset_empirical_audit": v8_quality,
            "applicability_boundary": "The <=8 m result is asserted only for the V7 acceptance profile after five warmup frames. Severe missing-data and V8 composite-interference runs are robustness evidence unless separately accepted.",
            "status": "quality conditions fixed in catalog, documentation, and reproducible evidence",
        },
        "review_item_11": {
            "opinion": "量化高机动场景的加速度、转弯角速度、速度变化率和最大过载。",
            "sample": {
                "v7_scenario_count": len(v7_paths),
                "v7_class_counts": class_counts(v7_paths),
                "v8_scenario_count": len(v8_paths),
            },
            "maneuver_audit": maneuver,
            "repository_status": "kinematic intensity quantified from registered truth",
            "external_confirmation_status": "pending owner/customer written confirmation of dataset quality and maneuver envelope",
        },
        "review_item_12": {
            "opinion": "核对协议经费明细与18号文及财务要求的一致性。",
            "agreement_audit": financial,
            "repository_status": "discrepancy, amount, and controlled change workflow recorded",
            "external_confirmation_status": "pending finance interpretation and written agreement amendment",
        },
    }
    output = Path(args.output).resolve()
    _write_json_atomic(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-v7-scenarios", type=int, default=64)
    parser.add_argument("--agreement", default=str(DEFAULT_AGREEMENT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
