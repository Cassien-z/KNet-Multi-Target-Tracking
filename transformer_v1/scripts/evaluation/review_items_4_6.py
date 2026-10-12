"""Generate auditable evidence for midterm-review items 4 through 6.

The audit keeps training data and model weights unchanged.  It inventories the
existing validation datasets, fixes the technical boundary of "end-to-end",
and recomputes measurement-only association accuracy on a deterministic sample
balanced across CV/CA/CT/CSSW under three individual radar interference modes
and one composite interference condition.

Example:
    python scripts/evaluation/review_items_4_6.py --max-scenarios 32 --workers 4
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import tempfile
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.io as sio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_association.pipeline_core import process_file
from tracking_system.backend.scenario_catalog import (
    SCENARIO_TAXONOMY,
    class_counts,
    dataset_files,
    scenario_class,
    scenario_dimensions,
    selected_dataset_files,
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "midterm_review"
    / "items_4_6_evidence.json"
)
THRESHOLD = 0.85
CONDITIONS = (
    "baseline",
    "radar_blinking",
    "radar_soft_suppression_70pct",
    "radar_range_smearing_100m",
    "composite_interference",
)
SENSOR_COMBINATIONS = {
    "radar_only": ("Radar",),
    "radar_eo": ("Radar", "EO"),
    "radar_esm": ("Radar", "ESM"),
    "radar_eo_esm": ("Radar", "EO", "ESM"),
}


def _uavs(frame: dict[str, Any]) -> list[dict[str, Any]]:
    value = frame["UAV"]
    return value if isinstance(value, list) else [value]


def _radar_mask(sensor: dict[str, Any]) -> np.ndarray:
    return np.asarray(sensor.get("Mask", []), dtype=float).reshape(-1) > 0.5


def _set_radar_mask(sensor: dict[str, Any], mask: np.ndarray) -> None:
    original = np.asarray(sensor.get("Mask", []))
    sensor["Mask"] = mask.astype(float).reshape(original.shape)


def _smear_radar_range(sensor: dict[str, Any], rng: np.random.Generator, sigma_m: float) -> None:
    measurements = np.asarray(sensor.get("Meas", []), dtype=float).copy()
    if measurements.size == 0:
        return
    one_dimensional = measurements.ndim == 1
    matrix = measurements.reshape(1, -1) if one_dimensional else measurements
    if matrix.shape[1] < 3:
        return
    mask = _radar_mask(sensor)
    count = min(len(mask), len(matrix))
    active = np.flatnonzero(mask[:count])
    if len(active):
        matrix[active, 2] += rng.normal(0.0, sigma_m, size=len(active))
    sensor["Meas"] = matrix.reshape(measurements.shape)


def apply_interference(
    data: dict[str, Any], condition: str, seed: int
) -> dict[str, int | float]:
    """Apply a deterministic validation-only radar disturbance in memory."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    frames = list(data["Data_Input"])
    rng = np.random.default_rng(seed)
    source_active = 0
    retained_active = 0
    for frame_index, frame in enumerate(frames):
        for uav in _uavs(frame):
            radar = uav["Radar"]
            mask = _radar_mask(radar)
            source_active += int(mask.sum())
            updated = mask.copy()
            blinking = condition in ("radar_blinking", "composite_interference")
            suppression = condition in (
                "radar_soft_suppression_70pct",
                "composite_interference",
            )
            smearing = condition in (
                "radar_range_smearing_100m",
                "composite_interference",
            )
            if blinking and frame_index % 30 < 2:
                updated[:] = False
            if suppression and np.any(updated):
                updated &= rng.random(len(updated)) >= 0.70
            _set_radar_mask(radar, updated)
            if smearing:
                _smear_radar_range(radar, rng, sigma_m=100.0)
            retained_active += int(updated.sum())
    return {
        "source_active_radar_measurements": source_active,
        "retained_active_radar_measurements": retained_active,
        "retained_ratio": retained_active / source_active if source_active else 0.0,
    }


def _read_stats(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        stats = pickle.load(handle).get("AssociationStats")
    if not isinstance(stats, dict):
        raise RuntimeError(f"AssociationStats missing from {path}")
    return stats


def _audit_one(task: tuple[str, str, int, str]) -> dict[str, Any]:
    mat_path_text, condition, base_seed, temporary_root = task
    mat_path = Path(mat_path_text)
    scenario_seed = base_seed ^ zlib.crc32(f"{mat_path.name}:{condition}".encode("utf-8"))
    process_id = os.getpid()
    output = Path(temporary_root) / f"{mat_path.stem}-{condition}-{process_id}.pkl"
    modified_mat = Path(temporary_root) / f"{mat_path.stem}-{condition}-{process_id}.mat"
    mutation = {
        "source_active_radar_measurements": 0,
        "retained_active_radar_measurements": 0,
        "retained_ratio": 1.0,
    }
    try:
        source = mat_path
        if condition != "baseline":
            data = sio.loadmat(mat_path, simplify_cells=True)
            mutation = apply_interference(data, condition, scenario_seed)
            payload = {
                key: copy.deepcopy(data[key])
                for key in ("Data_Input", "Data_Label", "truth", "T")
                if key in data
            }
            sio.savemat(modified_mat, payload, do_compression=True)
            source = modified_mat
        process_file(str(source), str(output), verbose=False)
        return {
            "scenario": mat_path.stem,
            "scenario_class": scenario_class(mat_path),
            "condition": condition,
            "mutation": mutation,
            "association": _read_stats(output),
        }
    finally:
        output.unlink(missing_ok=True)
        modified_mat.unlink(missing_ok=True)


def aggregate(records: Iterable[dict[str, Any]], sensors: tuple[str, ...]) -> dict[str, Any]:
    correct = 0
    total = 0
    leakage_files = 0
    retained = []
    record_list = list(records)
    for record in record_list:
        stats = record["association"]
        leakage_files += int(bool(stats.get("truth_used_for_observation_grouping", True)))
        retained.append(float(record["mutation"].get("retained_ratio", 1.0)))
        for sensor in sensors:
            values = (stats.get("per_sensor") or {}).get(sensor) or {}
            correct += int(values.get("correct", 0))
            total += int(values.get("total", 0))
    accuracy = correct / total if total else None
    return {
        "audited_scenarios": len(record_list),
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "threshold": THRESHOLD,
        "passed": bool(accuracy is not None and accuracy >= THRESHOLD and leakage_files == 0),
        "truth_used_for_observation_grouping": leakage_files > 0,
        "mean_retained_radar_ratio": float(np.mean(retained)) if retained else None,
    }


def build_association_matrix(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cell[(record["condition"], record["scenario_class"])].append(record)
    cells = []
    for condition in CONDITIONS:
        for class_name in ("CV", "CA", "CT", "CSSW"):
            cell_records = by_cell[(condition, class_name)]
            combinations = {
                name: aggregate(cell_records, sensors)
                for name, sensors in SENSOR_COMBINATIONS.items()
            }
            cells.append(
                {
                    "condition": condition,
                    "scenario_class": class_name,
                    "sensor_combinations": combinations,
                }
            )
    valid_cells = [
        {
            "condition": cell["condition"],
            "scenario_class": cell["scenario_class"],
            "sensor_combination": name,
            "accuracy": values["accuracy"],
        }
        for cell in cells
        for name, values in cell["sensor_combinations"].items()
        if values["accuracy"] is not None
    ]
    valid_accuracies = [cell["accuracy"] for cell in valid_cells]
    condition_summaries = []
    for condition in CONDITIONS:
        condition_cells = [cell for cell in cells if cell["condition"] == condition]
        all_sensor = [cell["sensor_combinations"]["radar_eo_esm"] for cell in condition_cells]
        correct = sum(int(item["correct"]) for item in all_sensor)
        total = sum(int(item["total"]) for item in all_sensor)
        condition_summaries.append(
            {
                "condition": condition,
                "all_sensor_accuracy": correct / total if total else None,
                "correct": correct,
                "total": total,
                "minimum_cell_accuracy": min(
                    float(values["accuracy"])
                    for cell in condition_cells
                    for values in cell["sensor_combinations"].values()
                    if values["accuracy"] is not None
                ),
            }
        )
    return {
        "cells": cells,
        "condition_summaries": condition_summaries,
        "minimum_cell_accuracy": min(valid_accuracies) if valid_accuracies else None,
        "worst_cell": min(valid_cells, key=lambda item: item["accuracy"]) if valid_cells else None,
        "all_cells_passed": bool(valid_accuracies) and all(
            value >= THRESHOLD for value in valid_accuracies
        ),
        "sensor_combination_method": (
            "Radar is the anchor. EO and ESM Hungarian assignments are independent given the "
            "radar tracks, so each combination is audited by aggregating only its included "
            "sensor-event records from the same association run."
        ),
    }


def _inventory() -> dict[str, Any]:
    v7 = dataset_files("v7", suffix=".pkl")
    v8 = dataset_files("v8_interference", suffix=".pkl")
    return {
        "v7": {
            "scenario_count": len(v7),
            "class_counts": class_counts(v7),
            "interference_modes": ["无专项干扰"],
        },
        "v8_interference": {
            "scenario_count": len(v8),
            "class_counts": class_counts(v8),
            "interference_modes": ["调幅闪烁", "软压制", "距离抖动"],
            "coverage_note": "现有1个复合强干扰验证场景，列入典型难场景，不作为大样本统计结论。",
        },
        "taxonomy": SCENARIO_TAXONOMY,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    selected = selected_dataset_files(
        "v7",
        args.max_scenarios,
        suffix=".mat",
        strategy="stratified_by_scenario_type",
    )
    tasks = []
    with tempfile.TemporaryDirectory(prefix="review-items-4-6-") as temporary_root:
        for path in selected:
            for condition in CONDITIONS:
                tasks.append((str(path), condition, args.seed, temporary_root))
        records = []
        if args.workers <= 1:
            for index, task in enumerate(tasks, start=1):
                records.append(_audit_one(task))
                if index % 20 == 0 or index == len(tasks):
                    print(f"association audit {index}/{len(tasks)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(_audit_one, task) for task in tasks]
                for index, future in enumerate(as_completed(futures), start=1):
                    records.append(future.result())
                    if index % 20 == 0 or index == len(tasks):
                        print(f"association audit {index}/{len(tasks)}", flush=True)

    audit_counts = class_counts(selected)
    dimensions = [scenario_dimensions(path, "v7") for path in selected]
    operation_counts: dict[str, int] = defaultdict(int)
    intensity_counts: dict[str, int] = defaultdict(int)
    difficult_count = 0
    for dimension in dimensions:
        operation_counts[str(dimension["operational_task"])] += 1
        intensity_counts[str(dimension["maneuver_intensity"])] += 1
        difficult_count += int("typical_difficult" in dimension["difficulty_tags"])

    matrix = build_association_matrix(records)
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_items": [4, 5, 6],
        "method_constraints": {
            "model_weights_changed": False,
            "training_data_changed": False,
            "interference_application": "deterministic in-memory validation copies",
            "truth_used_for_observation_grouping": False,
        },
        "review_item_4": {
            "title": "Scenario taxonomy and difficult-case coverage",
            "full_inventory": _inventory(),
            "audit_sample": {
                "scenario_count": len(selected),
                "class_counts": audit_counts,
                "operational_task_counts": dict(sorted(operation_counts.items())),
                "maneuver_intensity_counts": dict(sorted(intensity_counts.items())),
                "typical_difficult_scenario_count": difficult_count,
                "interference_conditions": list(CONDITIONS),
            },
        },
        "review_item_5": {
            "title": "End-to-end technical boundary",
            "system_pipeline": (
                "raw multi-source observation packets -> measurement-only target association and "
                "source-timestamp alignment -> registered target-wise sequences -> neural fusion "
                "tracking -> position and velocity tracks"
            ),
            "model_scope": (
                "registered, time-aligned and target-associated multi-source sequences -> continuous "
                "position and velocity estimates"
            ),
            "explicit_exclusion": (
                "The neural model does not jointly learn raw detection extraction or target identity "
                "association; those are deterministic preprocessing stages in the complete software chain."
            ),
            "repository_baseline_status": "fixed_and_documented",
            "external_confirmation_status": "requires_institute_and_customer_signoff",
        },
        "review_item_6": {
            "title": "Stratified association-accuracy audit",
            "threshold": THRESHOLD,
            "scenario_count": len(selected),
            "scenario_condition_runs": len(records),
            "scenario_class_counts": audit_counts,
            "conditions": list(CONDITIONS),
            "sensor_combinations": SENSOR_COMBINATIONS,
            "association_matrix": matrix,
        },
    }
    _write_json_atomic(Path(args.output).resolve(), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-scenarios", type=int, default=32)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    payload = run(args)
    summary = {
        "output": str(Path(args.output).resolve()),
        "item_4_sample": payload["review_item_4"]["audit_sample"],
        "item_5_boundary": payload["review_item_5"],
        "item_6": {
            key: payload["review_item_6"][key]
            for key in ("scenario_count", "scenario_condition_runs", "scenario_class_counts")
        },
        "minimum_cell_accuracy": payload["review_item_6"]["association_matrix"]["minimum_cell_accuracy"],
        "all_cells_passed": payload["review_item_6"]["association_matrix"]["all_cells_passed"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
