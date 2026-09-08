"""Generate same-scope data-association accuracy evidence for one batch test.

The script first reuses trustworthy ``AssociationStats`` embedded in registered
PKL files. If the selected validation set does not contain them, it reruns the
measurement-only association pipeline on the matching raw MAT scenarios in a
temporary directory. Ground truth is used only after grouping, for audit.

Example:
    python scripts/evaluation/test_association_accuracy.py \
        --batch-report artifacts/reports/batch_tests/<job_id>.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.acceptance_common import (
    REGISTERED_DATASETS,
    RAW_DATASETS,
    evidence_context,
    evidence_path,
    load_batch_report,
    selected_files,
    write_json_atomic,
)
from src.data_association.pipeline_core import process_file


THRESHOLD = 0.85


def _read_stats(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            stats = pickle.load(handle).get("AssociationStats")
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError):
        return None
    return stats if isinstance(stats, dict) and stats.get("total") else None


def _embedded_stats(directory: Path, names: list[str]) -> list[dict[str, Any]] | None:
    records = []
    for name in names:
        stats = _read_stats(directory / f"{name}.pkl")
        if stats is None:
            return None
        records.append(stats)
    return records


def _recompute_one(task: tuple[str, str]) -> dict[str, Any]:
    mat_path, temporary_dir = task
    output = Path(temporary_dir) / f"{Path(mat_path).stem}-{os.getpid()}.pkl"
    try:
        process_file(mat_path, str(output), verbose=False)
        stats = _read_stats(output)
        if stats is None:
            raise RuntimeError(f"Association audit was not produced for {mat_path}")
        return stats
    finally:
        output.unlink(missing_ok=True)


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_sensor: dict[str, dict[str, int | float | None]] = {}
    correct = 0
    total = 0
    leakage_files = 0
    for stats in records:
        correct += int(stats.get("correct", 0))
        total += int(stats.get("total", 0))
        leakage_files += int(bool(stats.get("truth_used_for_observation_grouping", True)))
        for sensor, values in (stats.get("per_sensor") or {}).items():
            target = per_sensor.setdefault(sensor, {"correct": 0, "total": 0})
            target["correct"] = int(target["correct"] or 0) + int(values.get("correct", 0))
            target["total"] = int(target["total"] or 0) + int(values.get("total", 0))
    for values in per_sensor.values():
        sensor_total = int(values["total"] or 0)
        values["accuracy"] = int(values["correct"] or 0) / sensor_total if sensor_total else None
    return {
        "method": "measurement_only_hungarian_with_temporal_memory",
        "audited_files": len(records),
        "truth_used_for_observation_grouping": leakage_files > 0,
        "files_flagged_for_truth_leakage": leakage_files,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "per_sensor": per_sensor,
    }


def evaluate(report: dict[str, Any], workers: int, force_recompute: bool) -> dict[str, Any]:
    raw_files = selected_files(report, ".mat")
    names = [path.stem for path in raw_files]
    dataset_id = str(report["config"]["dataset_id"])
    records = None
    source = None

    if not force_recompute:
        raw_root = RAW_DATASETS[dataset_id]
        candidates = [raw_root.with_name(f"{raw_root.name}_associated"), REGISTERED_DATASETS[dataset_id]]
        for candidate in candidates:
            records = _embedded_stats(candidate, names)
            if records is not None:
                source = f"embedded AssociationStats: {candidate}"
                break

    if records is None:
        source = f"recomputed from raw MAT: {RAW_DATASETS[dataset_id]}"
        records = []
        with tempfile.TemporaryDirectory(prefix="tracking-association-") as temporary_dir:
            tasks = [(str(path), temporary_dir) for path in raw_files]
            if workers <= 1:
                for index, task in enumerate(tasks, start=1):
                    records.append(_recompute_one(task))
                    if index % 25 == 0 or index == len(tasks):
                        print(f"association audit {index}/{len(tasks)}", flush=True)
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(_recompute_one, task) for task in tasks]
                    for index, future in enumerate(as_completed(futures), start=1):
                        records.append(future.result())
                        if index % 25 == 0 or index == len(tasks):
                            print(f"association audit {index}/{len(tasks)}", flush=True)

    summary = aggregate(records)
    value = summary["accuracy"]
    trustworthy = not summary["truth_used_for_observation_grouping"]
    return {
        "schema_version": "1.0",
        "metric_id": "association_accuracy",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **evidence_context(report),
        "value": value,
        "unit": "ratio",
        "operator": ">=",
        "threshold": THRESHOLD,
        "passed": bool(value is not None and value >= THRESHOLD and trustworthy),
        "valid_for_assessment": bool(value is not None and trustworthy),
        "source": source,
        "scenario_manifest": {
            "count": len(names),
            "first": names[0] if names else None,
            "last": names[-1] if names else None,
        },
        "association": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-report", required=True, help="Completed batch-test JSON report")
    parser.add_argument("--output", help="Output JSON; defaults to the job acceptance evidence directory")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force-recompute", action="store_true", help="Ignore embedded AssociationStats")
    args = parser.parse_args()

    report = load_batch_report(args.batch_report)
    result = evaluate(report, max(1, args.workers), args.force_recompute)
    output = Path(args.output) if args.output else evidence_path(report, "association_accuracy")
    saved = write_json_atomic(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved evidence: {saved}")


if __name__ == "__main__":
    main()
