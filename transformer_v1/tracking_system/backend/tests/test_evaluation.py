from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.acceptance_common import job_signature as script_job_signature
from tracking_system.backend.evaluation import EvaluationService, _job_signature


class _FakeBatchManager:
    def __init__(self, job: dict, report_dir: Path | None = None) -> None:
        self.job = job
        self.report_dir = report_dir or Path(tempfile.gettempdir())

    def get_job(self, job_id: str) -> dict:
        if job_id != self.job["id"]:
            raise KeyError(job_id)
        return copy.deepcopy(self.job)

    def list_jobs(self, limit: int = 200) -> list[dict]:
        return [copy.deepcopy(self.job)][:limit]


def _completed_job() -> dict:
    return {
        "id": "test-job",
        "name": "unit test",
        "status": "completed",
        "created_at": "2026-09-01T00:00:00+00:00",
        "finished_at": "2026-09-01T00:01:00+00:00",
        "config": {
            "model_id": "v7",
            "dataset_id": "v7",
            "device": "cpu",
            "batch_size": 4,
            "workers": 0,
            "max_scenarios": 1,
            "warmup_frames": 5,
            "sequence_length": 100,
        },
        "result": {
            "model_id": "v7",
            "model_path": "results/v7/best_model.pt",
            "dataset_id": "v7",
            "dataset_path": "data/val_data_v7_registered",
            "provenance": {
                "model_sha256": "model-digest",
                "dataset_sha256": "dataset-digest",
                "runtime_code_sha256": "code-digest",
            },
            "metrics": {
                "position_rmse_m": 7.0,
                "evaluated_scenarios": 1,
                "evaluated_tracks": 2,
            },
            "latency": {"estimated_frame_ms": 2.0},
            "association": {"accuracy": 0.9},
            "tracks": [
                {"track": "track_cv_0001_target_0", "rmse_m": 6.0},
                {"track": "track_ca_0001_target_0", "rmse_m": 8.0},
            ],
        },
    }


class EvaluationServiceTests(unittest.TestCase):
    def test_manual_baseline_and_embedded_association_cannot_formally_pass(self) -> None:
        result = EvaluationService(_FakeBatchManager(_completed_job())).assess_job(
            "test-job", baseline_rmse_m=10.0
        )
        self.assertEqual(result["overall_status"], "incomplete")
        self.assertFalse(result["formal_evidence_complete"])
        self.assertEqual(result["summary"], {"total": 4, "passed": 4, "failed": 0, "unavailable": 0})
        self.assertIn("排除每条航迹前5帧", result["criteria"][0]["note"])
        self.assertEqual(
            {item["maneuver"] for item in result["dimensions"]["by_maneuver"]},
            {"CV", "CA"},
        )

    def test_missing_evidence_is_incomplete_not_passed(self) -> None:
        job = _completed_job()
        job["result"]["association"] = None
        result = EvaluationService(_FakeBatchManager(job)).assess_job("test-job")
        self.assertEqual(result["overall_status"], "incomplete")
        self.assertEqual(result["summary"]["unavailable"], 2)

    def test_matching_script_evidence_completes_missing_metrics(self) -> None:
        job = _completed_job()
        job["result"]["association"] = None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            evidence_dir = root / job["id"]
            evidence_dir.mkdir(parents=True)
            common = {
                "job_id": job["id"],
                "job_signature": _job_signature(job),
                "valid_for_assessment": True,
            }
            (evidence_dir / "association_accuracy.json").write_text(
                json.dumps(
                    {
                        **common,
                        "metric_id": "association_accuracy",
                        "value": 0.9,
                        "correct": 90,
                        "total": 100,
                    }
                ),
                encoding="utf-8",
            )
            (evidence_dir / "rmse_improvement_pct.json").write_text(
                json.dumps(
                    {
                        **common,
                        "metric_id": "rmse_improvement_pct",
                        "value": 30.0,
                        "imm_jpda_rmse_m": 10.0,
                    }
                ),
                encoding="utf-8",
            )
            manager = _FakeBatchManager(job, Path(temporary) / "reports")
            result = EvaluationService(manager, evidence_root=root).assess_job("test-job")

        self.assertEqual(result["overall_status"], "passed")
        self.assertEqual(result["summary"]["unavailable"], 0)
        self.assertTrue(result["formal_evidence_complete"])
        self.assertTrue(result["evidence"]["association_accuracy"]["accepted"])
        self.assertTrue(result["evidence"]["rmse_improvement_pct"]["accepted"])

    def test_stale_script_evidence_is_rejected(self) -> None:
        job = _completed_job()
        job["result"]["association"] = None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            evidence_dir = root / job["id"]
            evidence_dir.mkdir(parents=True)
            (evidence_dir / "association_accuracy.json").write_text(
                json.dumps(
                    {
                        "job_id": job["id"],
                        "job_signature": "stale",
                        "metric_id": "association_accuracy",
                        "valid_for_assessment": True,
                        "value": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            manager = _FakeBatchManager(job, Path(temporary) / "reports")
            result = EvaluationService(manager, evidence_root=root).assess_job("test-job")

        self.assertEqual(result["overall_status"], "incomplete")
        self.assertFalse(result["evidence"]["association_accuracy"]["accepted"])

    def test_independent_forward_p95_replaces_sequence_division(self) -> None:
        job = _completed_job()
        values = list(range(10, 30))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            evidence_dir = root / job["id"]
            evidence_dir.mkdir(parents=True)
            (evidence_dir / "forward_latency.json").write_text(
                json.dumps({
                    "job_id": job["id"],
                    "job_signature": _job_signature(job),
                    "model_sha256": "model-digest",
                    "one_frame_input": {
                        "sequence_length": 1,
                        "test_iterations": len(values),
                        "samples_ms": values,
                        "p95_ms": 28.05,
                    },
                    "full_sequence_input": {"p95_ms": 80.0},
                }), encoding="utf-8"
            )
            result = EvaluationService(_FakeBatchManager(job), evidence_root=root).assess_job("test-job")
        self.assertTrue(result["evidence"]["forward_latency"]["accepted"])
        self.assertAlmostEqual(result["criteria"][3]["value"], 28.05)
        self.assertIn("完整100帧输入P95=80.000", result["criteria"][3]["note"])

    def test_stale_forward_evidence_does_not_fall_back_to_divided_sequence(self) -> None:
        job = _completed_job()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            evidence_dir = root / job["id"]
            evidence_dir.mkdir(parents=True)
            (evidence_dir / "forward_latency.json").write_text(
                json.dumps({
                    "job_id": job["id"], "job_signature": "stale",
                    "model_sha256": "model-digest",
                    "one_frame_input": {"sequence_length": 1, "test_iterations": 20,
                                        "samples_ms": [10] * 20, "p95_ms": 10},
                }), encoding="utf-8"
            )
            result = EvaluationService(_FakeBatchManager(job), evidence_root=root).assess_job("test-job")
        self.assertFalse(result["evidence"]["forward_latency"]["accepted"])
        self.assertEqual(result["criteria"][3]["status"], "unavailable")

    def test_historical_cross_dataset_matrix_has_twelve_cells(self) -> None:
        result = EvaluationService.cross_dataset()
        self.assertTrue(result["available"])
        self.assertEqual(len(result["matrix"]), 12)

    def test_completed_jobs_can_be_normalized_for_comparison(self) -> None:
        result = EvaluationService(_FakeBatchManager(_completed_job())).compare_jobs()
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["model_id"], "v7")
        self.assertEqual(result["items"][0]["position_rmse_m"], 7.0)

    def test_signature_covers_execution_and_artifact_identity(self) -> None:
        original = _completed_job()
        self.assertEqual(_job_signature(original), script_job_signature(original))
        changed = copy.deepcopy(original)
        changed["config"]["batch_size"] = 8
        self.assertNotEqual(_job_signature(original), _job_signature(changed))

        changed = copy.deepcopy(original)
        changed["result"]["provenance"] = {"model_sha256": "different"}
        self.assertNotEqual(_job_signature(original), _job_signature(changed))

    def test_track_distribution_includes_count_mean_and_standard_deviation(self) -> None:
        result = EvaluationService(_FakeBatchManager(_completed_job())).assess_job("test-job")
        distribution = result["dimensions"]["distribution"]
        self.assertEqual(distribution["track_count"], 2)
        self.assertEqual(distribution["mean_rmse_m"], 7.0)
        self.assertEqual(distribution["standard_deviation_rmse_m"], 1.0)

    def test_comparison_uses_config_when_list_result_is_compact(self) -> None:
        job = _completed_job()
        job["config"] = {"model_id": "v6", "dataset_id": "v8_interference"}
        job["result"].pop("model_id")
        job["result"].pop("dataset_id")
        result = EvaluationService(_FakeBatchManager(job)).compare_jobs()
        self.assertEqual(result["items"][0]["model_id"], "v6")
        self.assertEqual(result["items"][0]["dataset_id"], "v8_interference")

    def test_report_catalog_and_assessment_bundle(self) -> None:
        job = _completed_job()
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "test-job.json"
            report_path.write_text(json.dumps(job), encoding="utf-8")
            service = EvaluationService(_FakeBatchManager(job, Path(directory)))
            catalog = service.report_catalog()
            bundle = service.assessment_bundle("test-job", baseline_rmse_m=10.0)
        self.assertEqual(catalog["summary"]["assessable"], 1)
        self.assertTrue(catalog["items"][0]["artifacts"]["raw_json"])
        self.assertEqual(bundle["assessment"]["overall_status"], "incomplete")
        self.assertEqual(bundle["provenance"]["baseline_rmse_m"], 10.0)


if __name__ == "__main__":
    unittest.main()
