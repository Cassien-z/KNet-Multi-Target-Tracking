from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tracking_system.backend.batch_testing import BatchTestManager, ensure_finite_tensor


class _UnusedModelManager:
    pass


class BatchTestingSafetyTests(unittest.TestCase):
    def test_non_finite_tensor_is_rejected(self) -> None:
        ensure_finite_tensor("prediction", torch.tensor([1.0, 2.0]))
        with self.assertRaisesRegex(RuntimeError, "prediction"):
            ensure_finite_tensor("prediction", torch.tensor([1.0, float("nan")]))
        with self.assertRaisesRegex(RuntimeError, "prediction"):
            ensure_finite_tensor("prediction", torch.tensor([float("inf")]))

    def test_restart_recovery_is_written_back_to_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory)
            report_path = report_dir / "interrupted.json"
            report_path.write_text(
                json.dumps(
                    {
                        "id": "interrupted",
                        "name": "interrupted job",
                        "status": "running",
                        "created_at": "2026-09-01T00:00:00+00:00",
                        "finished_at": None,
                        "progress": {"stage": "running", "message": "执行中"},
                    }
                ),
                encoding="utf-8",
            )
            manager = BatchTestManager(_UnusedModelManager(), report_dir=report_dir)
            try:
                loaded = manager.get_job("interrupted")
                persisted = json.loads(report_path.read_text(encoding="utf-8"))
            finally:
                manager.shutdown()

        self.assertEqual(loaded["status"], "failed")
        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(persisted["progress"]["stage"], "failed")
        self.assertIsNotNone(persisted["finished_at"])


if __name__ == "__main__":
    unittest.main()
