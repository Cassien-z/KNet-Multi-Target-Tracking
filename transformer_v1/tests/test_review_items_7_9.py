import unittest

import torch

from scripts.evaluation.review_items_7_9 import (
    single_frame_batch,
    stable_position_errors,
    three_second_maneuver_comparison,
)
from tracking_system.backend.batch_testing import (
    ensure_finite_tensor,
    position_rmse_definition,
    stable_stage_mask,
)


class ReviewItemsSevenToNineTests(unittest.TestCase):
    def test_stable_stage_mask_excludes_transition_without_mutating_input(self):
        original = torch.tensor([[True, True, True, False, True]])
        result = stable_stage_mask(original, 2)
        self.assertEqual(result.tolist(), [[False, False, True, False, True]])
        self.assertEqual(original.tolist(), [[True, True, True, False, True]])
        self.assertFalse(position_rmse_definition(2)["cold_start_included"])

    def test_stable_position_errors_only_use_xyz_and_stable_valid_points(self):
        truth = torch.zeros((1, 4, 6))
        prediction = truth.clone()
        prediction[0, :, 0] = torch.tensor([100.0, 2.0, 3.0, 4.0])
        valid = torch.tensor([[True, True, False, True]])
        result = stable_position_errors(prediction, truth, valid, warmup_frames=1)
        self.assertEqual(result.tolist(), [2.0, 4.0])

    def test_three_second_criterion_is_independent_pointwise_comparison(self):
        truth = torch.zeros((1, 7, 6))
        truth[0, :, 0] = torch.arange(7, dtype=torch.float32)
        prediction = truth.clone()
        prediction[0, 1, 0] += 2.0
        prediction[0, 2, 0] += 4.0
        prediction[0, 3, 0] += 1.0
        valid = torch.ones((1, 7), dtype=torch.bool)
        result = three_second_maneuver_comparison(
            prediction,
            truth,
            valid,
            torch.tensor([1.0]),
            warmup_frames=1,
        )
        self.assertEqual(result["horizon_frames"], [3])
        self.assertEqual(result["eligible_points"], 3)
        self.assertEqual(result["compliant_points"], 2)
        self.assertEqual(result["compliance_rate"], 2 / 3)

    def test_single_frame_batch_preserves_static_reference_and_dt(self):
        batch = {
            "radar": torch.zeros((2, 100, 5, 3)),
            "ref_pos": torch.zeros((2, 3)),
            "dt": torch.tensor([0.1, 0.1]),
            "file_name": ["a", "b"],
        }
        result = single_frame_batch(batch)
        self.assertEqual(result["radar"].shape, (2, 1, 5, 3))
        self.assertEqual(result["ref_pos"].shape, (2, 3))
        self.assertEqual(result["dt"].shape, (2,))
        self.assertEqual(result["file_name"], ["a", "b"])

    def test_batch_numeric_safety_rejects_nan_and_infinity(self):
        ensure_finite_tensor("prediction", torch.tensor([0.0, 1.0]))
        with self.assertRaises(RuntimeError):
            ensure_finite_tensor("prediction", torch.tensor([float("nan")]))
        with self.assertRaises(RuntimeError):
            ensure_finite_tensor("prediction", torch.tensor([float("inf")]))


if __name__ == "__main__":
    unittest.main()
