from __future__ import annotations

import unittest

import torch

from scripts.evaluation.review_items_1_3 import apply_condition, operation_scenario


def _batch() -> dict:
    batch_size, steps, agents = 2, 10, 3
    ones = torch.ones
    zeros = torch.zeros
    return {
        "radar": ones((batch_size, steps, agents, 3)),
        "eo": ones((batch_size, steps, agents, 2)),
        "esm": ones((batch_size, steps, agents, 2)),
        "radar_mask": ones((batch_size, steps, agents)),
        "eo_mask": ones((batch_size, steps, agents)),
        "esm_mask": ones((batch_size, steps, agents)),
        "fast_link": ones((batch_size, steps, agents, 3)),
        "slow_link": ones((batch_size, steps, agents, 3)),
        "fast_update": ones((batch_size, steps, agents)),
        "slow_update": ones((batch_size, steps, agents)),
        "fast_age": zeros((batch_size, steps, agents)),
        "slow_age": zeros((batch_size, steps, agents)),
    }


class ReviewItemsOneToThreeTests(unittest.TestCase):
    def test_operation_scenario_mapping(self) -> None:
        self.assertEqual(operation_scenario("track_cv_0001.pkl_target_0"), "cooperative_detection")
        self.assertEqual(operation_scenario("track_ct_0001.pkl_target_0"), "cooperative_detection")
        self.assertEqual(operation_scenario("track_ca_0001.pkl_target_0"), "cooperative_strike")
        self.assertEqual(operation_scenario("track_cssw_0001.pkl_target_0"), "cooperative_strike")

    def test_random_loss_is_deterministic_and_zeroes_measurements(self) -> None:
        first, first_meta = apply_condition(_batch(), "random_measurement_drop_20pct", seed=7)
        second, second_meta = apply_condition(_batch(), "random_measurement_drop_20pct", seed=7)
        self.assertTrue(torch.equal(first["radar_mask"], second["radar_mask"]))
        self.assertEqual(first_meta, second_meta)
        self.assertTrue(torch.all(first["radar"][first["radar_mask"] == 0] == 0))
        self.assertLess(first_meta["retained_ratio"], 1.0)

    def test_all_source_occlusion_removes_five_central_frames(self) -> None:
        result, _ = apply_condition(_batch(), "all_source_occlusion_0_5s", seed=1)
        for sensor in ("radar", "eo", "esm"):
            self.assertEqual(int(result[f"{sensor}_mask"][:, 2:7].count_nonzero()), 0)
        self.assertEqual(int(result["fast_update"][:, 2:7].count_nonzero()), 0)
        self.assertTrue(torch.all(result["fast_age"][:, 2:7] == 999.0))

    def test_delay_shifts_datalink_by_two_frames_without_shifting_local_sensors(self) -> None:
        batch = _batch()
        batch["radar"][:, 0] = 3.0
        result, _ = apply_condition(batch, "datalink_delay_0_2s", seed=1)
        self.assertTrue(torch.all(result["radar"][:, 0] == 3.0))
        self.assertEqual(int(result["fast_update"][:, :2].count_nonzero()), 0)
        self.assertTrue(torch.all(result["fast_age"][:, 2:] == 0.2))


if __name__ == "__main__":
    unittest.main()
