import copy
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluation.review_items_4_6 import apply_interference, aggregate
from tracking_system.backend.scenario_catalog import scenario_dimensions


def _sample_data(frame_count=35):
    frames = []
    for _ in range(frame_count):
        frames.append(
            {
                "UAV": [
                    {
                        "Radar": {
                            "Mask": np.ones(3),
                            "Meas": np.array(
                                [[0.1, 0.2, 1000.0], [0.2, 0.3, 1100.0], [0.3, 0.4, 1200.0]]
                            ),
                        }
                    }
                ]
            }
        )
    return {"Data_Input": frames}


class ReviewItemsFourToSixTests(unittest.TestCase):
    def test_scenario_taxonomy_covers_operation_and_difficulty(self):
        cv = scenario_dimensions(Path("track_cv_0001.pkl"), "v7")
        cssw = scenario_dimensions(Path("track_cssw_0001.pkl"), "v7")
        self.assertEqual(cv["operational_task"], "cooperative_detection")
        self.assertEqual(cssw["operational_task"], "cooperative_strike")
        self.assertIn("typical_difficult", cssw["difficulty_tags"])

    def test_interference_is_deterministic_and_does_not_touch_source(self):
        source = _sample_data()
        first = copy.deepcopy(source)
        second = copy.deepcopy(source)
        first_summary = apply_interference(first, "composite_interference", 17)
        second_summary = apply_interference(second, "composite_interference", 17)
        self.assertEqual(first_summary, second_summary)
        for left, right in zip(first["Data_Input"], second["Data_Input"]):
            np.testing.assert_array_equal(
                left["UAV"][0]["Radar"]["Mask"], right["UAV"][0]["Radar"]["Mask"]
            )
            np.testing.assert_allclose(
                left["UAV"][0]["Radar"]["Meas"], right["UAV"][0]["Radar"]["Meas"]
            )
        self.assertTrue(
            all(np.all(frame["UAV"][0]["Radar"]["Mask"] == 1) for frame in source["Data_Input"])
        )

    def test_sensor_combination_aggregation_uses_only_included_modalities(self):
        record = {
            "mutation": {"retained_ratio": 0.5},
            "association": {
                "truth_used_for_observation_grouping": False,
                "per_sensor": {
                    "Radar": {"correct": 8, "total": 10},
                    "EO": {"correct": 9, "total": 10},
                    "ESM": {"correct": 10, "total": 10},
                },
            },
        }
        result = aggregate([record], ("Radar", "EO"))
        self.assertEqual(result["correct"], 17)
        self.assertEqual(result["total"], 20)
        self.assertAlmostEqual(result["accuracy"], 0.85)
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
