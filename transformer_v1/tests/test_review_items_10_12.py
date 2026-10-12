import unittest

import numpy as np

from scripts.evaluation.review_items_10_12 import kinematic_samples


class ReviewItemsTenToTwelveTests(unittest.TestCase):
    def test_constant_velocity_has_zero_intensity(self):
        velocity = np.tile(np.array([[10.0, 0.0, 0.0]]), (5, 1))
        result = kinematic_samples(velocity, 0.1)
        np.testing.assert_allclose(result["acceleration_magnitude_mps2"], 0.0)
        np.testing.assert_allclose(result["turn_rate_deg_s"], 0.0)
        np.testing.assert_allclose(result["speed_change_rate_mps2"], 0.0)

    def test_quarter_turn_reports_angle_rate_and_acceleration(self):
        velocity = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        result = kinematic_samples(velocity, 1.0)
        self.assertAlmostEqual(result["turn_rate_deg_s"][0], 90.0)
        self.assertAlmostEqual(result["acceleration_magnitude_mps2"][0], np.sqrt(200.0))
        self.assertAlmostEqual(result["speed_change_rate_mps2"][0], 0.0)

    def test_invalid_shape_or_period_is_rejected(self):
        with self.assertRaises(ValueError):
            kinematic_samples(np.zeros((1, 3)), 0.1)
        with self.assertRaises(ValueError):
            kinematic_samples(np.zeros((2, 3)), 0.0)


if __name__ == "__main__":
    unittest.main()
