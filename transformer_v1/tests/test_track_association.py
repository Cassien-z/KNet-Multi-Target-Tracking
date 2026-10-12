import unittest

import numpy as np

from src.data_association.pipeline_core import _align_stream
from src.data_association.track_association import TrackAssociation, hungarian_assignment


class TrackAssociationTests(unittest.TestCase):
    def test_time_alignment_uses_source_measurement_timestamps(self):
        t_grid = np.arange(5, dtype=float)
        raw = np.full((5, 1), np.nan)
        raw[1:4, 0] = [10.0, 20.0, 30.0]
        valid = np.array([False, True, True, True, False])
        source_times = np.array([np.nan, 0.0, 1.0, 2.0, np.nan])

        aligned, aligned_valid = _align_stream(raw, valid, t_grid, source_times)

        np.testing.assert_allclose(aligned[:3, 0], [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(aligned_valid, [True, True, True, False, False])

    def test_hungarian_assignment_is_one_to_one(self):
        costs = np.array([[5.0, 1.0, 9.0], [2.0, 8.0, 3.0], [9.0, 4.0, 1.0]])
        pairs = hungarian_assignment(costs, gate=5.0)
        self.assertEqual({(row, col) for row, col, _ in pairs}, {(0, 1), (1, 0), (2, 2)})

    def test_cross_sensor_association_handles_permuted_local_ids(self):
        associator = TrackAssociation()
        radar = np.array([[0.10, 0.01, 5000.0], [0.40, -0.02, 8000.0]])
        eo = np.array([[0.401, -0.019], [0.099, 0.011]])
        esm = np.array([[0.398], [0.102]])
        valid = np.ones(2, dtype=bool)

        eo_map, esm_map = associator.associate_same_platform(
            radar, eo, esm, valid, valid, valid, platform_id=0
        )

        self.assertEqual(eo_map, {0: 1, 1: 0})
        self.assertEqual(esm_map, {0: 1, 1: 0})

    def test_cross_platform_association_uses_world_geometry(self):
        associator = TrackAssociation()
        reference = np.array([[1000.0, 0.0, 0.0], [5000.0, 0.0, 0.0]])
        wingman = np.array([[5010.0, 5.0, 0.0], [990.0, -5.0, 0.0]])
        valid = np.ones(2, dtype=bool)

        mapping = associator.associate_radar_platforms(
            reference, wingman, valid, valid, platform_id=1
        )

        self.assertEqual(mapping, {0: 1, 1: 0})

    def test_cross_sensor_association_ignores_unmatched_clutter_in_topology(self):
        associator = TrackAssociation()
        radar = np.array([[0.10, 0.01, 5000.0], [0.40, -0.02, 8000.0]])
        eo = np.array([[0.401, -0.019], [1.50, 0.20], [0.099, 0.011]])
        esm = np.array([[0.398], [1.50], [0.102]])

        eo_map, esm_map = associator.associate_same_platform(
            radar,
            eo,
            esm,
            np.ones(2, dtype=bool),
            np.ones(3, dtype=bool),
            np.ones(3, dtype=bool),
            platform_id=0,
        )

        self.assertEqual(eo_map, {0: 2, 1: 0})
        self.assertEqual(esm_map, {0: 2, 1: 0})

    def test_cross_sensor_association_survives_temporary_dropout(self):
        associator = TrackAssociation()
        radar = np.array([[0.004, 0.0, 5000.0], [0.006, 0.0, 5000.0]])
        eo = np.array([[0.006, 0.0], [0.004, 0.0]])

        eo_map, _ = associator.associate_same_platform(
            radar,
            eo,
            np.empty((0, 1)),
            np.ones(2, dtype=bool),
            np.array([False, True]),
            np.zeros(0, dtype=bool),
            platform_id=0,
        )

        self.assertEqual(len(eo_map), 1)


if __name__ == "__main__":
    unittest.main()
