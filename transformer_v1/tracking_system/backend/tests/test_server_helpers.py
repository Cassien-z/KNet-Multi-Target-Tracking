from __future__ import annotations

import unittest

import numpy as np

from tracking_system.backend.time_utils import nearest_time_index


class ServerHelperTests(unittest.TestCase):
    def test_seek_uses_nearest_frame_and_prefers_earlier_tie(self) -> None:
        times = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
        self.assertEqual(nearest_time_index(times, -1.0), 0)
        self.assertEqual(nearest_time_index(times, 0.04), 0)
        self.assertEqual(nearest_time_index(times, 0.05), 0)
        self.assertEqual(nearest_time_index(times, 0.06), 1)
        self.assertEqual(nearest_time_index(times, 1.0), 2)

    def test_empty_seek_sequence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            nearest_time_index(np.asarray([], dtype=np.float64), 0.0)


if __name__ == "__main__":
    unittest.main()
