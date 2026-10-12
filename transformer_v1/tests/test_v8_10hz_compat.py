from __future__ import annotations

import unittest

import numpy as np

from scripts.data_generation.build_v8_10hz_compat import (
    resample_payload,
    select_resample_indices,
)
from src.dataset import _parse_registered_pkl


def _sensor(frames: int, width: int) -> dict[str, np.ndarray]:
    return {
        "Meas_ECEF": np.arange(frames * 3, dtype=np.float64).reshape(frames, 3),
        "Meas": np.arange(frames * width, dtype=np.float64).reshape(frames, width),
        "valid": np.ones(frames, dtype=bool),
    }


def _payload(frames: int = 400) -> dict:
    times = np.arange(frames, dtype=np.float64) * 0.025
    target = {
        "target_pos": np.arange(frames * 3, dtype=np.float64).reshape(frames, 3),
        "target_vel": np.ones((frames, 3), dtype=np.float64),
        "Radar": [_sensor(frames, 3)],
        "EO": [_sensor(frames, 2)],
        "ESM": [_sensor(frames, 1)],
        "FastLink": [
            {
                "Meas_ECEF": np.zeros((frames, 3), dtype=np.float64),
                "valid": np.ones(frames, dtype=bool),
            }
        ],
    }
    rotations = np.broadcast_to(np.eye(3), (frames, 1, 3, 3)).copy()
    return {
        "RegisteredTrack": [target],
        "t_grid": times,
        "UAV_Pos": np.zeros((frames, 1, 3), dtype=np.float64),
        "UAV_Rotmat": rotations,
    }


class V8CompatibilityTests(unittest.TestCase):
    def test_regular_40hz_grid_selects_every_fourth_frame(self) -> None:
        indices = select_resample_indices(np.arange(400) * 0.025, 0.1)
        np.testing.assert_array_equal(indices, np.arange(0, 400, 4))

    def test_every_time_aligned_field_uses_the_same_indices(self) -> None:
        source = _payload()
        converted, indices = resample_payload(source, 0.1)
        self.assertEqual(converted["t_grid"].shape, (100,))
        self.assertEqual(converted["UAV_Pos"].shape[0], 100)
        self.assertEqual(
            converted["RegisteredTrack"][0]["Radar"][0]["valid"].shape[0], 100
        )
        np.testing.assert_array_equal(
            converted["RegisteredTrack"][0]["target_pos"],
            source["RegisteredTrack"][0]["target_pos"][indices],
        )
        np.testing.assert_array_equal(
            converted["RegisteredTrack"][0]["Radar"][0]["Meas"],
            source["RegisteredTrack"][0]["Radar"][0]["Meas"][indices],
        )

    def test_parser_derives_period_and_keeps_global_time(self) -> None:
        converted, _ = resample_payload(_payload(), 0.1)
        sample = _parse_registered_pkl(converted, 100, 1, "sample.pkl")[0]
        self.assertAlmostEqual(float(sample["dt"]), 0.1, places=6)
        np.testing.assert_allclose(
            sample["global_time"].numpy(), converted["t_grid"].astype(np.float32)
        )

    def test_raw_40hz_parser_no_longer_claims_ten_hz(self) -> None:
        sample = _parse_registered_pkl(_payload(), 100, 1, "sample.pkl")[0]
        self.assertAlmostEqual(float(sample["dt"]), 0.025, places=6)


if __name__ == "__main__":
    unittest.main()
