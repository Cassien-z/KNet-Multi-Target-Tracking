from __future__ import annotations

import unittest
from pathlib import Path

from tracking_system.backend.scenario_catalog import (
    DATASET_CATALOG,
    V8_QUALITY_PROFILE,
    class_counts,
    select_stratified,
)


class ScenarioCatalogTests(unittest.TestCase):
    def test_sample_is_balanced_across_maneuver_types(self) -> None:
        paths = [
            Path(f"track_{maneuver}_{index:04d}.pkl")
            for maneuver in ("ca", "cv", "ct", "cssw")
            for index in range(8)
        ]
        selected = select_stratified(paths, 10)
        counts = class_counts(selected)

        self.assertEqual(len(selected), 10)
        self.assertEqual(set(counts), {"CA", "CV", "CT", "CSSW"})
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_full_scope_keeps_every_scenario(self) -> None:
        paths = [Path("track_cv_0001.pkl"), Path("track_ca_0001.pkl")]
        self.assertEqual(select_stratified(paths, None), sorted(paths))

    def test_v8_uses_ten_hz_model_compatibility_view(self) -> None:
        metadata = DATASET_CATALOG["v8_interference"]
        self.assertEqual(metadata["path"].name, "val_data_v8_interference_registered_10hz")
        self.assertEqual(
            metadata["source_registered_path"].name,
            "val_data_v8_interference_registered",
        )
        self.assertEqual(V8_QUALITY_PROFILE["frame_period_s"], 0.1)
        self.assertEqual(V8_QUALITY_PROFILE["source_frame_period_s"], 0.025)


if __name__ == "__main__":
    unittest.main()
