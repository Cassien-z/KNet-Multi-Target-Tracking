from __future__ import annotations

import unittest
from pathlib import Path

from tracking_system.backend.scenario_catalog import class_counts, select_stratified


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


if __name__ == "__main__":
    unittest.main()
