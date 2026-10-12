"""Compare event-purity and association-pair metrics on raw tracking data.

The historical acceptance figure reports event purity after assigning each
registered track a truth label.  This audit calculates the older MATLAB metric
directly from output Radar-EO and Radar-ESM pairs, and also reports recall and
F1 against all truth-compatible pair opportunities.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import scipy.io as sio

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_association.pipeline_core import _measurement_time, _radar_world, _sensor_frame
from src.data_association.track_association import TrackAssociation, hungarian_assignment, wrap_pi


PAIR_TYPES = ("Radar-EO", "Radar-ESM", "Radar-Radar")


class LegacyTrackAssociation(TrackAssociation):
    """Same-platform topology calculation used when the V7 figure was made."""

    def associate_same_platform(
        self, radar_meas, eo_meas, esm_meas, radar_valid, eo_valid, esm_valid, platform_id
    ):
        radar_ids = np.flatnonzero(radar_valid)
        eo_ids = np.flatnonzero(eo_valid)
        esm_ids = np.flatnonzero(esm_valid)

        eo_cost = np.full((len(radar_ids), len(eo_ids)), np.inf)
        for i, radar_id in enumerate(radar_ids):
            for j, eo_id in enumerate(eo_ids):
                angular = np.linalg.norm([
                    wrap_pi(eo_meas[eo_id, 0] - radar_meas[radar_id, 0]),
                    wrap_pi(eo_meas[eo_id, 1] - radar_meas[radar_id, 1]),
                ])
                if angular > self.gateEO:
                    continue
                radar_topology = self._topology_vector(radar_meas, radar_ids, radar_id, [0, 1])
                eo_topology = self._topology_vector(eo_meas, eo_ids, eo_id, [0, 1])
                topology = np.linalg.norm(radar_topology - eo_topology) / np.sqrt(self.topoK)
                if topology > self.gateTopoEO:
                    continue
                cost = angular + self.lambdaTopoEO * topology
                previous = self.prevEOForRadar.get((platform_id, int(radar_id)))
                if previous == int(eo_id):
                    cost *= 0.70
                elif previous is not None:
                    cost += self.switchPenaltyEO
                if cost <= self.totalGateEO:
                    eo_cost[i, j] = cost

        esm_cost = np.full((len(radar_ids), len(esm_ids)), np.inf)
        for i, radar_id in enumerate(radar_ids):
            for j, esm_id in enumerate(esm_ids):
                angular = abs(wrap_pi(esm_meas[esm_id, 0] - radar_meas[radar_id, 0]))
                if angular > self.gateESM:
                    continue
                radar_topology = self._topology_vector(radar_meas, radar_ids, radar_id, [0])
                esm_topology = self._topology_vector(esm_meas, esm_ids, esm_id, [0])
                topology = np.linalg.norm(radar_topology - esm_topology) / np.sqrt(self.topoK)
                if topology > self.gateTopoESM:
                    continue
                cost = angular + self.lambdaTopoESM * topology
                previous = self.prevESMForRadar.get((platform_id, int(radar_id)))
                if previous == int(esm_id):
                    cost *= 0.70
                elif previous is not None:
                    cost += self.switchPenaltyESM
                if cost <= self.totalGateESM:
                    esm_cost[i, j] = cost

        eo_pairs = hungarian_assignment(eo_cost, self.totalGateEO)
        esm_pairs = hungarian_assignment(esm_cost, self.totalGateESM)
        eo_by_radar = {int(radar_ids[i]): int(eo_ids[j]) for i, j, _ in eo_pairs}
        esm_by_radar = {int(radar_ids[i]): int(esm_ids[j]) for i, j, _ in esm_pairs}
        for radar_id, eo_id in eo_by_radar.items():
            self.prevEOForRadar[(platform_id, radar_id)] = eo_id
        for radar_id, esm_id in esm_by_radar.items():
            self.prevESMForRadar[(platform_id, radar_id)] = esm_id
        return eo_by_radar, esm_by_radar


def truth_segments(truth: dict[str, Any]) -> dict[tuple[str, int, int], list[tuple[int, float, float]]]:
    lookup: dict[tuple[str, int, int], list[tuple[int, float, float]]] = {}
    for sensor, key in (("Radar", "RadarSeg"), ("EO", "EOSeg"), ("ESM", "ESMSeg")):
        for row in np.atleast_2d(np.asarray(truth.get(key, []), dtype=float)):
            if row.size < 6:
                continue
            lookup.setdefault((sensor, int(row[4]) - 1, int(row[5]) - 1), []).append(
                (int(row[1]) - 1, float(row[2]), float(row[3]))
            )
    return lookup


def truth_at(lookup, sensor: str, uav_id: int, local_id: int, timestamp: float):
    for truth_id, start, end in lookup.get((sensor, uav_id, local_id), []):
        if start <= timestamp <= end:
            return truth_id
    return None


def empty_counts() -> dict[str, dict[str, int]]:
    return {
        pair_type: {"formed": 0, "correct": 0, "possible": 0, "unscored": 0, "duplicate": 0}
        for pair_type in PAIR_TYPES
    }


def possible_pairs(left_labels: list[int | None], right_labels: list[int | None]) -> int:
    left = Counter(value for value in left_labels if value is not None)
    right = Counter(value for value in right_labels if value is not None)
    return sum(min(left[key], right[key]) for key in left.keys() & right.keys())


def score_mapping(
    counts: dict[str, dict[str, int]], pair_type: str, mapping: dict[int, int],
    left_labels: dict[int, int | None], right_labels: dict[int, int | None],
) -> None:
    values = list(mapping.values())
    counts[pair_type]["duplicate"] += len(values) - len(set(values))
    for left_id, right_id in mapping.items():
        counts[pair_type]["formed"] += 1
        left_truth = left_labels.get(left_id)
        right_truth = right_labels.get(right_id)
        if left_truth is None or right_truth is None:
            counts[pair_type]["unscored"] += 1
        elif left_truth == right_truth:
            counts[pair_type]["correct"] += 1


def evaluate_file(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    mat = sio.loadmat(path, simplify_cells=True)
    frames = list(mat["Data_Input"])
    truth = mat.get("truth", {})
    t_grid = np.asarray(truth.get("t_grid", [frame["GlobalTime"] for frame in frames]), dtype=float).reshape(-1)
    lookup = truth_segments(truth)
    num_uavs = max(len(frame["UAV"]) for frame in frames)
    positions = np.zeros((len(frames), num_uavs, 3), dtype=float)
    for frame_id, frame in enumerate(frames):
        for uav_id, uav in enumerate(frame["UAV"]):
            positions[frame_id, uav_id] = uav["Pose"]["PositionWorld"]
    velocities = np.gradient(positions, t_grid, axis=0)

    modes = {"legacy": LegacyTrackAssociation(), "current": TrackAssociation()}
    results = {name: empty_counts() for name in modes}

    for frame_id, frame in enumerate(frames):
        sensor_data = [
            {sensor: _sensor_frame(uav, sensor) for sensor in ("Radar", "EO", "ESM")}
            for uav in frame["UAV"]
        ]
        reference_radar, reference_valid = sensor_data[0]["Radar"]
        reference_world = _radar_world(
            reference_radar, reference_valid, positions[frame_id, 0], velocities[frame_id, 0]
        )
        reference_times = {
            int(local_id): _measurement_time(frame["UAV"][0], "Radar", int(local_id), t_grid[frame_id])
            for local_id in np.flatnonzero(reference_valid)
        }
        reference_labels = {
            local_id: truth_at(lookup, "Radar", 0, local_id, timestamp)
            for local_id, timestamp in reference_times.items()
        }
        # Cross-platform history may legitimately attach a wingman radar event
        # to a reference track while the reference radar has no measurement in
        # the current frame.  Score against the reference track identity, not
        # only against currently valid reference measurements.
        reference_track_labels = {
            int(local_id): truth_at(
                lookup,
                "Radar",
                0,
                int(local_id),
                _measurement_time(frame["UAV"][0], "Radar", int(local_id), t_grid[frame_id]),
            )
            for local_id in range(len(reference_radar))
        }

        for uav_id, data in enumerate(sensor_data):
            radar, radar_valid = data["Radar"]
            eo, eo_valid = data["EO"]
            esm, esm_valid = data["ESM"]
            radar_times = {
                int(local_id): _measurement_time(frame["UAV"][uav_id], "Radar", int(local_id), t_grid[frame_id])
                for local_id in np.flatnonzero(radar_valid)
            }
            eo_times = {
                int(local_id): _measurement_time(frame["UAV"][uav_id], "EO", int(local_id), t_grid[frame_id])
                for local_id in np.flatnonzero(eo_valid)
            }
            esm_times = {
                int(local_id): _measurement_time(frame["UAV"][uav_id], "ESM", int(local_id), t_grid[frame_id])
                for local_id in np.flatnonzero(esm_valid)
            }
            radar_labels = {local_id: truth_at(lookup, "Radar", uav_id, local_id, timestamp) for local_id, timestamp in radar_times.items()}
            eo_labels = {local_id: truth_at(lookup, "EO", uav_id, local_id, timestamp) for local_id, timestamp in eo_times.items()}
            esm_labels = {local_id: truth_at(lookup, "ESM", uav_id, local_id, timestamp) for local_id, timestamp in esm_times.items()}

            eo_possible = possible_pairs(list(radar_labels.values()), list(eo_labels.values()))
            esm_possible = possible_pairs(list(radar_labels.values()), list(esm_labels.values()))

            platform_world = None
            if uav_id > 0:
                platform_world = _radar_world(
                    radar, radar_valid, positions[frame_id, uav_id], velocities[frame_id, uav_id]
                )
                radar_possible = possible_pairs(list(radar_labels.values()), list(reference_track_labels.values()))
            else:
                radar_possible = 0

            for name, associator in modes.items():
                counts = results[name]
                counts["Radar-EO"]["possible"] += eo_possible
                counts["Radar-ESM"]["possible"] += esm_possible
                eo_map, esm_map = associator.associate_same_platform(
                    radar, eo, esm, radar_valid, eo_valid, esm_valid, uav_id
                )
                score_mapping(counts, "Radar-EO", eo_map, radar_labels, eo_labels)
                score_mapping(counts, "Radar-ESM", esm_map, radar_labels, esm_labels)

                if uav_id > 0:
                    counts["Radar-Radar"]["possible"] += radar_possible
                    radar_map = associator.associate_radar_platforms(
                        reference_world, platform_world, reference_valid, radar_valid, uav_id
                    )
                    for local_id in np.flatnonzero(radar_valid):
                        previous = associator.previous_radar_mapping(uav_id, int(local_id))
                        if previous is not None:
                            radar_map.setdefault(int(local_id), int(previous))
                    # Mapping direction is platform-local -> reference-local.
                    score_mapping(counts, "Radar-Radar", radar_map, radar_labels, reference_track_labels)

    return {"file": path.stem, "modes": results}


def aggregate(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    counts = empty_counts()
    for record in records:
        for pair_type in PAIR_TYPES:
            for field in counts[pair_type]:
                counts[pair_type][field] += int(record["modes"][mode][pair_type][field])

    def metrics(items: list[str]) -> dict[str, Any]:
        formed = sum(counts[item]["formed"] for item in items)
        correct = sum(counts[item]["correct"] for item in items)
        possible = sum(counts[item]["possible"] for item in items)
        unscored = sum(counts[item]["unscored"] for item in items)
        duplicates = sum(counts[item]["duplicate"] for item in items)
        precision = correct / formed if formed else None
        recall = correct / possible if possible else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        return {
            "correct": correct,
            "formed": formed,
            "possible": possible,
            "unscored_formed": unscored,
            "duplicate_assignments": duplicates,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "formation_coverage": formed / possible if possible else None,
        }

    per_type = {pair_type: metrics([pair_type]) for pair_type in PAIR_TYPES}
    return {
        "pair_counts": counts,
        "per_type": per_type,
        "matlab_passive_pair_scope": metrics(["Radar-EO", "Radar-ESM"]),
        "all_pair_scope": metrics(list(PAIR_TYPES)),
    }


def draw(payload: dict[str, Any], output: Path) -> None:
    event = payload["historical_event_purity"]
    legacy = payload["pair_metrics"]["legacy"]["matlab_passive_pair_scope"]
    current = payload["pair_metrics"]["current"]["matlab_passive_pair_scope"]
    for font_path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    labels = [
        "历史观测\n事件纯度",
        "旧算法关联对\n正确率",
        "旧算法关联对\n召回率",
        "旧算法关联对\nF1",
        "当前算法关联对\n正确率",
        "当前算法关联对\n召回率",
    ]
    values = [event["accuracy"], legacy["precision"], legacy["recall"], legacy["f1"], current["precision"], current["recall"]]
    if any(value is None for value in values):
        return
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    colors = ["#245C83", "#5B9D85", "#C68938", "#9A6AB0", "#4C78A8", "#F58518"]
    bars = ax.bar(labels, np.asarray(values) * 100, color=colors, width=0.68)
    ax.axhline(85, color="#B4413E", linestyle="--", linewidth=1.8, label="合同阈值85%")
    ax.set_ylim(0, 108)
    ax.set_ylabel("比率（%）")
    ax.set_title("V7完整验证集关联统计口径对比")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 100 + 1.0, f"{value * 100:.2f}%", ha="center", fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "val_data_v7")
    parser.add_argument("--historical-evidence", type=Path, default=ROOT / "artifacts" / "metrics" / "acceptance" / "2ea4f9eb0f2845b599fcf4837d71a018" / "association_accuracy.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "metrics" / "association_metric_definition_comparison_v7.json")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-scenarios", type=int)
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("track_*.mat"))
    if args.max_scenarios is not None:
        files = files[: args.max_scenarios]
    if not files:
        raise FileNotFoundError(f"No track_*.mat files in {args.data_dir}")

    records = []
    if args.workers <= 1:
        for index, path in enumerate(files, 1):
            records.append(evaluate_file(str(path)))
            if index % 10 == 0 or index == len(files):
                print(f"pair audit {index}/{len(files)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_file, str(path)) for path in files]
            for index, future in enumerate(as_completed(futures), 1):
                records.append(future.result())
                if index % 25 == 0 or index == len(files):
                    print(f"pair audit {index}/{len(files)}", flush=True)

    historical = json.loads(args.historical_evidence.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "scenario_count": len(files),
        "historical_event_purity": historical["association"],
        "historical_evidence": str(args.historical_evidence.resolve()),
        "pair_metric_definition": {
            "precision": "correct output pairs / all output pairs",
            "recall": "correct output pairs / all truth-compatible pair opportunities",
            "f1": "harmonic mean of pair precision and pair recall",
            "matlab_passive_pair_scope": ["Radar-EO", "Radar-ESM"],
            "all_pair_scope": list(PAIR_TYPES),
        },
        "pair_metrics": {
            "legacy": aggregate(records, "legacy"),
            "current": aggregate(records, "current"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    figure = args.output.with_suffix(".png")
    draw(payload, figure)
    print(json.dumps({
        "scenarios": len(files),
        "event_purity": payload["historical_event_purity"]["accuracy"],
        "legacy_pair": payload["pair_metrics"]["legacy"]["matlab_passive_pair_scope"],
        "current_pair": payload["pair_metrics"]["current"]["matlab_passive_pair_scope"],
        "output": str(args.output),
        "figure": str(figure),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
