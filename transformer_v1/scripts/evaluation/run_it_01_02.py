"""Execute software-delivery test case IT-01-02 and generate evidence.

The test exercises measurement-only association, one-to-one assignment,
cross-platform gating, temporal continuity, dropout recovery and a fresh
end-to-end association audit on one frozen validation scenario.
"""

from __future__ import annotations

import csv
import inspect
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_association.pipeline_core import process_file
from src.data_association.track_association import TrackAssociation, hungarian_assignment


OUTPUT = ROOT / "artifacts" / "reports" / "software_test" / "IT-01-02" / "20260923_run01"
FIGURES = OUTPUT / "figures"
SAMPLE_MAT = ROOT / "data" / "val_data_v7" / "track_ca_0000.mat"


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def same_platform(associator: TrackAssociation, radar: np.ndarray, eo: np.ndarray, eo_valid=None):
    radar_valid = np.ones(len(radar), dtype=bool)
    eo_valid = np.ones(len(eo), dtype=bool) if eo_valid is None else np.asarray(eo_valid, dtype=bool)
    esm = np.empty((0, 1), dtype=float)
    esm_valid = np.zeros(0, dtype=bool)
    return associator.associate_same_platform(
        radar, eo, esm, radar_valid, eo_valid, esm_valid, platform_id=0
    )[0]


def run() -> dict[str, object]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    unit = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_track_association.py", "-v"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (OUTPUT / "unit_test_output.txt").write_text(unit.stdout, encoding="utf-8")

    scenario_rows: list[dict[str, object]] = []

    # 1. Manually auditable small matrix and one-to-one conflict resolution.
    cost = np.array([[5.0, 1.0, 9.0], [2.0, 8.0, 3.0], [9.0, 4.0, 1.0]])
    pairs = hungarian_assignment(cost, gate=5.0)
    pair_set = {(r, c) for r, c, _ in pairs}
    manual_ok = pair_set == {(0, 1), (1, 0), (2, 2)} and len({c for _, c in pair_set}) == len(pair_set)
    scenario_rows.append({"scenario": "manual_cost_and_one_to_one", "observed": str(sorted(pair_set)), "passed": manual_ok})

    # 2. Same-platform permutations plus out-of-gate clutter.
    associator = TrackAssociation()
    radar = np.array([[0.10, 0.01, 5000.0], [0.40, -0.02, 8000.0]])
    eo = np.array([[0.401, -0.019], [1.50, 0.20], [0.099, 0.011]])
    esm = np.array([[0.398], [1.50], [0.102]])
    radar_valid = np.ones(2, dtype=bool)
    passive_valid = np.ones(3, dtype=bool)
    eo_map, esm_map = associator.associate_same_platform(
        radar, eo, esm, radar_valid, passive_valid, passive_valid, platform_id=0
    )
    clutter_ok = eo_map == {0: 2, 1: 0} and esm_map == {0: 2, 1: 0}
    scenario_rows.append({"scenario": "permuted_ids_and_clutter", "observed": f"EO={eo_map}; ESM={esm_map}", "passed": clutter_ok})

    # 3. Inclusive 1500 m cross-platform gate boundary with history cleared per point.
    boundary_rows = []
    for distance in (1499.0, 1500.0, 1501.0):
        boundary_associator = TrackAssociation()
        mapping = boundary_associator.associate_radar_platforms(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[distance, 0.0, 0.0]]),
            np.array([True]),
            np.array([True]),
            platform_id=1,
            gate_m=1500.0,
        )
        boundary_rows.append({"distance_m": distance, "accepted": bool(mapping), "mapping": str(mapping)})
    boundary_ok = [row["accepted"] for row in boundary_rows] == [True, True, False]
    scenario_rows.append({"scenario": "cross_platform_gate_boundary", "observed": "1499=accept; 1500=accept; 1501=reject", "passed": boundary_ok})

    # 4. Crossing and temporary dropout. EO local IDs remain stable; angle-only
    # matching swaps them at the ambiguous frame, while retained history preserves IDs.
    frame0_radar = np.array([[0.000, 0.0, 5000.0], [0.010, 0.0, 5000.0]])
    frame0_eo = np.array([[0.000, 0.0], [0.010, 0.0]])
    frame1_radar = np.array([[0.004, 0.0, 5000.0], [0.006, 0.0, 5000.0]])
    frame1_eo = np.array([[0.006, 0.0], [0.004, 0.0]])

    with_history = TrackAssociation()
    map_h0 = same_platform(with_history, frame0_radar, frame0_eo)
    map_h1 = same_platform(with_history, frame1_radar, frame1_eo)
    cleared = TrackAssociation()
    map_c1 = same_platform(cleared, frame1_radar, frame1_eo)
    crossing_ok = map_h0 == {0: 0, 1: 1} and map_h1 == {0: 0, 1: 1} and map_c1 == {0: 1, 1: 0}
    scenario_rows.append({"scenario": "crossing_history_comparison", "observed": f"retained={map_h1}; cleared={map_c1}", "passed": crossing_ok})

    dropout = TrackAssociation()
    same_platform(dropout, frame0_radar, frame0_eo)
    map_drop = same_platform(dropout, frame1_radar, frame1_eo, eo_valid=[False, True])
    map_recovery = same_platform(dropout, frame1_radar, frame1_eo)
    dropout_ok = len(map_drop) == 1 and map_recovery == {0: 0, 1: 1}
    scenario_rows.append({"scenario": "temporary_dropout_and_recovery", "observed": f"drop={map_drop}; recovery={map_recovery}", "passed": dropout_ok})

    # 5. Passive-only and empty-radar input returns no fabricated association.
    empty = TrackAssociation()
    passive_only = empty.associate_same_platform(
        np.empty((0, 3)), np.array([[0.1, 0.0]]), np.array([[0.1]]),
        np.zeros(0, dtype=bool), np.array([True]), np.array([True]), platform_id=0
    )
    passive_ok = passive_only == ({}, {})
    scenario_rows.append({"scenario": "passive_only_no_radar_anchor", "observed": str(passive_only), "passed": passive_ok})

    # 6. Source-order audit: labels/truth are attached after measurement-only association.
    process_source = inspect.getsource(process_file)
    association_position = process_source.index("associate_same_platform")
    truth_position = process_source.index("_label_tracks")
    source_order_ok = association_position < truth_position and "truth" not in inspect.signature(TrackAssociation.associate_same_platform).parameters
    scenario_rows.append({"scenario": "truth_leakage_source_order", "observed": f"association_offset={association_position}; truth_label_offset={truth_position}", "passed": source_order_ok})

    # 7. Fresh raw-MAT processing, then audit the emitted AssociationStats.
    sample_output = OUTPUT / "track_ca_0000_registered.pkl"
    process_file(str(SAMPLE_MAT), str(sample_output), verbose=False)
    with sample_output.open("rb") as handle:
        stats = pickle.load(handle)["AssociationStats"]
    sensor_total = sum(int(item["total"]) for item in stats["per_sensor"].values())
    sensor_correct = sum(int(item["correct"]) for item in stats["per_sensor"].values())
    stats_ok = (
        stats["truth_used_for_observation_grouping"] is False
        and int(stats["total"]) > 0
        and sensor_total == int(stats["total"])
        and sensor_correct == int(stats["correct"])
    )
    scenario_rows.append({"scenario": "fresh_raw_mat_association_stats", "observed": f"{stats['correct']}/{stats['total']}={stats['accuracy']:.6f}", "passed": stats_ok})

    write_csv(OUTPUT / "scenario_checks.csv", scenario_rows)
    write_csv(OUTPUT / "gate_boundary.csv", boundary_rows)

    # Figure 1: small cost matrix and selected Hungarian pairs.
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    image = ax.imshow(cost, cmap="YlGnBu_r", vmin=0, vmax=9)
    for row in range(cost.shape[0]):
        for col in range(cost.shape[1]):
            selected = (row, col) in pair_set
            ax.text(col, row, f"{cost[row, col]:.0f}" + ("\nSELECT" if selected else ""), ha="center", va="center", color="white" if cost[row, col] > 5 else "black", weight="bold" if selected else "normal")
    ax.set_xticks(range(3), ["candidate 0", "candidate 1", "candidate 2"])
    ax.set_yticks(range(3), ["track 0", "track 1", "track 2"])
    ax.set_title("Hungarian one-to-one assignment (gate = 5)")
    fig.colorbar(image, ax=ax, label="association cost")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_1_cost_matrix.png", dpi=180)
    plt.close(fig)

    # Figure 2: inclusive cross-platform boundary.
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    distances = [row["distance_m"] for row in boundary_rows]
    accepted = [int(row["accepted"]) for row in boundary_rows]
    colors = ["#2E8B57" if value else "#C73E1D" for value in accepted]
    ax.bar([str(int(value)) for value in distances], accepted, color=colors, width=0.58)
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 1], ["Rejected", "Accepted"])
    ax.set_xlabel("cross-platform distance (m)")
    ax.set_title("Cross-platform gate boundary (gate = 1500 m, inclusive)")
    for index, value in enumerate(accepted):
        ax.text(index, value + 0.04, "PASS", ha="center", weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_2_gate_boundary.png", dpi=180)
    plt.close(fig)

    # Figure 3: mapping continuity at the crossing frame.
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2), sharey=True)
    for ax, title, mapping in (
        (axes[0], "History retained", map_h1),
        (axes[1], "History cleared", map_c1),
    ):
        radar_ids = sorted(mapping)
        eo_ids = [mapping[item] for item in radar_ids]
        ax.plot([0, 1], [0, eo_ids[0]], marker="o", linewidth=2, label="radar 0")
        ax.plot([0, 1], [1, eo_ids[1]], marker="o", linewidth=2, label="radar 1")
        ax.set_xticks([0, 1], ["previous EO ID", "crossing EO ID"])
        ax.set_yticks([0, 1])
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("associated local ID")
    axes[1].legend(loc="center right", fontsize=8)
    fig.suptitle("Temporal memory preserves IDs through an ambiguous crossing")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_3_history_continuity.png", dpi=180)
    plt.close(fig)

    # Figure 4: fresh sample AssociationStats with event counts printed on bars.
    sensors = ["Radar", "EO", "ESM", "Overall"]
    accuracies = [float(stats["per_sensor"][name]["accuracy"]) * 100 for name in sensors[:-1]] + [float(stats["accuracy"]) * 100]
    totals = [int(stats["per_sensor"][name]["total"]) for name in sensors[:-1]] + [int(stats["total"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(sensors, accuracies, color=["#4472C4", "#70AD47", "#ED7D31", "#5B9BD5"])
    ax.set_ylim(95, 100.8)
    ax.set_ylabel("association accuracy (%)")
    ax.set_title("Fresh audit: track_ca_0000 (truth attached after grouping)")
    ax.grid(axis="y", alpha=0.25)
    for bar, accuracy, total in zip(bars, accuracies, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, accuracy + 0.10, f"{accuracy:.2f}%\nn={total}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_4_sample_stats.png", dpi=180)
    plt.close(fig)

    passed_count = sum(bool(row["passed"]) for row in scenario_rows)
    result = {
        "schema_version": "1.0",
        "test_case": "IT-01-02",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_scope": "Python measurement-level association path",
        "unit_test": {"return_code": unit.returncode, "passed": unit.returncode == 0, "output_file": "unit_test_output.txt"},
        "scenario_checks": scenario_rows,
        "passed_checks": passed_count,
        "total_checks": len(scenario_rows),
        "all_checks_passed": unit.returncode == 0 and passed_count == len(scenario_rows),
        "boundary": boundary_rows,
        "crossing": {"history_retained": map_h1, "history_cleared": map_c1, "dropout_frame": map_drop, "recovery_frame": map_recovery},
        "association_stats": stats,
        "source_order": {"association_offset": association_position, "truth_label_offset": truth_position, "passed": source_order_ok},
        "figures": [path.name for path in sorted(FIGURES.glob("*.png"))],
    }
    save_json(OUTPUT / "result.json", result)
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({
        "test_case": outcome["test_case"],
        "passed": outcome["all_checks_passed"],
        "checks": f"{outcome['passed_checks']}/{outcome['total_checks']}",
        "association": f"{outcome['association_stats']['correct']}/{outcome['association_stats']['total']}",
        "output": str(OUTPUT),
    }, ensure_ascii=False, indent=2))
