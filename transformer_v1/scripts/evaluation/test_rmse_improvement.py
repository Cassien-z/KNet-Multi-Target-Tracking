"""Test RMSE reduction against an IMM-JPDA baseline on one batch-test scope.

The completed batch report supplies the Transformer RMSE and locks dataset,
scenario count, sequence length, and warm-up. The baseline evaluates every
target in the same leading raw MAT scenarios using one IMM per target and
competition-normalized JPDA marginal weights over shared measurements.

Example:
    python scripts/evaluation/test_rmse_improvement.py \
        --batch-report artifacts/reports/batch_tests/<job_id>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.acceptance_common import (
    evidence_context,
    evidence_path,
    load_batch_report,
    selected_files,
    write_json_atomic,
)
THRESHOLD_PCT = 15.0


def body_polar_to_global_cartesian(
    azimuth: float,
    elevation: float,
    distance: float,
    position: np.ndarray,
    euler_rpy: np.ndarray,
) -> np.ndarray:
    """Convert a leader-radar body-frame measurement to ECEF."""
    x, y, z = position
    semi_major = 6378137.0
    eccentricity_squared = 0.00669437999014
    horizontal = np.hypot(x, y)
    longitude = np.arctan2(y, x)
    latitude = np.arctan2(z, horizontal * (1.0 - eccentricity_squared))
    for _ in range(3):
        radius = semi_major / np.sqrt(
            1.0 - eccentricity_squared * np.sin(latitude) ** 2
        )
        latitude = np.arctan2(
            z + eccentricity_squared * radius * np.sin(latitude), horizontal
        )
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    sin_lon, cos_lon = np.sin(longitude), np.cos(longitude)
    enu_to_ecef = np.array(
        [
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0.0, cos_lat, sin_lat],
        ]
    )
    pitch, yaw = float(euler_rpy[1]), float(euler_rpy[2])
    forward_enu = np.array(
        [
            np.sin(yaw) * np.cos(pitch),
            np.cos(yaw) * np.cos(pitch),
            np.sin(pitch),
        ]
    )
    forward = enu_to_ecef @ forward_enu
    forward /= np.linalg.norm(forward) + 1e-12
    local_up = position / (np.linalg.norm(position) + 1e-12)
    right = np.cross(local_up, forward)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(forward, right)
    up /= np.linalg.norm(up) + 1e-12
    body_to_ecef = np.column_stack((forward, right, up))
    body_vector = np.array(
        [
            distance * np.cos(elevation) * np.cos(azimuth),
            distance * np.cos(elevation) * np.sin(azimuth),
            distance * np.sin(elevation),
        ]
    )
    return position + body_to_ecef @ body_vector


def _block_diagonal_3(matrix: np.ndarray) -> np.ndarray:
    result = np.zeros((9, 9), dtype=float)
    for start in (0, 3, 6):
        result[start : start + 3, start : start + 3] = matrix
    return result


def _white_noise_q(dt: float, variance: float) -> np.ndarray:
    base = np.array(
        [
            [0.25 * dt**4, 0.5 * dt**3, 0.5 * dt**2],
            [0.5 * dt**3, dt**2, dt],
            [0.5 * dt**2, dt, 1.0],
        ],
        dtype=float,
    )
    return _block_diagonal_3(base * variance)


class NumpyImmTracker:
    """Small NumPy IMM with CV, CA, left-CT, and right-CT models."""

    def __init__(self, initial_state: np.ndarray, dt: float = 0.1) -> None:
        cv = np.array([[1, dt, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
        ca = np.array(
            [[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]], dtype=float
        )

        def ct(omega: float) -> np.ndarray:
            transition = np.zeros((9, 9), dtype=float)
            sine, cosine = np.sin(omega * dt), np.cos(omega * dt)
            transition[0, 0] = 1
            transition[0, 1] = sine / omega
            transition[0, 4] = -(1 - cosine) / omega
            transition[1, 1] = cosine
            transition[1, 4] = -sine
            transition[3, 3] = 1
            transition[3, 1] = (1 - cosine) / omega
            transition[3, 4] = sine / omega
            transition[4, 1] = sine
            transition[4, 4] = cosine
            transition[6, 6] = 1
            transition[6, 7] = dt
            transition[7, 7] = 1
            return transition

        self.h = np.zeros((3, 9), dtype=float)
        self.h[0, 0], self.h[1, 3], self.h[2, 6] = 1.0, 1.0, 1.0
        self.r = np.diag([66.71**2] * 3)
        transitions = [_block_diagonal_3(cv), _block_diagonal_3(ca), ct(0.06), ct(-0.06)]
        noises = [_white_noise_q(dt, 52.21), _white_noise_q(dt, 33.96), _white_noise_q(dt, 40.0), _white_noise_q(dt, 40.0)]
        self.models = [
            {"x": initial_state.copy(), "p": np.eye(9) * 1000.0, "f": transition, "q": noise}
            for transition, noise in zip(transitions, noises)
        ]
        self.mu = np.array([0.85, 0.05, 0.05, 0.05], dtype=float)
        self.transition = np.array(
            [
                [0.94, 0.02, 0.02, 0.02],
                [0.05, 0.95, 0.00, 0.00],
                [0.05, 0.00, 0.95, 0.00],
                [0.05, 0.00, 0.00, 0.95],
            ],
            dtype=float,
        )
        self.x = initial_state.copy()
        self.p = np.eye(9) * 1000.0
        self._combine()

    def _combine(self) -> None:
        self.x = sum(weight * model["x"] for weight, model in zip(self.mu, self.models))
        covariance = np.zeros((9, 9), dtype=float)
        for weight, model in zip(self.mu, self.models):
            delta = model["x"] - self.x
            covariance += weight * (model["p"] + delta @ delta.T)
        self.p = (covariance + covariance.T) / 2.0

    def predict(self) -> None:
        normalization = self.mu @ self.transition
        mixed_models = []
        for destination in range(len(self.models)):
            weights = self.mu * self.transition[:, destination]
            weights /= max(normalization[destination], 1e-15)
            mixed_x = sum(weight * model["x"] for weight, model in zip(weights, self.models))
            mixed_p = np.zeros((9, 9), dtype=float)
            for weight, model in zip(weights, self.models):
                delta = model["x"] - mixed_x
                mixed_p += weight * (model["p"] + delta @ delta.T)
            mixed_models.append((mixed_x, mixed_p))
        self.mu = normalization / np.sum(normalization)
        for model, (mixed_x, mixed_p) in zip(self.models, mixed_models):
            transition, noise = model["f"], model["q"]
            model["x"] = transition @ mixed_x
            model["p"] = transition @ mixed_p @ transition.T + noise
        self._combine()

    def update(self, measurement: np.ndarray) -> None:
        log_likelihoods = []
        identity = np.eye(9)
        for model in self.models:
            innovation = measurement - self.h @ model["x"]
            covariance = self.h @ model["p"] @ self.h.T + self.r
            inverse = np.linalg.pinv(covariance)
            gain = model["p"] @ self.h.T @ inverse
            model["x"] = model["x"] + gain @ innovation
            residual = identity - gain @ self.h
            model["p"] = residual @ model["p"] @ residual.T + gain @ self.r @ gain.T
            sign, log_determinant = np.linalg.slogdet(2.0 * np.pi * covariance)
            distance_squared = float((innovation.T @ inverse @ innovation).item())
            log_likelihoods.append(
                -0.5 * (distance_squared + log_determinant) if sign > 0 else -1e12
            )
        posterior_log = np.log(np.maximum(self.mu, 1e-300)) + np.asarray(log_likelihoods)
        posterior_log -= np.max(posterior_log)
        posterior = np.exp(posterior_log)
        self.mu = posterior / np.sum(posterior)
        self._combine()


def _rows(values: Any, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.empty((0, width), dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] != width and array.shape[0] == width:
        array = array.T
    return array[:, :width]


def _mask(values: Any, count: int) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1) > 0.5
    if result.size < count:
        result = np.pad(result, (0, count - result.size), constant_values=False)
    return result[:count]


def load_multitarget_scenario(path: Path) -> tuple[list[list[np.ndarray]], np.ndarray]:
    data = scipy.io.loadmat(path, simplify_cells=True)
    frames = list(np.asarray(data["Data_Input"], dtype=object).reshape(-1))
    labels = np.asarray(data["Data_Label"], dtype=float)
    if labels.ndim == 2:
        labels = labels[:, None, :]
    if labels.shape[0] != len(frames) and labels.shape[1] == len(frames):
        labels = np.swapaxes(labels, 0, 1)
    labels = labels[: len(frames)]
    measurements_by_frame: list[list[np.ndarray]] = []

    for frame in frames:
        uavs = list(np.asarray(frame["UAV"], dtype=object).reshape(-1))
        measurements: list[np.ndarray] = []
        if uavs:
            leader = uavs[0]
            radar = leader.get("Radar", {})
            radar_rows = _rows(radar.get("Meas", []), 3)
            radar_valid = _mask(radar.get("Mask", []), len(radar_rows))
            pose = leader["Pose"]
            for index in np.flatnonzero(radar_valid):
                azimuth, elevation, distance = radar_rows[index]
                point = body_polar_to_global_cartesian(
                    azimuth,
                    elevation,
                    distance,
                    np.asarray(pose["PositionWorld"], dtype=float),
                    np.asarray(pose["EulerRPY"], dtype=float),
                )
                measurements.append(point.reshape(3))

        for wingman in uavs[1:]:
            fast_link = wingman.get("FastLink", {})
            link_rows = _rows(fast_link.get("Meas", []), 3)
            link_valid = _mask(fast_link.get("IsUpdate", []), len(link_rows))
            for index in np.flatnonzero(link_valid):
                point = link_rows[index]
                if np.all(np.isfinite(point)):
                    measurements.append(point.copy())
        measurements_by_frame.append(measurements)
    return measurements_by_frame, labels


def _initial_state(label: np.ndarray) -> np.ndarray:
    return np.asarray(
        [label[0], label[1], 0.0, label[2], label[3], 0.0, label[4], label[5], 0.0],
        dtype=float,
    ).reshape(9, 1)


def evaluate_file(
    path: Path,
    warmup: int,
    sequence_length: int,
    gate_squared: float,
    missed_weight: float,
) -> dict[str, Any]:
    measurements_by_frame, labels = load_multitarget_scenario(path)
    frame_count = min(sequence_length, len(measurements_by_frame), labels.shape[0])
    target_count = labels.shape[1]
    trackers = [
        NumpyImmTracker(_initial_state(labels[0, target]), dt=0.1)
        for target in range(target_count)
    ]

    errors: list[float] = []
    updated_tracks = 0
    for frame_index in range(frame_count):
        for tracker in trackers:
            tracker.predict()
        measurements = np.asarray(measurements_by_frame[frame_index], dtype=float).reshape(-1, 3)

        if len(measurements):
            likelihood = np.zeros((target_count, len(measurements)), dtype=float)
            predictions: list[np.ndarray] = []
            for target, tracker in enumerate(trackers):
                observation = tracker.h
                covariance = observation @ tracker.p @ observation.T + tracker.r
                inverse = np.linalg.pinv(covariance)
                predicted = (observation @ tracker.x).reshape(3)
                predictions.append(predicted)
                innovations = measurements - predicted
                distance_squared = np.einsum("ni,ij,nj->n", innovations, inverse, innovations)
                valid = np.isfinite(distance_squared) & (distance_squared <= gate_squared)
                likelihood[target, valid] = np.exp(-0.5 * distance_squared[valid])

            competition = likelihood.sum(axis=0) + 1e-12
            ownership = likelihood / competition[None, :]
            for target, tracker in enumerate(trackers):
                weights = ownership[target]
                weight_sum = float(np.sum(weights))
                if weight_sum <= 1e-9:
                    continue
                beta = weights / (weight_sum + missed_weight)
                innovation = np.sum(beta[:, None] * (measurements - predictions[target]), axis=0)
                tracker.update((predictions[target] + innovation).reshape(3, 1))
                updated_tracks += 1

        if frame_index >= warmup:
            for target, tracker in enumerate(trackers):
                estimate = tracker.x[[0, 3, 6], 0]
                truth = labels[frame_index, target, [0, 2, 4]]
                errors.append(float(np.linalg.norm(estimate - truth)))

    if not errors:
        raise RuntimeError(f"No baseline errors remained after warm-up for {path.name}")
    values = np.asarray(errors, dtype=np.float64)
    return {
        "file": path.name,
        "targets": int(target_count),
        "evaluated_points": int(values.size),
        "updated_track_frames": int(updated_tracks),
        "rmse_m": float(np.sqrt(np.mean(np.square(values)))),
        "mae_m": float(np.mean(values)),
        "p95_m": float(np.percentile(values, 95)),
    }


def evaluate(
    report: dict[str, Any], gate_squared: float, missed_weight: float
) -> dict[str, Any]:
    raw_files = selected_files(report, ".mat")
    config = report["config"]
    file_results = []
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    evaluated_points = 0
    for index, path in enumerate(raw_files, start=1):
        result = evaluate_file(
            path,
            warmup=int(config["warmup_frames"]),
            sequence_length=int(config["sequence_length"]),
            gate_squared=gate_squared,
            missed_weight=missed_weight,
        )
        file_results.append(result)
        count = int(result["evaluated_points"])
        squared_error_sum += float(result["rmse_m"]) ** 2 * count
        absolute_error_sum += float(result["mae_m"]) * count
        evaluated_points += count
        if index % 25 == 0 or index == len(raw_files):
            print(f"IMM-JPDA baseline {index}/{len(raw_files)}", flush=True)

    imm_rmse = float(np.sqrt(squared_error_sum / evaluated_points))
    transformer_rmse = float(report["result"]["metrics"]["position_rmse_m"])
    improvement = (imm_rmse - transformer_rmse) / imm_rmse * 100.0
    return {
        "schema_version": "1.0",
        "metric_id": "rmse_improvement_pct",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **evidence_context(report),
        "value": improvement,
        "unit": "%",
        "operator": ">=",
        "threshold": THRESHOLD_PCT,
        "passed": improvement >= THRESHOLD_PCT,
        "valid_for_assessment": True,
        "method": "NumPy IMM-JPDA competition-normalized marginal association, truth initialization",
        "settings": {
            "gate_squared": gate_squared,
            "missed_detection_weight": missed_weight,
            "measurement_sources": ["leader Radar", "wingman FastLink"],
        },
        "transformer_rmse_m": transformer_rmse,
        "imm_jpda_rmse_m": imm_rmse,
        "improvement_pct": improvement,
        "evaluated_points": evaluated_points,
        "per_file": file_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-report", required=True, help="Completed batch-test JSON report")
    parser.add_argument("--output", help="Output JSON; defaults to the job acceptance evidence directory")
    parser.add_argument("--gate-squared", type=float, default=25.0, help="Squared Mahalanobis validation gate")
    parser.add_argument("--missed-weight", type=float, default=0.15, help="Missed-detection weight in JPDA normalization")
    args = parser.parse_args()
    if args.gate_squared <= 0 or args.missed_weight <= 0:
        parser.error("gate-squared and missed-weight must be positive")

    report = load_batch_report(args.batch_report)
    result = evaluate(report, args.gate_squared, args.missed_weight)
    output = Path(args.output) if args.output else evidence_path(report, "rmse_improvement_pct")
    saved = write_json_atomic(output, result)
    console_result = {key: value for key, value in result.items() if key != "per_file"}
    console_result["per_file_records"] = len(result["per_file"])
    print(json.dumps(console_result, ensure_ascii=False, indent=2))
    print(f"saved evidence: {saved}")


if __name__ == "__main__":
    main()
