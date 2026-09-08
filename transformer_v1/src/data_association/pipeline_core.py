import pickle
import zlib

import numpy as np
import scipy.io as sio
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment

from src.data_association.track_association import TrackAssociation
from src.utils.ecef_utils import get_body_frame_axes, reconstruct_ecef_from_body_angles
from src.wingman_fusion.local_fusion import WingmanFusionEngine


SENSOR_DIMS = {"Radar": 3, "EO": 2, "ESM": 1}


def time_align_interp(t_meas, measurements, t_grid, active_mask, max_interp_gap=0.45):
    measurements = np.asarray(measurements, dtype=float)
    if measurements.ndim == 1:
        measurements = measurements.reshape(-1, 1)
    aligned = np.full((len(t_grid), measurements.shape[1]), np.nan)
    valid = np.zeros(len(t_grid), dtype=bool)
    if len(t_meas) < 2:
        return aligned, valid
    order = np.argsort(t_meas)
    t_meas = np.asarray(t_meas, dtype=float)[order]
    measurements = measurements[order]
    t_meas, unique = np.unique(t_meas, return_index=True)
    measurements = measurements[unique]
    if len(t_meas) < 2:
        return aligned, valid
    inside = (t_grid >= t_meas.min()) & (t_grid <= t_meas.max())
    nearest = np.min(np.abs(t_meas[None, :] - t_grid[:, None]), axis=1)
    valid = active_mask & inside & (nearest <= max_interp_gap)
    for dim in range(measurements.shape[1]):
        values = measurements[:, dim]
        good = np.isfinite(t_meas) & np.isfinite(values)
        if np.sum(good) < 2:
            continue
        interpolator = (
            PchipInterpolator(t_meas[good], values[good], extrapolate=False)
            if np.sum(good) >= 4
            else interp1d(t_meas[good], values[good], kind="linear", bounds_error=False, fill_value=np.nan)
        )
        aligned[valid, dim] = interpolator(t_grid[valid])
    valid &= np.all(np.isfinite(aligned), axis=1)
    aligned[~valid] = np.nan
    return aligned, valid


def _sensor_frame(uav, sensor_name):
    dim = SENSOR_DIMS[sensor_name]
    sensor = uav.get(sensor_name, {})
    mask = np.asarray(sensor.get("Mask", []), dtype=float).reshape(-1) > 0.5
    measurements = np.asarray(sensor.get("Meas", []), dtype=float)
    if measurements.size == 0 or mask.size == 0:
        return np.empty((0, dim)), np.zeros(0, dtype=bool)
    if measurements.ndim == 0:
        measurements = measurements.reshape(1, 1)
    elif measurements.ndim == 1:
        measurements = measurements.reshape(-1, 1) if dim == 1 else measurements.reshape(1, -1)
    if measurements.shape[0] != mask.size and measurements.shape[1] == mask.size:
        measurements = measurements.T
    count = min(measurements.shape[0], mask.size)
    result = np.full((count, dim), np.nan)
    width = min(dim, measurements.shape[1])
    result[:, :width] = measurements[:count, :width]
    return result, mask[:count] & np.all(np.isfinite(result), axis=1)


def _radar_world(measurements, valid, position, velocity):
    world = np.full((len(measurements), 3), np.nan)
    for local_id in np.flatnonzero(valid):
        azimuth, elevation, distance = measurements[local_id]
        world[local_id] = reconstruct_ecef_from_body_angles(azimuth, elevation, distance, position, velocity)
    return world


def _align_stream(raw, raw_valid, t_grid):
    valid_ids = np.flatnonzero(raw_valid)
    if len(valid_ids) < 2:
        aligned = np.full_like(raw, np.nan, dtype=float)
        aligned[raw_valid] = raw[raw_valid]
        return aligned, raw_valid.copy()
    active = np.zeros(len(t_grid), dtype=bool)
    active[valid_ids[0]:valid_ids[-1] + 1] = True
    return time_align_interp(t_grid[raw_valid], raw[raw_valid], t_grid, active, max_interp_gap=0.45)


def _remove_radar_outliers(aligned, valid):
    # Raw radar order is [azimuth, elevation, range].
    for dim, threshold in enumerate((0.05, 0.05, 100.0)):
        good = valid & np.isfinite(aligned[:, dim])
        if np.sum(good) < 3:
            continue
        filled = np.interp(np.arange(len(aligned)), np.flatnonzero(good), aligned[good, dim])
        baseline = median_filter(filled, size=5, mode="nearest")
        outliers = good & (np.abs(aligned[:, dim] - baseline) > threshold)
        valid[outliers] = False
    aligned[~valid] = np.nan
    return aligned, valid


def _label_tracks(registered_tracks, target_positions):
    costs = np.full((len(registered_tracks), target_positions.shape[1]), np.inf)
    for track_id, track in enumerate(registered_tracks):
        radar_positions = np.stack([entry["Meas_ECEF"] for entry in track["Radar"]], axis=1)
        estimate = np.full((len(radar_positions), 3), np.nan)
        for frame_id in range(len(radar_positions)):
            points = radar_positions[frame_id]
            points = points[np.all(np.isfinite(points), axis=1)]
            if len(points):
                estimate[frame_id] = np.median(points, axis=0)
        usable = np.all(np.isfinite(estimate), axis=1)
        for truth_id in range(target_positions.shape[1]):
            if np.any(usable):
                error = estimate[usable] - target_positions[usable, truth_id]
                costs[track_id, truth_id] = np.sqrt(np.mean(np.sum(error ** 2, axis=1)))
    mapping = {}
    finite_rows = np.any(np.isfinite(costs), axis=1)
    finite_cols = np.any(np.isfinite(costs), axis=0)
    if np.any(finite_rows) and np.any(finite_cols):
        row_ids = np.flatnonzero(finite_rows)
        col_ids = np.flatnonzero(finite_cols)
        reduced = costs[np.ix_(row_ids, col_ids)].copy()
        reduced[~np.isfinite(reduced)] = 1e12
        rows, cols = linear_sum_assignment(reduced)
        mapping = {int(row_ids[row]): int(col_ids[col]) for row, col in zip(rows, cols)}
    return mapping, costs


def _truth_segments(truth):
    lookup = {}
    for sensor, key in (("Radar", "RadarSeg"), ("EO", "EOSeg"), ("ESM", "ESMSeg")):
        rows = np.asarray(truth.get(key, []), dtype=float)
        if rows.size == 0:
            continue
        for row in np.atleast_2d(rows):
            lookup.setdefault((sensor, int(row[4]) - 1, int(row[5]) - 1), []).append(
                (int(row[1]) - 1, float(row[2]), float(row[3]))
            )
    return lookup


def _association_stats(events, label_mapping, truth, label_costs):
    lookup = _truth_segments(truth)
    per_sensor = {sensor: {"correct": 0, "total": 0} for sensor in SENSOR_DIMS}
    for sensor, uav_id, local_id, track_id, timestamp in events:
        expected_truth = label_mapping.get(track_id)
        if expected_truth is None:
            continue
        actual_truth = next(
            (truth_id for truth_id, start, end in lookup.get((sensor, uav_id, local_id), []) if start <= timestamp <= end),
            None,
        )
        if actual_truth is None:
            continue
        per_sensor[sensor]["total"] += 1
        per_sensor[sensor]["correct"] += int(actual_truth == expected_truth)
    correct = sum(item["correct"] for item in per_sensor.values())
    total = sum(item["total"] for item in per_sensor.values())
    for item in per_sensor.values():
        item["accuracy"] = item["correct"] / item["total"] if item["total"] else None
    return {
        "method": "measurement_only_hungarian_with_temporal_memory",
        "truth_used_for_observation_grouping": False,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "per_sensor": per_sensor,
        "track_to_truth": {int(k): int(v) for k, v in label_mapping.items()},
        "track_label_rmse_m": {
            int(track_id): float(label_costs[track_id, truth_id])
            for track_id, truth_id in label_mapping.items()
        },
    }


def process_file(mat_file_path, out_file_path, verbose=True):
    if verbose:
        print(f"Processing {mat_file_path} ...")
    mat = sio.loadmat(mat_file_path, simplify_cells=True)
    frames = list(mat["Data_Input"])
    truth = mat.get("truth", {})
    t_grid = np.asarray(truth.get("t_grid", [frame["GlobalTime"] for frame in frames]), dtype=float).reshape(-1)
    frame_count = len(t_grid)
    num_uavs = max(len(frame["UAV"]) for frame in frames)

    uav_positions = np.zeros((frame_count, num_uavs, 3))
    for frame_id, frame in enumerate(frames):
        for uav_id, uav in enumerate(frame["UAV"]):
            uav_positions[frame_id, uav_id] = uav["Pose"]["PositionWorld"]
    uav_velocities = np.gradient(uav_positions, t_grid, axis=0)
    uav_rotmats = np.zeros((frame_count, num_uavs, 3, 3))
    for frame_id in range(frame_count):
        for uav_id in range(num_uavs):
            # Preserve the body-frame convention used to generate the sensors
            # and to train V5: x follows ECEF velocity, y is global-Z cross x.
            forward, right, up = get_body_frame_axes(
                uav_velocities[frame_id, uav_id], uav_positions[frame_id, uav_id]
            )
            uav_rotmats[frame_id, uav_id] = np.column_stack((forward, right, up))

    track_count = max(len(_sensor_frame(frame["UAV"][0], "Radar")[0]) for frame in frames)
    if track_count == 0:
        raise ValueError("No reference-platform radar tracks were found")
    streams = {
        sensor: [[{"Meas": np.full((frame_count, dim), np.nan), "valid": np.zeros(frame_count, dtype=bool)} for _ in range(num_uavs)] for _ in range(track_count)]
        for sensor, dim in SENSOR_DIMS.items()
    }
    events = []
    associator = TrackAssociation()
    for frame_id, frame in enumerate(frames):
        sensor_data = [{sensor: _sensor_frame(uav, sensor) for sensor in SENSOR_DIMS} for uav in frame["UAV"]]
        reference_radar, reference_valid = sensor_data[0]["Radar"]
        reference_world = _radar_world(reference_radar, reference_valid, uav_positions[frame_id, 0], uav_velocities[frame_id, 0])
        for uav_id, data in enumerate(sensor_data):
            radar, radar_valid = data["Radar"]
            eo, eo_valid = data["EO"]
            esm, esm_valid = data["ESM"]
            eo_map, esm_map = associator.associate_same_platform(radar, eo, esm, radar_valid, eo_valid, esm_valid, uav_id)
            if uav_id == 0:
                radar_map = {int(local_id): int(local_id) for local_id in np.flatnonzero(radar_valid) if local_id < track_count}
            else:
                platform_world = _radar_world(radar, radar_valid, uav_positions[frame_id, uav_id], uav_velocities[frame_id, uav_id])
                radar_map = associator.associate_radar_platforms(reference_world, platform_world, reference_valid, radar_valid, uav_id)
                for local_id in np.flatnonzero(radar_valid):
                    previous = associator.previous_radar_mapping(uav_id, int(local_id))
                    if previous is not None:
                        radar_map.setdefault(int(local_id), int(previous))
            for local_radar_id, track_id in radar_map.items():
                if track_id >= track_count or not radar_valid[local_radar_id]:
                    continue
                streams["Radar"][track_id][uav_id]["Meas"][frame_id] = radar[local_radar_id]
                streams["Radar"][track_id][uav_id]["valid"][frame_id] = True
                events.append(("Radar", uav_id, local_radar_id, track_id, t_grid[frame_id]))
                eo_id = eo_map.get(local_radar_id)
                if eo_id is not None and eo_id < len(eo_valid) and eo_valid[eo_id]:
                    streams["EO"][track_id][uav_id]["Meas"][frame_id] = eo[eo_id]
                    streams["EO"][track_id][uav_id]["valid"][frame_id] = True
                    events.append(("EO", uav_id, eo_id, track_id, t_grid[frame_id]))
                esm_id = esm_map.get(local_radar_id)
                if esm_id is not None and esm_id < len(esm_valid) and esm_valid[esm_id]:
                    streams["ESM"][track_id][uav_id]["Meas"][frame_id] = esm[esm_id]
                    streams["ESM"][track_id][uav_id]["valid"][frame_id] = True
                    events.append(("ESM", uav_id, esm_id, track_id, t_grid[frame_id]))

    registered_tracks = []
    for track_id in range(track_count):
        track = {
            "target_pos": np.zeros((frame_count, 3)), "target_vel": np.zeros((frame_count, 3)),
            "Radar": [], "EO": [], "ESM": [],
            "FastLink": [{"Meas_ECEF": np.full((frame_count, 3), np.nan), "valid": np.zeros(frame_count, dtype=bool)} for _ in range(num_uavs)],
        }
        for sensor in SENSOR_DIMS:
            for uav_id in range(num_uavs):
                raw = streams[sensor][track_id][uav_id]["Meas"]
                raw_valid = streams[sensor][track_id][uav_id]["valid"]
                aligned, valid = _align_stream(raw, raw_valid, t_grid)
                if sensor == "Radar":
                    aligned, valid = _remove_radar_outliers(aligned, valid)
                    stored = aligned[:, [2, 0, 1]]
                else:
                    stored = aligned
                ecef = np.full((frame_count, 3), np.nan)
                for frame_id in np.flatnonzero(valid):
                    if sensor == "Radar":
                        azimuth, elevation, distance = aligned[frame_id]
                    elif sensor == "EO":
                        azimuth, elevation, distance = aligned[frame_id, 0], aligned[frame_id, 1], 10000.0
                    else:
                        azimuth, elevation, distance = aligned[frame_id, 0], 0.0, 10000.0
                    ecef[frame_id] = reconstruct_ecef_from_body_angles(azimuth, elevation, distance, uav_positions[frame_id, uav_id], uav_velocities[frame_id, uav_id])
                track[sensor].append({"Meas_ECEF": ecef, "Meas": stored, "valid": valid})
        fusion = WingmanFusionEngine()
        delay_steps = max(1, round(0.1 / float(t_grid[1] - t_grid[0]))) if frame_count > 1 else 1
        rng = np.random.default_rng(zlib.crc32(f"{mat_file_path}:{track_id}".encode("utf-8")))
        for uav_id in range(1, num_uavs):
            fused, fused_valid = fusion.fuse(
                track["Radar"][uav_id]["Meas"], track["EO"][uav_id]["Meas"], track["ESM"][uav_id]["Meas"],
                uav_positions[:, uav_id], uav_velocities[:, uav_id], track["Radar"][uav_id]["valid"],
                track["EO"][uav_id]["valid"], track["ESM"][uav_id]["valid"],
            )
            if delay_steps < frame_count:
                source, target = slice(0, frame_count - delay_steps), slice(delay_steps, frame_count)
                delayed, delayed_valid = fused[source].copy(), fused_valid[source].copy()
                noise_rows = delayed_valid & np.all(np.isfinite(delayed), axis=1)
                delayed[noise_rows] += rng.normal(0.0, 20.0, size=(np.sum(noise_rows), 3))
                track["FastLink"][uav_id]["Meas_ECEF"][target] = delayed
                track["FastLink"][uav_id]["valid"][target] = delayed_valid
        registered_tracks.append(track)

    labels = np.asarray(mat.get("Data_Label"), dtype=float)
    if labels.ndim == 2:
        labels = labels[:, None, :]
    target_positions, target_velocities = labels[:, :, [0, 2, 4]], labels[:, :, [1, 3, 5]]
    label_mapping, label_costs = _label_tracks(registered_tracks, target_positions)
    for track_id, track in enumerate(registered_tracks):
        truth_id = label_mapping.get(track_id)
        track["association_truth_id"] = -1 if truth_id is None else int(truth_id)
        if truth_id is not None:
            track["target_pos"], track["target_vel"] = target_positions[:, truth_id], target_velocities[:, truth_id]
    stats = _association_stats(events, label_mapping, truth, label_costs)
    output = {"RegisteredTrack": registered_tracks, "t_grid": t_grid, "UAV_Pos": uav_positions, "UAV_Rotmat": uav_rotmats, "AssociationStats": stats}
    with open(out_file_path, "wb") as handle:
        pickle.dump(output, handle)
    if verbose:
        print(f"Saved to {out_file_path} (association accuracy={stats['accuracy']})")
