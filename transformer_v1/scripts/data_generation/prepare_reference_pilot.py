"""Prepare a small reference-data pilot without changing legacy datasets.

Outputs standard RegisteredTrack PKL files. Four radar source slots and two
bearing-only channels share the actual ownship pose; they are not extra UAVs.
Calibration and splitting are fitted before validation data is transformed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.ecef_utils import ecef_to_geodetic, ecef_to_enu_matrix, get_body_frame_axes

SOURCE_NAMES = ['本机飞行数据', '目标飞行数据', '全维1', '全维2', '全维3', '全维4', '缺维1', '缺维2']
SENSOR_NAMES = SOURCE_NAMES[2:]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def wrap_degrees(values):
    return (np.asarray(values) + 180.0) % 360.0 - 180.0


def folded_pitch(values):
    return np.degrees(np.arcsin(np.sin(np.radians(values))))


def reference_direction(angles):
    """[pitch, bearing] in degrees -> unit ENU, including pitch branches."""
    pitch, bearing = np.radians(np.asarray(angles)).T
    return np.column_stack((np.abs(np.cos(pitch)) * np.sin(bearing),
                            np.abs(np.cos(pitch)) * np.cos(bearing), -np.sin(pitch)))


def transform_observations(raw, own_positions, enu_rotations, body_rotations,
                           calibration=None, full=True):
    """Transform measured directions only; never infer a missing range."""
    raw = np.asarray(raw, dtype=np.float64)
    valid = (raw[:, 0] > 0) if full else np.any(raw[:, 1:3] != 0.0, axis=1)
    valid &= np.all(np.isfinite(raw[:, :3]), axis=1)
    calibration = calibration or {'range_bias_m': 0., 'bearing_bias_deg': 0., 'pitch_bias_deg': 0.}
    angles = np.column_stack((folded_pitch(raw[:, 1]) - calibration['pitch_bias_deg'],
                              wrap_degrees(raw[:, 2] - calibration['bearing_bias_deg'])))
    enu = reference_direction(angles)
    world_direction = np.einsum('tji,tj->ti', enu_rotations, enu)
    body_direction = np.einsum('tji,tj->ti', body_rotations, world_direction)
    az = np.arctan2(body_direction[:, 1], body_direction[:, 0])
    el = np.arcsin(np.clip(body_direction[:, 2], -1., 1.))
    if full:
        distance = raw[:, 0] - calibration['range_bias_m']
        valid &= distance > 0.0
        meas = np.column_stack((distance, az, el))  # Registered PKL order
        world = own_positions + distance[:, None] * world_direction
    else:
        meas = np.column_stack((az, el))
        world = np.zeros((len(raw), 3))  # no Cartesian observation exists
    meas[~valid] = 0.0
    world[~valid] = 0.0
    return {'Meas': meas, 'Meas_ECEF': world, 'valid': valid}


def fit_calibration(raw, own, target, rotations, fit_mask, full=True):
    """Training-only mean residuals. fit_mask excludes every validation target."""
    fit_mask = np.asarray(fit_mask, bool)
    valid = fit_mask & ((raw[:, :, 0] > 0) if full else np.any(raw[:, :, 1:3] != 0., axis=2))
    if not np.any(valid):
        raise ValueError('No training observations available for calibration')
    relative = target - own[:, None, :]
    ranges = np.linalg.norm(relative, axis=2)
    enu = np.einsum('tij,tkj->tki', rotations, relative)
    bearing = np.degrees(np.arctan2(enu[:, :, 0], enu[:, :, 1]))
    pitch = -np.degrees(np.arcsin(np.clip(enu[:, :, 2] / np.maximum(ranges, 1.), -1., 1.)))
    return {'range_bias_m': float(np.mean((raw[:, :, 0] - ranges)[valid])) if full else 0.,
            'bearing_bias_deg': float(np.mean(wrap_degrees(raw[:, :, 2] - bearing)[valid])),
            'pitch_bias_deg': float(np.mean((folded_pitch(raw[:, :, 1]) - pitch)[valid])),
            'training_observation_count': int(valid.sum())}


def choose_tracks(targets, count, seed, seq_len):
    rng = np.random.default_rng(seed)
    eligible, rejected, seen = [], [], set()
    for idx in range(targets.shape[1]):
        track = targets[:, idx]
        active = np.linalg.norm(track, axis=1) > 1e6
        good = np.flatnonzero(active)
        if not len(good) or good[0] != 0 or not np.all(active[:good[-1]+1]):
            rejected.append({'target_id': idx+1, 'reason': 'noncontiguous_truth'}); continue
        length = len(good)
        if length < 2 * seq_len:
            rejected.append({'target_id': idx+1, 'reason': 'less_than_two_windows'}); continue
        track = track[:length]
        digest = hashlib.sha256(track.tobytes()).hexdigest()
        if digest in seen:
            rejected.append({'target_id': idx+1, 'reason': 'duplicate_truth'}); continue
        if np.max(np.linalg.norm(np.diff(track, axis=0), axis=1)) > 500.:
            rejected.append({'target_id': idx+1, 'reason': 'speed_over_500_mps'}); continue
        seen.add(digest); eligible.append((idx, length, digest))
    if len(eligible) < count:
        raise ValueError(f'Only {len(eligible)} eligible tracks for requested {count}')
    return eligible, rejected, rng



def measured_reference_origin(own, sources):
    """Initialize only when a real full-dimensional observation arrives.

    Prefix predictions before this frame are offline reconstructions and must
    not be counted as initialized tracking. No target truth is consulted.
    """
    for frame in range(len(own)):
        for name in SENSOR_NAMES[:4]:
            source = sources[name]
            if source['valid'][frame]:
                return (source['Meas_ECEF'][frame].astype(np.float32).astype(np.float64),
                        {'method':'first_actual_full_observation', 'frame':frame, 'sensor':name})
    return own[0].astype(np.float32).astype(np.float64), {'method':'ownship_fallback_no_full_observation','frame':len(own),'sensor':None}


def build_payload(target, velocity, own, rotations, sources, provenance):
    frames = len(target)
    def empty(width):
        return {'Meas': np.zeros((frames, width)), 'Meas_ECEF': np.zeros((frames, 3)),
                'valid': np.zeros(frames, dtype=bool)}
    track = {'target_pos': target.copy(), 'target_vel': velocity.copy(),
             'Radar': [empty(3) for _ in range(5)],
             'EO': [empty(2) for _ in range(5)],
             'ESM': [empty(1) for _ in range(5)],
             'FastLink': [{'Meas_ECEF': np.zeros((frames, 3)), 'valid': np.zeros(frames, bool)} for _ in range(5)]}
    for i, name in enumerate(SENSOR_NAMES[:4]):
        track['Radar'][i] = sources[name]
    for i, name in enumerate(SENSOR_NAMES[4:]):
        track['EO'][i] = sources[name]
    # All slots use one real ownship pose, and slot 4 has no active sensors.
    origin, initialization = measured_reference_origin(own, sources)
    return {'RegisteredTrack': [track], 't_grid': np.arange(frames, dtype=float),
            'UAV_Pos': np.repeat(own[:, None, :], 5, axis=1),
            'UAV_Rotmat': np.repeat(rotations[:, None, :, :], 5, axis=1),
            'ReferenceOrigin': origin, 'Initialization': initialization,
            'SourceMetadata': provenance}


def prepare(args):
    from python_calamine import CalamineWorkbook
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {output}')
    source = Path(args.source).resolve()
    manifest_sources = []
    def read(name):
        path = source / ('场景1_' + name + '.xlsx')
        workbook = CalamineWorkbook.from_path(str(path))
        matrix = np.array(workbook.get_sheet_by_index(0).to_python(), dtype=np.float64)
        manifest_sources.append({'path': str(path), 'sha256': sha256_file(path), 'shape': list(matrix.shape)})
        print(f'[source] {name} {matrix.shape}', flush=True)
        return matrix
    own = read(SOURCE_NAMES[0])
    targets = read(SOURCE_NAMES[1]).reshape(len(own), -1, 3)
    eligible, rejected, rng = choose_tracks(targets, args.count, args.seed, args.seq_len)
    distances = np.array([np.linalg.norm(targets[0, i] - own[0]) for i, _, _ in eligible])
    strata = np.array_split(np.argsort(distances), 4)
    picked = []
    group_sizes = [len(g) for g in np.array_split(np.arange(args.count), 4)]
    val_remaining = max(1, round(args.count * .25))
    for gi, (group, take) in enumerate(zip(strata, group_sizes)):
        selected = rng.choice(group, size=take, replace=False)
        val_count = min(take-1, int(np.ceil(val_remaining/(4-gi))))
        val_remaining -= val_count
        for j, item in enumerate(selected):
            idx, length, digest = eligible[int(item)]
            chunks = length // args.seq_len
            chunk_ids = np.unique(np.linspace(0, chunks-1, min(chunks, args.max_windows), dtype=int))
            picked.append({'target_index': idx, 'target_id': idx+1, 'valid_frames': length,
                           'truth_sha256': digest, 'range_stratum': gi,
                           'split': 'val' if j < val_count else 'train',
                           'window_starts': (chunk_ids*args.seq_len).tolist()})
    picked.sort(key=lambda item: item['target_id'])
    indices = [p['target_index'] for p in picked]
    truth = targets[:, indices, :].copy(); del targets
    enu_rot = np.array([ecef_to_enu_matrix(*ecef_to_geodetic(*p)[:2]) for p in own])
    own_velocity = np.gradient(own, 1., axis=0, edge_order=2)
    body_rot = np.array([np.column_stack(get_body_frame_axes(v, p)) for v, p in zip(own_velocity, own)])
    raw = {}
    for name in SENSOR_NAMES:
        matrix = read(name)
        raw[name] = matrix.reshape(len(own), -1, 6)[:, indices, :].copy()
        del matrix
    fit_mask = np.zeros(truth.shape[:2], dtype=bool)
    for i, item in enumerate(picked):
        if item['split'] == 'train':
            for start in item['window_starts']:
                fit_mask[start:start+args.seq_len, i] = True
    calibration = {name: fit_calibration(raw[name], own, truth, enu_rot, fit_mask, full=si<4)
                   for si, name in enumerate(SENSOR_NAMES)}
    train_ids = [p['target_id'] for p in picked if p['split']=='train']
    val_ids = [p['target_id'] for p in picked if p['split']=='val']
    assert set(train_ids).isdisjoint(val_ids)
    output.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'val_uncalibrated']:
        (output/split).mkdir(exist_ok=True)
    records, checks = [], {'max_raw_geometry_roundtrip_error_m': 0., 'finite_truth_velocities': True,
                           'truth_padding_exported': 0, 'max_velocity_mps': 0., 'calibration_validation_points': 0}
    for i, item in enumerate(picked):
        length = item['valid_frames']; target = truth[:length, i]
        # Differentiate before windowing, strictly within this valid lifetime.
        velocity = np.gradient(target, 1., axis=0, edge_order=2)
        checks['max_velocity_mps'] = max(checks['max_velocity_mps'], float(np.linalg.norm(velocity, axis=1).max()))
        for start in item['window_starts']:
            stop = start+args.seq_len; sl = slice(start, stop)
            for view in ([item['split'], 'val_uncalibrated'] if item['split']=='val' else ['train']):
                transformed = {}
                for si, name in enumerate(SENSOR_NAMES):
                    transformed[name] = transform_observations(raw[name][sl, i], own[sl], enu_rot[sl], body_rot[sl],
                        calibration=calibration[name] if view!='val_uncalibrated' else None, full=si<4)
                    if si<4 and view=='val_uncalibrated':
                        valid = transformed[name]['valid']
                        if valid.any():
                            error = np.linalg.norm(transformed[name]['Meas_ECEF'][valid]-raw[name][sl, i, 3:][valid], axis=1).max()
                            checks['max_raw_geometry_roundtrip_error_m'] = max(checks['max_raw_geometry_roundtrip_error_m'], float(error))
                metadata = {'source_target_id': item['target_id'], 'source_start_frame': start,
                    'source_stop_frame_exclusive': stop, 'source_dt_s': 1., 'split': item['split'],
                    'view': view, 'calibration_applied': view!='val_uncalibrated',
                    'physical_platform_count': 1, 'sensor_source_slots': ['全维1+缺维1','全维2+缺维2','全维3','全维4','empty'],
                    'reference_origin_method': 'first_actual_full_observation_no_target_truth',
                    'association_scope': 'source_column_correspondence; association accuracy not evaluated'}
                payload = build_payload(target[sl], velocity[sl], own[sl], body_rot[sl], transformed, metadata)
                if payload['Initialization']['frame'] >= args.seq_len:
                    raise ValueError('Window has no full-dimensional measurement for initialization')
                filename = f'ref_target_{item["target_id"]:04d}_frame_{start:04d}.pkl'
                with (output/view/filename).open('wb') as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                records.append({'file': str(Path(view)/filename), **metadata, 'initialization': payload['Initialization'],
                                'radar_valid_counts': [int(s['valid'].sum()) for s in payload['RegisteredTrack'][0]['Radar']],
                                'eo_valid_counts': [int(s['valid'].sum()) for s in payload['RegisteredTrack'][0]['EO']]})
                assert np.all(np.linalg.norm(payload['RegisteredTrack'][0]['target_pos'], axis=1)>1e6)
                assert np.isfinite(payload['RegisteredTrack'][0]['target_vel']).all()
    assert checks['max_raw_geometry_roundtrip_error_m'] < 2e-5
    manifest = {'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(), 'seed': args.seed,
        'selected_tracks': args.count, 'train_target_ids': train_ids, 'val_target_ids': val_ids,
        'seq_len': args.seq_len, 'dt_s': 1., 'max_windows_per_track': args.max_windows,
        'calibration_method': 'mean range, geographic bearing and folded pitch residuals on exported training frames only',
        'calibration_training_target_ids': train_ids, 'calibration': calibration, 'sources': manifest_sources,
        'tracks': picked, 'windows': records, 'quality_exclusions': rejected, 'checks': checks,
        'initialization': {'method': 'first actual full-dimensional source position, no target truth', 'score_from': 'arrival frame inclusive', 'prefix_scope': 'offline reconstruction excluded from training loss and primary validation metric'},
        'selection_note': 'random seed, stratified by initial range; valid length >= two windows; no exact duplicate truth; adjacent speed <= 500 m/s',
        'view_counts': {view: sum(r['view']==view for r in records) for view in ['train','val','val_uncalibrated']}}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('[prepared]', json.dumps({'output':str(output), 'train_tracks':len(train_ids), 'val_tracks':len(val_ids), 'views':manifest['view_counts'], 'checks':checks},ensure_ascii=False),flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(ROOT/'data/参考数据'))
    parser.add_argument('--output', default=str(ROOT/'data/reference_pilot_40_20261011_measured_anchor'))
    parser.add_argument('--count', type=int, default=40)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--max_windows', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261011)
    args = parser.parse_args()
    if not 20 <= args.count <= 50:
        parser.error('--count must be between 20 and 50')
    if args.seq_len < 3 or args.max_windows < 1:
        parser.error('--seq_len must be >= 3 and --max_windows >= 1')
    prepare(args)


if __name__=='__main__':
    main()
