import os
import pickle
from typing import Any, Dict

import h5py
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


def _to_fixed_array_64(value: Any, dim: int, fill_value: float = 0.0) -> np.ndarray:
    if value is None: return np.full((dim,), fill_value, dtype=np.float64)
    try: arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception: return np.full((dim,), fill_value, dtype=np.float64)
    if arr.size != dim: return np.full((dim,), fill_value, dtype=np.float64)
    return arr


def _to_fixed_array(value: Any, dim: int, fill_value: float = 0.0) -> np.ndarray:
    """
    Convert value to fixed-length float32 1D array.
    If invalid, return a filled vector.
    """
    if value is None:
        return np.full((dim,), fill_value, dtype=np.float32)

    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return np.full((dim,), fill_value, dtype=np.float32)

    if arr.size != dim:
        return np.full((dim,), fill_value, dtype=np.float32)
    return arr


def _to_scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(arr[0])
    except Exception:
        return float(default)


def _safe_get(d: Any, *keys, default=None):
    """
    Safe nested dict getter.
    Example:
        _safe_get(ud, "Pose", "PositionWorld", default=None)
    """
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_registered_multitarget_mat(mat_path: str, seq_len: int = 400, num_uavs: int = 5) -> list:
    """
    Parse MATLAB output 'RegisteredTrack_SpatialRegistration.mat'.
    Returns a list of dicts, where each dict represents a decoupled target
    formatted exactly like the output of SingerDataset.__getitem__.
    """
    if mat_path.endswith('.pkl'):
        try:
            with open(mat_path, 'rb') as f:
                data = pickle.load(f)
            return _parse_registered_pkl(data, seq_len, num_uavs, os.path.basename(mat_path))
        except Exception as e:
            print(f"Failed to load {mat_path} as pickle: {e}")
            return []

    try:
        f = h5py.File(mat_path, 'r')
    except Exception as e:
        print(f"Failed to load {mat_path} with h5py: {e}")
        return []

    if 'RegisteredTrack' not in f:
        print("RegisteredTrack not found in mat file.")
        return []

    # In h5py, MATLAB structs are groups, and arrays are datasets.
    registered_track = f['RegisteredTrack']
    if 'Target' not in registered_track:
        return []
        
    targets_ref = registered_track['Target']
    
    # In v7.3, struct arrays' fields are datasets of shape (1, N) containing references
    # Let's get the references for each field
    def get_field_refs(group, field):
        if field not in group: return []
        ds = group[field]
        if isinstance(ds, h5py.Dataset):
            return ds[()].flatten()
        return []

    pos_eval_refs = get_field_refs(targets_ref, 'pos_eval')
    valid_eval_refs = get_field_refs(targets_ref, 'valid_eval')
    sensor_refs = get_field_refs(targets_ref, 'Sensor')
    
    num_targets = max(len(pos_eval_refs), len(sensor_refs))
    samples = []

    for i in range(num_targets):
        try:
            sample = {
                "radar": np.zeros((seq_len, num_uavs, 3), dtype=np.float32),
                "eo": np.zeros((seq_len, num_uavs, 2), dtype=np.float32),
                "esm": np.zeros((seq_len, num_uavs, 2), dtype=np.float32),
                "uav_pos": np.zeros((seq_len, num_uavs, 3), dtype=np.float32),
                "rotmat_b2e": np.zeros((seq_len, num_uavs, 3, 3), dtype=np.float32),
                "radar_mask": np.zeros((seq_len, num_uavs), dtype=np.float32),
                "eo_mask": np.zeros((seq_len, num_uavs), dtype=np.float32),
                "esm_mask": np.zeros((seq_len, num_uavs), dtype=np.float32),
                "label": np.zeros((seq_len, 6), dtype=np.float32),
                "label_abs": np.zeros((seq_len, 6), dtype=np.float32),
                "label_pos_abs": np.zeros((seq_len, 3), dtype=np.float32),
                "label_pos_rel": np.zeros((seq_len, 3), dtype=np.float32),
                "ref_pos": np.zeros((3,), dtype=np.float32),
                "dt": 0.1,
                "target_valid_mask": np.zeros((seq_len,), dtype=np.float32),
                "file_name": f"target_{i}"
            }

            # Initialize identity rotation matrices
            for t in range(seq_len):
                for u in range(num_uavs):
                    sample["rotmat_b2e"][t, u] = np.eye(3, dtype=np.float32)

            # Ground truth reference
            if i < len(pos_eval_refs):
                pos_eval = f[pos_eval_refs[i]][()]
                if pos_eval.shape[0] == 3 and pos_eval.shape[1] > 3:
                    pos_eval = pos_eval.T  # (seq_len, 3)
                
                valid_eval = None
                if i < len(valid_eval_refs):
                    valid_eval = f[valid_eval_refs[i]][()].flatten()
                
                if len(pos_eval) > 0:
                    ref_pos = pos_eval[0].copy()
                    sample["ref_pos"] = ref_pos.astype(np.float32)

                    for t in range(min(seq_len, len(pos_eval))):
                        if valid_eval is not None and len(valid_eval) > t and valid_eval[t] > 0:
                            sample["label_pos_abs"][t] = pos_eval[t].astype(np.float32)
                            sample["label_pos_rel"][t] = (pos_eval[t] - ref_pos).astype(np.float32)
                            sample["label"][t, [0, 2, 4]] = sample["label_pos_rel"][t]
                            sample["label_abs"][t, [0, 2, 4]] = sample["label_pos_abs"][t]
                            sample["target_valid_mask"][t] = 1.0

            # Sensor parsing
            if i < len(sensor_refs):
                sensor_grp = f[sensor_refs[i]]
                
                def process_sensor(sensor_type_name, sensor_type):
                    if sensor_type_name not in sensor_grp: return
                    # Sensor entries are stored as groups in the v7.3 files used here.
                    s_grp = sensor_grp[sensor_type_name]
                    
                    if 'pos' not in s_grp or 'valid' not in s_grp: return
                    pos = s_grp['pos'][()]
                    valid = s_grp['valid'][()]
                    
                    if pos.ndim == 2 and pos.shape[0] == 3:
                        pos = pos.T # (T, 3)
                    if valid.ndim == 2 and valid.shape[0] == 1:
                        valid = valid.flatten() # (T,)
                        
                    for t in range(min(seq_len, len(pos))):
                        # The MATLAB code combines all UAV sensors into one single target observation!
                        # So it's effectively 1 "sensor stream" per target, we can just put it in UAV 0
                        u = 0 
                        if not valid[t]:
                            continue
                            
                        tgt_xyz = pos[t]
                        # We don't have UAV pos, assume UAV is at origin relative to world coordinates
                        rel_world = tgt_xyz
                        r = np.linalg.norm(rel_world)
                        
                        if r < 1e-6:
                            continue
                            
                        el = np.arcsin(rel_world[2] / r)
                        az = np.arctan2(rel_world[1], rel_world[0])
                        
                        if sensor_type == "radar":
                            sample["radar"][t, u] = np.array([az, el, r], dtype=np.float32)
                            sample["radar_mask"][t, u] = 1.0
                        elif sensor_type == "eo":
                            sample["eo"][t, u] = np.array([az, el], dtype=np.float32)
                            sample["eo_mask"][t, u] = 1.0
                        elif sensor_type == "esm":
                            sample["esm"][t, u, 0] = az
                            sample["esm_mask"][t, u] = 1.0
                        
                        # UAV pos is [0, 0, 0] relative to world origin, but we need it relative to ref_pos
                        sample["uav_pos"][t, u] = (np.array([0,0,0]) - sample["ref_pos"]).astype(np.float32)

                process_sensor("Radar_world", "radar")
                process_sensor("EO_world", "eo")
                process_sensor("ESM_world", "esm")
            
            # Convert to PyTorch tensors
            for k, v in sample.items():
                if isinstance(v, np.ndarray):
                    sample[k] = torch.from_numpy(v)
            
            samples.append(sample)
        except Exception as e:
            print(f"Error parsing target {i}: {e}")

    f.close()
    return samples


class SingerDataset(Dataset):
    """
    Load MATLAB traces generated by Dataset_Singer_v2.

    Output tensors (main):
    - radar:        [T, N, 3]
    - eo:           [T, N, 2]
    - esm:          [T, N, 2]
    - uav_pos:      [T, N, 3]
    - radar_mask:   [T, N]
    - eo_mask:      [T, N]
    - esm_mask:     [T, N]
    - label:        [T, 6]   (px, vx, py, vy, pz, vz)
    - dt:           scalar

    Extra fields (for future learned-K / timing / IMU use):
    - imu:          [T, N, 6]   [roll, pitch, yaw, p, q, r]
    - euler:        [T, N, 3]
    - quat:         [T, N, 4]
    - fast_link:    [T, N, 3]
    - slow_link:    [T, N, 3]
    - fast_age:     [T, N]
    - slow_age:     [T, N]
    - fast_update:  [T, N]
    - slow_update:  [T, N]
    - esm_mode:     [T, N]
    - global_time:  [T]
    """

    def __init__(
        self,
        data_dir: str = "./train_data",
        seq_len: int = 100,
        num_uavs: int = 5,
        return_extra: bool = True,
        use_dummy_on_error: bool = True,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.num_uavs = num_uavs
        self.return_extra = return_extra
        self.use_dummy_on_error = use_dummy_on_error

        abs_path = os.path.abspath(data_dir)
        print(f"[SingerDataset] loading from: {abs_path}")

        if not os.path.isdir(data_dir):
            self.file_list = []
        else:
            mat_files = []
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f.endswith(".mat"):
                        mat_files.append(os.path.relpath(os.path.join(root, f), data_dir))
            self.file_list = sorted(mat_files)

        print(f"[SingerDataset] found {len(self.file_list)} trace files")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx >= len(self.file_list):
            raise IndexError(idx)

        file_path = os.path.join(self.data_dir, self.file_list[idx])

        try:
            mat = scipy.io.loadmat(file_path, simplify_cells=True)
            sample = self._parse_mat(mat)
            sample["file_name"] = self.file_list[idx]
            return sample
        except Exception as e:
            if self.use_dummy_on_error:
                print(f"[SingerDataset] warning: failed to parse {file_path}, error={e}")
                sample = self._get_dummy_data()
                sample["file_name"] = self.file_list[idx]
                return sample
            raise

    def _parse_mat(self, mat: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "Data_Input" not in mat or "Data_Label" not in mat:
            raise ValueError("Missing Data_Input or Data_Label")

        data_input = mat["Data_Input"]
        label_raw = np.asarray(mat["Data_Label"], dtype=np.float64)
        dt = _to_scalar(mat.get("T", 0.1), default=0.1)

        # label: MATLAB usually stores [6, T]
        if label_raw.ndim != 2:
            raise ValueError(f"Data_Label ndim must be 2, got {label_raw.ndim}")

        if label_raw.shape[0] == 6:
            label_raw = label_raw.T
        ref_pos_64 = label_raw[0, [0, 2, 4]].copy()

        label_raw = self._pad_or_clip_2d(label_raw, self.seq_len, 6, dtype=np.float64)
        label_abs = label_raw.astype(np.float32)
        label_pos_abs_64 = label_raw[:, [0, 2, 4]]
        label_pos_abs = label_pos_abs_64.astype(np.float32)
        label_pos_rel = (label_pos_abs_64 - ref_pos_64[None, :]).astype(np.float32)
        ref_pos = ref_pos_64.astype(np.float32)

        # -------- allocate --------
        radar = np.zeros((self.seq_len, self.num_uavs, 3), dtype=np.float32)
        eo = np.zeros((self.seq_len, self.num_uavs, 2), dtype=np.float32)
        esm = np.zeros((self.seq_len, self.num_uavs, 2), dtype=np.float32)

        uav_pos = np.zeros((self.seq_len, self.num_uavs, 3), dtype=np.float32)
        rotmat_b2e = np.zeros((self.seq_len, self.num_uavs, 3, 3), dtype=np.float32)

        radar_mask = np.zeros((self.seq_len, self.num_uavs), dtype=np.float32)
        eo_mask = np.zeros((self.seq_len, self.num_uavs), dtype=np.float32)
        esm_mask = np.zeros((self.seq_len, self.num_uavs), dtype=np.float32)

        imu = np.zeros((self.seq_len, self.num_uavs, 6), dtype=np.float32)
        euler = np.zeros((self.seq_len, self.num_uavs, 3), dtype=np.float32)
        quat = np.zeros((self.seq_len, self.num_uavs, 4), dtype=np.float32)

        fast_link = np.zeros((self.seq_len, self.num_uavs, 3), dtype=np.float32)
        slow_link = np.zeros((self.seq_len, self.num_uavs, 3), dtype=np.float32)

        fast_age = np.zeros((self.seq_len, self.num_uavs), dtype=np.float32)
        slow_age = np.zeros((self.seq_len, self.num_uavs), dtype=np.float32)

        fast_update = np.zeros((self.seq_len, self.num_uavs), dtype=np.float32)
        slow_update = np.zeros((self.seq_len, self.num_uavs), dtype=np.float32)

        esm_mode = np.zeros((self.seq_len, self.num_uavs), dtype=np.float32)
        global_time = np.zeros((self.seq_len,), dtype=np.float32)

        # -------- parse --------
        max_t = min(self.seq_len, len(data_input))
        for t in range(max_t):
            frame = data_input[t]
            if not isinstance(frame, dict):
                continue

            global_time[t] = _to_scalar(frame.get("GlobalTime", 0.0), default=0.0)

            uavs = frame.get("UAV", None)
            if not isinstance(uavs, (list, tuple)):
                continue

            for u in range(min(self.num_uavs, len(uavs))):
                ud = uavs[u]
                if not isinstance(ud, dict):
                    continue

                # ---------------- Pose ----------------
                pos_64 = _to_fixed_array_64(_safe_get(ud, "Pose", "PositionWorld", default=None), 3)
                uav_pos[t, u] = (pos_64 - ref_pos_64).astype(np.float32)

                euler_rpy = _to_fixed_array(
                    _safe_get(ud, "Pose", "EulerRPY", default=None), 3
                )
                euler[t, u] = euler_rpy

                quat_wxyz = _to_fixed_array(
                    _safe_get(ud, "Pose", "QuaternionWXYZ", default=None), 4
                )
                quat[t, u] = quat_wxyz

                if np.linalg.norm(pos_64) < 1e-6:  # <--- 注意这里用 pos_64 算
                    rotmat_b2e[t, u] = np.eye(3, dtype=np.float32)
                else:
                    lon = np.arctan2(pos_64[1], pos_64[0])  # <--- 注意这里用 pos_64 算
                    p = np.hypot(pos_64[0], pos_64[1])
                    a_wgs84 = 6378137.0
                    e2_wgs84 = 0.00669437999014
                    lat = np.arctan2(pos_64[2], p * (1.0 - e2_wgs84))
                    for _ in range(3):
                        N_rad = a_wgs84 / np.sqrt(1.0 - e2_wgs84 * np.sin(lat) ** 2)
                        lat = np.arctan2(pos_64[2] + e2_wgs84 * N_rad * np.sin(lat), p)

                    # 2. 构建当地 ENU 到 ECEF 的旋转矩阵
                    slat, clat = np.sin(lat), np.cos(lat)
                    slon, clon = np.sin(lon), np.cos(lon)
                    R_n2e = np.array([
                        [-slon, -slat * clon, clat * clon],
                        [clon, -slat * slon, clat * slon],
                        [0.0, clat, slat]
                    ])

                    # 3. 解析记录的偏航和俯仰角 (注意这里使用刚刚提取的 euler_rpy)
                    pitch, yaw = euler_rpy[1], euler_rpy[2]
                    fwd_enu = np.array([
                        np.sin(yaw) * np.cos(pitch),
                        np.cos(yaw) * np.cos(pitch),
                        np.sin(pitch)
                    ])
                    unit_fwd = R_n2e @ fwd_enu
                    norm_fwd = np.linalg.norm(unit_fwd)
                    unit_fwd = unit_fwd / norm_fwd if norm_fwd > 1e-6 else np.array([1.0, 0.0, 0.0])

                    # 4. 在 ECEF 中构建正交的机体坐标系轴 (与 ecef_utils.py get_body_frame_axes 对齐)
                    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    unit_right = np.cross(world_up, unit_fwd)
                    norm_r = np.linalg.norm(unit_right)
                    if norm_r < 1e-6:
                        unit_right = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    else:
                        unit_right /= norm_r

                    unit_up = np.cross(unit_fwd, unit_right)
                    unit_up /= np.linalg.norm(unit_up)

                    # 5. 存入矩阵
                    rotmat_b2e[t, u] = np.column_stack((unit_fwd, unit_right, unit_up))

                # ---------------- IMU ----------------
                imu_vec = _to_fixed_array(
                    _safe_get(ud, "IMU", "Vector", default=None), 6
                )
                imu[t, u] = imu_vec

                # ---------------- Radar ----------------
                radar_meas = _to_fixed_array(
                    _safe_get(ud, "Radar", "Meas", default=None), 3
                )
                radar[t, u] = radar_meas
                radar_mask[t, u] = _to_scalar(
                    _safe_get(ud, "Radar", "Mask", default=0.0), default=0.0
                )

                # ---------------- EO ----------------
                eo_meas = _to_fixed_array(
                    _safe_get(ud, "EO", "Meas", default=None), 2
                )
                eo[t, u] = eo_meas
                eo_mask[t, u] = _to_scalar(
                    _safe_get(ud, "EO", "Mask", default=0.0), default=0.0
                )

                # ---------------- ESM ----------------
                esm_meas = _to_fixed_array(
                    _safe_get(ud, "ESM", "Meas", default=None), 2
                )
                esm[t, u] = esm_meas
                esm_mask[t, u] = _to_scalar(
                    _safe_get(ud, "ESM", "Mask", default=0.0), default=0.0
                )
                esm_mode[t, u] = _to_scalar(
                    _safe_get(ud, "ESM", "ModeCode", default=0.0), default=0.0
                )

                # ---------------- FastLink ----------------
                fast_meas_64 = _to_fixed_array_64(_safe_get(ud, "FastLink", "Meas", default=None), 3)
                if np.linalg.norm(fast_meas_64) > 1e-6:
                    fast_link[t, u] = (fast_meas_64 - ref_pos_64).astype(np.float32)
                fast_age[t, u] = _to_scalar(
                    _safe_get(ud, "FastLink", "Age", default=0.0), default=0.0
                )
                fast_update[t, u] = _to_scalar(
                    _safe_get(ud, "FastLink", "IsUpdate", default=0.0), default=0.0
                )

                # ---------------- SlowLink ----------------
                slow_meas_64 = _to_fixed_array_64(_safe_get(ud, "SlowLink", "Meas", default=None), 3)
                if np.linalg.norm(slow_meas_64) > 1e-6:
                    slow_link[t, u] = (slow_meas_64 - ref_pos_64).astype(np.float32)
                slow_age[t, u] = _to_scalar(
                    _safe_get(ud, "SlowLink", "Age", default=0.0), default=0.0
                )
                slow_update[t, u] = _to_scalar(
                    _safe_get(ud, "SlowLink", "IsUpdate", default=0.0), default=0.0
                )

        sample: Dict[str, torch.Tensor] = {
            "radar": torch.from_numpy(radar),               # [T, N, 3]
            "eo": torch.from_numpy(eo),                     # [T, N, 2]
            "esm": torch.from_numpy(esm),                   # [T, N, 2]
            "uav_pos": torch.from_numpy(uav_pos),           # [T, N, 3]
            "radar_mask": torch.from_numpy(radar_mask),     # [T, N]
            "eo_mask": torch.from_numpy(eo_mask),           # [T, N]
            "esm_mask": torch.from_numpy(esm_mask),         # [T, N]
            "label": torch.from_numpy(label_abs),           # [T, 6] (backward compatible)
            "label_abs": torch.from_numpy(label_abs),       # [T, 6]
            "label_pos_abs": torch.from_numpy(label_pos_abs),   # [T, 3]
            "label_pos_rel": torch.from_numpy(label_pos_rel),   # [T, 3], relative to frame-0
            "ref_pos": torch.from_numpy(ref_pos),           # [3], frame-0 absolute position
            "rotmat_b2e": torch.from_numpy(rotmat_b2e),     # [新增] [T, N, 3, 3]
            "dt": torch.tensor(dt, dtype=torch.float32),
        }

        if self.return_extra:
            sample.update(
                {
                    "imu": torch.from_numpy(imu),                 # [T, N, 6]
                    "euler": torch.from_numpy(euler),             # [T, N, 3]
                    "quat": torch.from_numpy(quat),               # [T, N, 4]
                    "fast_link": torch.from_numpy(fast_link),     # [T, N, 3]
                    "slow_link": torch.from_numpy(slow_link),     # [T, N, 3]
                    "fast_age": torch.from_numpy(fast_age),       # [T, N]
                    "slow_age": torch.from_numpy(slow_age),       # [T, N]
                    "fast_update": torch.from_numpy(fast_update), # [T, N]
                    "slow_update": torch.from_numpy(slow_update), # [T, N]
                    "esm_mode": torch.from_numpy(esm_mode),       # [T, N]
                    "global_time": torch.from_numpy(global_time), # [T]
                }
            )

        return sample

    def _pad_or_clip_2d(self, arr: np.ndarray, length: int, feat_dim: int, dtype=np.float32) -> np.ndarray:
        arr = np.asarray(arr, dtype=dtype)  # <--- 修改这里
        arr = arr[:length]
        if arr.shape[0] < length:
            pad = np.zeros((length - arr.shape[0], feat_dim), dtype=dtype)
            arr = np.concatenate([arr, pad], axis=0)
        return arr

    def _get_dummy_data(self) -> Dict[str, torch.Tensor]:
        zeros = torch.zeros
        sample = {
            "radar": zeros((self.seq_len, self.num_uavs, 3), dtype=torch.float32),
            "eo": zeros((self.seq_len, self.num_uavs, 2), dtype=torch.float32),
            "esm": zeros((self.seq_len, self.num_uavs, 2), dtype=torch.float32),
            "uav_pos": zeros((self.seq_len, self.num_uavs, 3), dtype=torch.float32),
            "radar_mask": zeros((self.seq_len, self.num_uavs), dtype=torch.float32),
            "eo_mask": zeros((self.seq_len, self.num_uavs), dtype=torch.float32),
            "esm_mask": zeros((self.seq_len, self.num_uavs), dtype=torch.float32),
            "label": zeros((self.seq_len, 6), dtype=torch.float32),
            "label_abs": zeros((self.seq_len, 6), dtype=torch.float32),
            "label_pos_abs": zeros((self.seq_len, 3), dtype=torch.float32),
            "label_pos_rel": zeros((self.seq_len, 3), dtype=torch.float32),
            "ref_pos": zeros((3,), dtype=torch.float32),
            "rotmat_b2e": torch.eye(3).view(1, 1, 3, 3).repeat(self.seq_len, self.num_uavs, 1, 1),
            "dt": torch.tensor(0.1, dtype=torch.float32),
        }

        if self.return_extra:
            sample.update(
                {
                    "imu": zeros((self.seq_len, self.num_uavs, 6), dtype=torch.float32),
                    "euler": zeros((self.seq_len, self.num_uavs, 3), dtype=torch.float32),
                    "quat": zeros((self.seq_len, self.num_uavs, 4), dtype=torch.float32),
                    "fast_link": zeros((self.seq_len, self.num_uavs, 3), dtype=torch.float32),
                    "slow_link": zeros((self.seq_len, self.num_uavs, 3), dtype=torch.float32),
                    "fast_age": zeros((self.seq_len, self.num_uavs), dtype=torch.float32),
                    "slow_age": zeros((self.seq_len, self.num_uavs), dtype=torch.float32),
                    "fast_update": zeros((self.seq_len, self.num_uavs), dtype=torch.float32),
                    "slow_update": zeros((self.seq_len, self.num_uavs), dtype=torch.float32),
                    "esm_mode": zeros((self.seq_len, self.num_uavs), dtype=torch.float32),
                    "global_time": zeros((self.seq_len,), dtype=torch.float32),
                }
            )
        return sample


def _parse_registered_pkl(data, seq_len, num_uavs, basename):
    samples = []
    registered_track = data['RegisteredTrack']
    uav_pos_all = data.get('UAV_Pos')

    # 优先使用 pkl 中存储的 UAV_Rotmat（经 fix_uav_rotmat_in_pkls.py 修正后为正确值）。
    # 若文件没有该字段，则实时用向量化差分重建（速度仍远快于旧的 Python 循环）。
    uav_rotmat_all = data.get('UAV_Rotmat')
    uav_rotmat_rebuilt = None
    if uav_pos_all is not None:
        if uav_rotmat_all is not None:
            uav_rotmat_rebuilt = np.array(uav_rotmat_all, dtype=np.float32)
        else:
            # 回退：向量化重建（无 Python 循环）
            T_total = uav_pos_all.shape[0]
            t_vals = np.arange(T_total, dtype=np.float64) * 0.1
            vel_all = np.gradient(uav_pos_all.astype(np.float64), t_vals, axis=0)
            world_up = np.array([0.0, 0.0, 1.0])
            speed = np.linalg.norm(vel_all, axis=-1, keepdims=True)
            speed = np.where(speed < 1e-6, 1.0, speed)
            unit_fwd = vel_all / speed
            wu = np.broadcast_to(world_up, unit_fwd.shape)
            unit_right = np.cross(wu, unit_fwd)
            norm_r = np.linalg.norm(unit_right, axis=-1, keepdims=True)
            fallback = np.broadcast_to(np.array([0.0, 1.0, 0.0]), unit_fwd.shape)
            unit_right = np.where(norm_r < 1e-6, fallback,
                                  unit_right / np.where(norm_r < 1e-6, 1.0, norm_r))
            unit_up = np.cross(unit_fwd, unit_right)
            norm_u = np.linalg.norm(unit_up, axis=-1, keepdims=True).clip(min=1e-12)
            unit_up = unit_up / norm_u
            uav_rotmat_rebuilt = np.stack([unit_fwd, unit_right, unit_up], axis=-1).astype(np.float32)


    for i, target_dict in enumerate(registered_track):
        # We need to construct the standard PyTorch dict
        sample = {
            "radar": np.zeros((seq_len, num_uavs, 3), dtype=np.float32),
            "eo": np.zeros((seq_len, num_uavs, 2), dtype=np.float32),
            "esm": np.zeros((seq_len, num_uavs, 2), dtype=np.float32),
            "uav_pos": np.zeros((seq_len, num_uavs, 3), dtype=np.float32),
            "rotmat_b2e": np.zeros((seq_len, num_uavs, 3, 3), dtype=np.float32),
            "fast_link": np.zeros((seq_len, num_uavs, 3), dtype=np.float32),
            "radar_mask": np.zeros((seq_len, num_uavs), dtype=np.float32),
            "eo_mask": np.zeros((seq_len, num_uavs), dtype=np.float32),
            "esm_mask": np.zeros((seq_len, num_uavs), dtype=np.float32),
            "label": np.zeros((seq_len, 6), dtype=np.float32),
            "label_abs": np.zeros((seq_len, 6), dtype=np.float32),
            "label_pos_abs": np.zeros((seq_len, 3), dtype=np.float32),
            "label_pos_rel": np.zeros((seq_len, 3), dtype=np.float32),
            "ref_pos": np.zeros((3,), dtype=np.float32),
            "dt": 0.1,
            "target_valid_mask": np.zeros((seq_len,), dtype=np.float32),
            "fast_update": np.zeros((seq_len, num_uavs), dtype=np.float32),
            "file_name": f"{basename}_target_{i}"
        }

        for t in range(seq_len):
            for u in range(num_uavs):
                sample["rotmat_b2e"][t, u] = np.eye(3, dtype=np.float32)

        # Ground truth
        target_pos = target_dict['target_pos']
        # If target_pos has non-zero values, it means we have truth
        has_truth = np.any(target_pos)
        ref_pos = np.zeros(3, dtype=np.float64)

        if has_truth:
            ref_pos = target_pos[0].copy()
            sample["ref_pos"] = ref_pos.astype(np.float32)
            target_vel = target_dict.get('target_vel')  # [T, 3]，可能为 None
            for t in range(min(seq_len, len(target_pos))):
                sample["target_valid_mask"][t] = 1.0
                sample["label_pos_abs"][t] = target_pos[t].astype(np.float32)
                sample["label_pos_rel"][t] = (target_pos[t] - ref_pos).astype(np.float32)
                sample["label"][t, [0, 2, 4]] = sample["label_pos_rel"][t]
                sample["label_abs"][t, [0, 2, 4]] = sample["label_pos_abs"][t]
                # 修复：填入速度分量（之前全为 0 导致速度监督失效）
                if target_vel is not None and t < len(target_vel):
                    sample["label"][t, [1, 3, 5]] = target_vel[t].astype(np.float32)
                    sample["label_abs"][t, [1, 3, 5]] = target_vel[t].astype(np.float32)

        # Fill sensors
        for u in range(num_uavs):
            # Fill UAV Poses — 使用重建的正确旋转矩阵（与 SingerDataset 一致）
            if uav_pos_all is not None and uav_rotmat_rebuilt is not None:
                for t in range(min(seq_len, uav_pos_all.shape[0])):
                    sample["uav_pos"][t, u] = (uav_pos_all[t, u] - ref_pos).astype(np.float32)
                    sample["rotmat_b2e"][t, u] = uav_rotmat_rebuilt[t, u]

            # Radar
            radar = target_dict['Radar'][u]
            meas = radar.get('Meas') # [r, az, el]
            meas_ecef = radar.get('Meas_ECEF')
            valid = radar['valid']
            for t in range(min(seq_len, len(valid))):
                if valid[t]:
                    if meas is not None:
                        r_val, az_val, el_val = meas[t]
                        sample["radar"][t, u] = np.array([az_val, el_val, r_val], dtype=np.float32)
                    else:
                        sample["radar"][t, u] = (meas_ecef[t] - sample["ref_pos"]).astype(np.float32)
                    sample["radar_mask"][t, u] = 1.0
            
            # EO
            eo = target_dict['EO'][u]
            meas_eo = eo.get('Meas') # [az, el]
            meas_ecef_eo = eo.get('Meas_ECEF')
            valid_eo = eo['valid']
            for t in range(min(seq_len, len(valid_eo))):
                if valid_eo[t]:
                    if meas_eo is not None:
                        az_val, el_val = meas_eo[t]
                        sample["eo"][t, u] = np.array([az_val, el_val], dtype=np.float32)
                    else:
                        direction = meas_ecef_eo[t] - sample["ref_pos"]
                        sample["eo"][t, u] = direction[:2].astype(np.float32)
                    sample["eo_mask"][t, u] = 1.0

            # ESM
            esm = target_dict['ESM'][u]
            meas_esm = esm.get('Meas') # [az]
            meas_ecef_esm = esm.get('Meas_ECEF')
            valid_esm = esm['valid']
            for t in range(min(seq_len, len(valid_esm))):
                if valid_esm[t]:
                    if meas_esm is not None:
                        sample["esm"][t, u, 0] = meas_esm[t][0]
                    else:
                        direction = meas_ecef_esm[t] - sample["ref_pos"]
                        sample["esm"][t, u, 0] = direction[0]
                    sample["esm_mask"][t, u] = 1.0
            
            # FastLink (from Wingman Fusion)
            if 'FastLink' in target_dict and u > 0: # Wingman only
                fastlink = target_dict['FastLink'][u]
                meas_ecef_fl = fastlink.get('Meas_ECEF')
                valid_fl = fastlink['valid']
                for t in range(min(seq_len, len(valid_fl))):
                    if valid_fl[t] and np.all(np.isfinite(meas_ecef_fl[t])):
                        sample["fast_link"][t, u] = (meas_ecef_fl[t] - sample["ref_pos"]).astype(np.float32)
                        if "fast_update" in sample:
                            sample["fast_update"][t, u] = 1.0

        # Convert to torch tensors
        for k in sample:
            if isinstance(sample[k], np.ndarray):
                sample[k] = torch.from_numpy(sample[k])
            elif k == "dt":
                sample[k] = torch.tensor(sample[k], dtype=torch.float32)
                
        samples.append(sample)
    
    return samples
