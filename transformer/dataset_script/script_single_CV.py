# script_single_CV.py — 匀速 (CV) 单目标轨迹生成，全程在 WGS84 ECEF 坐标系建模
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import datetime
from datetime import timedelta
import traceback
import collections
import numpy as np
import scipy.io

from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.detection import TrueDetection
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.base import Property
from stonesoup.sensor.sensor import SimpleSensor

from ecef_utils import (
    geodetic_to_ecef, ecef_to_geodetic,
    ecef_to_enu_matrix, enu_to_ecef_matrix,
    enu_vel_to_ecef, get_body_frame_axes,
    reconstruct_ecef_from_body_angles,
)

# ─────────────────────────── 全局仿真参数 ───────────────────────────
START_TIME = datetime.datetime.now()
DURATION   = 10.0   # s
DT         = 0.1    # s

SENSOR_PERIODS = {
    0: timedelta(seconds=0.1),   # 雷达
    1: timedelta(seconds=0.1),   # EO
    2: timedelta(seconds=0.2),   # ESM
}

# ─────────────────────────── 数据链节点 ──────────────────────────────
class DatalinkNode:
    """僚机数据链：将目标 ECEF 坐标（加噪声）延迟传输给长机"""
    def __init__(self, delay_s=0.1, interval_s=0.1, noise_r_m=100.0):
        self.delay_s    = delay_s
        self.interval_s = interval_s
        self.noise_r    = noise_r_m
        self.queue      = collections.deque()
        self.last_tx    = -999.0

    def ingest(self, t_s, target_ecef):
        if t_s - self.last_tx >= self.interval_s - 1e-5:
            noisy = np.array(target_ecef, dtype=float) + np.random.randn(3) * self.noise_r
            self.queue.append({
                "arrival": t_s + self.delay_s,
                "age":     self.delay_s,
                "xyz":     noisy,
            })
            self.last_tx = t_s

    def pop(self, t_s):
        latest = None
        while self.queue and self.queue[0]["arrival"] <= t_s + 1e-5:
            latest = self.queue.popleft()
        return latest


# ─────────────────────────── ECEF 平台 ───────────────────────────────
class ECEFPlatform:
    """
    在 ECEF 坐标系中积分运动的平台。
    机动加速度在当地 ENU 坐标系中施加（侧向 + 垂向正弦），再转回 ECEF 积分。
    """
    def __init__(self, pos_ecef, vel_ecef, dt=0.1, phase=0.0):
        self.pos        = np.array(pos_ecef, dtype=float)
        self.vel        = np.array(vel_ecef, dtype=float)
        self.base_speed = np.linalg.norm(vel_ecef)
        self.dt         = dt
        self.time       = 0.0
        self.phase      = phase

    def step(self):
        self.time += self.dt
        lat, lon, _ = ecef_to_geodetic(*self.pos)
        R_e2n = ecef_to_enu_matrix(lat, lon)   # ECEF → ENU
        R_n2e = R_e2n.T                         # ENU  → ECEF

        vel_enu = R_e2n @ self.vel
        vE, vN = vel_enu[0], vel_enu[1]
        speed_h = np.hypot(vE, vN)

        # 水平面内垂直于前进方向的侧向单位矢（ENU）
        if speed_h > 1e-6:
            lat_dir_enu = np.array([-vN / speed_h, vE / speed_h, 0.0])
        else:
            lat_dir_enu = np.array([1.0, 0.0, 0.0])

        a_lat = 3.0 * np.sin(2 * np.pi * self.time / 20.0 + self.phase)
        a_vert= 1.0 * np.sin(2 * np.pi * self.time / 15.0 + self.phase * 1.5)
        acc_enu = a_lat * lat_dir_enu + np.array([0.0, 0.0, a_vert])

        # 速度幅值修正（向前方向恢复额定速度）
        cur_speed  = np.linalg.norm(self.vel)
        fwd_ecef   = self.vel / max(cur_speed, 1e-6)
        fwd_enu    = R_e2n @ fwd_ecef
        acc_enu   += 0.2 * (self.base_speed - cur_speed) * fwd_enu

        acc_ecef = R_n2e @ acc_enu
        self.vel = self.vel + acc_ecef * self.dt
        self.pos = self.pos + self.vel * self.dt
        return self.pos.copy(), self.vel.copy(), acc_ecef.copy()


# ─────────────────────────── 传感器 ──────────────────────────────────
class ECEFBodySensor(SimpleSensor):
    """
    在 ECEF 坐标中计算机体系相对矢量，然后喂给 StoneSoup 量测模型。
    机体坐标系使用当地地理"上"方向，而非全局 Z 轴。
    """
    measurement_model = Property("量测模型")
    position_mapping  = Property("位置映射", default=(0, 2, 4))
    az_fov   = Property("方位视场 (rad)", default=None)
    el_fov   = Property("俯仰视场 (rad)", default=None)
    max_range = Property("最大探测距离 (m)", default=None)

    def _body_vec(self, truth, plat_state):
        sv_tgt = truth.state_vector.flatten()
        sv_plt = plat_state.state_vector.flatten()
        pos_tgt = np.array([sv_tgt[0], sv_tgt[2], sv_tgt[4]])
        pos_plt = np.array([sv_plt[0], sv_plt[2], sv_plt[4]])
        vel_plt = np.array([sv_plt[1], sv_plt[3], sv_plt[5]])
        rel_ecef = pos_tgt - pos_plt
        fwd, rgt, up = get_body_frame_axes(vel_plt, pos_plt)
        return np.dot(rel_ecef, fwd), np.dot(rel_ecef, rgt), np.dot(rel_ecef, up)

    def measure(self, ground_truths, noise=True, platform_state=None, **kwargs):
        detections = []
        for truth in ground_truths:
            if not self.is_detectable(truth, platform_state):
                continue
            dx, dy, dz = self._body_vec(truth, platform_state)
            rs = truth.state_vector.copy()
            rs[self.position_mapping[0]] = dx
            rs[self.position_mapping[1]] = dy
            rs[self.position_mapping[2]] = dz
            rel_truth = GroundTruthState(rs, timestamp=truth.timestamp)
            meas_vec  = self.measurement_model.function(rel_truth, noise=noise, **kwargs)
            detections.append(TrueDetection(
                meas_vec,
                measurement_model=self.measurement_model,
                timestamp=truth.timestamp,
                groundtruth_path=truth,
            ))
        return detections

    def is_detectable(self, truth, platform_state):
        dx, dy, dz = self._body_vec(truth, platform_state)
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        if r < 1e-6:
            return False
        az = np.arctan2(dy, dx)
        el = np.arcsin(np.clip(dz / r, -1.0, 1.0))
        if self.az_fov  and not (self.az_fov[0]  <= az <= self.az_fov[1]):  return False
        if self.el_fov  and not (self.el_fov[0]  <= el <= self.el_fov[1]):  return False
        if self.max_range and r > self.max_range: return False
        return True

    def is_clutter_detectable(self, clutter):
        return True


def _reorder(stone_vec):
    """StoneSoup 输出顺序 (el, az, r) → (az, el, r)"""
    arr = np.asarray(stone_vec).flatten()
    if arr.size >= 3:
        return float(arr[1]), float(arr[0]), float(arr[2])
    return tuple(float(x) for x in arr)


def _make_sensors():
    """构造雷达 / EO / ESM 三个传感器"""
    radar = ECEFBodySensor(
        measurement_model=CartesianToElevationBearingRange(
            ndim_state=6, mapping=(0, 2, 4),
            noise_covar=np.diag([0.002**2, 0.002**2, 80.0**2])),
        az_fov=(-np.pi/3,    np.pi/3),
        el_fov=(-np.radians(3), np.radians(3)),
        max_range=200_000.0,
    )
    eo = ECEFBodySensor(
        measurement_model=CartesianToElevationBearingRange(
            ndim_state=6, mapping=(0, 2, 4),
            noise_covar=np.diag([0.002**2, 0.002**2, 1e20])),
        az_fov=(-np.pi/3,    np.pi/3),
        el_fov=(-np.radians(3), np.radians(3)),
        max_range=200_000.0,
    )
    esm = ECEFBodySensor(
        measurement_model=CartesianToElevationBearingRange(
            ndim_state=6, mapping=(0, 2, 4),
            noise_covar=np.diag([0.005**2, 0.005**2, 1e20])),
        az_fov=(-np.radians(60), np.radians(60)),
        el_fov=(-np.radians(3),  np.radians(3)),
        max_range=200_000.0,
    )
    return radar, eo, esm


# ─────────────────────────── 目标运动：CV ────────────────────────────
def generate_cv_target_ecef(pos0_ecef, vel0_ecef, steps, dt):
    """匀速运动 (CV)：在 ECEF 中匀速积分"""
    pos = np.array(pos0_ecef, dtype=float)
    vel = np.array(vel0_ecef, dtype=float)
    states = []
    t = START_TIME
    for _ in range(steps):
        pos = pos + vel * dt
        sv  = np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]])
        st  = GroundTruthState(sv, timestamp=t)
        st.true_acc = np.zeros(3)
        states.append(st)
        t += timedelta(seconds=dt)
    return states


# ─────────────────────────── 主仿真函数 ──────────────────────────────
def run_simulation(master_lat, master_lon, master_alt_m,
                   master_heading_deg, master_speed_mps,
                   target_pos_ecef, target_vel_ecef,
                   seed=None):
    if seed is not None:
        np.random.seed(seed)

    steps = int(DURATION / DT)
    truth = generate_cv_target_ecef(target_pos_ecef, target_vel_ecef, steps, DT)

    # ── 平台初始化 ──
    R_n2e = enu_to_ecef_matrix(master_lat, master_lon)
    h_rad = np.radians(master_heading_deg)
    vel_enu_master = np.array([master_speed_mps * np.sin(h_rad),
                                master_speed_mps * np.cos(h_rad),
                                0.0])
    master_vel = R_n2e @ vel_enu_master
    master_pos = geodetic_to_ecef(master_lat, master_lon, master_alt_m)

    # 僚机 ENU 偏置 → ECEF 绝对位置
    wm_enu_offsets = [
        np.array([ 0.0,  3000.0, 0.0]),
        np.array([ 0.0, -3000.0, 0.0]),
        np.array([ 3000.0,  3000.0, 0.0]),
        np.array([ 3000.0, -3000.0, 0.0]),
    ]
    wingmen_pos = [master_pos + R_n2e @ off for off in wm_enu_offsets]

    sensor_configs = ["all", "radar_only", "radar_eo", "radar_esm", "radar_only"]
    platforms = []
    for i, cfg in enumerate(sensor_configs):
        pos0 = master_pos if i == 0 else wingmen_pos[i - 1]
        plat = {
            "motion":  ECEFPlatform(pos0, master_vel.copy(), dt=DT, phase=float(i)),
            "is_master": (i == 0),
            "datalink": None if i == 0 else DatalinkNode(),
            "cached_dl": None,
        }
        r, eo, esm = _make_sensors()
        if cfg == "all":
            plat["sensors"] = [r, eo, esm]
        elif cfg == "radar_only":
            plat["sensors"] = [r, None, None]
        elif cfg == "radar_esm":
            plat["sensors"] = [r, None, esm]
        elif cfg == "radar_eo":
            plat["sensors"] = [r, eo, None]
        else:
            plat["sensors"] = [None, None, None]
        plat["last_meas"] = {
            idx: START_TIME - p for idx, p in SENSOR_PERIODS.items()
        }
        platforms.append(plat)

    T = len(truth)
    Data_Input = np.empty((T,), dtype=object)
    Data_Label  = np.zeros((T, 6), dtype=np.float64)
    consec_invis = 0

    for t, state in enumerate(truth):
        t_s = float((state.timestamp - START_TIME).total_seconds())
        sv  = state.state_vector.flatten()
        # 标签：绝对 ECEF [X, Vx, Y, Vy, Z, Vz]
        Data_Label[t] = [sv[0], sv[1], sv[2], sv[3], sv[4], sv[5]]

        uav_list   = np.empty((len(platforms),), dtype=object)
        step_visible = False

        for u, plat in enumerate(platforms):
            p_pos, p_vel, p_acc = plat["motion"].step()

            plat_sv = np.array([
                p_pos[0], p_vel[0], p_pos[1], p_vel[1], p_pos[2], p_vel[2]
            ])
            plat_state = GroundTruthState(plat_sv, timestamp=state.timestamp)

            # 传感器观测
            sensor_out = {}
            reconstructed_ecef = None
            for idx, sensor in enumerate(plat["sensors"]):
                if sensor is None:
                    continue
                period = SENSOR_PERIODS[idx]
                if state.timestamp - plat["last_meas"][idx] >= period - timedelta(microseconds=10):
                    dets = sensor.measure([state], platform_state=plat_state)
                    obs  = []
                    for d in dets:
                        if isinstance(d, TrueDetection):
                            az, el, r = _reorder(d.state_vector)
                            obs.append({"meas": [az, el, r]})
                            if idx == 0 and reconstructed_ecef is None:
                                reconstructed_ecef = reconstruct_ecef_from_body_angles(
                                    az, el, r, p_pos, p_vel)
                    if obs and np.random.rand() >= 0.15:   # 15% 干扰
                        sensor_out[f"sensor_{idx}"] = obs
                    plat["last_meas"][idx] = state.timestamp

            if plat["is_master"]:
                dl_xyz, dl_age = None, 0.0
                sens = sensor_out
            else:
                if reconstructed_ecef is not None:
                    plat["datalink"].ingest(t_s, reconstructed_ecef)
                arrived = plat["datalink"].pop(t_s)
                if arrived is not None:
                    plat["cached_dl"] = arrived
                if plat["cached_dl"]:
                    dl_xyz = plat["cached_dl"]["xyz"]
                    dl_age = plat["cached_dl"]["age"]
                else:
                    dl_xyz, dl_age = None, -1.0
                sens = {}

            def _get(key, dim):
                obs = sens.get(key, [])
                if obs:
                    return obs[0]["meas"][:dim], 1.0
                return [0.0] * dim, 0.0

            r_meas, r_mask = _get("sensor_0", 3)
            e_meas, e_mask = _get("sensor_1", 2)
            s_meas, s_mask = _get("sensor_2", 2)

            if sensor_out:
                step_visible = True

            # Euler (使用 ENU 速度计算航向/俯仰)
            lat_p, lon_p, _ = ecef_to_geodetic(*p_pos)
            vel_enu_p = ecef_to_enu_matrix(lat_p, lon_p) @ p_vel
            yaw   = np.arctan2(vel_enu_p[0], vel_enu_p[1])
            pitch = np.arctan2(vel_enu_p[2], np.hypot(vel_enu_p[0], vel_enu_p[1]))
            euler = [0.0, float(pitch), float(yaw)]

            uav_d = {
                "Pose": {
                    "PositionWorld": p_pos.tolist(),   # 绝对 ECEF
                    "EulerRPY":      euler,
                    "QuaternionWXYZ": [1.0, 0.0, 0.0, 0.0],
                },
                "IMU": {"Vector": [0.0, 0.0, 0.0,
                                   float(p_acc[0]), float(p_acc[1]), float(p_acc[2])]},
                "FastLink": {
                    "Meas":     dl_xyz.tolist() if dl_xyz is not None else [0.0, 0.0, 0.0],
                    "Age":      dl_age,
                    "IsUpdate": 1.0 if dl_xyz is not None else 0.0,
                },
            }
            if plat["is_master"]:
                uav_d["Radar"]    = {"Meas": r_meas, "Mask": r_mask}
                uav_d["EO"]       = {"Meas": e_meas, "Mask": e_mask}
                uav_d["ESM"]      = {"Meas": s_meas, "Mask": s_mask, "ModeCode": 0.0}
                uav_d["SlowLink"] = {"Meas": [0.0, 0.0, 0.0], "Age": 0.0, "IsUpdate": 0.0}

            uav_list[u] = uav_d

        Data_Input[t] = {"GlobalTime": t_s, "UAV": uav_list}

        if not step_visible:
            consec_invis += 1
        else:
            consec_invis = 0
        if consec_invis > 4:
            return None

    if consec_invis > 0:
        return None

    return {"Data_Input": Data_Input, "Data_Label": Data_Label, "T": DT}


# ─────────────────────────── 随机初始化帮助函数 ───────────────────────
def _random_scenario():
    """随机生成长机初始状态 + 目标初始状态（全在 ECEF）"""
    # 长机：随机纬度 / 经度 / 高度 / 航向 / 速度
    lat  = np.random.uniform(30.0, 50.0)
    lon  = np.random.uniform(100.0, 130.0)
    alt  = np.random.uniform(5000.0, 8000.0)
    hdg  = np.random.uniform(0.0, 360.0)
    spd  = np.random.uniform(220.0, 300.0)

    master_pos = geodetic_to_ecef(lat, lon, alt)
    R_n2e      = enu_to_ecef_matrix(lat, lon)
    h_rad      = np.radians(hdg)
    master_vel = R_n2e @ np.array([spd * np.sin(h_rad), spd * np.cos(h_rad), 0.0])

    # 目标：在长机机体坐标系前方 8–25 km，方位 ±45°，俯仰 ±2°
    fwd, rgt, up = get_body_frame_axes(master_vel, master_pos)
    rng    = np.random.uniform(8000.0, 25000.0)
    az_off = np.random.uniform(-np.radians(45), np.radians(45))
    el_off = np.random.uniform(-np.radians(2),  np.radians(2))
    dx = rng * np.cos(el_off) * np.cos(az_off)
    dy = rng * np.cos(el_off) * np.sin(az_off)
    dz = rng * np.sin(el_off)
    tgt_pos = master_pos + dx * fwd + dy * rgt + dz * up

    # 目标速度（ENU → ECEF，在目标当地坐标系定义）
    tgt_lat, tgt_lon, _ = ecef_to_geodetic(*tgt_pos)
    t_hdg = hdg + np.random.uniform(-108.0, 108.0)   # ±0.6π 偏转
    t_spd = np.random.uniform(200.0, 350.0)
    t_vz  = np.random.uniform(-30.0, 30.0)
    th    = np.radians(t_hdg)
    tgt_vel = enu_to_ecef_matrix(tgt_lat, tgt_lon) @ np.array(
        [t_spd * np.sin(th), t_spd * np.cos(th), t_vz])

    return lat, lon, alt, hdg, spd, tgt_pos, tgt_vel


# ─────────────────────────── 入口 ────────────────────────────────────
if __name__ == "__main__":
    try:
        num_traj = 500
        save_dir = "train_data_cv"
        os.makedirs(save_dir, exist_ok=True)
        print(f"🚀 开始生成 {num_traj} 条匀速 (CV) 单目标轨迹 [ECEF 地心坐标系]...")

        count, attempts = 0, 0
        while count < num_traj:
            attempts += 1
            lat, lon, alt, hdg, spd, tgt_pos, tgt_vel = _random_scenario()
            data = run_simulation(lat, lon, alt, hdg, spd, tgt_pos, tgt_vel)
            if data is None:
                continue
            fname = os.path.join(save_dir, f"track_data_cv_{count:04d}.mat")
            scipy.io.savemat(fname, data)
            count += 1
            if count % 50 == 0:
                print(f"  ✅ {count}/{num_traj}  (尝试 {attempts})")

        print(f"🎉 完成！数据保存于 {save_dir}/")
    except Exception:
        traceback.print_exc()
