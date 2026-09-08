# script_generate_dataset_unified.py
#
# 统一数据生成脚本：训练集（many2one）和验证集（many2many）共用同一套仿真框架。
#
# 用法示例：
#   # 生成训练集（单目标，2500条，各模型625条）
#   python scripts/data_generation/script_generate_dataset_unified.py \
#       --mode train --count_per_model 625 --num_targets 1 \
#       --save_dir data/train_data/single_target_new
#
#   # 生成验证集（3目标，500条，各模型125条）
#   python scripts/data_generation/script_generate_dataset_unified.py \
#       --mode val --count_per_model 125 --num_targets 3 \
#       --save_dir data/val_data_fused_new
#
# 生成后需执行注册流程（.mat → .pkl）：
#   python scripts/pipelines/run_register_pipeline.py \
#       --in_dir data/train_data/single_target_new \
#       --out_dir data/train_data/single_target_new_registered
#
# 运动模型说明（已简化，降低机动强度以提高训练精度）：
#   CV   : 纯匀速，不变
#   CA   : 恒定加速度 ±6 m/s²（原 ±15）
#   CT   : 协同转弯，转弯率 ω ∈ [-0.08, 0.08] rad/s（原 [-0.15, 0.15]）
#   CSSW : 复合机动，最大加速度 ±40 m/s²，延长模式持续时间（原最大 ±150, 短burst）
#   目标速度范围：200-300 m/s（原 200-350 m/s）

import os
import sys
import datetime
import collections
import argparse
import traceback
from datetime import timedelta

import numpy as np
import scipy.io

# ── 路径设置 ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.detection import TrueDetection
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.base import Property
from stonesoup.sensor.sensor import SimpleSensor

from src.utils.ecef_utils import (
    geodetic_to_ecef, ecef_to_geodetic,
    ecef_to_enu_matrix, enu_to_ecef_matrix,
    get_body_frame_axes, reconstruct_ecef_from_body_angles,
)

# ── 全局仿真参数 ──────────────────────────────────────────────────────────────
START_TIME = datetime.datetime.now()
DURATION   = 10.0    # 仿真时长 (s)，与现有数据集一致
DT         = 0.1     # 时间步长 (s)

SENSOR_PERIODS = {
    0: timedelta(seconds=0.1),   # 雷达：10 Hz
    1: timedelta(seconds=0.1),   # EO：10 Hz
    2: timedelta(seconds=0.2),   # ESM：5 Hz
}

# ── 传感器类 ──────────────────────────────────────────────────────────────────
class ECEFBodySensor(SimpleSensor):
    """在体坐标系中计算相对矢量，喂给 StoneSoup 量测模型。"""
    measurement_model = Property("量测模型")
    position_mapping  = Property("位置映射", default=(0, 2, 4))
    az_fov    = Property("方位视场 (rad)", default=None)
    el_fov    = Property("俯仰视场 (rad)", default=None)
    max_range = Property("最大探测距离 (m)", default=None)

    def _body_vec(self, truth, plat_state):
        sv_t = truth.state_vector.flatten()
        sv_p = plat_state.state_vector.flatten()
        rel  = np.array([sv_t[0]-sv_p[0], sv_t[2]-sv_p[2], sv_t[4]-sv_p[4]])
        vel_p = np.array([sv_p[1], sv_p[3], sv_p[5]])
        pos_p = np.array([sv_p[0], sv_p[2], sv_p[4]])
        fwd, rgt, up = get_body_frame_axes(vel_p, pos_p)
        return np.dot(rel, fwd), np.dot(rel, rgt), np.dot(rel, up)

    def measure(self, ground_truths, noise=True, platform_state=None, **kw):
        dets = []
        for truth in ground_truths:
            if not self.is_detectable(truth, platform_state):
                continue
            dx, dy, dz = self._body_vec(truth, platform_state)
            rs = truth.state_vector.copy()
            rs[self.position_mapping[0]] = dx
            rs[self.position_mapping[1]] = dy
            rs[self.position_mapping[2]] = dz
            mv = self.measurement_model.function(
                GroundTruthState(rs, timestamp=truth.timestamp), noise=noise, **kw)
            dets.append(TrueDetection(mv, measurement_model=self.measurement_model,
                                      timestamp=truth.timestamp, groundtruth_path=truth))
        return dets

    def is_detectable(self, truth, plat_state):
        dx, dy, dz = self._body_vec(truth, plat_state)
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        if r < 1e-6:
            return False
        az = np.arctan2(dy, dx)
        el = np.arcsin(np.clip(dz / r, -1.0, 1.0))
        if self.az_fov  and not (self.az_fov[0]  <= az <= self.az_fov[1]):  return False
        if self.el_fov  and not (self.el_fov[0]  <= el <= self.el_fov[1]):  return False
        if self.max_range and r > self.max_range: return False
        return True

    def is_clutter_detectable(self, c):
        return True


def _ro(sv):
    """StoneSoup (el, az, r) → (az, el, r)"""
    a = np.asarray(sv).flatten()
    return float(a[1]), float(a[0]), float(a[2])


def _make_sensors():
    """构造雷达 / EO / ESM 三个传感器（参数与 script_single_target_fused 一致）。"""
    radar = ECEFBodySensor(
        measurement_model=CartesianToElevationBearingRange(
            ndim_state=6, mapping=(0, 2, 4),
            noise_covar=np.diag([0.002**2, 0.002**2, 48.0**2])),
        az_fov=(-np.pi/3, np.pi/3),
        el_fov=(-np.radians(3), np.radians(3)),
        max_range=200_000.0,
    )
    eo = ECEFBodySensor(
        measurement_model=CartesianToElevationBearingRange(
            ndim_state=6, mapping=(0, 2, 4),
            noise_covar=np.diag([0.002**2, 0.002**2, 1e20])),
        az_fov=(-np.pi/3, np.pi/3),
        el_fov=(-np.radians(3), np.radians(3)),
        max_range=200_000.0,
    )
    esm = ECEFBodySensor(
        measurement_model=CartesianToElevationBearingRange(
            ndim_state=6, mapping=(0, 2, 4),
            noise_covar=np.diag([0.005**2, 0.005**2, 1e20])),
        az_fov=(-np.radians(60), np.radians(60)),
        el_fov=(-np.radians(3), np.radians(3)),
        max_range=200_000.0,
    )
    return radar, eo, esm


# ── 数据链节点 ────────────────────────────────────────────────────────────────
class DatalinkNode:
    """模拟僚机数据链：0.1s 延迟 + 20m 传输噪声 + 零阶保持。"""
    def __init__(self, delay_s=0.1, noise_m=20.0):
        self.delay_s  = delay_s
        self.noise_m  = noise_m
        self.queue    = collections.deque()
        self.last_val = None

    def ingest(self, t_s, xyz_fused):
        noisy = np.array(xyz_fused, float) + np.random.randn(3) * self.noise_m
        self.queue.append({"arrival": t_s + self.delay_s, "xyz": noisy})

    def pop(self, t_s):
        latest = None
        while self.queue and self.queue[0]["arrival"] <= t_s + 1e-6:
            latest = self.queue.popleft()
        if latest is not None:
            self.last_val = latest["xyz"]
        return self.last_val


# ── 僚机单步融合 ──────────────────────────────────────────────────────────────
_W_RADAR_AZ = 1.0 / (0.002 ** 2)
_W_RADAR_EL = 1.0 / (0.002 ** 2)
_W_EO_AZ    = 1.0 / (0.002 ** 2)
_W_EO_EL    = 1.0 / (0.002 ** 2)
_W_ESM_AZ   = 1.0 / (0.005 ** 2)


def _fuse_step(radar_meas, eo_meas, esm_meas, p_pos, p_vel):
    """多传感器角度融合 → ECEF XYZ。无雷达时返回 (None, False)。"""
    if radar_meas is None:
        return None, False
    az_radar, el_radar, r_radar = radar_meas
    az_fused = az_radar * _W_RADAR_AZ
    el_fused = el_radar * _W_RADAR_EL
    sum_w_az = _W_RADAR_AZ
    sum_w_el = _W_RADAR_EL
    if eo_meas is not None:
        az_eo, el_eo = eo_meas
        az_fused += az_eo * _W_EO_AZ
        el_fused += el_eo * _W_EO_EL
        sum_w_az += _W_EO_AZ
        sum_w_el += _W_EO_EL
    if esm_meas is not None:
        az_fused += esm_meas[0] * _W_ESM_AZ
        sum_w_az += _W_ESM_AZ
    az_fused /= sum_w_az
    el_fused /= sum_w_el
    xyz = reconstruct_ecef_from_body_angles(az_fused, el_fused, r_radar, p_pos, p_vel)
    return xyz, True


# ── UAV 平台（轻微机动飞行） ──────────────────────────────────────────────────
class ECEFPlatform:
    """在 ECEF 坐标系中积分的平台（含轻微周期性机动）。"""
    def __init__(self, pos_ecef, vel_ecef, dt=0.1, phase=0.0):
        self.pos = np.array(pos_ecef, float)
        self.vel = np.array(vel_ecef, float)
        self.base_speed = np.linalg.norm(vel_ecef)
        self.dt   = dt
        self.time = 0.0
        self.phase = phase

    def step(self):
        self.time += self.dt
        lat, lon, _ = ecef_to_geodetic(*self.pos)
        Re = ecef_to_enu_matrix(lat, lon)
        Rn = Re.T
        vel_enu = Re @ self.vel
        vE, vN = vel_enu[0], vel_enu[1]
        sh = np.hypot(vE, vN)
        ld = np.array([-vN/sh, vE/sh, 0.0]) if sh > 1e-6 else np.array([1., 0., 0.])
        al = 3.0 * np.sin(2*np.pi*self.time/20.0 + self.phase)
        av = 1.0 * np.sin(2*np.pi*self.time/15.0 + self.phase*1.5)
        ae = al*ld + np.array([0., 0., av])
        cs = np.linalg.norm(self.vel)
        fe = Re @ (self.vel / max(cs, 1e-6))
        ae += 0.2 * (self.base_speed - cs) * fe
        ac = Rn @ ae
        self.vel += ac * self.dt
        self.pos += self.vel * self.dt
        return self.pos.copy(), self.vel.copy(), ac.copy()


# ── 目标运动模型（简化版） ────────────────────────────────────────────────────
def generate_cv_target_ecef(pos0, vel0, steps, dt):
    """匀速 (CV)：ECEF 中匀速积分，无改动。"""
    pos = np.array(pos0, float)
    vel = np.array(vel0, float)
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


def generate_ca_target_ecef(pos0, vel0, steps, dt):
    """匀加速 (CA)：简化为 ±6 m/s²（原 ±15），减少激烈机动。"""
    pos = np.array(pos0, float)
    vel = np.array(vel0, float)
    acc = np.array([
        np.random.uniform(-6., 6.),
        np.random.uniform(-6., 6.),
        np.random.uniform(-2., 2.),   # 垂直方向更小
    ])
    states = []
    t = START_TIME
    for _ in range(steps):
        vel = vel + acc * dt
        pos = pos + vel * dt + 0.5 * acc * dt**2
        sv  = np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]])
        st  = GroundTruthState(sv, timestamp=t)
        st.true_acc = acc.copy()
        states.append(st)
        t += timedelta(seconds=dt)
    return states


def generate_ct_target_ecef(pos0, vel0, steps, dt):
    """协同转弯 (CT)：转弯率 ω ∈ [-0.08, 0.08] rad/s（原 [-0.15, 0.15]）。"""
    pos = np.array(pos0, float)
    vel = np.array(vel0, float)
    omega = np.random.uniform(-0.08, 0.08)
    if abs(omega) < 0.02:                          # 最小转弯率保证轨迹有弧度
        omega = 0.02 * (1 if omega >= 0 else -1)
    states = []
    t = START_TIME
    for _ in range(steps):
        lat, lon, _ = ecef_to_geodetic(*pos)
        Re = ecef_to_enu_matrix(lat, lon)
        Rn = Re.T
        vel_enu = Re @ vel
        vE, vN, vU = vel_enu
        sin_o = np.sin(omega*dt)
        cos_o = np.cos(omega*dt)
        vE_new =  vE*cos_o - vN*sin_o
        vN_new =  vE*sin_o + vN*cos_o
        if abs(omega) > 1e-8:
            pE_new =  vE*sin_o/omega - vN*(1-cos_o)/omega
            pN_new =  vE*(1-cos_o)/omega + vN*sin_o/omega
        else:
            pE_new = vE*dt
            pN_new = vN*dt
        pU_new = vU * dt
        pos = pos + Rn @ np.array([pE_new, pN_new, pU_new])
        vel = Rn @ np.array([vE_new, vN_new, vU])
        acc_ecef = Rn @ np.array([-omega*vN, omega*vE, 0.0])
        sv = np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]])
        st = GroundTruthState(sv, timestamp=t)
        st.true_acc = acc_ecef
        states.append(st)
        t += timedelta(seconds=dt)
    return states


# CSSW 模式定义（加速度上限降至 ±40 m/s²，增大转移到 CV 模式的概率）
CSSW_MODES = [
    {"a_bar": [ 0.0,  0.0,  0.0], "omega0": 0.0},   # 模式 0：CV
    {"a_bar": [-8.,   12.,  0.0], "omega0": 0.6},    # 模式 1：左转（原 -15/20，简化）
    {"a_bar": [ 8.,  -12.,  0.0], "omega0": 0.6},    # 模式 2：右转
    {"a_bar": [ 0.0,  0.0, -10.], "omega0": 0.3},    # 模式 3：俯冲（原 -20）
    {"a_bar": [ 0.0,  0.0,  12.], "omega0": 0.3},    # 模式 4：爬升（原 25）
    {"a_bar": [ 12.,  12.,  0.0], "omega0": 2.0},    # 模式 5：蛇形（原 25/25，omega0=4）
]
# 转移矩阵：增大停留在 CV 模式（行0）的概率
_T_MAT = np.zeros((6, 6))
for _i in range(1, 6):
    _T_MAT[_i][_i] = 1.0
_T_MAT[0] = [0.97, 0.008, 0.008, 0.004, 0.004, 0.006]   # 原 [0.96, 0.012, ...]


class CSSWAccelModel:
    def __init__(self, a_bar, omega0=0.6, alpha=0.3, sigma=3.0, dt=0.1):
        # sigma 降为 3.0（原 5.0），限制随机波动幅度
        self.a_bar  = np.array(a_bar, float)
        self.omega0 = omega0
        self.alpha  = alpha
        self.sigma  = sigma
        self.dt     = dt
        self.z1 = np.zeros(3)
        self.z2 = np.zeros(3)

    def set(self, a_bar, omega0):
        self.a_bar  = np.array(a_bar, float)
        self.omega0 = omega0

    def step(self):
        w   = np.random.randn(3) * self.sigma
        dz2 = (-2*self.alpha*self.z2 - (self.alpha**2 + self.omega0**2)*self.z1 + w)
        self.z2 += dz2 * self.dt
        self.z1 += self.z2 * self.dt
        self.z1 = np.clip(self.z1, -40., 40.)          # 原 ±80，简化为 ±40
        return np.clip(self.a_bar + self.z1, -40., 40.) # 原 ±150，简化为 ±40


def generate_cssw_target_ecef(pos0, vel0, steps, dt):
    """CS-SW 复合机动（简化版）：
    - 机动幅度上限 ±40 m/s²（原 ±150）
    - 模式切换 burst 时长延长（更平滑）
    - 转移矩阵倾向 CV（0.97，原 0.96）
    """
    pos = np.array(pos0, float)
    vel = np.array(vel0, float)
    acc_model = CSSWAccelModel(CSSW_MODES[0]["a_bar"], dt=dt)
    states = []
    t = START_TIME
    mode_idx = 0
    burst = 0.
    maneuvered = False

    for si in range(steps):
        if burst > 0:
            burst -= dt
            if burst <= 0:
                mode_idx = 0
        else:
            probs = _T_MAT[mode_idx]
            nxt = np.random.choice(6, p=probs)
            # 保证在轨迹中段至少有一次机动
            if not maneuvered and si >= 30 and mode_idx == 0:
                nxt = np.random.randint(1, 6)
            if nxt != mode_idx:
                mode_idx  = nxt
                maneuvered = True
                # 延长 burst：更平滑的机动（原 3-6s，现 4-8s）
                if mode_idx in [1, 2]:   burst = np.random.uniform(4., 8.)
                elif mode_idx in [3, 4]: burst = np.random.uniform(3., 5.)
                elif mode_idx == 5:      burst = np.random.uniform(4., 7.)

        if mode_idx == 0:
            lat, lon, _ = ecef_to_geodetic(*pos)
            vU = (ecef_to_enu_matrix(lat, lon) @ vel)[2]
            acc_model.set([0., 0., float(np.clip(-0.5*vU, -8., 8.))], 0.)
        else:
            acc_model.set(**CSSW_MODES[mode_idx])

        acc_enu = acc_model.step()
        lat, lon, _ = ecef_to_geodetic(*pos)
        Re = ecef_to_enu_matrix(lat, lon)
        Rn = Re.T
        acc_ecef = Rn @ acc_enu
        vel = vel + acc_ecef * dt
        pos = pos + vel * dt + 0.5 * acc_ecef * dt**2
        sv = np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]])
        st = GroundTruthState(sv, timestamp=t)
        st.true_acc = acc_ecef
        states.append(st)
        t += timedelta(seconds=dt)
    return states


# ── 随机场景初始化 ─────────────────────────────────────────────────────────────
def _random_scenario():
    """随机生成主机（UAV编队长机）的初始位置/航向/速度。"""
    lat = np.random.uniform(30., 50.)
    lon = np.random.uniform(100., 130.)
    alt = np.random.uniform(5000., 8000.)
    hdg = np.random.uniform(0., 360.)
    spd = np.random.uniform(220., 300.)
    mp  = geodetic_to_ecef(lat, lon, alt)
    R   = enu_to_ecef_matrix(lat, lon)
    h   = np.radians(hdg)
    mv  = R @ np.array([spd*np.sin(h), spd*np.cos(h), 0.])
    fwd, rgt, up = get_body_frame_axes(mv, mp)
    return lat, lon, alt, hdg, spd, mp, mv, fwd, rgt, up


def _random_targets(mp, mv, fwd, rgt, up, lat, lon, hdg, num_targets):
    """在长机前方随机生成 num_targets 个目标的初始状态。
    目标速度范围：200-300 m/s（简化，缩小范围）。
    """
    initial_targets = []
    for _ in range(num_targets):
        rng = np.random.uniform(5000., 10000.)
        az  = np.random.uniform(-np.radians(45), np.radians(45))
        el  = np.random.uniform(-np.radians(2),  np.radians(2))
        tp  = mp + rng*np.cos(el)*np.cos(az)*fwd \
                 + rng*np.cos(el)*np.sin(az)*rgt \
                 + rng*np.sin(el)*up
        tl, tlo, _ = ecef_to_geodetic(*tp)
        th  = np.radians(hdg + np.random.uniform(-90., 90.))   # 原 ±108°，收窄为 ±90°
        ts  = np.random.uniform(200., 300.)                     # 目标速度（简化：200-300 m/s）
        tv  = enu_to_ecef_matrix(tl, tlo) @ np.array([
            ts*np.sin(th), ts*np.cos(th), np.random.uniform(-20., 20.)  # 垂直速度 ±20（原 ±30）
        ])
        initial_targets.append((tp, tv))
    return initial_targets


# ── 主仿真函数 ─────────────────────────────────────────────────────────────────
def run_simulation(master_lat, master_lon, master_alt_m,
                   master_heading_deg, master_speed_mps,
                   mp, mv,
                   initial_targets, motion_model="cv"):
    """
    运行一次多目标仿真，返回 .mat 格式字典（含 Data_Input/Data_Label/truth）。
    支持 num_targets=1（many2one 训练集格式）和 num_targets>1（many2many 验证集格式）。
    """
    steps       = int(DURATION / DT)
    num_targets = len(initial_targets)

    # 1. 生成目标轨迹
    _GENERATORS = {
        "cv":   generate_cv_target_ecef,
        "ca":   generate_ca_target_ecef,
        "ct":   generate_ct_target_ecef,
        "cssw": generate_cssw_target_ecef,
    }
    gen_fn = _GENERATORS[motion_model]
    truths = [gen_fn(tp, tv, steps, DT) for tp, tv in initial_targets]

    # 2. 构建 UAV 编队（长机 + 4 架僚机）
    # 使用 V 型/伴飞阵型，沿相对机体坐标系排列，确保全部对准前方且高度差极小
    fwd, rgt, up = get_body_frame_axes(mv, mp)
    wm_pos = []
    for i in range(4):
        # 长机后方 1000~3000m
        dx = -np.random.uniform(1000., 3000.)
        # 左右展开 1000~4000m
        dy = np.random.uniform(1000., 4000.) * (-1 if i % 2 == 0 else 1)
        # 高度差控制在 ±200m 以内，防止目标脱离 ±3° 俯仰视场
        dz = np.random.uniform(-200., 200.)
        wm_pos.append(mp + dx * fwd + dy * rgt + dz * up)

    # 僚机传感器配置（4种组合覆盖多元传感器场景）
    sensor_cfgs = ["all", "radar_only", "radar_eo", "radar_esm"]
    platforms = []

    # 长机（u=0）：全传感器，有 Radar/EO/ESM，无 FastLink
    r0, eo0, esm0 = _make_sensors()
    platforms.append({
        "motion":    ECEFPlatform(mp, mv.copy(), dt=DT, phase=0.0),
        "is_master": True,
        "sensors":   [r0, eo0, esm0],
        "last_meas": {idx: START_TIME - p for idx, p in SENSOR_PERIODS.items()},
        "datalink":  None,
    })

    # 僚机（u=1..4）：各自不同传感器组合，有 FastLink（僚机融合后发给长机）
    for i, cfg in enumerate(sensor_cfgs):
        r, eo, esm = _make_sensors()
        if   cfg == "all":       sens = [r, eo, esm]
        elif cfg == "radar_only": sens = [r, None, None]
        elif cfg == "radar_eo":   sens = [r, eo, None]
        elif cfg == "radar_esm":  sens = [r, None, esm]
        else:                     sens = [None, None, None]
        platforms.append({
            "motion":    ECEFPlatform(wm_pos[i], mv.copy(), dt=DT, phase=float(i+1)),
            "is_master": False,
            "sensors":   sens,
            "last_meas": {idx: START_TIME - p for idx, p in SENSOR_PERIODS.items()},
            # 每个僚机为每个目标维护一条数据链队列
            "datalink":  [DatalinkNode(delay_s=0.1, noise_m=60.0) for _ in range(num_targets)],
        })

    # 3. 逐帧仿真
    T = len(truths[0])
    Data_Input  = np.empty((T,), dtype=object)
    Data_Label  = np.zeros((T, num_targets, 6), dtype=np.float64)
    consec_invis = 0

    for t in range(T):
        current_time = truths[0][t].timestamp
        t_s = float((current_time - START_TIME).total_seconds())

        for k in range(num_targets):
            sv = truths[k][t].state_vector.flatten()
            Data_Label[t, k] = sv[[0, 1, 2, 3, 4, 5]]

        uav_list    = np.empty((len(platforms),), dtype=object)
        visible_targets = set()

        for u, plat in enumerate(platforms):
            p_pos, p_vel, p_acc = plat["motion"].step()
            plat_sv    = np.array([p_pos[0], p_vel[0], p_pos[1], p_vel[1], p_pos[2], p_vel[2]])
            plat_state = GroundTruthState(plat_sv, timestamp=current_time)

            # 各传感器采集
            obs_dict = collections.defaultdict(dict)
            for idx, sensor in enumerate(plat["sensors"]):
                if sensor is None:
                    continue
                if current_time - plat["last_meas"][idx] >= SENSOR_PERIODS[idx] - timedelta(microseconds=10):
                    dets = sensor.measure([tr[t] for tr in truths], platform_state=plat_state)
                    if dets:
                        plat["last_meas"][idx] = current_time
                    for d in dets:
                        # 长机 15% 丢包（模拟干扰）
                        if plat["is_master"] and np.random.rand() < 0.15:
                            continue
                        tgt_id = [tr[t] for tr in truths].index(d.groundtruth_path)
                        az, el, r = _ro(d.state_vector)
                        obs_dict[tgt_id][idx] = [az, el, r]

            lat_p, lon_p, _ = ecef_to_geodetic(*p_pos)
            vu    = ecef_to_enu_matrix(lat_p, lon_p) @ p_vel
            yaw   = np.arctan2(vu[0], vu[1])
            pitch = np.arctan2(vu[2], np.hypot(vu[0], vu[1]))

            base_uav = {
                "Pose": {
                    "PositionWorld": p_pos.tolist(),
                    "VelocityWorld": p_vel.tolist(),
                    "EulerRPY":      [0., float(pitch), float(yaw)],
                    "QuaternionWXYZ":[1., 0., 0., 0.],
                },
                "IMU": {"Vector": [0., 0., 0., float(p_acc[0]), float(p_acc[1]), float(p_acc[2])]},
            }

            # 传感器量测打包（列表顺序 = 目标序号）
            radar_meas, radar_mask = [], []
            eo_meas,    eo_mask    = [], []
            esm_meas,   esm_mask   = [], []
            for k in range(num_targets):
                if 0 in obs_dict[k]:
                    radar_meas.append(obs_dict[k][0]); radar_mask.append(1); visible_targets.add(k)
                else:
                    radar_meas.append([0., 0., 0.]); radar_mask.append(0)
                if 1 in obs_dict[k]:
                    eo_meas.append(obs_dict[k][1][:2]); eo_mask.append(1)
                else:
                    eo_meas.append([0., 0.]); eo_mask.append(0)
                if 2 in obs_dict[k]:
                    esm_meas.append([obs_dict[k][2][0]]); esm_mask.append(1)
                else:
                    esm_meas.append([0.]); esm_mask.append(0)

            base_uav["Radar"] = {"Meas": radar_meas, "Mask": radar_mask}
            base_uav["EO"]    = {"Meas": eo_meas,    "Mask": eo_mask}
            base_uav["ESM"]   = {"Meas": esm_meas,   "Mask": esm_mask, "ModeCode": [0]*num_targets}

            if plat["is_master"]:
                base_uav["FastLink"] = {"Meas": [[0.,0.,0.]]*num_targets,
                                        "Age": [0]*num_targets, "IsUpdate": [0]*num_targets}
                base_uav["SlowLink"] = {"Meas": [[0.,0.,0.]]*num_targets,
                                        "Age": [0]*num_targets, "IsUpdate": [0]*num_targets}
            else:
                dl_meas, dl_isupdate = [], []
                for k in range(num_targets):
                    rm  = obs_dict[k].get(0)
                    em  = obs_dict[k].get(1)
                    em2 = em[:2] if em else None
                    sm  = obs_dict[k].get(2)
                    sm1 = [sm[0]] if sm else None
                    xyz_fused, valid = _fuse_step(rm, em2, sm1, p_pos, p_vel)
                    if valid:
                        plat["datalink"][k].ingest(t_s, xyz_fused)
                    dl_xyz = plat["datalink"][k].pop(t_s)
                    if dl_xyz is not None:
                        dl_meas.append(dl_xyz.tolist()); dl_isupdate.append(1)
                    else:
                        dl_meas.append([0., 0., 0.]); dl_isupdate.append(0)
                base_uav["FastLink"] = {"Meas": dl_meas, "Age": [0]*num_targets, "IsUpdate": dl_isupdate}

            uav_list[u] = base_uav

        Data_Input[t] = {"GlobalTime": t_s, "UAV": uav_list}

        if len(visible_targets) < num_targets:
            return None    # 全段严格可见性检查：任何目标在该帧丢失任何观测即丢弃该场景

    # 4. 生成真值段（用于 pipeline_core 配准）
    radar_seg, eo_seg, esm_seg = [], [], []
    for k in range(num_targets):
        for u_id in range(len(platforms)):
            radar_seg.append([k+1, k+1, 0, T*DT, u_id+1, k+1])
            eo_seg.append(   [k+1, k+1, 0, T*DT, u_id+1, k+1])
            esm_seg.append(  [k+1, k+1, 0, T*DT, u_id+1, k+1])

    t_grid = np.arange(T, dtype=np.float64) * DT   # 时间网格 [0.0, 0.1, ..., 9.9]
    truth_dict = {
        "N_true":   num_targets,
        "t_grid":   t_grid,                          # pipeline_core.py 必需字段
        "RadarSeg": radar_seg,
        "EOSeg":    eo_seg,
        "ESMSeg":   esm_seg,
    }
    return {"Data_Input": Data_Input, "Data_Label": Data_Label, "truth": truth_dict, "T": DT}


# ── 入口 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="统一数据生成脚本，支持训练集（many2one）和验证集（many2many）。"
    )
    parser.add_argument("--mode",           type=str, default="train",
                        choices=["train", "val"],
                        help="train=单目标训练集，val=多目标验证集")
    parser.add_argument("--num_targets",    type=int, default=None,
                        help="每场景目标数（默认：train→1，val→3）")
    parser.add_argument("--count_per_model",type=int, default=625,
                        help="每种运动模型生成的场景数（默认：625→共2500）")
    parser.add_argument("--save_dir",       type=str, default=None,
                        help="保存目录（默认：train→data/train_data/single_target_new，"
                             "val→data/val_data_fused_new）")
    parser.add_argument("--seed",           type=int, default=None,
                        help="随机种子（可选）")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # 默认值
    if args.num_targets is None:
        args.num_targets = 1 if args.mode == "train" else 3
    if args.save_dir is None:
        args.save_dir = ("data/train_data/single_target_new" if args.mode == "train"
                         else "data/val_data_fused_new")

    os.makedirs(args.save_dir, exist_ok=True)

    CONFIGS = [
        ("cv",   args.count_per_model),
        ("ca",   args.count_per_model),
        ("ct",   args.count_per_model),
        ("cssw", args.count_per_model),
    ]
    total_scenes = sum(n for _, n in CONFIGS)

    print(f"[Info] 模式: {args.mode.upper()}")
    print(f"[Info] 目标数/场景: {args.num_targets}")
    print(f"[Info] 运动模型: CV/CA/CT/CSSW，各 {args.count_per_model} 条")
    print(f"[Info] 总场景数: {total_scenes}")
    print(f"[Info] 保存目录: {os.path.abspath(args.save_dir)}")
    print(f"[Info] 注意：生成后需运行 scripts/pipelines/run_register_pipeline.py 进行注册")
    print()

    total_done = 0
    for motion_model, num_traj in CONFIGS:
        print(f"==> 生成 {num_traj} 条 [{motion_model.upper()}] 场景 ...")
        count    = 0
        attempts = 0
        while count < num_traj:
            attempts += 1
            try:
                lat, lon, alt, hdg, spd, mp, mv, fwd, rgt, up = _random_scenario()
                initial_targets = _random_targets(mp, mv, fwd, rgt, up, lat, lon, hdg,
                                                  args.num_targets)
                data = run_simulation(lat, lon, alt, hdg, spd, mp, mv,
                                      initial_targets, motion_model)
            except Exception:
                traceback.print_exc()
                data = None

            if data is None:
                continue

            fname = os.path.join(args.save_dir, f"track_{motion_model}_{count:04d}.mat")
            scipy.io.savemat(fname, data)
            count     += 1
            total_done += 1

            if count % 50 == 0 or count == num_traj:
                print(f"    [{motion_model.upper()}] {count}/{num_traj}  (尝试 {attempts} 次)")

        print(f"  [{motion_model.upper()}] 完成 {count} 条 (尝试 {attempts} 次)")

    print(f"\n全部完成！共生成 {total_done} 个场景，保存于 {os.path.abspath(args.save_dir)}/")
    print(f"\n下一步：运行注册流程")
    print(f"  python scripts/pipelines/run_register_pipeline.py \\")
    print(f"      --in_dir {args.save_dir} \\")
    print(f"      --out_dir {args.save_dir}_registered")


if __name__ == "__main__":
    main()
