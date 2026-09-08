# script_multitarget_ecef.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import datetime
from datetime import timedelta
import traceback
import collections
import numpy as np
import scipy.io

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.base import Property
from stonesoup.sensor.sensor import SimpleSensor

from src.utils.ecef_utils import (
    geodetic_to_ecef, ecef_to_geodetic,
    ecef_to_enu_matrix, enu_to_ecef_matrix,
    get_body_frame_axes, reconstruct_ecef_from_body_angles,
)

START_TIME = datetime.datetime.now()
DURATION = 10.0
SIM_DT = 0.001  # 物理主仿真步长: 1毫秒
FRAME_DT = 0.025  # 输出打包帧率: 25毫秒 (40Hz)
STEPS_PER_FRAME = int(FRAME_DT / SIM_DT)

SENSOR_PERIODS = {
    0: 0.025,  # Radar: 25ms
    1: 0.050,  # EO: 50ms
    2: 0.200  # ESM: 200ms
}

MODES = [
    {"a_bar": [0.0, 0.0, 0.0], "omega0": 0.0},
    {"a_bar": [-15.0, 20.0, 0.0], "omega0": 1.0},
    {"a_bar": [15.0, -20.0, 0.0], "omega0": 1.0},
    {"a_bar": [0.0, 0.0, -20.0], "omega0": 0.5},
    {"a_bar": [0.0, 0.0, 25.0], "omega0": 0.5},
    {"a_bar": [25.0, 25.0, 0.0], "omega0": 4.0},
]

transition_matrix = np.zeros((6, 6))
for i in range(1, 6): transition_matrix[i][i] = 1.0
t_prob = 0.003 / 4.0;
s_prob = 0.001 / 4.0;
d_prob = 0.0015 / 4.0
transition_matrix[0] = [1.0 - 2 * t_prob - 2 * d_prob - s_prob, t_prob, t_prob, d_prob, d_prob, s_prob]


class DatalinkNode:
    """运行在 1ms 时钟下的独立数据链节点"""

    def __init__(self, delay_s, interval_s, noise_r_m):
        self.delay_s = delay_s
        self.interval_s = interval_s
        self.noise_r = noise_r_m
        self.queue = collections.deque()

        self.nominal_tx = 0.0
        # 初始化加入随机抖动 (已注释时序偏差)
        # self.next_tx = self.nominal_tx + np.clip(np.random.randn() * 0.0005, -0.002, 0.002)
        self.next_tx = self.nominal_tx

    def step_ingest(self, current_time_s, targets_xyz_ecef_dict):
        """1ms时钟检询：是否到达发送时间"""
        if current_time_s >= self.next_tx:
            noisy_targets = {}
            for tgt_idx, xyz in targets_xyz_ecef_dict.items():
                noisy_targets[tgt_idx] = np.array(xyz, float) + np.random.randn(3) * self.noise_r

            self.queue.append({
                "arrival_time": current_time_s + self.delay_s,
                "data_age": self.delay_s,
                "targets_xyz_ecef": noisy_targets,
                "tx_time": current_time_s
            })
            # 安排下一次发送，加入时钟漂移
            self.nominal_tx += self.interval_s
            # self.next_tx = self.nominal_tx + np.clip(np.random.randn() * 0.0005, -0.002, 0.002)
            self.next_tx = self.nominal_tx

    def get_arrived_data(self, current_time_s):
        latest_data = None
        while self.queue and self.queue[0]["arrival_time"] <= current_time_s + 1e-6:
            latest_data = self.queue.popleft()
        return latest_data


class ECEFPlatform:
    def __init__(self, pos_ecef, vel_ecef, dt=0.001, phase=0.0):
        self.pos = np.array(pos_ecef, float)
        self.vel = np.array(vel_ecef, float)
        self.acc = np.zeros(3, float)  # 🆕 增加：初始化 acc 属性
        self.base_speed = np.linalg.norm(vel_ecef)
        self.dt = dt
        self.time = 0.0
        self.phase = phase

    def step(self):
        self.time += self.dt
        lat, lon, _ = ecef_to_geodetic(*self.pos)
        Re = ecef_to_enu_matrix(lat, lon)
        Rn = Re.T

        vu = Re @ self.vel
        vE, vN = vu[0], vu[1]
        sh = np.hypot(vE, vN)
        ld = np.array([-vN / sh, vE / sh, 0.0]) if sh > 1e-6 else np.array([1.0, 0.0, 0.0])

        al = 3.0 * np.sin(2 * np.pi * self.time / 20.0 + self.phase)
        av = 1.0 * np.sin(2 * np.pi * self.time / 15.0 + self.phase * 1.5)
        ae = al * ld + np.array([0.0, 0.0, av])

        cs = np.linalg.norm(self.vel)
        fe = Re @ (self.vel / max(cs, 1e-6))
        ae += 0.2 * (self.base_speed - cs) * fe

        ac = Rn @ ae
        self.acc = ac.copy()  # 🆕 增加：每步更新存储真实的 ECEF 加速度
        self.vel += ac * self.dt
        self.pos += self.vel * self.dt
        return self.pos.copy(), self.vel.copy(), self.acc.copy()


class ECEFBodySensor(SimpleSensor):
    measurement_model = Property("Measurement model")
    position_mapping = Property("Position mapping", default=(0, 2, 4))
    az_fov = Property("Azimuth FOV", default=None);
    el_fov = Property("Elevation FOV", default=None)
    max_range = Property("Max range", default=None)

    def _get_body_frame_vector(self, truth, platform_state):
        st = truth.state_vector.flatten();
        sp = platform_state.state_vector.flatten()
        rel = np.array([st[0] - sp[0], st[2] - sp[2], st[4] - sp[4]])
        fwd, right, up = get_body_frame_axes(np.array([sp[1], sp[3], sp[5]]), np.array([sp[0], sp[2], sp[4]]))
        return np.dot(rel, fwd), np.dot(rel, right), np.dot(rel, up)

    def measure(self, ground_truths, noise=True, platform_state=None, **kwargs):
        detections = []
        for truth in ground_truths:
            if not self.is_detectable(truth, platform_state): continue
            dx, dy, dz = self._get_body_frame_vector(truth, platform_state)
            relative_state = truth.state_vector.copy()
            relative_state[self.position_mapping[0]] = dx;
            relative_state[self.position_mapping[1]] = dy;
            relative_state[self.position_mapping[2]] = dz
            meas_vec = self.measurement_model.function(GroundTruthState(relative_state, timestamp=truth.timestamp),
                                                       noise=noise, **kwargs)
            detections.append(
                TrueDetection(meas_vec, measurement_model=self.measurement_model, timestamp=truth.timestamp,
                              groundtruth_path=truth))
        return detections

    def is_detectable(self, truth, platform_state):
        dx, dy, dz = self._get_body_frame_vector(truth, platform_state)
        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if r < 1e-6: return False
        az = np.arctan2(dy, dx);
        el = np.arcsin(np.clip(dz / r, -1.0, 1.0))
        if self.az_fov and not (self.az_fov[0] <= az <= self.az_fov[1]): return False
        if self.el_fov and not (self.el_fov[0] <= el <= self.el_fov[1]): return False
        if self.max_range and r > self.max_range: return False
        return True

    def is_clutter_detectable(self, clutter):
        return True


def reorder_measurement(stone_vec):
    arr = np.asarray(stone_vec).flatten()
    return float(arr[1]), float(arr[0]), float(arr[2])


class CSSWAccelModel:
    def __init__(self, a_bar, omega0=1.2, alpha=0.3, sigma=5.0, dt=0.001):
        self.a_bar = np.array(a_bar, float)
        self.omega0 = omega0;
        self.alpha = alpha;
        self.dt = dt
        # 严格数学定义：Wiener过程方差需按 sqrt(dt) 缩放。为了匹配原0.025s体感，放大基准sigma
        self.sigma = sigma * np.sqrt(0.025 / dt)
        self.z1 = np.zeros(3);
        self.z2 = np.zeros(3)

    def set_params(self, a_bar, omega0):
        self.a_bar = np.array(a_bar, float);
        self.omega0 = omega0

    def step(self):
        w = np.random.randn(3) * self.sigma * np.sqrt(self.dt)
        dz2 = (-2 * self.alpha * self.z2 - (self.alpha ** 2 + self.omega0 ** 2) * self.z1) * self.dt + w
        self.z2 += dz2;
        self.z1 += self.z2 * self.dt
        self.z1 = np.clip(self.z1, -80.0, 80.0)
        return np.clip(self.a_bar + self.z1, -150.0, 150.0)


class ManeuveringTargetECEF:
    """融入 1ms 主循环的目标实体类"""

    def __init__(self, pos_init, vel_init, start_time, dt=0.001):
        self.pos = np.array(pos_init, float)
        self.vel = np.array(vel_init, float)
        self.acc_model = CSSWAccelModel(MODES[0]["a_bar"], dt=dt)
        self.dt = dt
        self.current_time = start_time
        self.mode_idx = 0
        self.burst_timer = 0.0
        self.state = None

    def step(self):
        if self.burst_timer > 0:
            self.burst_timer -= self.dt
            if self.burst_timer <= 0: self.mode_idx = 0
        else:
            nxt = np.random.choice(6, p=transition_matrix[self.mode_idx])
            if nxt != self.mode_idx:
                self.mode_idx = nxt
                if self.mode_idx in [1, 2]:
                    self.burst_timer = np.random.uniform(3.0, 6.0)
                elif self.mode_idx in [3, 4]:
                    self.burst_timer = np.random.uniform(2.0, 3.5)
                elif self.mode_idx == 5:
                    self.burst_timer = np.random.uniform(4.0, 7.0)

        if self.mode_idx == 0:
            lat, lon, _ = ecef_to_geodetic(*self.pos)
            Re = ecef_to_enu_matrix(lat, lon)
            vU = (Re @ self.vel)[2]
            self.acc_model.set_params([0.0, 0.0, np.clip(-0.5 * vU, -10.0, 10.0)], 0.0)
        else:
            self.acc_model.set_params(**MODES[self.mode_idx])

        acc_enu = self.acc_model.step()
        Rn = ecef_to_enu_matrix(*ecef_to_geodetic(*self.pos)[:2]).T
        acc_ecef = Rn @ acc_enu

        self.vel += acc_ecef * self.dt
        
        # 柔性速度限制 (空气阻力/推力模型)，避免之前直接 clip 的生硬感
        speed = np.linalg.norm(self.vel)
        if speed > 350.0:
            self.vel -= 1.0 * (speed - 350.0) * (self.vel / speed) * self.dt
        elif speed < 150.0:
            self.vel += 1.0 * (150.0 - speed) * (self.vel / speed) * self.dt
            
        self.pos += self.vel * self.dt + 0.5 * acc_ecef * (self.dt ** 2)

        self.current_time += timedelta(seconds=self.dt)
        self.state = GroundTruthState(
            [self.pos[0], self.vel[0], self.pos[1], self.vel[1], self.pos[2], self.vel[2]],
            timestamp=self.current_time
        )
        self.state.true_acc = acc_ecef.copy()


def get_meas_array(obs_dict, dim, num_targets):
    meas_arr = np.zeros((num_targets, dim), dtype=np.float32)
    mask_arr = np.zeros((num_targets,), dtype=np.float32)
    if obs_dict:
        for tgt_idx, meas in obs_dict.items():
            if tgt_idx < num_targets:
                meas_arr[tgt_idx, :dim] = meas[:dim]
                mask_arr[tgt_idx] = 1.0
    return meas_arr.tolist(), mask_arr.tolist()


def run_simulation(seed, initial_targets, master_lat, master_lon, master_alt_m, master_hdg_deg, master_spd_mps,
                   num_targets=10):
    if seed is not None: np.random.seed(seed)

    targets = [ManeuveringTargetECEF(initial_targets[i][0::2], initial_targets[i][1::2], START_TIME, dt=SIM_DT) for i in
               range(num_targets)]

    mp = geodetic_to_ecef(master_lat, master_lon, master_alt_m)
    R_n2e = enu_to_ecef_matrix(master_lat, master_lon)
    h_rad = np.radians(master_hdg_deg)
    mv = R_n2e @ np.array([master_spd_mps * np.sin(h_rad), master_spd_mps * np.cos(h_rad), 0.0])

    wm_offsets = [np.array([0.0, 3000.0, 0.0]), np.array([0.0, -3000.0, 0.0]), np.array([3000.0, 3000.0, 0.0]),
                  np.array([3000.0, -3000.0, 0.0])]
    platforms_info = []
    cfgs = ["all", "radar_only", "radar_esm", "radar_eo", "radar_only"]

    for i, cfg in enumerate(cfgs):
        p0 = mp if i == 0 else (mp + R_n2e @ wm_offsets[i - 1])
        radar = ECEFBodySensor(measurement_model=CartesianToElevationBearingRange(ndim_state=6, mapping=(0, 2, 4),
                                                                                  noise_covar=np.diag(
                                                                                      [4e-6, 4e-6, 6400.0])),
                               az_fov=(-np.pi / 3, np.pi / 3), el_fov=(-np.radians(3), np.radians(3)),
                               max_range=200000.0)
        eo = ECEFBodySensor(measurement_model=CartesianToElevationBearingRange(ndim_state=6, mapping=(0, 2, 4),
                                                                               noise_covar=np.diag([4e-6, 4e-6, 1e20])),
                            az_fov=(-np.pi / 3, np.pi / 3), el_fov=(-np.radians(3), np.radians(3)), max_range=200000.0)
        esm = ECEFBodySensor(measurement_model=CartesianToElevationBearingRange(ndim_state=6, mapping=(0, 2, 4),
                                                                                noise_covar=np.diag(
                                                                                    [2.5e-5, 2.5e-5, 1e20])),
                             az_fov=(-np.radians(60), np.radians(60)), el_fov=(-np.radians(3), np.radians(3)),
                             max_range=200000.0)

        sensors = ({"all": [radar, eo, esm], "radar_only": [radar, None, None], "radar_esm": [radar, None, esm],
                    "radar_eo": [radar, eo, None]})[cfg]

        # 建立独立时钟与 Buffer
        nominals = {idx: 0.0 for idx in range(3)}
        # nexts = {idx: np.clip(np.random.randn() * 0.0005, -0.002, 0.002) for idx in range(3)}
        nexts = {idx: 0.0 for idx in range(3)}

        platforms_info.append({
            "id": f"plat_{i}", "motion": ECEFPlatform(p0, mv.copy(), dt=SIM_DT, phase=float(i)),
            "is_master": i == 0,
            "datalink": None if i == 0 else DatalinkNode(delay_s=0.1, interval_s=0.1, noise_r_m=100.0),
            "sensors": sensors,
            "nominal_fire": nominals, "next_fire": nexts,
            "meas_buffer": {0: None, 1: None, 2: None},  # 缓存此帧内发生的测量
            "reconstructed_ecef": {}  # 供僚机发送
        })

    total_sim_steps = int(DURATION / SIM_DT)
    total_frames = int(DURATION / FRAME_DT)
    Data_Input = np.empty((total_frames,), dtype=object)
    Data_Label = np.zeros((total_frames, num_targets, 6), dtype=np.float64)
    
    consec_invis = np.zeros(num_targets, dtype=int)

    for step in range(total_sim_steps):
        current_time_s = step * SIM_DT
        current_datetime = START_TIME + timedelta(seconds=current_time_s)

        # 1. 微步进：物理实体
        for tgt in targets: tgt.step()
        for p in platforms_info: p["motion"].step()

        # 2. 微步进：传感器独立触发
        for p in platforms_info:
            p_pos = p["motion"].pos;
            p_vel = p["motion"].vel
            p_state = GroundTruthState([p_pos[0], p_vel[0], p_pos[1], p_vel[1], p_pos[2], p_vel[2]],
                                       timestamp=current_datetime)

            for idx, sensor in enumerate(p["sensors"]):
                if sensor is None: continue
                # 如果时钟敲到了带抖动的触发点
                if current_time_s >= p["next_fire"][idx]:
                    # 取出当前 1ms 的绝对真值进行测量，并强行挂载目标索引 ID
                    tgt_states = []
                    for tgt_i, tgt in enumerate(targets):
                        tgt.state.tgt_idx = tgt_i  # 将索引绑定到状态对象上
                        tgt_states.append(tgt.state)
                        # for tgt in targets:
                        # pass
                        # # if (tgt.state.position[0] < -40000 or tgt.state.position[0] > 40000 or
                        # #     tgt.state.position[1] < -40000 or tgt.state.position[1] > 40000 or
                        # #     tgt.state.position[2] < -10000 or tgt.state.position[2] > 10000):
                        # #     all_valid = False
                        # #     breaktections = sensor.measure(tgt_states, platform_state=p_state)

                    # StoneSoup 传感器处理
                    detections = sensor.measure(tgt_states, platform_state=p_state)
                    obs_dict = {}
                    for d in detections:
                        if isinstance(d, TrueDetection):
                            tgt_idx = d.groundtruth_path.tgt_idx  # 直接读取我们刚刚挂载的索引
                            az, el, r = reorder_measurement(d.state_vector)

                            # ==================== 💥 干扰注入模块 (Data-Level) 💥 ====================
                            is_jammed_missing = False

                            # 只对主雷达 (idx == 0) 进行干扰，光电和ESM保持原样
                            # 对所有目标进行干扰
                            if idx == 0:
                                
                                # 1. 调幅干扰 (Blinking): 周期 3 秒，其中 0.2 秒处于完全盲区
                                blink_phase = current_time_s % 3.0
                                if blink_phase < 0.2:
                                    is_jammed_missing = True
                                    
                                # 2. 温和压制干扰 (Soft Suppression): 距离越远，被底噪淹没的概率越大
                                elif r > 18000.0:
                                    if np.random.rand() < 0.25:  # 25% 概率漏检
                                        is_jammed_missing = True
                                elif r > 12000.0:
                                    if np.random.rand() < 0.10:  # 10% 概率漏检
                                        is_jammed_missing = True
                                
                                # 3. 相位/距离抖动 (Range Smearing): 如果没漏检，注入距离跳变
                                if not is_jammed_missing:
                                    if np.random.rand() < 0.15:  # 15% 的回波受到距离欺骗
                                        # 额外注入标准差为 400 米的距离噪声
                                        r += np.random.normal(0.0, 400.0)

                            # =========================================================================

                            # 如果被压制或闪烁掉了，直接跳过，不录入 obs_dict
                            if is_jammed_missing:
                                continue

                            # 僚机雷达反算用于数据链
                            if idx == 0 and not p["is_master"]:
                                p["reconstructed_ecef"][tgt_idx] = reconstruct_ecef_from_body_angles(az, el, r, p_pos, p_vel)
                            
                            # 将处理（并可能注入了干扰）后的量测数据存入 obs_dict
                            obs_dict[tgt_idx] = [az, el, r]

                    if obs_dict:
                        p["meas_buffer"][idx] = {
                            "observations": obs_dict,
                            "meas_time": current_time_s  # 记录发生测量的绝对时标
                        }

                    # 安排下一次触发 (基础周期 + 新的随机抖动)
                    p["nominal_fire"][idx] += SENSOR_PERIODS[idx]
                    # p["next_fire"][idx] = p["nominal_fire"][idx] + np.clip(np.random.randn() * 0.0005, -0.002, 0.002)
                    p["next_fire"][idx] = p["nominal_fire"][idx]

            # 数据链节点跟进微步进
            if not p["is_master"]:
                if p["reconstructed_ecef"]:
                    p["datalink"].step_ingest(current_time_s, p["reconstructed_ecef"])
                    p["reconstructed_ecef"] = {}  # 发送后清空

        # 3. 宏观帧打包：按照 FRAME_DT 打包输出一次
        if step % STEPS_PER_FRAME == 0:
            frame_idx = step // STEPS_PER_FRAME

            # 记录此帧目标物理真值 (使用 float32 降维)
            for i in range(num_targets):
                st = targets[i].state.state_vector.flatten()
                Data_Label[frame_idx, i] = np.array([st[0], st[1], st[2], st[3], st[4], st[5]], dtype=np.float64)

            uav_list = np.empty((len(platforms_info),), dtype=object)
            step_visible = np.zeros(num_targets, dtype=bool)

            for u, p in enumerate(platforms_info):
                p_pos = p["motion"].pos;
                p_vel = p["motion"].vel;
                p_acc = p["motion"].acc
                lp, lop, _ = ecef_to_geodetic(*p_pos)
                vu = ecef_to_enu_matrix(lp, lop) @ p_vel
                yaw = np.arctan2(vu[0], vu[1]);
                pitch = np.arctan2(vu[2], np.hypot(vu[0], vu[1]))

                def _pop_buffer(idx, dim):
                    buf = p["meas_buffer"][idx]
                    if buf:
                        meas, mask = get_meas_array(buf["observations"], dim, num_targets)
                        m_time = buf["meas_time"]
                        p["meas_buffer"][idx] = None
                        return meas, mask, m_time
                    else:
                        return [[0.0] * dim for _ in range(num_targets)], [0.0] * num_targets, -1.0

                r_m, r_mask, r_time = _pop_buffer(0, 3)
                e_m, e_mask, e_time = _pop_buffer(1, 2)
                s_m, s_mask, s_time = _pop_buffer(2, 2)

                uav_dict = {
                    "Pose": {
                        "PositionWorld": p_pos.tolist(),
                        "EulerRPY": [0.0, float(pitch), float(yaw)],
                        "QuaternionWXYZ": [1.0, 0.0, 0.0, 0.0]
                    },
                    "IMU": {"Vector": [0.0, 0.0, 0.0, float(p_acc[0]), float(p_acc[1]), float(p_acc[2])]}
                }

                if p["is_master"]:
                    uav_dict["Radar"] = {"Meas": r_m, "Mask": r_mask, "MeasTime": max(0.0, float(r_time))}
                    uav_dict["EO"] = {"Meas": e_m, "Mask": e_mask, "MeasTime": max(0.0, float(e_time))}
                    uav_dict["ESM"] = {"Meas": s_m, "Mask": s_mask, "MeasTime": max(0.0, float(s_time)), "ModeCode": 0.0}
                    
                    # 更新主机的目标可见性
                    for t_idx in range(num_targets):
                        if r_mask[t_idx] > 0 or e_mask[t_idx] > 0 or s_mask[t_idx] > 0:
                            step_visible[t_idx] = True
                else:
                    arr_data = p["datalink"].get_arrived_data(current_time_s)
                    dl_meas = [[0.0] * 3 for _ in range(num_targets)]
                    dl_mask = [0.0] * num_targets
                    dl_age = -1.0
                    if arr_data:
                        dl_age = float(arr_data["data_age"])
                        for tgt_idx, xyz in arr_data["targets_xyz_ecef"].items():
                            if tgt_idx < num_targets:
                                dl_meas[tgt_idx] = xyz.astype(np.float32).tolist()
                                dl_mask[tgt_idx] = 1.0
                                step_visible[tgt_idx] = True # 僚机数据链收到也算可见
                    uav_dict["FastLink"] = {"Meas": dl_meas, "Age": dl_age, "IsUpdate": dl_mask}

                uav_list[u] = uav_dict

            Data_Input[frame_idx] = {"GlobalTime": float(current_time_s), "UAV": uav_list}

            # 检查可见性 (任意一个目标如果没看到，对应计数加1)
            for t_idx in range(num_targets):
                if not step_visible[t_idx]:
                    consec_invis[t_idx] += 1
                else:
                    consec_invis[t_idx] = 0

            if np.any(consec_invis > 100):
                return None

    return {"Data_Input": Data_Input, "Data_Label": Data_Label, "T": FRAME_DT}


if __name__ == "__main__":
    try:
        num_trajectories = 1
        num_targets = 5
        save_dir = "data/val_data_v8_interference"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Start generating {num_trajectories} multi-target tracks with heavy interference...")

        count, attempts = 0, 0
        while count < num_trajectories:
            attempts += 1
            if attempts > num_trajectories * 200:
                print("Too many failed attempts.")
                break
            master_lat = np.random.uniform(30.0, 50.0);
            master_lon = np.random.uniform(100.0, 130.0)
            master_alt = np.random.uniform(5000.0, 8000.0);
            master_hdg = np.random.uniform(0.0, 360.0)
            master_spd = np.random.uniform(220.0, 300.0)

            mp = geodetic_to_ecef(master_lat, master_lon, master_alt);
            R_n2e = enu_to_ecef_matrix(master_lat, master_lon)
            h = np.radians(master_hdg)
            mv = R_n2e @ np.array([master_spd * np.sin(h), master_spd * np.cos(h), 0.0])
            fwd, rgt, up = get_body_frame_axes(mv, mp)

            initial_targets = []
            for j in range(num_targets):
                rng = np.random.uniform(8000.0, 25000.0)
                az = np.random.uniform(-np.radians(45), np.radians(45));
                el = np.random.uniform(-np.radians(2), np.radians(2))
                tp = mp + rng * np.cos(el) * np.cos(az) * fwd + rng * np.cos(el) * np.sin(az) * rgt + rng * np.sin(
                    el) * up

                tl, tlo, _ = ecef_to_geodetic(*tp);
                th = np.radians(master_hdg + np.random.uniform(-108.0, 108.0))
                ts = np.random.uniform(200.0, 350.0)
                tv = enu_to_ecef_matrix(tl, tlo) @ np.array(
                    [ts * np.sin(th), ts * np.cos(th), np.random.uniform(-30.0, 30.0)])
                initial_targets.append(
                    [float(tp[0]), float(tv[0]), float(tp[1]), float(tv[1]), float(tp[2]), float(tv[2])])

            data = run_simulation(seed=count, initial_targets=initial_targets, master_lat=master_lat, master_lon=master_lon,
                                  master_alt_m=master_alt, master_hdg_deg=master_hdg, master_spd_mps=master_spd,
                                  num_targets=num_targets)
            if data is None:
                continue

            filename = os.path.join(save_dir, f"track_data_highfi_{count:04d}.mat")
            scipy.io.savemat(filename, data)
            count += 1
            print(f"  Successfully generated {count} / {num_trajectories} (attempt {attempts})...")
        print(f"Finished! Data saved in {save_dir}/.")
    except Exception:
        traceback.print_exc()