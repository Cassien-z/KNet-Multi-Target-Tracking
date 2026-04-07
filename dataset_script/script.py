# script_final_stonesoup_v1_4_fixed.py
import datetime
from datetime import timedelta
import json
import traceback

# StoneSoup core (ensure stonesoup is installed in your env)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection, Clutter
from stonesoup.platform.base import MovingPlatform
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.base import Property
from stonesoup.sensor.sensor import SimpleSensor

import numpy as np

import numpy as np
from datetime import timedelta
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection

import collections


class DatalinkNode:
    """数据链模拟节点（部署在僚机上）"""

    def __init__(self, delay_s, interval_s, noise_ang_rad, noise_r_m):
        self.delay_s = delay_s  # 传输延迟 (例如 1.0秒)
        self.interval_s = interval_s  # 更新周期 (例如 1.0秒)
        self.noise_ang = noise_ang_rad  # 角度退化精度 (例如 0.010 rad)
        self.noise_r = noise_r_m  # 距离退化精度 (例如 100 m)

        self.queue = collections.deque()  # 延迟缓冲区
        self.last_tx_time = -999.0  # 上次发送时间

    def ingest_data(self, current_time_s, platform_state, sensor_outputs):
        """僚机产生观测后，送入数据链"""
        # 1. 模拟更新周期 (降采样)
        if current_time_s - self.last_tx_time >= self.interval_s - 1e-5:
            # 2. 注入数据链精度退化噪声 (仅对观测值，不破坏平台真实位置记录)
            degraded_sensors = {}
            for s_name, s_data in sensor_outputs.items():
                degraded_obs = []
                for obs in s_data["observations"]:
                    az, el, r = obs["meas"]
                    # 叠加上数据链的额外高斯噪声
                    noisy_az = az + np.random.randn() * self.noise_ang
                    noisy_el = el + np.random.randn() * self.noise_ang
                    noisy_r = r + np.random.randn() * self.noise_r
                    degraded_obs.append({"meas": [noisy_az, noisy_el, noisy_r], "type": obs["type"]})
                degraded_sensors[s_name] = {"observations": degraded_obs}

            # 3. 打包放入队列，标记“到达长机的时间”
            arrival_time = current_time_s + self.delay_s
            self.queue.append({
                "arrival_time": arrival_time,
                "data_age": self.delay_s,
                "platform_state": platform_state.copy(),  # 必须记录僚机当时的物理坐标
                "sensors": degraded_sensors
            })
            self.last_tx_time = current_time_s

    def get_arrived_data(self, master_time_s):
        """长机提取当前已到达的最新数据"""
        latest_data = None
        # 把所有早于等于长机当前时间的数据包都拆开，只保留最新的一个
        while self.queue and self.queue[0]["arrival_time"] <= master_time_s + 1e-5:
            latest_data = self.queue.popleft()

        # 注意：由于是按周期更新，如果这帧没有新数据到达，长机应该保持上一次的数据（零阶保持）
        # 在实际代码里，我们可以返回最新数据，让外部主循环决定是否缓存。
        return latest_data

class CSSWAccelerationModel:
    """
    CS-SW 复合机动加速度模型 (重构版)
    a(t) = a_bar + a_tilde(t)
    """

    def __init__(self, a_bar, sigma_a=5.0, alpha=0.3, omega0=1.0, dt=0.1):
        self.a_bar = np.array(a_bar, dtype=float)
        self.sigma_a = sigma_a
        self.alpha = alpha
        self.omega0 = omega0
        self.dt = dt

        self.jump_prob = 0.005  # 降低单步突变概率，避免过于频繁 (0.1秒步长下约20秒触发一次)
        self.jump_scale = 15.0  # 突变幅度

        # SW 内部状态（等效二阶系统）
        self.z1 = np.zeros(3)
        self.z2 = np.zeros(3)

    def set_params(self, a_bar, omega0):
        """用于 IMM 模式切换时更新意图参数"""
        self.a_bar = np.array(a_bar, dtype=float)
        self.omega0 = omega0

    def step(self):
        # -------- 意图突变触发 --------
        # 修复：突变现在直接作用于 a_bar (长期意图)，模拟飞行员猛拉杆
        if np.random.rand() < self.jump_prob:
            jump = np.random.randn(3) * self.jump_scale
            self.a_bar += jump

        w = np.random.randn(3) * self.sigma_a
        dz1 = self.z2
        dz2 = (
                -2 * self.alpha * self.z2
                - (self.alpha ** 2 + self.omega0 ** 2) * self.z1
                + w
        )

        self.z1 += dz1 * self.dt
        self.z2 += dz2 * self.dt

        return self.a_bar + self.z1


class KinematicMotionModel:
    """
    标准运动学模型 (替代原有的错误 3DVT)
    既然 CSSW 已经生成了总加速度，直接进行二次积分即可保证物理连贯性。
    """

    def __init__(self, dt=0.1):
        self.dt = dt

    def step(self, pos, vel, acc):
        # 修复：标准匀加速运动学积分，避免加速度被重复计算
        vel_next = vel + acc * self.dt
        pos_next = pos + vel * self.dt + 0.5 * acc * (self.dt ** 2)
        return pos_next, vel_next


class ManeuveringTarget3D:
    """
    统一的目标实体，管理唯一的位置和速度状态
    """

    def __init__(self, init_pos, init_vel, init_a_bar, dt=0.1, omega0=1.2):
        self.pos = np.array(init_pos, dtype=float)
        self.vel = np.array(init_vel, dtype=float)

        self.acc_model = CSSWAccelerationModel(
            a_bar=init_a_bar, dt=dt, omega0=omega0
        )
        self.motion_model = KinematicMotionModel(dt=dt)

    def switch_mode(self, a_bar, omega0):
        """平滑切换机动模式，只改参数，不改物理坐标"""
        self.acc_model.set_params(a_bar, omega0)

    def step(self):
        acc = self.acc_model.step()
        self.pos, self.vel = self.motion_model.step(self.pos, self.vel, acc)
        return self.pos, self.vel, acc


class GuidedPlatformModel:
    """
    闭环引导平台模型：始终尝试指向并接近目标
    """

    def __init__(self, init_pos, init_vel, dt=0.1, max_acc=20.0, gain=0.5):
        self.pos = np.array(init_pos, dtype=float)
        self.vel = np.array(init_vel, dtype=float)
        self.dt = dt
        self.max_acc = max_acc  # 平台机动能力限制
        self.gain = gain  # 响应灵敏度系数

    def step(self, target_pos):
        # 1. 计算指向目标的期望速度方向
        rel_pos = np.array(target_pos) - self.pos
        dist = np.linalg.norm(rel_pos)

        # 如果距离太近，可以停止加速以避免碰撞
        if dist < 100:
            desired_vel = self.vel
        else:
            # 期望速度方向指向目标，速率保持不变
            desired_vel = (rel_pos / dist) * np.linalg.norm(self.vel)

        # 2. 计算需要的加速度 (反馈控制)
        # a = k * (v_desired - v_current)
        acc = self.gain * (desired_vel - self.vel) / self.dt

        # 3. 物理限制：限制平台最大过载，防止“瞬移”
        acc_mag = np.linalg.norm(acc)
        if acc_mag > self.max_acc:
            acc = (acc / acc_mag) * self.max_acc

        # 4. 运动学积分
        self.vel += acc * self.dt
        self.pos += self.vel * self.dt

        return self.pos, self.vel, acc

# ---------------------------
# 通用可实例化传感器（v1.4 方案）
# ---------------------------

class GenericDetectionSensor(SimpleSensor):
    measurement_model = Property("Measurement model")
    position_mapping = Property("传感器位置", default=(0, 2, 4))
    az_fov = Property("方位视场", default=None)
    el_fov = Property("俯仰视场", default=None)
    max_range = Property("最大探测距离", default=None)
    clutter_rate = Property("每帧期望杂波数（虚警率）", default=0.0)

    def _get_relative_vector(self, truth, platform_state):
        pos = truth.state_vector.flatten()
        x, y, z = pos[0], pos[2], pos[4]

        p = platform_state.state_vector.flatten()
        px, py, pz = p[0], p[2], p[4]

        return x - px, y - py, z - pz

    def _get_body_frame_vector(self, truth, platform_state):
        """
        将目标相对矢量转换到以平台速度方向为 X 轴的机载坐标系
        """
        # 1. 全局相对矢量
        pos = truth.state_vector.flatten()
        p = platform_state.state_vector.flatten()
        dx_w = pos[0] - p[0]
        dy_w = pos[2] - p[2]
        dz_w = pos[4] - p[4]
        rel_w = np.array([dx_w, dy_w, dz_w])

        # 2. 获取平台速度向量作为机载 X 轴
        v = np.array([p[1], p[3], p[5]])
        speed = np.linalg.norm(v)
        if speed < 1e-6:  # 平台静止时退化为世界坐标系
            return dx_w, dy_w, dz_w

        unit_x = v / speed

        # 3. 构建旋转矩阵 (简单姿态估计：假设无倾斜，Z轴向上)
        # 侧向单位向量 (Y)
        unit_y = np.cross([0, 0, 1], unit_x)
        if np.linalg.norm(unit_y) < 1e-6:  # 垂直飞行特例
            unit_y = np.array([0, 1, 0])
        unit_y /= np.linalg.norm(unit_y)

        # 上方单位向量 (Z)
        unit_z = np.cross(unit_x, unit_y)

        # 4. 投影到机载轴上 (旋转矩阵转置)
        dx_b = np.dot(rel_w, unit_x)
        dy_b = np.dot(rel_w, unit_y)
        dz_b = np.dot(rel_w, unit_z)

        return dx_b, dy_b, dz_b

    def measure(self, ground_truths, noise=True, platform_state=None, **kwargs):
        if platform_state is None:
            raise ValueError("platform_state must be provided for relative measurement")

        detections = []

        for truth in ground_truths:
            if not self.is_detectable(truth, platform_state):
                continue

            dx, dy, dz = self._get_body_frame_vector(truth, platform_state)

            relative_state = truth.state_vector.copy()
            relative_state[self.position_mapping[0]] = dx
            relative_state[self.position_mapping[1]] = dy
            relative_state[self.position_mapping[2]] = dz

            relative_truth = GroundTruthState(
                relative_state,
                timestamp=truth.timestamp
            )

            meas_vec = self.measurement_model.function(
                relative_truth,
                noise=noise,
                **kwargs
            )

            detections.append(
                TrueDetection(
                    meas_vec,
                    measurement_model=self.measurement_model,
                    timestamp=truth.timestamp,
                    groundtruth_path=truth
                )
            )

        return detections

    def is_detectable(self, truth, platform_state):
        # 🟢 修复：在判断视场时，也必须使用机载坐标系！
        dx, dy, dz = self._get_body_frame_vector(truth, platform_state)

        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if r < 1e-6:
            return False

        az = np.arctan2(dy, dx)
        el = np.arcsin(dz / r)

        # 调试输出（可选，建议保留用来观察机头对准情况）
        # print(f"[RADAR CHECK] body_dx={dx:.1f}, body_dy={dy:.1f}, body_dz={dz:.1f}, "
        #       f"body_az={np.degrees(az):.2f}°, body_el={np.degrees(el):.2f}°, r={r / 1000:.1f}km")

        if self.az_fov and not (self.az_fov[0] <= az <= self.az_fov[1]):
            return False
        if self.el_fov and not (self.el_fov[0] <= el <= self.el_fov[1]):
            return False
        if self.max_range and r > self.max_range:
            return False

        return True

    def is_clutter_detectable(self, clutter):
        return True

# ---------------------------
# 工具：统一测量顺序（StoneSoup -> 我们的 JSON）
# StoneSoup 的 CartesianToElevationBearingRange 返回 [el, az, r]
# 我们要的顺序为 [az, el, r]
# ---------------------------
def reorder_measurement(stone_vec):
    """
    stone_vec: np.array or list from StoneSoup detection.state_vector
               expected shape >=3 as [el, az, r]
    returns: list [az, el, r] (floats)
    """
    arr = np.asarray(stone_vec).flatten()
    if arr.size >= 3:
        el = float(arr[0])
        az = float(arr[1])
        r  = float(arr[2])
        return [az, el, r]
    else:
        # fallback: return as floats
        return [float(x) for x in arr]

# ---------------------------
# 配置参数
# ---------------------------
START_TIME = datetime.datetime.now()
DURATION = 60
DT = 0.1
PROJECTION_DIST = 60000.0  # 用于 EO / Passive 可视化投影（若需要）

# --- 异步性配置：每个传感器的测量周期 (T_meas) ---
SENSOR_MEASUREMENT_PERIODS = {
    0: timedelta(seconds=0.1),
    1: timedelta(seconds=0.1),
    2: timedelta(seconds=0.1)
}

# ---------------------------
# IMM 风格切换的 3D 机动目标
# ---------------------------
# --- 定义三种机动意图 (仅参数，不再实例化为独立目标) ---
MODES = [
    {"a_bar": [-2.0,  5.0,  3.0], "omega0": 1.2}, # 模式0：爬升转弯 (Z轴为正)
    {"a_bar": [-5.0, -5.0,  0.0], "omega0": 0.5}, # 模式1：水平大过载左转
    {"a_bar": [ 2.0,  3.0, -4.0], "omega0": 2.0}, # 模式2：俯冲俯冲 (Z轴为负)
]

# 修复：提高对角线概率。假设 dt=0.1，0.98 表示平均在一个模式停留 50 步 (即 5 秒)
transition_matrix = np.array([
    [0.98, 0.01, 0.01],
    [0.01, 0.98, 0.01],
    [0.01, 0.01, 0.98]
])


def generate_maneuvering_target(start_time, duration, dt, initial_target=None):
    steps = int(duration / dt)
    current_time = start_time
    current_model_idx = 0

    # 如果外部没传，使用默认值；如果传了，解包位置和速度
    if initial_target is None:
        pos_init = [100000.0, 0.0, 10000.0]
        vel_init = [-250.0, 30.0, 0.0]
    else:
        # 假设传入格式为 [x, vx, y, vy, z, vz]
        pos_init = [initial_target[0], initial_target[2], initial_target[4]]
        vel_init = [initial_target[1], initial_target[3], initial_target[5]]

    target = ManeuveringTarget3D(
        init_pos=pos_init,
        init_vel=vel_init,
        init_a_bar=MODES[0]["a_bar"],
        dt=dt,
        omega0=MODES[0]["omega0"]
    )

    truth = GroundTruthPath()

    for _ in range(steps):
        # 马尔可夫链状态转移
        probs = transition_matrix[current_model_idx]
        next_model_idx = np.random.choice(len(MODES), p=probs)

        if next_model_idx != current_model_idx:
            current_model_idx = next_model_idx
            # 仅切换意图参数，保持状态连续
            target.switch_mode(**MODES[current_model_idx])

        # 步进更新
        pos, vel, _ = target.step()
        state_vec = np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]])
        truth.append(GroundTruthState(state_vec, timestamp=current_time))

        current_time += timedelta(seconds=dt)

    return truth


# ---------------------------
# 创建传感器平台
# ---------------------------
def create_sensor_platform(start_time):
    dt = DT

    # ---------------------------
    # 1. 平台机动模型
    # ---------------------------
    platform_motion = ManeuveringTarget3D(
        init_pos=[0.0, 0.0, 10000.0],
        init_vel=[250.0, 0.0, 0.0],
        init_a_bar=[0.0, 0.0, 0.0],  # 初始匀速
        dt=DT,
        omega0=0.0  # 禁用简谐波动
    )

    # 覆盖平台的随机参数，确保其“老实”飞行
    platform_motion.acc_model.sigma_a = 2.0  # 极小噪声
    platform_motion.acc_model.jump_prob = 0.0  # 严禁加速度突变
    platform_motion.acc_model.alpha = 1.5  # 高阻尼，确保 z1 迅速回归 0

    pos, vel, _ = platform_motion.step()
    platform_state = GroundTruthState(
        [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]],
        timestamp=start_time
    )

    # ---------------------------
    # 2. 定义传感器（不要管 position）
    # ---------------------------
    radar_model = CartesianToElevationBearingRange(
        ndim_state=6,
        mapping=(0, 2, 4),
        noise_covar=np.diag([
            (0.002)**2,
            (0.002)**2,
            80**2
        ])
    )

    radar = GenericDetectionSensor(
        measurement_model=radar_model,
        az_fov=(-np.pi/3, np.pi/3),
        el_fov=(-np.radians(3), np.radians(3)),
        max_range=200000.0,
        clutter_rate=0.0
    )

    eo_model = CartesianToElevationBearingRange(
        ndim_state=6,
        mapping=(0, 2, 4),
        noise_covar=np.diag([
            (0.002)**2,
            (0.002)**2,
            1e20
        ])
    )

    eo = GenericDetectionSensor(
        measurement_model=eo_model,
        az_fov=(-np.pi/3, np.pi/3),
        el_fov=(-np.radians(3), np.radians(3)),
        max_range=200000.0,
        clutter_rate=0.0
    )

    esm_model = CartesianToElevationBearingRange(
        ndim_state=6,
        mapping=(0, 2, 4),
        noise_covar=np.diag([
            1e20,
            (0.025)**2,
            1e20
        ])
    )

    esm = GenericDetectionSensor(
        measurement_model=esm_model,
        az_fov=(-np.pi, np.pi),
        el_fov=None,
        max_range=200000.0,
        clutter_rate=0.0
    )

    sensors = [radar, eo, esm]

    # ---------------------------
    # 3. 构造平台（不传 sensors！）
    # ---------------------------
    platform = MovingPlatform(
        states=[platform_state],
        transition_model=None,
        position_mapping=(0, 2, 4),
        velocity_mapping=(1, 3, 5),
        sensors=[]
    )
    platform.motion = platform_motion

    return platform, sensors


# ---------------------------
# 主仿真
# ---------------------------
def run_simulation(seed=None, initial_target=None):
    if seed is not None:
        np.random.seed(seed)

    # === 1. 生成目标的 GroundTruth 轨迹 ===
    truth = generate_maneuvering_target(START_TIME, DURATION, DT, initial_target)

    # === 2. 初始化编队平台 ===
    # 长机 (Master 0) - 零延迟
    master_motion = GuidedPlatformModel(
        init_pos=[0.0, 0.0, 10000.0], init_vel=[250.0, 0.0, 0.0], dt=DT, gain=2.5, max_acc=40.0)

    # 僚机 1 (Wingman 1 - 左侧 10km) -> 挂载 快数据链
    wm1_motion = GuidedPlatformModel(
        init_pos=[0.0, 10000.0, 10000.0], init_vel=[250.0, 0.0, 0.0], dt=DT, gain=2.5, max_acc=40.0)
    datalink_wm1 = DatalinkNode(delay_s=0.1, interval_s=0.1, noise_ang_rad=0.005, noise_r_m=100.0)

    # 🌟 重点修改这里：僚机 2 (Wingman 2 - 右后方 10km) -> 也统一挂载 快数据链
    wm2_motion = GuidedPlatformModel(
        init_pos=[-5000.0, -10000.0, 10000.0], init_vel=[250.0, 0.0, 0.0], dt=DT, gain=2.5, max_acc=40.0)
    datalink_wm2 = DatalinkNode(delay_s=0.1, interval_s=0.1, noise_ang_rad=0.005, noise_r_m=100.0)

    platforms_info = [
        {"id": "master_0", "motion": master_motion, "is_master": True, "datalink": None},
        {"id": "wingman_1", "motion": wm1_motion, "is_master": False, "datalink": datalink_wm1},
        {"id": "wingman_2", "motion": wm2_motion, "is_master": False, "datalink": datalink_wm2},
    ]

    # 给每个平台分配一套独立的传感器实例，并初始化最后测量时间
    for p in platforms_info:
        _, p["sensors"] = create_sensor_platform(START_TIME)
        p["last_meas_time"] = {idx: START_TIME - period for idx, period in SENSOR_MEASUREMENT_PERIODS.items()}
        # 缓存僚机的最后一帧有效数据（用于零阶保持）
        p["cached_arrived_data"] = None

    # === 3. 核心仿真循环 ===
    dataset = []
    for state in truth:
        current_time_s = float((state.timestamp - START_TIME).total_seconds())
        target_pos = [state.state_vector[0], state.state_vector[2], state.state_vector[4]]

        frame_platforms_data = {}

        # 遍历编队中的每一架飞机
        for p in platforms_info:
            # 更新平台自身物理状态
            p_pos, p_vel, p_acc = p["motion"].step(target_pos)
            new_p_state = GroundTruthState(
                [p_pos[0], p_vel[0], p_pos[1], p_vel[1], p_pos[2], p_vel[2]],
                timestamp=state.timestamp
            )

            # --- 传感器测量逻辑 (恢复实际的测量触发机制) ---
            sensor_outputs = {}
            for idx, sensor in enumerate(p["sensors"]):
                period = SENSOR_MEASUREMENT_PERIODS[idx]
                # 判断当前时间是否达到了传感器的更新周期
                if state.timestamp - p["last_meas_time"][idx] >= period - timedelta(microseconds=10):
                    detections = sensor.measure([state], platform_state=new_p_state)
                    obs_list = []
                    for d in detections:
                        if isinstance(d, TrueDetection):
                            # 转为 [az, el, r] 格式
                            az, el, r = reorder_measurement(d.state_vector)
                            obs_list.append({"meas": [az, el, r], "type": "true"})

                    if obs_list:
                        sensor_outputs[f"sensor_{idx}"] = {"observations": obs_list}

                    p["last_meas_time"][idx] = state.timestamp

            # --- 数据链处理逻辑 ---
            if p["is_master"]:
                # 长机：直接享受无延迟、无损的传感器数据
                frame_platforms_data[p["id"]] = {
                    "data_age": 0.0,
                    "platform_state": new_p_state.state_vector.flatten().tolist(),
                    "sensors": sensor_outputs
                }
            else:
                # 僚机：先将刚出炉的数据灌入数据链 (即使 sensor_outputs 是空的，也要维持平台状态的更新)
                p["datalink"].ingest_data(
                    current_time_s,
                    new_p_state.state_vector.flatten().tolist(),
                    sensor_outputs
                )

                # 然后长机尝试从该数据链中提取“已经到达”的数据包
                arrived_data = p["datalink"].get_arrived_data(current_time_s)

                if arrived_data is not None:
                    p["cached_arrived_data"] = arrived_data  # 更新缓存

                # 如果有缓存数据，则写入当前帧；如果没有（比如仿真第一秒），则写入缺省值
                if p["cached_arrived_data"]:
                    frame_platforms_data[p["id"]] = p["cached_arrived_data"]
                else:
                    # 慢数据链还没建立首次通信
                    frame_platforms_data[p["id"]] = {
                        "data_age": -1.0,
                        "platform_state": None,
                        "sensors": {}
                    }

        # --- 存帧 ---
        dataset.append({
            "time_s": current_time_s,
            "ground_truth_state": [float(x) for x in state.state_vector.flatten()],
            "platforms": frame_platforms_data
        })

    return dataset

# ---------------------------
# CLI 主体
# ---------------------------
if __name__ == "__main__":
    try:
        data = run_simulation()
        print("生成完成，共 {} 帧".format(len(data)))

        with open("dataset_final_stonesoup_v1_4_asynchronous.json", "w") as f:
            json.dump(data, f, indent=2)

        print("已保存 dataset_final_stonesoup_v1_4_asynchronous.json")

    except Exception:
        traceback.print_exc()