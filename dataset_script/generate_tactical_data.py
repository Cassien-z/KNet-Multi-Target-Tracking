# script.py
import datetime
from datetime import timedelta
import json
import traceback
import collections
import numpy as np

# StoneSoup core
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection
from stonesoup.platform.base import MovingPlatform
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.base import Property
from stonesoup.sensor.sensor import SimpleSensor


class DatalinkNode:
    """数据链模拟节点（部署在僚机上）"""

    def __init__(self, delay_s, interval_s, noise_ang_rad, noise_r_m):
        self.delay_s = delay_s
        self.interval_s = interval_s
        self.noise_ang = noise_ang_rad
        self.noise_r = noise_r_m

        self.queue = collections.deque()
        self.last_tx_time = -999.0

    def ingest_data(self, current_time_s, platform_state, sensor_outputs):
        if current_time_s - self.last_tx_time >= self.interval_s - 1e-5:
            degraded_sensors = {}
            for s_name, s_data in sensor_outputs.items():
                degraded_obs = []
                for obs in s_data["observations"]:
                    az, el, r = obs["meas"]
                    noisy_az = az + np.random.randn() * self.noise_ang
                    noisy_el = el + np.random.randn() * self.noise_ang
                    noisy_r = r + np.random.randn() * self.noise_r
                    degraded_obs.append({"meas": [noisy_az, noisy_el, noisy_r], "type": obs["type"]})
                degraded_sensors[s_name] = {"observations": degraded_obs}

            arrival_time = current_time_s + self.delay_s
            self.queue.append({
                "arrival_time": arrival_time,
                "data_age": self.delay_s,
                "platform_state": platform_state.copy(),
                "sensors": degraded_sensors
            })
            self.last_tx_time = current_time_s

    def get_arrived_data(self, master_time_s):
        latest_data = None
        while self.queue and self.queue[0]["arrival_time"] <= master_time_s + 1e-5:
            latest_data = self.queue.popleft()
        return latest_data


class Figure8Target3D:
    """🌟 新增：目标8字机动模型（利萨如曲线）"""

    def __init__(self, cx, cy, cz, A, B, omega, dt):
        self.cx = cx  # 8字中心X
        self.cy = cy  # 8字中心Y
        self.cz = cz  # 高度Z
        self.A = A  # X轴振幅
        self.B = B  # Y轴振幅
        self.omega = omega  # 角速度
        self.dt = dt
        self.t = 0.0

    def step(self):
        # 位置
        x = self.cx + self.A * np.sin(self.omega * self.t)
        y = self.cy + self.B * np.sin(2 * self.omega * self.t)
        z = self.cz
        pos = np.array([x, y, z])

        # 速度 (位置的导数)
        vx = self.A * self.omega * np.cos(self.omega * self.t)
        vy = 2 * self.B * self.omega * np.cos(2 * self.omega * self.t)
        vz = 0.0
        vel = np.array([vx, vy, vz])

        # 加速度 (速度的导数)
        ax = -self.A * (self.omega ** 2) * np.sin(self.omega * self.t)
        ay = -4 * self.B * (self.omega ** 2) * np.sin(2 * self.omega * self.t)
        az = 0.0
        acc = np.array([ax, ay, az])

        self.t += self.dt
        return pos, vel, acc


# -------------------------------------------------------------------------
# 以下保留原有的加速度模型和运动学模型，仅供平台初始化构建时占位使用
# -------------------------------------------------------------------------
class CSSWAccelerationModel:
    def __init__(self, a_bar, sigma_a=5.0, alpha=0.3, omega0=1.0, dt=0.1):
        self.a_bar = np.array(a_bar, dtype=float)
        self.sigma_a = sigma_a
        self.alpha = alpha
        self.omega0 = omega0
        self.dt = dt
        self.jump_prob = 0.005
        self.jump_scale = 15.0
        self.z1 = np.zeros(3)
        self.z2 = np.zeros(3)

    def set_params(self, a_bar, omega0):
        self.a_bar = np.array(a_bar, dtype=float)
        self.omega0 = omega0

    def step(self):
        if np.random.rand() < self.jump_prob:
            jump = np.random.randn(3) * self.jump_scale
            self.a_bar += jump
        w = np.random.randn(3) * self.sigma_a
        dz1 = self.z2
        dz2 = -2 * self.alpha * self.z2 - (self.alpha ** 2 + self.omega0 ** 2) * self.z1 + w
        self.z1 += dz1 * self.dt
        self.z2 += dz2 * self.dt
        return self.a_bar + self.z1


class KinematicMotionModel:
    def __init__(self, dt=0.1):
        self.dt = dt

    def step(self, pos, vel, acc):
        vel_next = vel + acc * self.dt
        pos_next = pos + vel * self.dt + 0.5 * acc * (self.dt ** 2)
        return pos_next, vel_next


class ManeuveringTarget3D:
    def __init__(self, init_pos, init_vel, init_a_bar, dt=0.1, omega0=1.2):
        self.pos = np.array(init_pos, dtype=float)
        self.vel = np.array(init_vel, dtype=float)
        self.acc_model = CSSWAccelerationModel(a_bar=init_a_bar, dt=dt, omega0=omega0)
        self.motion_model = KinematicMotionModel(dt=dt)

    def switch_mode(self, a_bar, omega0):
        self.acc_model.set_params(a_bar, omega0)

    def step(self):
        acc = self.acc_model.step()
        self.pos, self.vel = self.motion_model.step(self.pos, self.vel, acc)
        return self.pos, self.vel, acc


# -------------------------------------------------------------------------


class GuidedPlatformModel:
    """闭环引导平台模型"""

    def __init__(self, init_pos, init_vel, dt=0.1, max_acc=20.0, gain=0.5):
        self.pos = np.array(init_pos, dtype=float)
        self.vel = np.array(init_vel, dtype=float)
        self.dt = dt
        self.max_acc = max_acc
        self.gain = gain

    def step(self, target_pos):
        rel_pos = np.array(target_pos) - self.pos
        dist = np.linalg.norm(rel_pos)

        if dist < 100:
            desired_vel = self.vel
        else:
            desired_vel = (rel_pos / dist) * np.linalg.norm(self.vel)

        acc = self.gain * (desired_vel - self.vel) / self.dt
        acc_mag = np.linalg.norm(acc)
        if acc_mag > self.max_acc:
            acc = (acc / acc_mag) * self.max_acc

        self.vel += acc * self.dt
        self.pos += self.vel * self.dt

        return self.pos, self.vel, acc


class GenericDetectionSensor(SimpleSensor):
    measurement_model = Property("Measurement model")
    position_mapping = Property("传感器位置", default=(0, 2, 4))
    az_fov = Property("方位视场", default=None)
    el_fov = Property("俯仰视场", default=None)
    max_range = Property("最大探测距离", default=None)
    clutter_rate = Property("每帧期望杂波数（虚警率）", default=0.0)

    def _get_body_frame_vector(self, truth, platform_state):
        pos = truth.state_vector.flatten()
        p = platform_state.state_vector.flatten()
        dx_w = pos[0] - p[0]
        dy_w = pos[2] - p[2]
        dz_w = pos[4] - p[4]
        rel_w = np.array([dx_w, dy_w, dz_w])

        v = np.array([p[1], p[3], p[5]])
        speed = np.linalg.norm(v)
        if speed < 1e-6:
            return dx_w, dy_w, dz_w

        unit_x = v / speed
        unit_y = np.cross([0, 0, 1], unit_x)
        if np.linalg.norm(unit_y) < 1e-6:
            unit_y = np.array([0, 1, 0])
        unit_y /= np.linalg.norm(unit_y)
        unit_z = np.cross(unit_x, unit_y)

        dx_b = np.dot(rel_w, unit_x)
        dy_b = np.dot(rel_w, unit_y)
        dz_b = np.dot(rel_w, unit_z)

        return dx_b, dy_b, dz_b

    def measure(self, ground_truths, noise=True, platform_state=None, **kwargs):
        if platform_state is None:
            raise ValueError("platform_state must be provided")

        detections = []
        for truth in ground_truths:
            if not self.is_detectable(truth, platform_state):
                continue

            dx, dy, dz = self._get_body_frame_vector(truth, platform_state)
            relative_state = truth.state_vector.copy()
            relative_state[self.position_mapping[0]] = dx
            relative_state[self.position_mapping[1]] = dy
            relative_state[self.position_mapping[2]] = dz

            relative_truth = GroundTruthState(relative_state, timestamp=truth.timestamp)
            meas_vec = self.measurement_model.function(relative_truth, noise=noise, **kwargs)

            detections.append(TrueDetection(
                meas_vec, measurement_model=self.measurement_model,
                timestamp=truth.timestamp, groundtruth_path=truth
            ))
        return detections

    def is_detectable(self, truth, platform_state):
        dx, dy, dz = self._get_body_frame_vector(truth, platform_state)
        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if r < 1e-6: return False

        az = np.arctan2(dy, dx)
        el = np.arcsin(dz / r)

        if self.az_fov and not (self.az_fov[0] <= az <= self.az_fov[1]): return False
        if self.el_fov and not (self.el_fov[0] <= el <= self.el_fov[1]): return False
        if self.max_range and r > self.max_range: return False
        return True

    def is_clutter_detectable(self, clutter):
        return True


def reorder_measurement(stone_vec):
    arr = np.asarray(stone_vec).flatten()
    if arr.size >= 3:
        return [float(arr[1]), float(arr[0]), float(arr[2])]  # az, el, r
    return [float(x) for x in arr]


START_TIME = datetime.datetime.now()
DURATION = 60
DT = 0.1

SENSOR_MEASUREMENT_PERIODS = {
    0: timedelta(seconds=0.1),
    1: timedelta(seconds=0.1),
    2: timedelta(seconds=0.1)
}


def generate_figure8_target(start_time, duration, dt):
    """🌟 新增：生成8字机动目标的 GroundTruthPath"""
    steps = int(duration / dt)
    current_time = start_time

    # 初始化8字轨迹参数：中心在原点上方2000米处，完成一个8字需要40秒
    target = Figure8Target3D(
        cx=0.0, cy=0.0, cz=2000.0,
        A=6000.0, B=3000.0,
        omega=2 * np.pi / 80.0,
        dt=dt
    )

    truth = GroundTruthPath()

    for _ in range(steps):
        pos, vel, acc = target.step()

        # 兼容 StoneSoup 的 6维状态向量 [x, vx, y, vy, z, vz]
        state_vec = np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]])
        gt_state = GroundTruthState(state_vec, timestamp=current_time)
        gt_state.true_acc = acc.copy()  # 挂载真实的物理加速度
        truth.append(gt_state)

        current_time += timedelta(seconds=dt)
    return truth


def create_sensor_platform(start_time):
    # 此部分保留，用于挂载传感器的平台占位符
    platform_motion = ManeuveringTarget3D(
        init_pos=[0.0, 0.0, 10000.0], init_vel=[250.0, 0.0, 0.0],
        init_a_bar=[0.0, 0.0, 0.0], dt=DT, omega0=0.0
    )
    platform_motion.acc_model.sigma_a = 2.0
    platform_motion.acc_model.jump_prob = 0.0
    platform_motion.acc_model.alpha = 1.5

    pos, vel, _ = platform_motion.step()
    platform_state = GroundTruthState(
        [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]], timestamp=start_time
    )

    radar_model = CartesianToElevationBearingRange(ndim_state=6, mapping=(0, 2, 4),
                                                   noise_covar=np.diag([(0.002) ** 2, (0.002) ** 2, 80 ** 2]))
    radar = GenericDetectionSensor(measurement_model=radar_model, az_fov=(-np.pi / 3, np.pi / 3),
                                   el_fov=(-np.radians(3), np.radians(3)), max_range=200000.0)

    eo_model = CartesianToElevationBearingRange(ndim_state=6, mapping=(0, 2, 4),
                                                noise_covar=np.diag([(0.002) ** 2, (0.002) ** 2, 1e20]))
    eo = GenericDetectionSensor(measurement_model=eo_model, az_fov=(-np.pi / 3, np.pi / 3),
                                el_fov=(-np.radians(3), np.radians(3)), max_range=200000.0)

    esm_model = CartesianToElevationBearingRange(ndim_state=6, mapping=(0, 2, 4),
                                                 noise_covar=np.diag([1e20, (0.025) ** 2, 1e20]))
    esm = GenericDetectionSensor(measurement_model=esm_model, az_fov=(-np.pi, np.pi), el_fov=None, max_range=200000.0)

    platform = MovingPlatform(
        states=[platform_state], transition_model=None,
        position_mapping=(0, 2, 4), velocity_mapping=(1, 3, 5), sensors=[]
    )
    platform.motion = platform_motion
    return platform, [radar, eo, esm]


def run_simulation(seed=None):
    if seed is not None: np.random.seed(seed)

    # 8字机动目标生成器
    truth = generate_figure8_target(START_TIME, DURATION, DT)

    # 🌟 修改：重新设计三机编队初始位置与速度
    # 目标中心高度 2000m。我方分布在目标四周，水平距离约 15000m，高度 18000m。
    # 初始速度指向目标中心并带有-200m/s的下洗速度，形成完美的包围+俯冲态势。

    # 长机：正东方向，向西俯冲
    master_motion = GuidedPlatformModel(
        init_pos=[30000.0, 0.0, 36000.0],
        init_vel=[-100.0, 0.0, -133.0],
        dt=DT, gain=3.0, max_acc=50.0
    )

    # 僚机1：西北方向，向东南俯冲
    wm1_motion = GuidedPlatformModel(
        init_pos=[-15000.0, 26000.0, 36000.0],
        init_vel=[50.0, -87.0, -133.0],
        dt=DT, gain=3.0, max_acc=50.0
    )

    # 僚机2：西南方向，向东北俯冲
    wm2_motion = GuidedPlatformModel(
        init_pos=[-15000.0, -26000.0, 36000.0],
        init_vel=[50.0, 87.0, -133.0],
        dt=DT, gain=3.0, max_acc=50.0
    )

    platforms_info = [
        {"id": "master_0", "motion": master_motion, "is_master": True, "datalink": None},
        {"id": "wingman_1", "motion": wm1_motion, "is_master": False, "datalink": DatalinkNode(0.1, 0.1, 0.005, 100.0)},
        {"id": "wingman_2", "motion": wm2_motion, "is_master": False, "datalink": DatalinkNode(0.1, 0.1, 0.005, 100.0)},
    ]

    # ... [保持下方的传感器挂载与主循环代码不变] ...
    for p in platforms_info:
        _, p["sensors"] = create_sensor_platform(START_TIME)
        p["last_meas_time"] = {idx: START_TIME - period for idx, period in SENSOR_MEASUREMENT_PERIODS.items()}
        p["cached_arrived_data"] = None

    dataset = []
    for state in truth:
        current_time_s = float((state.timestamp - START_TIME).total_seconds())

        target_pos = [state.state_vector[0], state.state_vector[2], state.state_vector[4]]
        target_vel = [state.state_vector[1], state.state_vector[3], state.state_vector[5]]
        target_acc = getattr(state, 'true_acc', np.array([0.0, 0.0, 0.0]))

        gt_9d = [
            float(target_pos[0]), float(target_vel[0]), float(target_acc[0]),
            float(target_pos[1]), float(target_vel[1]), float(target_acc[1]),
            float(target_pos[2]), float(target_vel[2]), float(target_acc[2])
        ]

        frame_platforms_data = {}

        for p in platforms_info:
            p_pos, p_vel, p_acc = p["motion"].step(target_pos)

            new_p_state = GroundTruthState(
                [p_pos[0], p_vel[0], p_pos[1], p_vel[1], p_pos[2], p_vel[2]],
                timestamp=state.timestamp
            )

            p_state_9d = [
                float(p_pos[0]), float(p_vel[0]), float(p_acc[0]),
                float(p_pos[1]), float(p_vel[1]), float(p_acc[1]),
                float(p_pos[2]), float(p_vel[2]), float(p_acc[2])
            ]

            sensor_outputs = {}
            for idx, sensor in enumerate(p["sensors"]):
                period = SENSOR_MEASUREMENT_PERIODS[idx]
                if state.timestamp - p["last_meas_time"][idx] >= period - timedelta(microseconds=10):
                    detections = sensor.measure([state], platform_state=new_p_state)
                    obs_list = []
                    for d in detections:
                        if isinstance(d, TrueDetection):
                            az, el, r = reorder_measurement(d.state_vector)
                            obs_list.append({"meas": [az, el, r], "type": "true"})

                    if obs_list:
                        sensor_outputs[f"sensor_{idx}"] = {"observations": obs_list}
                    p["last_meas_time"][idx] = state.timestamp

            if p["is_master"]:
                frame_platforms_data[p["id"]] = {
                    "data_age": 0.0,
                    "platform_state": p_state_9d,
                    "sensors": sensor_outputs
                }
            else:
                p["datalink"].ingest_data(current_time_s, p_state_9d, sensor_outputs)
                arrived_data = p["datalink"].get_arrived_data(current_time_s)
                if arrived_data is not None:
                    p["cached_arrived_data"] = arrived_data

                if p["cached_arrived_data"]:
                    frame_platforms_data[p["id"]] = p["cached_arrived_data"]
                else:
                    frame_platforms_data[p["id"]] = {"data_age": -1.0, "platform_state": None, "sensors": {}}

        dataset.append({
            "time_s": current_time_s,
            "ground_truth_state": gt_9d,
            "platforms": frame_platforms_data
        })

    return dataset


if __name__ == "__main__":
    try:
        data = run_simulation()
        print("生成完成，共 {} 帧".format(len(data)))

        with open("dataset_strike_figure8_dive.json", "w") as f:
            json.dump(data, f, indent=2)

        print("已保存 dataset_strike_figure8_dive.json")

    except Exception:
        traceback.print_exc()