# script_multitarget.py
import datetime
from datetime import timedelta
import traceback
import collections
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import scipy.io

# StoneSoup core
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection
from stonesoup.platform.base import MovingPlatform
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.base import Property
from stonesoup.sensor.sensor import SimpleSensor

def get_body_to_world_matrix(vx, vy, vz):
    v = np.array([vx, vy, vz])
    speed = np.linalg.norm(v)
    if speed < 1e-6: return np.eye(3)
    unit_x = v / speed
    unit_y = np.cross([0, 0, 1], unit_x)
    if np.linalg.norm(unit_y) < 1e-6:
        unit_y = np.array([0, 1, 0])
    else:
        unit_y /= np.linalg.norm(unit_y)
    unit_z = np.cross(unit_x, unit_y)
    return np.column_stack((unit_x, unit_y, unit_z))

def reconstruct_ecef_from_angles(az, el, r, platform_pos, platform_vel):
    """从方位角/俯仰角/距离 + 平台状态，反算目标ECEF坐标。"""
    dx = r * np.cos(el) * np.cos(az)
    dy = r * np.cos(el) * np.sin(az)
    dz = r * np.sin(el)
    vec_body = np.array([dx, dy, dz])
    R_b2w = get_body_to_world_matrix(*platform_vel)
    return np.array(platform_pos) + R_b2w @ vec_body


class CSSWAccelerationModel:
    """CS-SW 复合机动加速度模型"""

    def __init__(self, a_bar, sigma_a=5.0, alpha=0.3, omega0=1.0, dt=0.1):
        self.a_bar = np.array(a_bar, dtype=float)
        self.sigma_a = sigma_a
        self.alpha = alpha
        self.omega0 = omega0
        self.dt = dt

        self.z1 = np.zeros(3)
        self.z2 = np.zeros(3)

    def set_params(self, a_bar, omega0):
        self.a_bar = np.array(a_bar, dtype=float)
        self.omega0 = omega0

    def step(self):
        w = np.random.randn(3) * self.sigma_a

        # 1. 计算二阶变化率
        dz2 = (
                -2 * self.alpha * self.z2
                - (self.alpha ** 2 + self.omega0 ** 2) * self.z1
                + w
        )

        # 2. 半隐式欧拉法
        self.z2 += dz2 * self.dt
        self.z1 += self.z2 * self.dt

        # 3. 物理过载绝对限幅 (弹簧震荡不超过 8G)
        self.z1 = np.clip(self.z1, -80.0, 80.0)

        final_acc = self.a_bar + self.z1
        return np.clip(final_acc, -150.0, 150.0)


class KinematicMotionModel:
    """标准运动学模型"""

    def __init__(self, dt=0.1):
        self.dt = dt

    def step(self, pos, vel, acc):
        vel_next = vel + acc * self.dt
        pos_next = pos + vel * self.dt + 0.5 * acc * (self.dt ** 2)
        return pos_next, vel_next


class ManeuveringTarget3D:
    """统一的目标实体"""

    def __init__(self, init_pos, init_vel, init_a_bar, dt=0.1, omega0=1.2):
        self.pos = np.array(init_pos, dtype=float)
        self.vel = np.array(init_vel, dtype=float)

        self.acc_model = CSSWAccelerationModel(
            a_bar=init_a_bar, dt=dt, omega0=omega0
        )
        self.motion_model = KinematicMotionModel(dt=dt)

    def switch_mode(self, a_bar, omega0):
        self.acc_model.set_params(a_bar, omega0)

    def step(self):
        acc = self.acc_model.step()
        self.pos, self.vel = self.motion_model.step(self.pos, self.vel, acc)
        return self.pos, self.vel, acc


class ConstantVelocityPlatform:
    """匀速直线飞行平台（我方，无引导）"""

    def __init__(self, init_pos, init_vel, dt=0.1):
        self.pos = np.array(init_pos, dtype=float)
        self.vel = np.array(init_vel, dtype=float)
        self.dt = dt

    def step(self):
        acc = np.zeros(3)
        self.pos = self.pos + self.vel * self.dt
        return self.pos.copy(), self.vel.copy(), acc

class SlightManeuverPlatform:
    """轻微机动平台：沿主航向进行缓和的蛇形机动，确保目标不脱离视场"""

    def __init__(self, init_pos, init_vel, dt=0.025, phase_offset=0.0):
        self.pos = np.array(init_pos, dtype=float)
        self.vel = np.array(init_vel, dtype=float)
        self.base_vel = np.array(init_vel, dtype=float)
        self.dt = dt
        self.time = 0.0
        self.phase = phase_offset
        
        speed = np.linalg.norm(self.base_vel)
        self.dir_x = self.base_vel / speed
        self.dir_y = np.cross([0, 0, 1], self.dir_x)
        if np.linalg.norm(self.dir_y) < 1e-6:
             self.dir_y = np.array([0, 1, 0])
        else:
             self.dir_y /= np.linalg.norm(self.dir_y)
        self.dir_z = np.cross(self.dir_x, self.dir_y)

    def step(self):
        self.time += self.dt
        # 侧向 3m/s^2，垂向 1m/s^2 的低频正弦过载
        acc_y = 3.0 * np.sin(2 * np.pi * self.time / 20.0 + self.phase)
        acc_z = 1.0 * np.sin(2 * np.pi * self.time / 15.0 + self.phase * 1.5)
        
        acc = acc_y * self.dir_y + acc_z * self.dir_z
        
        # 速度修正项，防止偏离基准速度太远
        vel_error = self.base_vel - self.vel
        acc += 0.2 * vel_error

        self.vel = self.vel + acc * self.dt
        self.pos = self.pos + self.vel * self.dt
        return self.pos.copy(), self.vel.copy(), acc.copy()


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
    elif arr.size == 2:
        return [float(arr[1]), float(arr[0])] # az, el
    return [float(x) for x in arr]


START_TIME = datetime.datetime.now()
DURATION = 20
DT = 0.025

SENSOR_MEASUREMENT_PERIODS = {
    0: timedelta(milliseconds=25),
    1: timedelta(milliseconds=50),
    2: timedelta(milliseconds=200)
}

MODES = [
    {"a_bar": [0.0, 0.0, 0.0], "omega0": 0.0},
    {"a_bar": [-5.0, 5.0, 0.0], "omega0": 0.2},
    {"a_bar": [5.0, -5.0, 0.0], "omega0": 0.2},
    {"a_bar": [0.0, 0.0, -5.0], "omega0": 0.1},
    {"a_bar": [0.0, 0.0, 5.0], "omega0": 0.1},
    {"a_bar": [5.0, 5.0, 0.0], "omega0": 0.5},
]

transition_matrix = np.zeros((6, 6))
transition_matrix[0][0] = 1.0


def generate_maneuvering_target(start_time, duration, dt, initial_target=None):
    steps = int(duration / dt)
    current_time = start_time

    if initial_target is None:
        pos_init = [10000.0, 0.0, 10000.0]
        vel_init = [-250.0, 30.0, 0.0]
    else:
        pos_init = [initial_target[0], initial_target[2], initial_target[4]]
        vel_init = [initial_target[1], initial_target[3], initial_target[5]]

    target = ManeuveringTarget3D(
        init_pos=pos_init, init_vel=vel_init,
        init_a_bar=MODES[0]["a_bar"], dt=dt, omega0=MODES[0]["omega0"]
    )

    truth = GroundTruthPath()

    current_model_idx = 0
    burst_timer = 0.0

    for _ in range(steps):
        if burst_timer > 0:
            burst_timer -= dt
            if burst_timer <= 0:
                current_model_idx = 0
        else:
            probs = transition_matrix[current_model_idx]
            next_model_idx = np.random.choice(len(MODES), p=probs)

            if next_model_idx != current_model_idx:
                current_model_idx = next_model_idx

                if current_model_idx in [1, 2]:
                    burst_timer = np.random.uniform(3.0, 6.0)
                elif current_model_idx in [3, 4]:
                    burst_timer = np.random.uniform(2.0, 3.5)
                elif current_model_idx == 5:
                    burst_timer = np.random.uniform(4.0, 7.0)

        if current_model_idx == 0:
            damp_acc_z = -0.5 * target.vel[2]
            damp_acc_z = np.clip(damp_acc_z, -10.0, 10.0)
            target.switch_mode([0.0, 0.0, damp_acc_z], omega0=0.0)
        else:
            target.switch_mode(**MODES[current_model_idx])

        pos, vel, acc = target.step()
        state_vec = np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]])
        gt_state = GroundTruthState(state_vec, timestamp=current_time)
        gt_state.true_acc = acc.copy()
        truth.append(gt_state)

        current_time += timedelta(seconds=dt)

    return truth


def create_sensor_platform(start_time, config="all"):
    platform_motion = ManeuveringTarget3D(
        init_pos=[0.0, 0.0, 10000.0], init_vel=[250.0, 0.0, 0.0],
        init_a_bar=[0.0, 0.0, 0.0], dt=DT, omega0=0.0
    )
    platform_motion.acc_model.sigma_a = 2.0
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
                                                 noise_covar=np.diag([(0.005) ** 2, (0.005) ** 2, 1e20]))
    esm = GenericDetectionSensor(measurement_model=esm_model, az_fov=(-np.radians(60), np.radians(60)), el_fov=(-np.radians(3), np.radians(3)), max_range=200000.0)

    platform = MovingPlatform(
        states=[platform_state], transition_model=None,
        position_mapping=(0, 2, 4), velocity_mapping=(1, 3, 5), sensors=[]
    )
    platform.motion = platform_motion
    
    sensors = [None, None, None]
    if config == "all":
        sensors = [radar, eo, esm]
    elif config == "radar_only":
        sensors = [radar, None, None]
    elif config == "radar_esm":
        sensors = [radar, None, esm]
    elif config == "radar_eo":
        sensors = [radar, eo, None]

    return platform, sensors



import random

class LocalTrackManager:
    def __init__(self, sensor_type_name):
        self.sensor_type_name = sensor_type_name
        self.next_local_id = 1
        self.target_to_local = {}
        self.missed_frames = {}
        self.start_times = {}
        self.closed_segments = []
        
    def process(self, obs_dict, current_time_s):
        local_obs_dict = {}
        current_targets = set(obs_dict.keys())
        
        # 1. Check existing tracks
        for tgt in list(self.target_to_local.keys()):
            if tgt in current_targets:
                self.missed_frames[tgt] = 0
                if random.random() < 0.0: # 2% chance to randomly drop track -> Disabled
                    local_id = self.target_to_local[tgt]
                    self.closed_segments.append([local_id, tgt+1, self.start_times[tgt], current_time_s])
                    del self.target_to_local[tgt]
                    del self.missed_frames[tgt]
                    del self.start_times[tgt]
            else:
                self.missed_frames[tgt] += 1
                if self.missed_frames[tgt] > 99999: # Drop after 20 missed frames -> Disabled
                    local_id = self.target_to_local[tgt]
                    end_time = current_time_s - self.missed_frames[tgt] * 0.025
                    self.closed_segments.append([local_id, tgt+1, self.start_times[tgt], end_time])
                    del self.target_to_local[tgt]
                    del self.missed_frames[tgt]
                    del self.start_times[tgt]
                    
        # 2. Assign IDs
        for tgt in current_targets:
            if tgt not in self.target_to_local:
                self.target_to_local[tgt] = self.next_local_id
                self.missed_frames[tgt] = 0
                self.start_times[tgt] = current_time_s
                self.next_local_id += 1
            local_id = self.target_to_local[tgt]
            local_obs_dict[local_id] = obs_dict[tgt]
            
        return local_obs_dict
        
    def close_all(self, end_time):
        for tgt in list(self.target_to_local.keys()):
            local_id = self.target_to_local[tgt]
            self.closed_segments.append([local_id, tgt+1, self.start_times[tgt], end_time])
        self.target_to_local.clear()

def get_meas_array(obs_dict, dim, max_local_id):
    size = max(max_local_id, 1)
    meas_arr = np.zeros((size, dim), dtype=np.float32)
    mask_arr = np.zeros((size,), dtype=np.float32)
    if obs_dict:
        for local_id, meas in obs_dict.items():
            if local_id <= size:
                meas_arr[local_id - 1, :dim] = meas[:dim]
                mask_arr[local_id - 1] = 1.0
    return meas_arr.tolist(), mask_arr.tolist()



def run_simulation(seed=None, initial_targets=None, platform_vel=None, num_targets=10):
    if seed is not None: np.random.seed(seed)
    
    if initial_targets is None or len(initial_targets) != num_targets:
        raise ValueError(f"initial_targets must be provided and have length {num_targets}")

    # 生成多条目标轨迹
    truths = []
    for i in range(num_targets):
        truths.append(generate_maneuvering_target(START_TIME, DURATION, DT, initial_targets[i]))

    # 我方平台轻微机动飞行，总体速度方向保持一致
    if platform_vel is None:
        platform_vel = [250.0, 0.0, 0.0]

    master_motion = SlightManeuverPlatform(init_pos=[0.0, 0.0, 10000.0], init_vel=platform_vel, dt=DT, phase_offset=0.0)
    wm1_motion    = SlightManeuverPlatform(init_pos=[0.0, 3000.0, 10000.0], init_vel=platform_vel, dt=DT, phase_offset=1.0)
    wm2_motion    = SlightManeuverPlatform(init_pos=[0.0, -3000.0, 10000.0], init_vel=platform_vel, dt=DT, phase_offset=2.0)
    wm3_motion    = SlightManeuverPlatform(init_pos=[3000.0, 3000.0, 10000.0], init_vel=platform_vel, dt=DT, phase_offset=3.0)
    wm4_motion    = SlightManeuverPlatform(init_pos=[3000.0, -3000.0, 10000.0], init_vel=platform_vel, dt=DT, phase_offset=4.0)

    # 初始化 LocalTrackManagers
    track_managers = {}
    for p_idx in range(5): # 5 platforms
        track_managers[(p_idx, 0)] = LocalTrackManager("Radar")
        track_managers[(p_idx, 1)] = LocalTrackManager("EO")
        track_managers[(p_idx, 2)] = LocalTrackManager("ESM")

    platforms_info = [
        {"id": "master_0", "motion": master_motion, "is_master": True,  "sensor_config": "all"},
        {"id": "wingman_1", "motion": wm1_motion,   "is_master": False, "sensor_config": "radar_only"},
        {"id": "wingman_2", "motion": wm2_motion,   "is_master": False, "sensor_config": "radar_esm"},
        {"id": "wingman_3", "motion": wm3_motion,   "is_master": False, "sensor_config": "radar_eo"},
        {"id": "wingman_4", "motion": wm4_motion,   "is_master": False, "sensor_config": "radar_only"},
    ]

    for p in platforms_info:
        _, p["sensors"] = create_sensor_platform(START_TIME, config=p.get("sensor_config", "all"))
        p["last_meas_time"] = {idx: START_TIME - period for idx, period in SENSOR_MEASUREMENT_PERIODS.items()}
        p["cached_arrived_data"] = None

    T_steps = len(truths[0])
    Data_Input = np.empty((T_steps,), dtype=object)
    
    # 标签扩展为 (T_steps, num_targets, 6)
    Data_Label = np.zeros((T_steps, num_targets, 6), dtype=np.float32)

    consec_invis_targets = np.zeros(num_targets, dtype=int)

    for t in range(T_steps):
        step_vis_targets = np.zeros(num_targets, dtype=bool)

        current_time_s = float((truths[0][t].timestamp - START_TIME).total_seconds())
        
        current_truths = [truths[i][t] for i in range(num_targets)]

        # 提取各个目标位置、速度，构建 Data_Label
        for i in range(num_targets):
            state = current_truths[i]
            target_pos = [state.state_vector[0], state.state_vector[2], state.state_vector[4]]
            target_vel = [state.state_vector[1], state.state_vector[3], state.state_vector[5]]
            Data_Label[t, i] = [target_pos[0], target_vel[0], target_pos[1], target_vel[1], target_pos[2], target_vel[2]]

        uav_list = np.empty((len(platforms_info),), dtype=object)

        for u, p in enumerate(platforms_info):
            p_pos, p_vel, p_acc = p["motion"].step()

            new_p_state = GroundTruthState(
                [p_pos[0], p_vel[0], p_pos[1], p_vel[1], p_pos[2], p_vel[2]],
                timestamp=current_truths[0].timestamp
            )

            p_state_9d = [
                float(p_pos[0]), float(p_vel[0]), float(p_acc[0]),
                float(p_pos[1]), float(p_vel[1]), float(p_acc[1]),
                float(p_pos[2]), float(p_vel[2]), float(p_acc[2])
            ]

            sensor_outputs = {}
            reconstructed_ecef_dict = {}  

            for idx, sensor in enumerate(p["sensors"]):
                period = SENSOR_MEASUREMENT_PERIODS[idx]
                if sensor is None:
                    continue
                if current_truths[0].timestamp - p["last_meas_time"][idx] >= period - timedelta(microseconds=10):
                    # measure函数能一次性接收多目标，返回所有探测结果
                    detections = sensor.measure(current_truths, platform_state=new_p_state)
                    obs_dict = {}
                    for d in detections:
                        if isinstance(d, TrueDetection):
                            tgt_idx = current_truths.index(d.groundtruth_path)
                            az, el, r = reorder_measurement(d.state_vector)
                            obs_dict[tgt_idx] = [az, el, r]
                            
                            if u == 0 and idx == 0:
                                step_vis_targets[tgt_idx] = True
                    if obs_dict:
                        is_jammed = False # np.random.rand() < 0.15 -> Disabled
                        if not is_jammed:
                            # Apply LocalTrackManager mapping
                            manager = track_managers[(u, idx)]
                            local_obs_dict = manager.process(obs_dict, current_time_s)
                            sensor_outputs[f"sensor_{idx}"] = {"observations": local_obs_dict}
                    p["last_meas_time"][idx] = current_truths[0].timestamp

            data_age = -1.0
            dl_xyz_dict = {}
            sens = sensor_outputs

            radar_obs = sens.get("sensor_0", {}).get("observations", {})
            eo_obs    = sens.get("sensor_1", {}).get("observations", {})
            esm_obs   = sens.get("sensor_2", {}).get("observations", {})
            
            radar_max_id = track_managers[(u, 0)].next_local_id - 1
            eo_max_id = track_managers[(u, 1)].next_local_id - 1
            esm_max_id = track_managers[(u, 2)].next_local_id - 1

            radar_meas, radar_mask = get_meas_array(radar_obs, 3, radar_max_id)
            eo_meas,    eo_mask    = get_meas_array(eo_obs,    2, eo_max_id)
            esm_meas,   esm_mask   = get_meas_array(esm_obs,   2, esm_max_id)

            cur_pos = p_pos 
            
            fast_meas_arr = np.zeros((num_targets, 3), dtype=np.float32)
            fast_mask_arr = np.zeros((num_targets,), dtype=np.float32)
            for tgt_idx, xyz in dl_xyz_dict.items():
                if tgt_idx < num_targets:
                    fast_meas_arr[tgt_idx] = xyz
                    fast_mask_arr[tgt_idx] = 1.0
            
            yaw = np.arctan2(p_vel[1], p_vel[0])
            pitch = np.arctan2(-p_vel[2], np.hypot(p_vel[0], p_vel[1]))
            euler_rpy = [0.0, float(pitch), float(yaw)]

            uav_data = {
                "Pose": {
                    "PositionWorld": [float(cur_pos[0]), float(cur_pos[1]), float(cur_pos[2])],
                    "VelocityWorld": [float(p_vel[0]), float(p_vel[1]), float(p_vel[2])],
                    "EulerRPY": euler_rpy,
                    "QuaternionWXYZ": [1.0, 0.0, 0.0, 0.0]
                },
                "IMU": {"Vector": [0.0, 0.0, 0.0, float(p_acc[0]), float(p_acc[1]), float(p_acc[2])]},
                
                # 更新为多目标格式：Meas 和 Mask 长度/行数均为 num_targets
                "Radar": {"Meas": radar_meas, "Mask": radar_mask},
                "EO":    {"Meas": eo_meas,    "Mask": eo_mask},
                "ESM":   {"Meas": esm_meas,   "Mask": esm_mask, "ModeCode": 0.0},
                
                "FastLink": {
                    "Meas": fast_meas_arr.tolist(), 
                    "Age": float(data_age), 
                    "IsUpdate": fast_mask_arr.tolist()
                },
                "SlowLink": {
                    "Meas": [[0.0, 0.0, 0.0] for _ in range(num_targets)], 
                    "Age": 0.0, 
                    "IsUpdate": [0.0]*num_targets
                }
            }
                
            uav_list[u] = uav_data

        Data_Input[t] = {
            "GlobalTime": current_time_s,
            "UAV": uav_list
        }

        for i in range(num_targets):
            if step_vis_targets[i]:
                consec_invis_targets[i] = 0
            else:
                consec_invis_targets[i] += 1
                
        if np.any(consec_invis_targets > 4):
            return None

    if np.any(consec_invis_targets > 0):
        return None

    end_time_s = float((truths[0][-1].timestamp - START_TIME).total_seconds())
    
    # Close all tracks and collect segments
    radar_segs, eo_segs, esm_segs = [], [], []
    nr_total, ne_total, nm_total = 0, 0, 0
    
    for (u, idx), manager in track_managers.items():
        manager.close_all(end_time_s)
        # Shift local IDs so they are globally unique across platforms (for the truth file)
        # However, MATLAB script uses `uavID` inferred from localID via a fixed array!
        # Wait, the MATLAB scripts: `Radar_uav = [1 1 1 2 2 2 3 3 3 4 4 4 1 2 3 4]`
        # We need to assign global localIDs and output a new format, or rely on our new batch script to read the truth!
        # Actually, let's just make the segments
        # To avoid MATLAB indexing issues, we will make localIDs globally unique across UAVs!
        pass
        
    # Wait, in the MATLAB Data_Input, local tracks are per UAV (so local_id 1 is for UAV 0, and another local_id 1 is for UAV 1).
    # But in the Truth Segments, they need global localIDs if they are going to map to the demo script.
    # Let's generate a general Truth array:
    # We will compute total tracks across all UAVs to populate SegTables.
    # We'll map (uav, local_id) -> global_local_id
    global_local_id_radar = 1
    global_local_id_eo = 1
    global_local_id_esm = 1
    
    radar_segs = []
    eo_segs = []
    esm_segs = []
    
    for (u, idx), manager in sorted(track_managers.items()):
        for seg in manager.closed_segments:
            # seg is [local_id, true_id, start, end]
            # convert to global_local_id
            if idx == 0:
                radar_segs.append([global_local_id_radar + seg[0] - 1, seg[1], seg[2], seg[3], u+1, seg[0]])
            elif idx == 1:
                eo_segs.append([global_local_id_eo + seg[0] - 1, seg[1], seg[2], seg[3], u+1, seg[0]])
            elif idx == 2:
                esm_segs.append([global_local_id_esm + seg[0] - 1, seg[1], seg[2], seg[3], u+1, seg[0]])
                
        if idx == 0: global_local_id_radar += (manager.next_local_id - 1)
        elif idx == 1: global_local_id_eo += (manager.next_local_id - 1)
        elif idx == 2: global_local_id_esm += (manager.next_local_id - 1)

    truth = {
        "Nr": global_local_id_radar - 1,
        "Ne": global_local_id_eo - 1,
        "Nm": global_local_id_esm - 1,
        "RadarSeg": radar_segs,
        "EOSeg": eo_segs,
        "ESMSeg": esm_segs,
        "N_true": num_targets,
        "t_grid": np.arange(0, end_time_s + DT, DT).tolist()
    }

    return {
        "Data_Input": Data_Input,
        "Data_Label": Data_Label,
        "T": DT,
        "truth": truth
    }


if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_targets", type=int, default=1)
        parser.add_argument("--num_trajectories", type=int, default=3000)
        parser.add_argument("--save_dir", type=str, default="data/train_data_cv_many2one")
        args = parser.parse_args()

        num_trajectories = args.num_trajectories
        num_targets = args.num_targets
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)

        print(f"Start generating {num_trajectories} trajectories (multi-target, {num_targets} targets)...")

        i = 0
        attempts = 0
        while i < num_trajectories:
            attempts += 1
            platform_heading = np.random.uniform(0, 2 * np.pi)
            platform_speed   = np.random.uniform(220, 300)
            platform_vel = [
                platform_speed * np.cos(platform_heading),
                platform_speed * np.sin(platform_heading),
                0.0
            ]

            fwd = np.array([np.cos(platform_heading), np.sin(platform_heading), 0.0])
            right = np.array([-np.sin(platform_heading), np.cos(platform_heading), 0.0])

            initial_targets = []
            for j in range(num_targets):
                range_init  = np.random.uniform(5000, 10000)
                az_offset   = np.random.uniform(-np.radians(45), np.radians(45))
                el_offset   = np.random.uniform(-np.radians(2), np.radians(2))

                dx = range_init * np.cos(el_offset) * np.cos(az_offset)
                dy = range_init * np.cos(el_offset) * np.sin(az_offset)
                dz = range_init * np.sin(el_offset)
                target_init_pos = np.array([0.0, 0.0, 10000.0]) + dx * fwd + dy * right + np.array([0, 0, dz])

                initial_speed  = platform_speed + np.random.uniform(-20, 20)
                target_heading = platform_heading + np.random.uniform(-np.radians(10), np.radians(10))
                vz_init = np.random.uniform(-10, 10)
                target_init_vel = np.array([
                    initial_speed * np.cos(target_heading),
                    initial_speed * np.sin(target_heading),
                    vz_init
                ])

                random_init_target = [
                    float(target_init_pos[0]), float(target_init_vel[0]),
                    float(target_init_pos[1]), float(target_init_vel[1]),
                    float(target_init_pos[2]), float(target_init_vel[2])
                ]
                initial_targets.append(random_init_target)

            data = run_simulation(seed=None, initial_targets=initial_targets, platform_vel=platform_vel, num_targets=num_targets)
            if data is None:
                continue

            filename = os.path.join(save_dir, f"track_data_{i:04d}.mat")
            scipy.io.savemat(filename, data, do_compression=True)
            i += 1
            if i % 10 == 0:
                print(f"  Generated {i} / {num_trajectories} trajectories...")

        print(f"All generation complete! Saved in {save_dir}/")

    except Exception:
        traceback.print_exc()
