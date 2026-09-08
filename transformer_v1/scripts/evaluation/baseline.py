import json
import os
import numpy as np
import scipy.io
from filterpy.kalman import KalmanFilter, IMMEstimator
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag


# ========================================================
# 1. 坐标逆解算模块
# ========================================================

def body_polar_to_global_cartesian_json(az, el, r, platform_state):
    """【旧版 JSON 逆解算】将极坐标转换回全局笛卡尔坐标系 (依赖速度计算朝向)"""
    px, pvx = platform_state[0], platform_state[1]
    py, pvy = platform_state[3], platform_state[4]
    pz, pvz = platform_state[6], platform_state[7]

    dx_b = r * np.cos(el) * np.cos(az)
    dy_b = r * np.cos(el) * np.sin(az)
    dz_b = r * np.sin(el)
    vec_b = np.array([dx_b, dy_b, dz_b])

    v = np.array([pvx, pvy, pvz])
    speed = np.linalg.norm(v)
    if speed < 1e-6:
        vec_w = vec_b
    else:
        unit_x = v / speed
        unit_y = np.cross([0, 0, 1], unit_x)
        if np.linalg.norm(unit_y) < 1e-6:
            unit_y = np.array([0, 1, 0])
        unit_y /= np.linalg.norm(unit_y)
        unit_z = np.cross(unit_x, unit_y)

        R_bg = np.column_stack((unit_x, unit_y, unit_z))
        vec_w = R_bg @ vec_b

    return np.array([[vec_w[0] + px], [vec_w[1] + py], [vec_w[2] + pz]])


def ecef_to_latlon(x, y, z):
    """ECEF 转经纬度，用于推导东北天坐标系"""
    a_wgs84 = 6378137.0
    e2_wgs84 = 0.00669437999014
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p * (1.0 - e2_wgs84))
    for _ in range(3):
        N_rad = a_wgs84 / np.sqrt(1.0 - e2_wgs84 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2_wgs84 * N_rad * np.sin(lat), p)
    return lat, lon


def body_polar_to_global_cartesian_mat(az, el, r, u_pos, euler_rpy):
    """【新版 MAT 逆解算】完美复刻 ECEF 地心坐标系下的机体坐标系反推"""
    pitch, yaw = euler_rpy[1], euler_rpy[2]

    lat, lon = ecef_to_latlon(u_pos[0], u_pos[1], u_pos[2])
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    # ENU -> ECEF
    R_n2e = np.array([
        [-slon, -slat * clon, clat * clon],
        [clon, -slat * slon, clat * slon],
        [0.0, clat, slat]
    ])

    fwd_enu = np.array([
        np.sin(yaw) * np.cos(pitch),
        np.cos(yaw) * np.cos(pitch),
        np.sin(pitch)
    ])
    unit_fwd = R_n2e @ fwd_enu
    unit_fwd = unit_fwd / (np.linalg.norm(unit_fwd) + 1e-8)

    local_up = np.array(u_pos) / np.linalg.norm(u_pos)
    unit_right = np.cross(local_up, unit_fwd)
    unit_right = unit_right / (np.linalg.norm(unit_right) + 1e-8)
    unit_up = np.cross(unit_fwd, unit_right)
    unit_up = unit_up / (np.linalg.norm(unit_up) + 1e-8)

    # Body -> ECEF
    R_body2ecef = np.column_stack((unit_fwd, unit_right, unit_up))

    dx_b = r * np.cos(el) * np.cos(az)
    dy_b = r * np.cos(el) * np.sin(az)
    dz_b = r * np.sin(el)
    rel_body = np.array([dx_b, dy_b, dz_b])

    rel_ecef = R_body2ecef @ rel_body
    t_pos = np.array(u_pos) + rel_ecef

    return np.array([[t_pos[0]], [t_pos[1]], [t_pos[2]]])


# ========================================================
# 2. 数据读取模块
# ========================================================

def load_dataset_json(json_file):
    """读取旧版 JSON 格式"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    all_measurements = []
    ground_truths = []

    for frame in data:
        ground_truths.append(np.array(frame['ground_truth_state']))
        frame_measurements = []

        for platform_id, p_data in frame['platforms'].items():
            if p_data.get('data_age', -1) < 0 or p_data.get('platform_state') is None:
                continue

            p_state = p_data['platform_state']
            sensors = p_data.get('sensors', {})

            if 'sensor_0' in sensors:
                for obs in sensors['sensor_0'].get('observations', []):
                    az, el, r = obs['meas']
                    global_xyz = body_polar_to_global_cartesian_json(az, el, r, p_state)
                    frame_measurements.append(global_xyz)

        all_measurements.append(frame_measurements)
    return all_measurements, ground_truths


def load_dataset_mat(mat_file):
    """读取新版 MAT 格式，并适配异构无人机蜂群数据"""
    mat_data = scipy.io.loadmat(mat_file, simplify_cells=True)

    data_label = mat_data['Data_Label']
    if data_label.ndim == 3:
        # For multi-target data (T, num_targets, 6), take target 0
        data_label = data_label[:, 0, :]
    # 自动处理转置
    if data_label.shape[0] == 6 and data_label.shape[1] > 6:
        data_label = data_label.T

    data_input = mat_data['Data_Input']
    if isinstance(data_input, np.ndarray) and data_input.ndim > 1:
        data_input = data_input.flatten()

    all_measurements = []
    ground_truths = []

    for i in range(len(data_input)):
        frame = data_input[i]

        # 将 6D 标签扩充为 IMM 所需的 9D [X, Vx, Ax, Y, Vy, Ay, Z, Vz, Az]
        gt_6d = data_label[i]
        gt_9d = np.array([
            gt_6d[0], gt_6d[1], 0.0,
            gt_6d[2], gt_6d[3], 0.0,
            gt_6d[4], gt_6d[5], 0.0
        ])
        ground_truths.append(gt_9d)

        frame_measurements = []
        uavs = frame['UAV']
        if isinstance(uavs, np.ndarray):
            uavs = uavs.flatten()

        for u in range(len(uavs)):
            uav_data = uavs[u]
            pose = uav_data['Pose']
            u_pos = pose['PositionWorld']
            euler_rpy = pose['EulerRPY']

            if u == 0:
                # 长机：有原生雷达视野，需要逆解算
                radar = uav_data.get('Radar', {})
                if np.any(radar.get('Mask', 0.0)):
                    meas_array = np.atleast_2d(radar['Meas'])
                    for meas in meas_array:
                        az, el, r = meas
                        global_xyz = body_polar_to_global_cartesian_mat(az, el, r, u_pos, euler_rpy)
                        frame_measurements.append(global_xyz)
            else:
                # 僚机：利用数据链中的 FastLink 获取目标的三维坐标 (已在生成时加噪并推算)
                fast_link = uav_data.get('FastLink', {})
                if np.any(fast_link.get('IsUpdate', 0.0)):
                    meas_array = np.atleast_2d(fast_link['Meas'])
                    for global_xyz in meas_array:
                        if global_xyz.shape == (3,):
                            global_xyz = global_xyz.reshape(3, 1)
                        frame_measurements.append(global_xyz)

        all_measurements.append(frame_measurements)

    return all_measurements, ground_truths


# ========================================================
# 3. 跟踪与评估模块
# ========================================================

def build_imm_tracker(dt=0.1):
    """使用 FilterPy 构建 IMM 滤波器"""
    # 1. 构建等速 (CV) 滤波器
    kf_cv = KalmanFilter(dim_x=9, dim_z=3)
    F_cv_1d = np.array([[1, dt, 0],
                        [0, 1, 0],
                        [0, 0, 0]])
    kf_cv.F = block_diag(F_cv_1d, F_cv_1d, F_cv_1d)

    # 2. 构建等加速 (CA) 滤波器
    kf_ca = KalmanFilter(dim_x=9, dim_z=3)
    F_ca_1d = np.array([[1, dt, 0.5 * dt ** 2],
                        [0, 1, dt],
                        [0, 0, 1]])
    kf_ca.F = block_diag(F_ca_1d, F_ca_1d, F_ca_1d)

    # 3. 构建左转弯 (CT-L) 和 右转弯 (CT-R) 滤波器
    def make_ct_model(omega):
        kf = KalmanFilter(dim_x=9, dim_z=3)
        F = np.zeros((9, 9))
        sin_wt = np.sin(omega * dt)
        cos_wt = np.cos(omega * dt)
        
        F[0, 0] = 1; F[0, 1] = sin_wt / omega; F[0, 4] = -(1 - cos_wt) / omega
        F[1, 1] = cos_wt; F[1, 4] = -sin_wt
        F[3, 3] = 1; F[3, 1] = (1 - cos_wt) / omega; F[3, 4] = sin_wt / omega
        F[4, 1] = sin_wt; F[4, 4] = cos_wt
        
        # Z轴保持 CV
        F[6, 6] = 1; F[6, 7] = dt
        F[7, 7] = 1
        kf.F = F
        
        H = np.zeros((3, 9))
        H[0, 0], H[1, 3], H[2, 6] = 1, 1, 1
        kf.H = H
        
        q = Q_discrete_white_noise(dim=3, dt=dt, var=40.0)
        kf.Q = block_diag(q, q, q)
        kf.R = np.diag([66.71 ** 2, 66.71 ** 2, 66.71 ** 2])
        return kf

    kf_ct_l = make_ct_model(0.06)   # 左转 ~3.4 deg/s
    kf_ct_r = make_ct_model(-0.06)  # 右转 ~3.4 deg/s

    # 4. 观测矩阵与过程噪声 (CV 和 CA)
    H = np.zeros((3, 9))
    H[0, 0], H[1, 3], H[2, 6] = 1, 1, 1
    kf_cv.H = kf_ca.H = H

    q_cv = Q_discrete_white_noise(dim=3, dt=dt, var=52.21)
    q_ca = Q_discrete_white_noise(dim=3, dt=dt, var=33.96)
    kf_cv.Q = block_diag(q_cv, q_cv, q_cv)
    kf_ca.Q = block_diag(q_ca, q_ca, q_ca)

    # 5. 量测噪声 (CV 和 CA)
    R = np.diag([66.71 ** 2, 66.71 ** 2, 66.71 ** 2])
    kf_cv.R = kf_ca.R = R

    # 6. 初始概率和马尔可夫转移矩阵 (4个模型)
    # 【修复核心】之前 M 矩阵对角线只有 0.70，导致模型每 0.3 秒就在互相高频切换
    # 在量测噪声大 (R=66.7^2) 的情况下，会引发严重的“概率稀释”，四个模型的预测被强行平均混合
    # 现在改为 0.95 级别，确保每个模型能稳定生效。
    filters = [kf_cv, kf_ca, kf_ct_l, kf_ct_r]
    mu = [0.85, 0.05, 0.05, 0.05]
    M = np.array([
        [0.94, 0.02, 0.02, 0.02],  # 从 CV 状态可以小概率切换到机动状态
        [0.05, 0.95, 0.00, 0.00],  # 从 CA 状态倾向于保持或回到 CV，避免与 CT 串扰
        [0.05, 0.00, 0.95, 0.00],  # 从 左转 倾向于保持或回到 CV
        [0.05, 0.00, 0.00, 0.95]   # 从 右转 倾向于保持或回到 CV
    ])

    return IMMEstimator(filters, mu, M)


def evaluate_per_frame(file_path):
    """
    逐帧评估单条轨迹，自动适配 JSON 或 MAT 格式
    """
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    # 动态路由解析器
    if file_path.endswith('.json'):
        measurements_seq, ground_truths = load_dataset_json(file_path)
    elif file_path.endswith('.mat'):
        measurements_seq, ground_truths = load_dataset_mat(file_path)
    else:
        print("❌ 仅支持 .json 或 .mat 格式的数据！")
        return

    imm = build_imm_tracker(dt=0.1)

    # 赋予初始状态
    init_gt = ground_truths[0].reshape(9, 1)
    for f in imm.filters:
        f.x = init_gt.copy()
        f.P *= 1000
    imm.x = init_gt.copy()

    errors_pos = []

    print(f"📊 基线 (IMM) 评估：{file_path}")
    print(f"{'Frame':<6} | {'PosErr(m)':<10} | {'Status':<15}")

    for i, frame_measurements in enumerate(measurements_seq):
        imm.predict()

        # 波门内最近邻 (NN) 关联
        best_z = None
        min_dist = float('inf')
        z_pred = imm.x[[0, 3, 6], :]

        for z in frame_measurements:
            dist = np.linalg.norm(z - z_pred)
            if dist < min_dist:
                min_dist = dist
                best_z = z

        status = "Predict Only"
        if best_z is not None and min_dist < 2000:
            imm.update(best_z)
            status = "IMM Updated"

        gt_pos = ground_truths[i][[0, 3, 6]]
        est_pos = imm.x[[0, 3, 6], 0]

        err_pos = np.linalg.norm(est_pos - gt_pos)
        errors_pos.append(err_pos)

        # 减小打印频率以避免刷屏，每 100 帧打印一次
        if i % 10 == 0 or i == len(measurements_seq) - 1:
            print(f"{i:<6} | {err_pos:<10.1f} | {status}")

    rmse = np.sqrt(np.mean(np.square(errors_pos)))
    print("\n" + "=" * 40)
    print(f"传统基线 (IMM) 总 RMSE: {rmse:.2f} m")
    print("=" * 40)


if __name__ == "__main__":
    # 指向你刚刚生成好的 .mat 文件
    test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "train_data", "single_target", "train_data_ca", "track_data_ca_0342.mat")
    evaluate_per_frame(test_file)