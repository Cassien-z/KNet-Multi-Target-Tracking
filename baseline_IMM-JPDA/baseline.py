import json
import glob
import os
import numpy as np
from filterpy.kalman import KalmanFilter, IMMEstimator
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag


def body_polar_to_global_cartesian(az, el, r, platform_state):
    """逆解算：将极坐标转换回全局笛卡尔坐标系"""
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


def load_dataset(json_file):
    """从 JSON 中提取真实状态与所有雷达三维量测"""
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

            # 作为基线，这里依然只提取有距离信息的 sensor_0(雷达)
            if 'sensor_0' in sensors:
                for obs in sensors['sensor_0'].get('observations', []):
                    az, el, r = obs['meas']
                    global_xyz = body_polar_to_global_cartesian(az, el, r, p_state)
                    frame_measurements.append(global_xyz)

        all_measurements.append(frame_measurements)
    return all_measurements, ground_truths


def build_imm_tracker(dt=0.1):
    """使用 FilterPy 构建 IMM 滤波器"""
    # 1. 构建等速 (CV) 滤波器
    kf_cv = KalmanFilter(dim_x=9, dim_z=3)
    # 状态转移：加速度设为0
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

    # 3. 观测矩阵 (仅观测 x, y, z)
    H = np.zeros((3, 9))
    H[0, 0], H[1, 3], H[2, 6] = 1, 1, 1
    kf_cv.H = kf_ca.H = H

    # 4. 过程噪声 (CV噪声小，CA噪声大适应机动)
    q_cv = Q_discrete_white_noise(dim=3, dt=dt, var=1.0)
    q_ca = Q_discrete_white_noise(dim=3, dt=dt, var=10.0)
    kf_cv.Q = block_diag(q_cv, q_cv, q_cv)
    kf_ca.Q = block_diag(q_ca, q_ca, q_ca)

    # 5. 量测噪声 (设为固定大范围)
    R = np.diag([100 ** 2, 100 ** 2, 100 ** 2])
    kf_cv.R = kf_ca.R = R

    # 6. 初始概率和马尔可夫转移矩阵
    filters = [kf_cv, kf_ca]
    mu = [0.8, 0.2]
    M = np.array([[0.95, 0.05],
                  [0.05, 0.95]])

    return IMMEstimator(filters, mu, M)

def evaluate_per_frame(json_file):
    """
    逐帧评估单条轨迹，打印格式对齐 inference.py
    """
    if not os.path.exists(json_file):
        print(f"❌ 找不到文件: {json_file}")
        return

    measurements_seq, ground_truths = load_dataset(json_file)
    imm = build_imm_tracker(dt=0.1)

    # 赋予初始状态 (统一注入各个子滤波器)
    init_gt = ground_truths[0].reshape(9, 1)
    for f in imm.filters:
        f.x = init_gt.copy()
        f.P *= 1000
    imm.x = init_gt.copy()

    errors_pos = []

    print(f"📊 基线 (IMM) 评估：{json_file}")
    print(f"{'Frame':<6} | {'PosErr(m)':<10} | {'Status':<15}")

    for i, frame_measurements in enumerate(measurements_seq):
        # 第一步：IMM 预测
        imm.predict()

        # 第二步：单目标关联 (波门内最近邻)
        best_z = None
        min_dist = float('inf')
        z_pred = imm.x[[0, 3, 6], :]  # 提取预测的 XYZ

        for z in frame_measurements:
            dist = np.linalg.norm(z - z_pred)
            if dist < min_dist:
                min_dist = dist
                best_z = z

        # 门限：只有距离在 2000 米内的量测才被接受更新
        status = "Predict Only" # 默认状态为仅预测（丢失目标）
        if best_z is not None and min_dist < 2000:
            imm.update(best_z)
            status = "IMM Updated"

        # 第三步：提取最终后验状态计算误差
        gt_pos = ground_truths[i][[0, 3, 6]]
        est_pos = imm.x[[0, 3, 6], 0]

        err_pos = np.linalg.norm(est_pos - gt_pos)
        errors_pos.append(err_pos)

        # 逐帧打印输出
        print(f"{i:<6} | {err_pos:<10.1f} | {status}")

    # 计算整体 RMSE
    rmse = np.sqrt(np.mean(np.square(errors_pos)))
    print("\n" + "=" * 40)
    print(f"传统基线 (IMM) 总 RMSE: {rmse:.2f} m")
    print("=" * 40)


if __name__ == "__main__":
    # 🌟 修改这里：填入你想要逐帧测试的具体文件路径
    # 比如这里填入你之前在 inference.py 里测试用的文件
    test_file = "knet_verify_data/track_data_0985.json"

    # 调用我们刚刚写好的逐帧评估函数
    evaluate_per_frame(test_file)