# inference.py
import torch
import numpy as np
import json
import math
import os
from knet_model import KNet_Tracker

NORM_CFG = {
    "pos": 5000.0,   # 🌟 必须改为 5000.0
    "vel": 500.0,
    "acc": 50.0,     # 🌟 补上加速度基准
    "r": 150000.0,
    "ang": np.pi
}


def convert_body_to_world_meas(az_body, el_body, r_body, vx, vy, vz):
    r_calc = r_body if r_body > 0 else 1.0
    vec_body = np.array([
        r_calc * np.cos(el_body) * np.cos(az_body),
        r_calc * np.cos(el_body) * np.sin(az_body),
        r_calc * np.sin(el_body)
    ])

    v = np.array([vx, vy, vz])
    speed = np.linalg.norm(v)
    if speed < 1e-6:
        R_b2w = np.eye(3)
    else:
        unit_x = v / speed
        unit_y = np.cross([0, 0, 1], unit_x)
        if np.linalg.norm(unit_y) < 1e-6:
            unit_y = np.array([0, 1, 0])
        else:
            unit_y /= np.linalg.norm(unit_y)
        unit_z = np.cross(unit_x, unit_y)
        R_b2w = np.column_stack((unit_x, unit_y, unit_z))

    vec_world = R_b2w @ vec_body

    r_xy = np.sqrt(vec_world[0] ** 2 + vec_world[1] ** 2)
    az_world = np.arctan2(vec_world[1], vec_world[0])
    el_world = np.arctan2(vec_world[2], r_xy)

    return az_world, el_world, r_body


class KNet_Engine:
    def __init__(self, weights_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = KNet_Tracker().to(self.device)

        # 加载权重
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"✅ 成功加载推理模型权重: {weights_path}")
        else:
            print(f"⚠️ 找不到权重文件 {weights_path}，将使用随机初始化权重！")

        self.model.eval()

        # 🌟 修复报错：在这里初始化所有跨帧所需的变量
        self.prev_t = None
        self.ref_pos = None

        # 🌟 初始化 RNN 的跨帧状态（极其重要）
        self.h = None  # GRU 的隐藏层状态
        self.curr_state_in = None  # 物理运动学模型的上一帧后验状态

    def reset(self):
        self.hx = None
        self.curr_state = None
        self.last_time = None
        self.step_count = 0

    def _extract_platform_meas(self, p_data, p_state):
        meas_arr = np.zeros(9, dtype=np.float32)
        mask_arr = np.zeros(9, dtype=np.float32)

        if not p_data or p_state is None:
            return meas_arr, mask_arr

        # 🌟 智能兼容 6维或9维的飞机自身状态提取
        if len(p_state) == 9:
            vx, vy, vz = p_state[1], p_state[4], p_state[7]
        else:
            vx, vy, vz = p_state[1], p_state[3], p_state[5]

        sensors = p_data.get('sensors', {})

        if 'sensor_0' in sensors and sensors['sensor_0']['observations']:
            m = sensors['sensor_0']['observations'][0]['meas']
            az_w, el_w, r_w = convert_body_to_world_meas(m[0], m[1], m[2], vx, vy, vz)
            meas_arr[0:3] = [az_w / NORM_CFG['ang'], el_w / NORM_CFG['ang'], r_w / NORM_CFG['r']]
            mask_arr[0:3] = 1.0

        if 'sensor_1' in sensors and sensors['sensor_1']['observations']:
            m = sensors['sensor_1']['observations'][0]['meas']
            az_w, el_w, _ = convert_body_to_world_meas(m[0], m[1], 1.0, vx, vy, vz)
            meas_arr[3:6] = [az_w / NORM_CFG['ang'], el_w / NORM_CFG['ang'], 0.0]
            mask_arr[3:5] = 1.0

        if 'sensor_2' in sensors and sensors['sensor_2']['observations']:
            m = sensors['sensor_2']['observations'][0]['meas']
            az_w, _, _ = convert_body_to_world_meas(m[0], 0.0, 1.0, vx, vy, vz)
            meas_arr[6:9] = [az_w / NORM_CFG['ang'], 0.0, 0.0]
            mask_arr[6] = 1.0

        return meas_arr, mask_arr

    def _extract_platform_data(self, p_data, p_state):
        """提取单个平台的 9维测量 和 9维掩码，全部转化为笛卡尔空间特征！"""
        meas_arr = np.zeros(9, dtype=np.float32)
        mask_arr = np.zeros(9, dtype=np.float32)

        if not p_data or p_state is None:
            return meas_arr, mask_arr

        vx, vy, vz = p_state[1], p_state[4], p_state[7]
        sensors = p_data.get('sensors', {})

        # 🌟 核心修改 1：雷达直接算好笛卡尔坐标 dx, dy, dz
        if 'sensor_0' in sensors and sensors['sensor_0'].get('observations'):
            meas = sensors['sensor_0']['observations'][0]['meas']
            az_w, el_w, r_w = convert_body_to_world_meas(meas[0], meas[1], meas[2], vx, vy, vz)
            dx = r_w * np.cos(el_w) * np.cos(az_w)
            dy = r_w * np.cos(el_w) * np.sin(az_w)
            dz = r_w * np.sin(el_w)
            meas_arr[0:3] = [dx, dy, dz]
            mask_arr[0:3] = [1.0, 1.0, 1.0]

        # 🌟 核心修改 2：光电算好单位方向的笛卡尔射线 (放大至 NORM_CFG 以便归一化后为 1)
        if 'sensor_1' in sensors and sensors['sensor_1'].get('observations'):
            meas = sensors['sensor_1']['observations'][0]['meas']
            az_w, el_w, _ = convert_body_to_world_meas(meas[0], meas[1], 1.0, vx, vy, vz)
            r_pseudo = 5000.0  # NORM_CFG['pos'] 的值
            dx = r_pseudo * np.cos(el_w) * np.cos(az_w)
            dy = r_pseudo * np.cos(el_w) * np.sin(az_w)
            dz = r_pseudo * np.sin(el_w)
            meas_arr[3:6] = [dx, dy, dz]
            mask_arr[3:6] = [1.0, 1.0, 0.0]

        # 🌟 核心修改 3：ESM 算好 XY 平面射线
        if 'sensor_2' in sensors and sensors['sensor_2'].get('observations'):
            meas = sensors['sensor_2']['observations'][0]['meas']
            az_w, _, _ = convert_body_to_world_meas(meas[0], 0.0, 1.0, vx, vy, vz)
            r_pseudo = 5000.0
            dx = r_pseudo * np.cos(az_w)
            dy = r_pseudo * np.sin(az_w)
            dz = 0.0
            meas_arr[6:9] = [dx, dy, dz]
            mask_arr[6:9] = [1.0, 0.0, 0.0]

        return meas_arr, mask_arr

    def _get_observation_vector(self, platforms):
        """整合三个平台的观测与掩码，生成 27 维向量"""
        master = platforms.get('master_0', {})
        wm1 = platforms.get('wingman_1', {})
        wm2 = platforms.get('wingman_2', {})

        m_state = master.get("platform_state")
        wm1_state = wm1.get("platform_state")
        wm2_state = wm2.get("platform_state")

        m_meas, m_mask = self._extract_platform_data(master, m_state)
        w1_meas, w1_mask = self._extract_platform_data(wm1, wm1_state)
        w2_meas, w2_mask = self._extract_platform_data(wm2, wm2_state)

        meas_27 = np.concatenate([m_meas, w1_meas, w2_meas])
        mask_27 = np.concatenate([m_mask, w1_mask, w2_mask])
        return meas_27, mask_27

    def _get_baseline_vector(self, platforms, m_state):
        """计算僚机相对于长机的 18 维基线特征"""
        wm1 = platforms.get('wingman_1', {})
        wm2 = platforms.get('wingman_2', {})

        wm1_state = np.array(wm1["platform_state"]) if wm1.get("platform_state") else m_state
        wm2_state = np.array(wm2["platform_state"]) if wm2.get("platform_state") else m_state

        base1 = wm1_state - m_state
        base2 = wm2_state - m_state

        return np.concatenate([base1, base2])

    def predict_step(self, platforms, t):
        """
        单帧推理函数
        platforms: 当前帧从 JSON 读取的平台字典
        t: 当前帧的时间戳 (s)
        """
        if self.prev_t is None:
            dt_val = 0.1  # 第一帧默认间隔
        else:
            dt_val = t - self.prev_t
        self.prev_t = t
        dt_tensor = torch.tensor([[dt_val]], dtype=torch.float32).to(self.device)

        master_data = platforms.get('master_0', {})
        if not master_data:
            return None
        m_state = np.array(master_data['platform_state'])

        meas_27, mask_27 = self._get_observation_vector(platforms)
        baselines_18 = self._get_baseline_vector(platforms, m_state)

        # ========== 归一化输入 ==========
        meas_norm = meas_27.copy()
        base_norm = baselines_18.copy()

        # 🌟 修复一：将僚机观测统一到长机坐标系
        meas_norm[9:12] += base_norm[0:3]
        meas_norm[18:21] += base_norm[9:12]

        # 🌟 修复二：减去初始参考点 ref_pos，进入局部坐标系
        if hasattr(self, 'ref_pos') and self.ref_pos is not None:
            for i in [0, 9, 18]: meas_norm[i] -= self.ref_pos[0]
            for i in [1, 10, 19]: meas_norm[i] -= self.ref_pos[1]
            for i in [2, 11, 20]: meas_norm[i] -= self.ref_pos[2]

        meas_norm /= NORM_CFG['pos']

        base_norm[[0, 3, 6, 9, 12, 15]] /= NORM_CFG['pos']
        base_norm[[1, 4, 7, 10, 13, 16]] /= NORM_CFG['vel']
        base_norm[[2, 5, 8, 11, 14, 17]] /= NORM_CFG['acc']

        meas_tensor = torch.from_numpy(meas_norm).float().unsqueeze(0).unsqueeze(0).to(self.device)
        base_tensor = torch.from_numpy(base_norm).float().unsqueeze(0).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask_27).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # ========== 模型前向推理 ==========
        self.model.eval()
        with torch.no_grad():

            # 🌟 终极修复 1：安全提取历史记忆，防止张量污染
            safe_h_in = self.h.detach().clone() if self.h is not None else None
            safe_state_in = self.curr_state_in.detach().clone() if self.curr_state_in is not None else None

            pred_states_norm, next_h, next_state = self.model(
                meas_tensor, base_tensor, mask_tensor, dt_tensor,
                h_in=safe_h_in, state_in=safe_state_in
            )

            # 🌟 终极修复 2：将网络的新记忆安全保存
            self.h = next_h.detach().clone()
            self.curr_state_in = next_state.detach().clone()

            # 提取网络最新一帧预测的 9 维状态 (相对坐标)
            curr_state_norm = pred_states_norm[0, -1, :].cpu().numpy()

        # ========== 绝对坐标还原 ==========
        final_state = curr_state_norm.copy()

        # 乘回物理缩放尺度
        final_state[[0, 3, 6]] *= NORM_CFG['pos']
        final_state[[1, 4, 7]] *= NORM_CFG['vel']
        final_state[[2, 5, 8]] *= NORM_CFG['acc']

        # 加回初始参考点 ref_pos
        if hasattr(self, 'ref_pos') and self.ref_pos is not None:
            final_state[[0, 3, 6]] += self.ref_pos

        # 加回长机当前帧的绝对坐标
        final_state += m_state

        # 🌟 返回 6 维 [X, Vx, Y, Vy, Z, Vz] 供脚本评估
        return final_state[[0, 1, 3, 4, 6, 7]]


def evaluate(json_path, weights_path):
    engine = KNet_Engine(weights_path)
    with open(json_path, 'r') as f:
        data = json.load(f)

    errors_pos = []
    print(f"📊 协同评估：{json_path}")
    print(f"{'Frame':<6} | {'PosErr(m)':<10} | {'Status':<15}")

    # 🌟 新增：用于记录第一帧的局部坐标系原点
    ref_pos = None

    for i, frame in enumerate(data):
        t = frame['time_s']
        gt_w = np.array(frame['ground_truth_state'])
        platforms = frame.get('platforms', {})
        master_state = platforms.get('master_0', {}).get('platform_state')

        if master_state is None: continue

        # 🌟 新增：提取并初始化 ref_pos
        if ref_pos is None:
            # 提取真实目标的绝对初始位置 [X, Y, Z]
            if len(gt_w) == 9:
                gt_pos = gt_w[[0, 3, 6]]
            else:
                gt_pos = gt_w[[0, 2, 4]]

            # 提取长机的绝对初始位置 [X, Y, Z]
            m_pos = np.array(master_state)[[0, 3, 6]]

            # 记录“目标 - 长机”的初始相对距离作为原点
            ref_pos = gt_pos - m_pos
            # 把原点告诉推理引擎
            engine.ref_pos = ref_pos

        # 🌟 动态切片：无论 JSON 是旧的 6维还是新的 9维，都能精准切到 XYZ
        if len(gt_w) == 9:
            gt_abs_pos = gt_w[[0, 3, 6]]  # [X, 加速度X, 速度Y...] -> 修复为直接取 0, 3, 6 提取绝对位置
        else:
            gt_abs_pos = gt_w[[0, 2, 4]]

        pred_abs_state = engine.predict_step(platforms, t)
        # 统一约定的输出始终是 6 维 [X, Vx, Y, Vy, Z, Vz]
        pred_abs_pos = pred_abs_state[[0, 2, 4]]

        err_pos = np.linalg.norm(pred_abs_pos - gt_abs_pos)
        errors_pos.append(err_pos)

        if i % 1 == 0 or i == len(data) - 1:
            status = "Coop Active" if (len(platforms) > 1) else "Single Only"
            print(f"{i:<6} | {err_pos:<10.1f} | {status}")

    rmse = np.sqrt(np.mean(np.square(errors_pos)))
    print("\n" + "=" * 40)
    print(f"多机协同总 RMSE: {rmse:.2f} m")
    print("=" * 40)


if __name__ == "__main__":
    # 使用你的验证集进行测试
    evaluate("knet_verify_data/track_data_109.json", "knet_weights_finetuned.pth")