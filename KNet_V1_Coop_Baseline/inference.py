# inference.py
import torch
import numpy as np
import json
import math
import os
from knet_model import KNet_Tracker

# 🟢 必须与 data_loader.py 严格一致
NORM_CFG = {
    "pos": 150000.0,
    "vel": 500.0,
    "r": 150000.0,
    "ang": np.pi
}


def convert_body_to_world_meas(az_body, el_body, r_body, vx, vy, vz):
    """
    将机载坐标系下的观测值，利用飞机速度矢量，配准到世界坐标系
    """
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
        # 🌟 初始化支持 3 平台的模型
        self.model = KNet_Tracker(num_platforms=3).to(self.device)

        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"✅ 成功加载多机协同权重：{weights_path}")
        else:
            raise FileNotFoundError(f"❌ 未找到权重文件：{weights_path}")

        self.model.eval()
        self.reset()

    def reset(self):
        self.hx = None
        self.curr_state = None
        self.last_time = None

    def _extract_platform_meas(self, p_data, p_state):
        """🌟 修复：从字典中提取数据，并利用 p_state 进行空间配准"""
        meas_arr = np.zeros(9, dtype=np.float32)
        mask_arr = np.zeros(9, dtype=np.float32)

        if not p_data or p_state is None:
            return meas_arr, mask_arr

        vx, vy, vz = p_state[1], p_state[3], p_state[5]
        sensors = p_data.get('sensors', {})

        # Sensor 0: Radar [az, el, r]
        if 'sensor_0' in sensors and sensors['sensor_0']['observations']:
            m = sensors['sensor_0']['observations'][0]['meas']
            az_w, el_w, r_w = convert_body_to_world_meas(m[0], m[1], m[2], vx, vy, vz)
            meas_arr[0:3] = [az_w / NORM_CFG['ang'], el_w / NORM_CFG['ang'], r_w / NORM_CFG['r']]
            mask_arr[0:3] = 1.0

        # Sensor 1: EO [az, el]
        if 'sensor_1' in sensors and sensors['sensor_1']['observations']:
            m = sensors['sensor_1']['observations'][0]['meas']
            az_w, el_w, _ = convert_body_to_world_meas(m[0], m[1], 1.0, vx, vy, vz)
            meas_arr[3:6] = [az_w / NORM_CFG['ang'], el_w / NORM_CFG['ang'], 0.0]
            mask_arr[3:5] = 1.0

        # Sensor 2: ESM [az]
        if 'sensor_2' in sensors and sensors['sensor_2']['observations']:
            m = sensors['sensor_2']['observations'][0]['meas']
            az_w, _, _ = convert_body_to_world_meas(m[0], 0.0, 1.0, vx, vy, vz)
            meas_arr[6:9] = [az_w / NORM_CFG['ang'], 0.0, 0.0]
            mask_arr[6] = 1.0

        return meas_arr, mask_arr

    def predict_step(self, platforms_dict, t):
        """多机协同单步推理"""
        master = platforms_dict.get('master_0', {})
        wm1 = platforms_dict.get('wingman_1', {})
        wm2 = platforms_dict.get('wingman_2', {})

        # 1. 提取长机状态用于初始化或基线计算
        m_state = np.array(master["platform_state"])
        wm1_state = np.array(wm1["platform_state"]) if wm1.get("platform_state") else m_state
        wm2_state = np.array(wm2["platform_state"]) if wm2.get("platform_state") else m_state

        # --- 第一帧初始化 (以长机视角) ---
        if self.curr_state is None:
            # 🌟 使用配准后的长机雷达初始化
            m_meas, _ = self._extract_platform_meas(master, m_state)
            az, el, r = m_meas[0] * NORM_CFG['ang'], m_meas[1] * NORM_CFG['ang'], m_meas[2] * NORM_CFG['r']
            x = r * math.cos(el) * math.cos(az)
            y = r * math.cos(el) * math.sin(az)
            z = r * math.sin(el)

            init_rel = np.array([x / NORM_CFG['pos'], 0.0, y / NORM_CFG['pos'], 0.0, z / NORM_CFG['pos'], 0.0],
                                dtype=np.float32)
            self.curr_state = torch.tensor(init_rel).unsqueeze(0).to(self.device)
            self.last_time = t
            return init_rel

        # --- 正常推理 ---
        dt = torch.tensor([max(0.0, t - self.last_time)], dtype=torch.float32).to(self.device)

        # 2. 构建 27维量测和掩码 (🌟 传入各自的物理状态进行空间配准)
        m_meas, m_mask = self._extract_platform_meas(master, m_state)
        w1_meas, w1_mask = self._extract_platform_meas(wm1, wm1_state)
        w2_meas, w2_mask = self._extract_platform_meas(wm2, wm2_state)

        meas_27 = torch.tensor(np.concatenate([m_meas, w1_meas, w2_meas])).unsqueeze(0).to(self.device)
        mask_27 = torch.tensor(np.concatenate([m_mask, w1_mask, w2_mask])).unsqueeze(0).to(self.device)

        # 3. 构建 12维基线向量 (Wingman - Master)
        base1 = (wm1_state - m_state)
        base2 = (wm2_state - m_state)

        # 归一化基线
        base_12_np = np.concatenate([base1, base2]).astype(np.float32)
        base_12_np[[0, 2, 4, 6, 8, 10]] /= NORM_CFG['pos']
        base_12_np[[1, 3, 5, 7, 9, 11]] /= NORM_CFG['vel']
        base_12 = torch.tensor(base_12_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.curr_state, self.hx = self.model(
                meas_27, mask_27, dt, base_12, self.curr_state, self.hx
            )

        self.last_time = t
        return self.curr_state.cpu().numpy()[0]


def evaluate(json_path, weights_path):
    engine = KNet_Engine(weights_path)
    with open(json_path, 'r') as f:
        data = json.load(f)

    errors_pos = []
    print(f"📊 协同评估：{json_path}")
    print(f"{'Frame':<6} | {'PosErr(m)':<10} | {'Status':<15}")

    for i, frame in enumerate(data):
        t = frame['time_s']
        gt_w = np.array(frame['ground_truth_state'])
        platforms = frame.get('platforms', {})
        master_state = platforms.get('master_0', {}).get('platform_state')

        if master_state is None: continue

        # 真值（相对于长机）
        gt_rel = gt_w - np.array(master_state)

        # 推理
        pred_norm = engine.predict_step(platforms, t)
        pred_pos = pred_norm[[0, 2, 4]] * NORM_CFG['pos']

        # 误差计算
        err_pos = np.linalg.norm(pred_pos - gt_rel[[0, 2, 4]])
        errors_pos.append(err_pos)

        if i % 1 == 0 or i == len(data) - 1:
            status = "Coop Active" if (len(platforms) > 1) else "Single Only"
            print(f"{i:<6} | {err_pos:<10.1f} | {status}")

    rmse = np.sqrt(np.mean(np.square(errors_pos)))
    print("\n" + "=" * 40)
    print(f"多机协同总 RMSE: {rmse:.2f} m")
    print("=" * 40)


if __name__ == "__main__":
    # 使用刚生成的验证集进行测试
    evaluate("knet_verify_data/track_data_000.json", "knet_weights_finetuned.pth")