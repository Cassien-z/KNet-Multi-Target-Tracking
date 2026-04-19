# inference.py
import torch
import numpy as np
import json
import math
import os
from knet_model import KNet_Tracker

NORM_CFG = {
    "pos": 150000.0,
    "vel": 500.0,
    "acc": 50.0,
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

    def predict_step(self, platforms_dict, t):
        self.step_count += 1
        master = platforms_dict.get('master_0', {})
        wm1 = platforms_dict.get('wingman_1', {})
        wm2 = platforms_dict.get('wingman_2', {})

        m_state = np.array(master["platform_state"])
        wm1_state = np.array(wm1["platform_state"]) if wm1.get("platform_state") else m_state
        wm2_state = np.array(wm2["platform_state"]) if wm2.get("platform_state") else m_state

        pos_idx = [0, 3, 6] if len(m_state) == 9 else [0, 2, 4]
        vel_idx = [1, 4, 7] if len(m_state) == 9 else [1, 3, 5]

        # ==========================================
        # 🌟 修复：两点差分高精度冷启动
        # ==========================================
        m_meas, _ = self._extract_platform_meas(master, m_state)
        az, el, r = m_meas[0] * NORM_CFG['ang'], m_meas[1] * NORM_CFG['ang'], m_meas[2] * NORM_CFG['r']
        x = r * math.cos(el) * math.cos(az)
        y = r * math.cos(el) * math.sin(az)
        z = r * math.sin(el)

        # ==========================================
        # 🌟 安全冷启动：完全信任第一帧雷达，速度给 0
        # ==========================================
        if self.curr_state is None:
            m_meas, _ = self._extract_platform_meas(master, m_state)
            az, el, r = m_meas[0] * NORM_CFG['ang'], m_meas[1] * NORM_CFG['ang'], m_meas[2] * NORM_CFG['r']
            x = r * math.cos(el) * math.cos(az)
            y = r * math.cos(el) * math.sin(az)
            z = r * math.sin(el)

            init_rel = np.array([
                x / NORM_CFG['pos'], 0.0, 0.0,
                y / NORM_CFG['pos'], 0.0, 0.0,
                z / NORM_CFG['pos'], 0.0, 0.0
            ], dtype=np.float32)

            self.curr_state = torch.tensor(init_rel).unsqueeze(0).to(self.device)
            self.last_time = t

            return np.array([
                m_state[pos_idx[0]] + x, m_state[vel_idx[0]],
                m_state[pos_idx[1]] + y, m_state[vel_idx[1]],
                m_state[pos_idx[2]] + z, m_state[vel_idx[2]]
            ])

        # --- 网络正式推断 ---
        # (保留原来正常的 dt, meas_27 构建及 model forward 逻辑)

        # --- 以下是网络正式推断逻辑 (从第 2 帧开始) ---
        dt = torch.tensor([max(0.0, t - self.last_time)], dtype=torch.float32).to(self.device)
        self.last_time = t

        m_meas, m_mask = self._extract_platform_meas(master, m_state)
        w1_meas, w1_mask = self._extract_platform_meas(wm1, wm1_state)
        w2_meas, w2_mask = self._extract_platform_meas(wm2, wm2_state)

        meas_27 = torch.tensor(np.concatenate([m_meas, w1_meas, w2_meas])).unsqueeze(0).to(self.device)
        mask_27 = torch.tensor(np.concatenate([m_mask, w1_mask, w2_mask])).unsqueeze(0).to(self.device)

        base1 = (wm1_state - m_state)
        base2 = (wm2_state - m_state)
        base_np = np.concatenate([base1, base2]).astype(np.float32)

        # 🌟 智能基线归一化
        if len(m_state) == 9:
            base_np[[0, 3, 6, 9, 12, 15]] /= NORM_CFG['pos']
            base_np[[1, 4, 7, 10, 13, 16]] /= NORM_CFG['vel']
            base_np[[2, 5, 8, 11, 14, 17]] /= NORM_CFG['acc']
        else:
            base_np[[0, 2, 4, 6, 8, 10]] /= NORM_CFG['pos']
            base_np[[1, 3, 5, 7, 9, 11]] /= NORM_CFG['vel']

        base_tensor = torch.tensor(base_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.curr_state, self.hx, _ = self.model(
                meas_27, mask_27, dt, base_tensor, self.curr_state, self.hx, step=self.step_count
            )

        out_np = self.curr_state.cpu().numpy()[0]
        pred_rel_pos = np.array([out_np[0], out_np[3], out_np[6]]) * NORM_CFG['pos']
        pred_rel_vel = np.array([out_np[1], out_np[4], out_np[7]]) * NORM_CFG['vel']

        # 最终恢复绝对坐标返回
        pred_abs_pos = m_state[pos_idx] + pred_rel_pos
        pred_abs_vel = m_state[vel_idx] + pred_rel_vel

        return np.array([
            pred_abs_pos[0], pred_abs_vel[0],
            pred_abs_pos[1], pred_abs_vel[1],
            pred_abs_pos[2], pred_abs_vel[2]
        ])


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