# inference.py
import torch
import numpy as np
import json
import math
import os
from knet_model import KNet_Tracker

# 🟢 必须与 data_loader.py 和 knet_model.py 严格一致
NORM_CFG = {
    "pos": 150000.0,  # 150km
    "vel": 500.0,  # 500m/s
    "r": 150000.0,  # 150km
    "ang": np.pi  # 角度除以 pi
}


class KNet_Engine:
    def __init__(self, weights_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = KNet_Tracker().to(self.device)

        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"✅ 成功加载权重：{weights_path}")
        else:
            raise FileNotFoundError(f"❌ 未找到权重文件：{weights_path}")

        self.model.eval()

        # 统一状态重置
        self.reset()

    def reset(self):
        """重置引擎状态（建议外部每次开始新序列时调用此方法）"""
        self.hx = None
        self.curr_state = None
        self.last_time = None
        # 🌟 新增：用于两点法差分初始化的缓存
        self.first_pos = None
        self.first_time = None

    def _sph2cart(self, meas_sph):
        """极坐标转直角坐标辅助函数"""
        az, el, r = meas_sph[0], meas_sph[1], meas_sph[2]
        x = r * math.cos(el) * math.cos(az)
        y = r * math.cos(el) * math.sin(az)
        z = r * math.sin(el)
        return np.array([x, y, z])

    def predict_step(self, sensors_dict, t):
        """单步推理
        sensors_dict: frame['sensors']
        """
        # 构建 9维 meas 和 9维 mask
        frame_meas = np.zeros(9, dtype=np.float32)
        frame_mask = np.zeros(9, dtype=np.float32)

        if 'sensor_0' in sensors_dict and sensors_dict['sensor_0']['observations']:
            meas = sensors_dict['sensor_0']['observations'][0]['meas']
            frame_meas[0:3] = meas
            frame_mask[0:3] = [1.0, 1.0, 1.0]

        if 'sensor_1' in sensors_dict and sensors_dict['sensor_1']['observations']:
            meas = sensors_dict['sensor_1']['observations'][0]['meas']
            frame_meas[3:6] = [meas[0], meas[1], 0.0]
            frame_mask[3:6] = [1.0, 1.0, 0.0]

        if 'sensor_2' in sensors_dict and sensors_dict['sensor_2']['observations']:
            meas = sensors_dict['sensor_2']['observations'][0]['meas']
            frame_meas[6:9] = [meas[0], 0.0, 0.0]
            frame_mask[6:9] = [1.0, 0.0, 0.0]

            # --- 零速度安全初始化逻辑 ---
        if self.curr_state is None:
            # 初始化必须依赖至少有方位角和距离的传感器，默认用雷达数据起步
            az, el, r = frame_meas[0], frame_meas[1], frame_meas[2]
            x = r * math.cos(el) * math.cos(az)
            y = r * math.cos(el) * math.sin(az)
            z = r * math.sin(el)

            state = np.array([x, 0.0, y, 0.0, z, 0.0])

            norm_state = state.copy()
            norm_state[[0, 2, 4]] /= NORM_CFG['pos']
            norm_state[[1, 3, 5]] /= NORM_CFG['vel']

            self.curr_state = torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.last_time = t
            return self.curr_state.detach().cpu().numpy()[0]

        # --- 正常推理阶段 ---
        dt_val = t - self.last_time
        if dt_val < 0: dt_val = 0.0
        dt = torch.tensor([dt_val], dtype=torch.float32).to(self.device)

        # 对 9 维数据归一化
        norm_meas = frame_meas.copy()
        norm_meas[[0, 1, 3, 4, 6]] /= NORM_CFG['ang']
        norm_meas[2] /= NORM_CFG['r']

        norm_meas_t = torch.tensor(norm_meas, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_t = torch.tensor(frame_mask, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.curr_state, self.hx = self.model(
                norm_meas_t, mask_t, dt, self.curr_state, self.hx
            )

        self.last_time = t
        return self.curr_state.detach().cpu().numpy()[0]


def evaluate(json_path, weights_path):
    engine = KNet_Engine(weights_path)

    with open(json_path, 'r') as f:
        data = json.load(f)

    errors_pos = []
    errors_range = []
    errors_az = []

    start_time = data[0]['time_s'] if data else 0

    print(f"📊 开始评估：{json_path}")
    print(f"{'Frame':<6} | {'Time':<6} | {'PosErr(m)':<10} | {'RangeErr(m)':<10} | {'AzErr(deg)':<10}")

    for i, frame in enumerate(data):
        t = frame['time_s']

        # 真值
        gt_w = np.array(frame['ground_truth_state'])
        p_w = np.array(frame['platform_state'])
        gt_rel = gt_w[[0, 2, 4]] - p_w[[0, 2, 4]]

        true_r = np.linalg.norm(gt_rel)
        true_az = math.atan2(gt_rel[1], gt_rel[0])
        true_el = math.asin(gt_rel[2] / (true_r + 1e-6))

        # ---------- 修复前：----------
        # obs = frame['sensors']['sensor_0']['observations']
        # if not obs: continue
        # meas_sph = obs[0]['meas']  # [az, el, r] (弧度，米)
        # pred_norm = engine.predict_step(meas_sph, t)

        # ========== 修复后：==========
        # 将整个传感器字典提取出来，支持多模态解析
        sensors_dict = frame.get('sensors', {})

        # 必须确保至少有雷达数据来起步
        if 'sensor_0' not in sensors_dict or not sensors_dict['sensor_0']['observations']:
            continue

        # 预测：把整个字典喂给 KNet_Engine
        pred_norm = engine.predict_step(sensors_dict, t)
        pred_pos = pred_norm[[0, 2, 4]] * NORM_CFG['pos']

        # 误差
        err_pos = np.linalg.norm(pred_pos - gt_rel)
        pred_r = np.linalg.norm(pred_pos)
        err_range = abs(pred_r - true_r)

        pred_az = math.atan2(pred_pos[1], pred_pos[0])
        diff_az = true_az - pred_az
        while diff_az > np.pi: diff_az -= 2 * np.pi
        while diff_az < -np.pi: diff_az += 2 * np.pi
        err_az = abs(diff_az) * 180.0 / np.pi

        errors_pos.append(err_pos)
        errors_range.append(err_range)
        errors_az.append(err_az)

        if i % 50 == 0 or i < 5 or i > len(data) - 5:
            print(f"{i:<6} | {t:<6.1f} | {err_pos:<10.1f} | {err_range:<10.1f} | {err_az:<10.4f}")

    rmse_pos = np.sqrt(np.mean(np.square(errors_pos)))
    steady_errors = errors_pos[int(len(errors_pos) * 0.1):]  # 去掉前10%
    rmse_steady = np.sqrt(np.mean(np.square(steady_errors))) if steady_errors else 0

    print("\n" + "=" * 50)
    print(f"总 RMSE: {rmse_pos:.2f} m")
    print(f"稳态 RMSE (>10%): {rmse_steady:.2f} m")
    print(f"平均角度误差: {np.mean(errors_az):.4f} °")
    print("=" * 50)


if __name__ == "__main__":
    evaluate("knet_verify_data/track_data_008.json", "knet_weights_finetuned.pth")