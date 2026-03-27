# data_loader.py
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import math
import os

# 🟢 统一归一化配置 (必须与 inference.py 和 knet_model.py 严格一致)
NORM_POS = 150000.0  # 150km
NORM_VEL = 500.0  # 500m/s
NORM_R = 150000.0  # 150km
NORM_ANG = np.pi  # 角度除以 pi


def wrap_to_pi(angle):
    """将角度映射到 [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class MultiTrackingDataset(Dataset):
    def __init__(self, data_dir, seq_length=600, seq_len=None):
        """
        初始化数据集
        :param seq_length: 序列长度 (优先使用)
        :param seq_len: 兼容旧代码的参数别名
        """
        # 兼容处理：如果传了 seq_len 则使用它
        if seq_len is not None:
            self.seq_length = seq_len
        else:
            self.seq_length = seq_length

        self.data_dir = data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在：{data_dir}")

        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])

        if not self.files:
            raise ValueError(f"未在 {data_dir} 找到任何 JSON 文件")

        print(f"📂 加载数据集：{len(self.files)} 个轨迹文件 (Dir: {data_dir})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise e

        states = []
        measurements = []
        times = []

        # 如果数据长度超过设定长度，截取；如果不足，保留原长 (collate_fn会处理padding)
        # 这里我们先读取全部，后续在 collate_fn 或 train 中处理截断，或者在这里截断
        # 为了简单，这里读取全部，由 collate_fn 处理

        for i, frame in enumerate(data):
            # 1. 解析真值
            gt_w = np.array(frame.get('ground_truth_state', [0] * 6))
            p_w = np.array(frame.get('platform_state', [0] * 6))

            rel_pos = gt_w[[0, 2, 4]] - p_w[[0, 2, 4]]
            rel_vel = gt_w[[1, 3, 5]] - p_w[[1, 3, 5]]

            state_cart = np.zeros(6, dtype=np.float32)
            state_cart[0::2] = rel_pos
            state_cart[1::2] = rel_vel
            states.append(state_cart)

            # 2. 解析观测
            obs_list = frame.get('sensors', {}).get('sensor_0', {}).get('observations', [])

            if not obs_list:
                if len(measurements) > 0:
                    measurements.append(measurements[-1])
                else:
                    measurements.append([0.0, 0.0, 1000.0])
            else:
                meas = obs_list[0].get('meas', [0, 0, 1000])
                az = wrap_to_pi(meas[0])
                el = wrap_to_pi(meas[1])
                r = meas[2]
                measurements.append([az, el, r])

            times.append(frame.get('time_s', 0.0))

        states = np.array(states, dtype=np.float32)
        measurements = np.array(measurements, dtype=np.float32)
        times = np.array(times, dtype=np.float32)

        # --- 归一化 ---
        norm_states = states.copy()
        norm_states[:, 0::2] /= NORM_POS
        norm_states[:, 1::2] /= NORM_VEL

        norm_meas = measurements.copy()
        norm_meas[:, 0] /= NORM_ANG
        norm_meas[:, 1] /= NORM_ANG
        norm_meas[:, 2] /= NORM_R

        # 计算 dt
        if len(times) > 1:
            dts = np.diff(times)
            first_dt = times[1] - times[0]
            dts = np.insert(dts, 0, first_dt)
        else:
            dts = np.array([0.1])
        dts = np.array(dts, dtype=np.float32)

        return {
            'states': torch.tensor(norm_states, dtype=torch.float32),
            'meas': torch.tensor(norm_meas, dtype=torch.float32),
            'dt': torch.tensor(dts, dtype=torch.float32),
            'file_name': self.files[idx]
        }


def collate_fn(batch):
    lengths = [item['states'].shape[0] for item in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    # Padding
    padded_states = torch.zeros(batch_size, max_len, 6)
    padded_meas = torch.zeros(batch_size, max_len, 3)
    padded_dt = torch.zeros(batch_size, max_len)

    # Mask for loss calculation (1 for real data, 0 for padding)
    mask_tensor = torch.zeros(batch_size, max_len, 3)

    for i, item in enumerate(batch):
        l = item['states'].shape[0]
        padded_states[i, :l, :] = item['states']
        padded_meas[i, :l, :] = item['meas']
        padded_dt[i, :l] = item['dt']
        mask_tensor[i, :l, :] = 1.0  # 有效数据标记

    return {
        'states': padded_states,
        'meas': padded_meas,
        'dt': padded_dt,
        'mask': mask_tensor,  # 新增 mask 用于 Loss 计算
        'lengths': lengths
    }