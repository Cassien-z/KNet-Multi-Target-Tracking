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
        all_meas = []
        all_masks = []  # 🌟 新增：记录每个时间步的 9 维细粒度掩码
        times = []
        # 如果数据长度超过设定长度，截取；如果不足，保留原长 (collate_fn会处理padding)
        # 这里我们先读取全部，后续在 collate_fn 或 train 中处理截断，或者在这里截断
        # 为了简单，这里读取全部，由 collate_fn 处理

        for frame in data:
            # 提取真值
            gt_w = np.array(frame['ground_truth_state'])
            p_w = np.array(frame['platform_state'])
            gt_rel = gt_w[[0, 2, 4]] - p_w[[0, 2, 4]]
            vel_rel = gt_w[[1, 3, 5]] - p_w[[1, 3, 5]]
            states.append([gt_rel[0], vel_rel[0], gt_rel[1], vel_rel[1], gt_rel[2], vel_rel[2]])

            # 🌟 新增：多传感器量测扩维与掩码生成
            frame_meas = np.zeros(9, dtype=np.float32)
            frame_mask = np.zeros(9, dtype=np.float32)
            sensors = frame.get('sensors', {})

            # 1. 雷达 (Sensor_0): [az, el, r]
            if 'sensor_0' in sensors and sensors['sensor_0']['observations']:
                meas = sensors['sensor_0']['observations'][0]['meas']
                frame_meas[0:3] = meas
                frame_mask[0:3] = [1.0, 1.0, 1.0]

            # 2. 光电 (Sensor_1): [az, el]
            if 'sensor_1' in sensors and sensors['sensor_1']['observations']:
                meas = sensors['sensor_1']['observations'][0]['meas']
                frame_meas[3:6] = [meas[0], meas[1], 0.0]
                frame_mask[3:6] = [1.0, 1.0, 0.0] # 距离项无数据

            # 3. 电子战 (Sensor_2): [az]
            if 'sensor_2' in sensors and sensors['sensor_2']['observations']:
                meas = sensors['sensor_2']['observations'][0]['meas']
                frame_meas[6:9] = [meas[0], 0.0, 0.0]
                frame_mask[6:9] = [1.0, 0.0, 0.0] # 俯仰、距离项无数据

            all_meas.append(frame_meas)
            all_masks.append(frame_mask)
            times.append(frame['time_s'])

        # ========== 归一化逻辑 ==========
        # 🌟 修复：新增状态真值的归一化逻辑
        norm_states = np.array(states, dtype=np.float32)
        norm_states[:, [0, 2, 4]] /= NORM_POS
        norm_states[:, [1, 3, 5]] /= NORM_VEL

        # 下面保留原有的量测归一化逻辑
        norm_meas = np.array(all_meas, dtype=np.float32)
        # 对所有的方位角和俯仰角位置（0,1, 3,4, 6）进行角度归一化
        norm_meas[:, [0, 1, 3, 4, 6]] /= NORM_ANG
        # 对雷达斜距（2）进行归一化（光电和电子战对应位置为0，除法不受影响）
        norm_meas[:, 2] /= NORM_R

        # ... (前面保留你的归一化逻辑和 dt 计算逻辑) ...

        # 计算 dt
        if len(times) > 1:
            dts = np.diff(times)
            first_dt = times[1] - times[0]
            dts = np.insert(dts, 0, first_dt)
        else:
            dts = np.array([0.1])
        dts = np.array(dts, dtype=np.float32)

        # 🌟 修复：在这里执行截断！防止超长序列导致 GPU OOM
        if len(norm_states) > self.seq_length:
            norm_states = norm_states[:self.seq_length]
            norm_meas = norm_meas[:self.seq_length]
            all_masks = all_masks[:self.seq_length]
            dts = dts[:self.seq_length]

        return {
            'states': torch.tensor(norm_states, dtype=torch.float32),
            'meas': torch.tensor(norm_meas, dtype=torch.float32),
            'mask': torch.tensor(np.array(all_masks), dtype=torch.float32),
            'dt': torch.tensor(dts, dtype=torch.float32),
            'file_name': self.files[idx]
        }


def collate_fn(batch):
    lengths = [item['states'].shape[0] for item in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    # Padding
    padded_states = torch.zeros(batch_size, max_len, 6)
    padded_meas = torch.zeros(batch_size, max_len, 9) # 🌟 测量维度变为 9
    padded_mask_input = torch.zeros(batch_size, max_len, 9) # 🌟 细粒度掩码维度变为 9
    padded_dt = torch.zeros(batch_size, max_len)

    # Loss 掩码保持 1 维 (表示该帧是否有效)
    mask_loss = torch.zeros(batch_size, max_len)

    for i, item in enumerate(batch):
        seq_len = item['states'].shape[0]
        padded_states[i, :seq_len, :] = item['states']
        padded_meas[i, :seq_len, :] = item['meas']
        padded_mask_input[i, :seq_len, :] = item['mask'] # 🌟 填入细粒度掩码
        padded_dt[i, :seq_len] = item['dt']
        mask_loss[i, :seq_len] = 1.0

    return {
        'states': padded_states,
        'meas': padded_meas,
        'mask_input': padded_mask_input, # 🌟 给网络输入的 9 维 Mask
        'mask_loss': mask_loss,          # 给损失函数的 1 维 Mask
        'dt': padded_dt
    }