# data_loader.py
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

# 🟢 统一归一化配置
NORM_POS = 150000.0  # 150km
NORM_VEL = 500.0  # 500m/s
NORM_R = 150000.0  # 150km
NORM_ANG = np.pi  # 角度除以 pi


class MultiTrackingDataset(Dataset):
    def __init__(self, data_dir, seq_length=100, seq_len=None):
        if seq_len is not None:
            self.seq_length = seq_len
        else:
            self.seq_length = seq_length
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
        print(f"📂 加载多机协同数据集：{len(self.files)} 个文件 (Dir: {data_dir})")

    def __len__(self):
        return len(self.files)

    def _extract_platform_data(self, p_data, p_state):
        """提取单个平台的 9维测量 和 9维掩码，并进行空间配准"""
        meas_arr = np.zeros(9, dtype=np.float32)
        mask_arr = np.zeros(9, dtype=np.float32)

        if p_data is None or p_state is None:
            return meas_arr, mask_arr

        # 获取平台速度
        vx, vy, vz = p_state[1], p_state[3], p_state[5]
        sensors = p_data.get('sensors', {})

        if 'sensor_0' in sensors and sensors['sensor_0']['observations']:
            meas = sensors['sensor_0']['observations'][0]['meas']
            # 🌟 配准：Body -> World
            az_w, el_w, r_w = convert_body_to_world_meas(meas[0], meas[1], meas[2], vx, vy, vz)
            meas_arr[0:3] = [az_w, el_w, r_w]
            mask_arr[0:3] = [1.0, 1.0, 1.0]

        if 'sensor_1' in sensors and sensors['sensor_1']['observations']:
            meas = sensors['sensor_1']['observations'][0]['meas']
            # 🌟 配准：缺省距离用 1.0 代替运算
            az_w, el_w, _ = convert_body_to_world_meas(meas[0], meas[1], 1.0, vx, vy, vz)
            meas_arr[3:6] = [az_w, el_w, 0.0]
            mask_arr[3:6] = [1.0, 1.0, 0.0]

        if 'sensor_2' in sensors and sensors['sensor_2']['observations']:
            meas = sensors['sensor_2']['observations'][0]['meas']
            # 🌟 配准：缺省仰角 0.0，距离 1.0
            az_w, _, _ = convert_body_to_world_meas(meas[0], 0.0, 1.0, vx, vy, vz)
            meas_arr[6:9] = [az_w, 0.0, 0.0]
            mask_arr[6:9] = [1.0, 0.0, 0.0]

        return meas_arr, mask_arr

    def __getitem__(self, idx):
        with open(os.path.join(self.data_dir, self.files[idx]), 'r') as f:
            data = json.load(f)

        states, baselines, all_meas, all_masks, times = [], [], [], [], []

        for frame in data:
            gt_w = np.array(frame['ground_truth_state'])
            platforms = frame.get('platforms', {})

            master = platforms.get('master_0', {})
            wm1 = platforms.get('wingman_1', {})
            wm2 = platforms.get('wingman_2', {})

            # 必须有长机状态，否则跳过
            if master.get("platform_state") is None: continue

            m_state = np.array(master["platform_state"])
            wm1_state = np.array(wm1["platform_state"]) if wm1.get("platform_state") else m_state
            wm2_state = np.array(wm2["platform_state"]) if wm2.get("platform_state") else m_state

            # 1. 目标相对长机的状态
            gt_rel = gt_w - m_state
            states.append([gt_rel[0], gt_rel[1], gt_rel[2], gt_rel[3], gt_rel[4], gt_rel[5]])

            # 2. 僚机相对长机的基线向量 (Wingman - Master)
            base1 = wm1_state - m_state
            base2 = wm2_state - m_state
            baselines.append(np.concatenate([base1, base2]))

            # 3. 提取 3 个平台的量测 (27维)
            m_meas, m_mask = self._extract_platform_data(master, m_state)
            w1_meas, w1_mask = self._extract_platform_data(wm1, wm1_state)
            w2_meas, w2_mask = self._extract_platform_data(wm2, wm2_state)

            all_meas.append(np.concatenate([m_meas, w1_meas, w2_meas]))
            all_masks.append(np.concatenate([m_mask, w1_mask, w2_mask]))
            times.append(frame['time_s'])

        # ========== 归一化 ==========
        norm_states = np.array(states, dtype=np.float32)
        norm_states[:, [0, 2, 4]] /= NORM_POS
        norm_states[:, [1, 3, 5]] /= NORM_VEL

        # 基线向量同样需要归一化 (让网络在同一尺度下运算)
        norm_bases = np.array(baselines, dtype=np.float32)
        norm_bases[:, [0, 2, 4, 6, 8, 10]] /= NORM_POS
        norm_bases[:, [1, 3, 5, 7, 9, 11]] /= NORM_VEL

        norm_meas = np.array(all_meas, dtype=np.float32)
        # 27维中，角度索引和距离索引规律排布
        ang_idx = [0, 1, 3, 4, 6, 9, 10, 12, 13, 15, 18, 19, 21, 22, 24]
        r_idx = [2, 11, 20]
        norm_meas[:, ang_idx] /= NORM_ANG
        norm_meas[:, r_idx] /= NORM_R

        dts = np.diff(times, prepend=times[0] if len(times) == 1 else times[1] - times[0])
        dts = np.array(dts, dtype=np.float32)

        if len(norm_states) > self.seq_length:
            norm_states = norm_states[:self.seq_length]
            norm_bases = norm_bases[:self.seq_length]
            norm_meas = norm_meas[:self.seq_length]
            all_masks = all_masks[:self.seq_length]
            dts = dts[:self.seq_length]

        return {
            'states': torch.tensor(norm_states),
            'baselines': torch.tensor(norm_bases),  # 🌟 新增：[12]维的编队几何信息
            'meas': torch.tensor(norm_meas),  # 🌟 扩展为 27 维
            'mask': torch.tensor(np.array(all_masks), dtype=torch.float32),
            'dt': torch.tensor(dts)
        }

def convert_body_to_world_meas(az_body, el_body, r_body, vx, vy, vz):
    """
    将机载坐标系下的观测值，利用飞机速度矢量，配准到世界坐标系
    """
    # 1. 计算身体坐标系下的单位/实际方向向量 (若只有角度，用 r=1.0 替代)
    r_calc = r_body if r_body > 0 else 1.0
    vec_body = np.array([
        r_calc * np.cos(el_body) * np.cos(az_body),
        r_calc * np.cos(el_body) * np.sin(az_body),
        r_calc * np.sin(el_body)
    ])

    # 2. 计算机载到世界的旋转矩阵 (基于速度矢量)
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

    # 3. 旋转到世界坐标系
    vec_world = R_b2w @ vec_body

    # 4. 反解世界坐标系下的 az, el
    r_xy = np.sqrt(vec_world[0]**2 + vec_world[1]**2)
    az_world = np.arctan2(vec_world[1], vec_world[0])
    el_world = np.arctan2(vec_world[2], r_xy)

    return az_world, el_world, r_body


def collate_fn(batch):
    lengths = [item['states'].shape[0] for item in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    padded_states = torch.zeros(batch_size, max_len, 6)
    padded_bases = torch.zeros(batch_size, max_len, 12)
    padded_meas = torch.zeros(batch_size, max_len, 27)
    padded_mask_input = torch.zeros(batch_size, max_len, 27)
    padded_dt = torch.zeros(batch_size, max_len)
    mask_loss = torch.zeros(batch_size, max_len)

    for i, item in enumerate(batch):
        seq_len = item['states'].shape[0]
        padded_states[i, :seq_len, :] = item['states']
        padded_bases[i, :seq_len, :] = item['baselines']
        padded_meas[i, :seq_len, :] = item['meas']
        padded_mask_input[i, :seq_len, :] = item['mask']
        padded_dt[i, :seq_len] = item['dt']
        mask_loss[i, :seq_len] = 1.0

    return {
        'states': padded_states, 'baselines': padded_bases,
        'meas': padded_meas, 'mask_input': padded_mask_input,
        'mask_loss': mask_loss, 'dt': padded_dt
    }