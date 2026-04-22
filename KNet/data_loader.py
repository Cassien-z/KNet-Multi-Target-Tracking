# data_loader.py
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

NORM_CFG = {
    "pos": 5000.0,   # 从 15万 降到 5千！
    "vel": 500.0,
    "acc": 50.0,
    "r": 150000.0,   # 极坐标测距可以保留原样，因为它本身就是相对距离
    "ang": np.pi
}


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
            states.append(gt_rel)
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
        norm_bases = np.array(baselines, dtype=np.float32)
        norm_meas = np.array(all_meas, dtype=np.float32)

        # 🌟 修复一：将僚机的雷达观测，补偿基线偏移，统一拉回长机坐标系！
        # 僚机1 的雷达位置 (9:12) 加上 僚机1 的相对基线 (0:3)
        norm_meas[:, 9:12] += norm_bases[:, 0:3]
        # 僚机2 的雷达位置 (18:21) 加上 僚机2 的相对基线 (9:12)
        norm_meas[:, 18:21] += norm_bases[:, 9:12]

        # 🌟 修复二：局部坐标系平移
        ref_pos = norm_states[0, [0, 3, 6]].copy()

        # 1. 目标真实状态减去原点
        norm_states[:, 0] -= ref_pos[0]
        norm_states[:, 3] -= ref_pos[1]
        norm_states[:, 6] -= ref_pos[2]

        # 2. 所有雷达观测值也必须减去 同一个原点！（确保输入输出在同一个局部空间）
        for i in [0, 9, 18]: norm_meas[:, i] -= ref_pos[0]  # X轴
        for i in [1, 10, 19]: norm_meas[:, i] -= ref_pos[1]  # Y轴
        for i in [2, 11, 20]: norm_meas[:, i] -= ref_pos[2]  # Z轴

        # 缩放 (除以物理极限)
        norm_states[:, [0, 3, 6]] /= NORM_CFG['pos']
        norm_states[:, [1, 4, 7]] /= NORM_CFG['vel']
        norm_states[:, [2, 5, 8]] /= NORM_CFG['acc']

        norm_bases[:, [0, 3, 6, 9, 12, 15]] /= NORM_CFG['pos']
        norm_bases[:, [1, 4, 7, 10, 13, 16]] /= NORM_CFG['vel']
        norm_bases[:, [2, 5, 8, 11, 14, 17]] /= NORM_CFG['acc']

        norm_meas /= NORM_CFG['pos']

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

    # 🌟 修复：states pad 的最后一维改为 9
    padded_states = torch.zeros(batch_size, max_len, 9)
    padded_bases = torch.zeros(batch_size, max_len, 18)
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