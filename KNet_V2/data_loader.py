# data_loader.py
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

NORM_CFG = {
    "pos": 50000.0,
    "vel": 500.0,
    "acc": 50.0,
    "r": 50000.0,    # 🌟 从 15万 改成 5万！与 pos 保持绝对一致
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
        meas_arr = np.zeros(9, dtype=np.float32)
        mask_arr = np.zeros(9, dtype=np.float32)
        if not p_data or p_state is None:
            return meas_arr, mask_arr

        vx, vy, vz = p_state[1], p_state[4], p_state[7]
        sensors = p_data.get('sensors', {})

        # 雷达：真实 [方位, 俯仰, 距离]
        if 'sensor_0' in sensors and sensors['sensor_0'].get('observations'):
            meas = sensors['sensor_0']['observations'][0]['meas']
            az, el, r = convert_body_to_world_meas(meas[0], meas[1], meas[2], vx, vy, vz)
            meas_arr[0:3] = [az / NORM_CFG['ang'], el / NORM_CFG['ang'], r / NORM_CFG['r']]
            mask_arr[0:3] = [1.0, 1.0, 1.0]

        # 光电：真实 [方位, 俯仰, 无效0]
        if 'sensor_1' in sensors and sensors['sensor_1'].get('observations'):
            meas = sensors['sensor_1']['observations'][0]['meas']
            az, el, _ = convert_body_to_world_meas(meas[0], meas[1], 1.0, vx, vy, vz)
            meas_arr[3:6] = [az / NORM_CFG['ang'], el / NORM_CFG['ang'], 0.0]
            mask_arr[3:6] = [1.0, 1.0, 0.0]

        # ESM：真实 [方位, 无效0, 无效0]
        if 'sensor_2' in sensors and sensors['sensor_2'].get('observations'):
            meas = sensors['sensor_2']['observations'][0]['meas']
            az, _, _ = convert_body_to_world_meas(meas[0], 0.0, 1.0, vx, vy, vz)
            meas_arr[6:9] = [az / NORM_CFG['ang'], 0.0, 0.0]
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

            # --- 🌟 核心修改开始：时间对齐补偿 ---

            # 1. 提取僚机的旧状态 (可能包含 0.1s 的延迟)
            wm1_state_old = np.array(wm1["platform_state"]) if wm1.get("platform_state") else m_state.copy()
            wm2_state_old = np.array(wm2["platform_state"]) if wm2.get("platform_state") else m_state.copy()

            # 2. 定义状态外推函数 (根据速度和加速度补偿时间差)
            def extrapolate(state, dt):
                s = state.copy()
                if dt > 1e-4:
                    # 位置补偿: P_now = P_old + V_old*dt + 0.5*A_old*dt^2
                    s[0] += s[1] * dt + 0.5 * s[2] * (dt ** 2)
                    s[3] += s[4] * dt + 0.5 * s[5] * (dt ** 2)
                    s[6] += s[7] * dt + 0.5 * s[8] * (dt ** 2)
                    # 速度补偿: V_now = V_old + A_old*dt
                    s[1] += s[2] * dt
                    s[4] += s[5] * dt
                    s[7] += s[8] * dt
                return s

            # 3. 计算对齐到长机当前时间戳的僚机新状态
            wm1_state_now = extrapolate(wm1_state_old, wm1.get("data_age", 0.0))
            wm2_state_now = extrapolate(wm2_state_old, wm2.get("data_age", 0.0))

            # 1. 目标相对长机的真实状态
            gt_rel = gt_w - m_state
            states.append(gt_rel)

            # 2. 僚机相对长机的基线向量 (使用外推后的最新状态，保证空间拓扑正确)
            base1 = wm1_state_now - m_state
            base2 = wm2_state_now - m_state
            baselines.append(np.concatenate([base1, base2]))

            # ... 前面的代码不变 ...
            # 3. 提取观测
            m_meas, m_mask = self._extract_platform_data(master, m_state)
            w1_meas, w1_mask = self._extract_platform_data(wm1, wm1_state_old)
            w2_meas, w2_mask = self._extract_platform_data(wm2, wm2_state_old)

            # 🚨 以前这里有一大堆 offset_w1 的加法，全部删掉！

            all_meas.append(np.concatenate([m_meas, w1_meas, w2_meas]))
            all_masks.append(np.concatenate([m_mask, w1_mask, w2_mask]))
            times.append(frame['time_s'])

            # ========== 归一化 ==========
        norm_states = np.array(states, dtype=np.float32)
        norm_bases = np.array(baselines, dtype=np.float32)
        norm_meas = np.array(all_meas, dtype=np.float32)  # 已经是归一化好的极坐标了！

        # 🚨 以前这里的 ref_pos 平移和 norm_meas /= 5000，全部删掉！

        norm_states[:, [0, 3, 6]] /= NORM_CFG['pos']
        norm_states[:, [1, 4, 7]] /= NORM_CFG['vel']
        norm_states[:, [2, 5, 8]] /= NORM_CFG['acc']

        norm_bases[:, [0, 3, 6, 9, 12, 15]] /= NORM_CFG['pos']
        norm_bases[:, [1, 4, 7, 10, 13, 16]] /= NORM_CFG['vel']
        norm_bases[:, [2, 5, 8, 11, 14, 17]] /= NORM_CFG['acc']
        # ... 返回字典代码不变 ...


        dts = np.diff(times, prepend=times[0] if len(times) == 1 else times[1] - times[0])
        dts = np.array(dts, dtype=np.float32)

        # ====== 🌟 核心升级：时间轴随机滑动窗口裁剪 ======
        full_len = len(norm_states)
        if full_len > self.seq_length:
            # 随机选择一个起始点，确保切出来的长度刚好是 seq_length
            start_idx = np.random.randint(0, full_len - self.seq_length + 1)
            end_idx = start_idx + self.seq_length

            norm_states = norm_states[start_idx:end_idx]
            norm_bases = norm_bases[start_idx:end_idx]
            norm_meas = norm_meas[start_idx:end_idx]
            all_masks = all_masks[start_idx:end_idx]
            dts = dts[start_idx:end_idx]
        # ===============================================

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