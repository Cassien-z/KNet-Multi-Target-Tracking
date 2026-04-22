import torch
import torch.nn as nn
import math
# 🌟 导入归一化配置，用于物理尺度补偿
from data_loader import NORM_CFG


class KNet_Tracker(nn.Module):
    def __init__(self, state_dim=9, num_platforms=3, meas_dim_per_platform=9, hidden_size=256, embed_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.num_platforms = num_platforms
        self.total_meas_dim = num_platforms * meas_dim_per_platform

        # 特征提取与注意力机制部分 (保持你原来的设计)
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(state_dim, embed_dim)
        platform_feat_dim = meas_dim_per_platform * 2 + state_dim
        self.kv_proj = nn.Linear(platform_feat_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.attn_ln = nn.LayerNorm(embed_dim)

        # RNN 输入维度：27维量测 + 27维掩码 + 18维基线 + embed_dim + 9维状态 + 1维dt
        input_dim = self.total_meas_dim * 2 + 18 + embed_dim + state_dim + 1
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 🌟 修改：直接输出 9 维的状态修正量，不再输出庞大的 K 矩阵
            nn.Linear(256, state_dim)
        )

    def forward(self, meas, baselines, mask_input, dts, h_in=None, state_in=None):
        """
        meas: [B, T, 27]
        baselines: [B, T, 18]
        mask_input: [B, T, 27]
        dts: [B, T]
        """
        batch_size, seq_len, _ = meas.shape
        device = meas.device

        # 如果有外部传入的历史状态，就继承它；否则从 0 开始
        if h_in is not None:
            h = h_in
        else:
            h = torch.zeros(1, batch_size, self.gru.hidden_size).to(device)

        if state_in is not None:
            curr_state = state_in
        else:
            curr_state = torch.zeros(batch_size, self.state_dim).to(device)

        predictions = []

        for t in range(seq_len):
            # --- 1. 物理预测步 (Kinematic Predict) ---
            dt = dts[:, t].unsqueeze(-1)
            pred_state = curr_state.clone()

            scale_v2p = NORM_CFG['vel'] / NORM_CFG['pos']
            scale_a2v = NORM_CFG['acc'] / NORM_CFG['vel']

            pred_state[:, [0, 3, 6]] += curr_state[:, [1, 4, 7]] * dt * scale_v2p
            pred_state[:, [1, 4, 7]] += curr_state[:, [2, 5, 8]] * dt * scale_a2v

            # --- 2. 提取当前帧特征 (Attention) ---
            q = self.query_proj(pred_state).unsqueeze(1)  # [B, 1, embed_dim]

            # 🌟 终极修复：显式计算残差 (Innovation)
            pred_pos = pred_state[:, [0, 3, 6]]  # 提取预测的三维相对位置 [B, 3]

            meas_innov = meas[:, t, :].clone()
            # 此时所有的 meas 都已经在上一级被统一拉到了局部的长机坐标系，所以直接减去 pred_pos 即可！
            meas_innov[:, 0:3] = meas[:, t, 0:3] - pred_pos  # 长机残差
            meas_innov[:, 9:12] = meas[:, t, 9:12] - pred_pos  # 僚机1残差
            meas_innov[:, 18:21] = meas[:, t, 18:21] - pred_pos  # 僚机2残差

            # 分离特征，注意这里把原始的 meas 替换成了算好的残差 meas_innov
            m_feat = torch.cat([meas_innov[:, 0:9], mask_input[:, t, 0:9], torch.zeros_like(baselines[:, t, 0:9])],
                               dim=-1).unsqueeze(1)
            w1_feat = torch.cat([meas_innov[:, 9:18], mask_input[:, t, 9:18], baselines[:, t, 0:9]], dim=-1).unsqueeze(
                1)
            w2_feat = torch.cat([meas_innov[:, 18:27], mask_input[:, t, 18:27], baselines[:, t, 9:18]],
                                dim=-1).unsqueeze(1)

            # 拼接并投影 KV
            kv_input = torch.cat([m_feat, w1_feat, w2_feat], dim=1)
            kv = self.kv_proj(kv_input)

            # 交叉注意力计算
            attn_out, _ = self.attention(q, kv, kv)
            attn_feat = self.attn_ln(attn_out.squeeze(1))

            # --- 3. RNN 更新 ---
            # 同样送入带残差的 meas_innov，让 GRU 直观感受到“预测到底偏了多少”
            rnn_input = torch.cat([
                meas_innov, mask_input[:, t, :], baselines[:, t, :],
                attn_feat, pred_state, dt
            ], dim=-1).unsqueeze(1)

            out, h = self.gru(rnn_input, h)
            correction = self.fc(out.squeeze(1))  # [B, 9]

            # --- 4. 修正步 (Update) ---
            # 🌟 修复 Bug 1：冷启动特权门限
            # 如果是第一帧且状态为全0，放宽限幅，让网络迅速捕捉初始速度
            is_cold_start = (t == 0) and (torch.norm(curr_state) < 1e-5)

            # A. 位置限幅：冷启动允许修 5000 米，稳态只允许修 100 米
            max_p_corr = (5000.0 if is_cold_start else 100.0) / NORM_CFG['pos']
            correction[:, [0, 3, 6]] = torch.clamp(correction[:, [0, 3, 6]], -max_p_corr, max_p_corr)

            # B. 速度限幅：冷启动允许修 1000 m/s，稳态只允许修 30 m/s
            max_v_corr = (1000.0 if is_cold_start else 30.0) / NORM_CFG['vel']
            correction[:, [1, 4, 7]] = torch.clamp(correction[:, [1, 4, 7]], -max_v_corr, max_v_corr)

            # C. 加速度限幅：冷启动允许修 100 m/s²，稳态只允许修 20 m/s²
            max_a_corr = (100.0 if is_cold_start else 20.0) / NORM_CFG['acc']
            correction[:, [2, 5, 8]] = torch.clamp(correction[:, [2, 5, 8]], -max_a_corr, max_a_corr)

            curr_state = pred_state + correction
            predictions.append(curr_state.unsqueeze(1))

        return torch.cat(predictions, dim=1), h, curr_state