import torch
import torch.nn as nn
from data_loader import NORM_CFG


class KNet_Tracker(nn.Module):
    def __init__(self, state_dim=9, num_platforms=3, meas_dim_per_platform=9, hidden_size=256, embed_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.num_platforms = num_platforms

        self.embed_dim = embed_dim

        # --- 🌟 全新设计 1：可学习的 Track Query ---
        # 替代了以前用 pred_state 映射生成的 Query。
        # 它的物理意义是：“网络在没有任何先验时，向当前所有传感器询问目标在哪里的探针”
        self.track_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        # --- 🌟 初始化修改点：Token 维度变为 9 维 ---
        # 3维观测(极坐标) + 3维掩码 + 3维平台相对长机的XYZ坐标 = 9维
        sensor_feat_dim = 3 + 3 + 3
        self.kv_proj = nn.Linear(sensor_feat_dim, embed_dim)

        # 交叉注意力
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.attn_ln = nn.LayerNorm(embed_dim)

        # --- 🌟 全新设计 2：RNN 输入极度精简 ---
        # 现在的输入只有：融合后的注意力特征 + dt，抛弃了原来庞杂的残差和掩码展平
        input_dim = embed_dim + 1
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)

        # 神经 IMM 架构保持不变，但现在的意义是直接输出状态，而不是输出修正残差
        self.num_models = 3
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, state_dim)
            ) for _ in range(self.num_models)
        ])

        self.gating = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_models),
            nn.Softmax(dim=-1)
        )

    def forward(self, meas, baselines, mask_input, dts, h_in=None, state_in=None):
        batch_size, seq_len, _ = meas.shape
        device = meas.device

        # 初始化 GRU 隐藏状态
        if h_in is not None:
            h = h_in
        else:
            h = torch.zeros(1, batch_size, self.gru.hidden_size).to(device)

        predictions = []

        for t in range(seq_len):
            dt = dts[:, t].unsqueeze(-1)

            # --- 1. 组装 Sensor Tokens ---
            sensor_tokens = []

            # 分解当前帧的 baselines [B, 18] 为长机和僚机的相对位置 (XYZ)
            # 长机相对于自己是 0
            base_master = torch.zeros(batch_size, 3, device=device)
            # 僚机 1 和 僚机 2 的相对位置 [B, 3]
            base_w1 = baselines[:, t, 0:3]
            base_w2 = baselines[:, t, 9:12]
            platform_bases = [base_master, base_w1, base_w2]

            for p_i, p_idx in enumerate([0, 9, 18]):
                for s_idx in [0, 3, 6]:
                    start_idx = p_idx + s_idx
                    end_idx = start_idx + 3

                    # 🌟 核心：Token = 原始极坐标观测 + 掩码 + 该平台的位置
                    token = torch.cat([
                        meas[:, t, start_idx:end_idx],
                        mask_input[:, t, start_idx:end_idx],
                        platform_bases[p_i]  # 告诉网络这个数据是在哪里测的
                    ], dim=-1)
                    sensor_tokens.append(token.unsqueeze(1))

            kv_input = torch.cat(sensor_tokens, dim=1)  # [B, 9, 6]
            kv = self.kv_proj(kv_input)  # [B, 9, embed_dim]

            # --- 2. 基于 Track Query 的交叉注意力 ---
            # 扩展可学习的 query 以匹配 Batch Size
            q = self.track_query.expand(batch_size, -1, -1)  # [B, 1, embed_dim]

            # 生成掩码 (如果某个传感器的 3 个 mask 位全是 0，说明它完全瞎了，屏蔽它)
            mask_sum = kv_input[:, :, 3:6].sum(dim=-1)
            key_padding_mask = (mask_sum < 1e-5)
            all_blind = key_padding_mask.all(dim=-1, keepdim=True)
            key_padding_mask = key_padding_mask.masked_fill(all_blind, False)

            # Track Query 去 9 个传感器里“捞”出当前帧最有用的空间信息
            attn_out, _ = self.attention(q, kv, kv, key_padding_mask=key_padding_mask)
            attn_feat = self.attn_ln(attn_out.squeeze(1))  # [B, embed_dim]

            # --- 3. RNN 时序推演 ---
            # 现在的输入非常干净：注意力捞出来的空间特征 + 当前时间间隔
            rnn_input = torch.cat([attn_feat, dt], dim=-1).unsqueeze(1)
            out, h = self.gru(rnn_input, h)
            gru_feat = out.squeeze(1)

            # --- 4. 神经 IMM 直接输出状态 ---
            mode_probs = self.gating(gru_feat)
            expert_preds = torch.stack([expert(gru_feat) for expert in self.experts], dim=1)

            # 🔪 【已更改】以前这里算出来的是 correction，现在直接就是 final_state
            curr_state = torch.sum(mode_probs.unsqueeze(-1) * expert_preds, dim=1)

            # (可选) 如果你觉得网络初期瞎猜太离谱，可以在这里加一点简单的 clamp 限制输出范围
            # curr_state = torch.clamp(curr_state, -2.0, 2.0)

            predictions.append(curr_state.unsqueeze(1))

        return torch.cat(predictions, dim=1), h, curr_state