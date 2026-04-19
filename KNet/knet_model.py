import torch
import torch.nn as nn
import math


class KNet_Tracker(nn.Module):
    def __init__(self, state_dim=9, num_platforms=3, meas_dim_per_platform=9, hidden_size=256, embed_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.num_platforms = num_platforms
        self.total_meas_dim = num_platforms * meas_dim_per_platform

        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(state_dim, embed_dim)

        platform_feat_dim = meas_dim_per_platform * 2 + state_dim
        self.kv_proj = nn.Linear(platform_feat_dim, embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.attn_ln = nn.LayerNorm(embed_dim)

        input_dim = self.total_meas_dim * 2 + 18 + embed_dim + state_dim + 1

        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim * self.total_meas_dim)
        )

        self.register_buffer('norm_ang', torch.tensor(math.pi))
        self.register_buffer('norm_r', torch.tensor(150000.0))

    def f_predict(self, x, dt):
        B = x.shape[0]
        if dt.dim() == 1: dt = dt.unsqueeze(-1)

        # 构建 9x9 状态转移矩阵
        F = torch.eye(9, device=x.device).unsqueeze(0).repeat(B, 1, 1)

        # 🌟 结合归一化系数的动力学积分 (pos/150000, vel/500, acc/50)
        dt_pos_v = dt.squeeze(-1) * (500.0 / 150000.0)
        dt_pos_a = 0.5 * (dt.squeeze(-1) ** 2) * (50.0 / 150000.0)
        dt_vel_a = dt.squeeze(-1) * (50.0 / 500.0)

        for i in range(3):
            idx = i * 3  # 0, 3, 6 分别对应 x, y, z 轴的起始索引
            F[:, idx, idx+1] = dt_pos_v    # pos = pos + vel * dt
            F[:, idx, idx+2] = dt_pos_a    # pos = pos + 0.5 * acc * dt^2
            F[:, idx+1, idx+2] = dt_vel_a  # vel = vel + acc * dt

        return torch.bmm(F, x.unsqueeze(-1)).squeeze(-1)

    def _wrap_angle(self, angle):
        two_pi = 2.0 * math.pi
        return angle - two_pi * torch.round(angle / two_pi)

    def _get_platform_y_pred(self, x_rel):
        y_phys = torch.zeros(x_rel.shape[0], 9, device=x_rel.device)
        # 🌟 修复：9维状态下，y的索引是3，z的索引是6
        r_xy = torch.sqrt(x_rel[:, 0] ** 2 + x_rel[:, 3] ** 2 + 1e-9)
        az = torch.atan2(x_rel[:, 3], x_rel[:, 0])
        el = torch.atan2(x_rel[:, 6], r_xy)
        r = torch.sqrt(x_rel[:, 0] ** 2 + x_rel[:, 3] ** 2 + x_rel[:, 6] ** 2 + 1e-9)

        az_norm = az / self.norm_ang
        el_norm = el / self.norm_ang

        y_phys[:, 0:3] = torch.stack([az_norm, el_norm, r], dim=-1)
        y_phys[:, 3:6] = torch.stack([az_norm, el_norm, torch.zeros_like(r)], dim=-1)
        y_phys[:, 6:9] = torch.stack([az_norm, torch.zeros_like(r), torch.zeros_like(r)], dim=-1)
        return y_phys

    def forward(self, meas, mask, dt, baselines, x_prev=None, hx=None, step=100):
        batch_size = meas.shape[0]
        if x_prev is None:
            x_prev = torch.zeros(batch_size, self.state_dim, device=meas.device)

        x_minus = self.f_predict(x_prev, dt)

        # ================= 预测量测计算 =================
        # 🌟 状态升维后，位置的索引变成了 0, 3, 6
        x_pos = x_minus[:, 0] * 150000.0
        y_pos = x_minus[:, 3] * 150000.0
        z_pos = x_minus[:, 6] * 150000.0

        y_pred_m0 = self._get_platform_y_pred(x_minus)
        y_pred_w1 = self._get_platform_y_pred(x_minus - baselines[:, 0:9])
        y_pred_w2 = self._get_platform_y_pred(x_minus - baselines[:, 9:18])
        y_pred = torch.cat([y_pred_m0, y_pred_w1, y_pred_w2], dim=1)

        innov = meas - y_pred
        ang_idx = [0, 1, 3, 4, 6, 9, 10, 12, 13, 15, 18, 19, 21, 22, 24]
        for idx in ang_idx:
            innov[:, idx] = self._wrap_angle(innov[:, idx] * self.norm_ang.item()) / self.norm_ang.item()
        innov = innov * mask

        innov_3d = innov.view(batch_size, self.num_platforms, -1)
        mask_3d = mask.view(batch_size, self.num_platforms, -1)

        base_m0 = torch.zeros(batch_size, 1, self.state_dim, device=meas.device)
        base_w1 = baselines[:, 0:9].unsqueeze(1)
        base_w2 = baselines[:, 9:18].unsqueeze(1)
        bases_3d = torch.cat([base_m0, base_w1, base_w2], dim=1)

        platform_feats = torch.cat([innov_3d, mask_3d, bases_3d], dim=-1)

        Q = self.query_proj(x_minus).unsqueeze(1)
        KV = self.kv_proj(platform_feats)

        attn_out, attn_weights = self.attention(Q, KV, KV)
        attn_out = self.attn_ln(Q + attn_out).squeeze(1)

        dt_input = dt.unsqueeze(-1) if dt.dim() == 1 else dt
        rnn_in = torch.cat([innov, mask, baselines, attn_out, x_minus, dt_input], dim=1).unsqueeze(1)

        gru_out, hx_new = self.gru(rnn_in, hx)
        gru_out = self.ln(gru_out)

        # 5. 卡尔曼增益修正
        # 🌟 修复：修正变量名 batch_size
        K = self.fc(gru_out[:, -1, :]).view(batch_size, self.state_dim, self.total_meas_dim)

        # ================= 终极物理保险：自适应残差门控 =================
        # 🌟 修复：移除未定义的 self.fc_gain, innov_27 和 B，直接使用 K 和 innov 矩阵相乘
        correction = torch.bmm(K, innov.unsqueeze(-1)).squeeze(-1)

        # 🌟 智能判定：用长机雷达残差的模长来衡量目标的机动剧烈程度
        m_innov_pos = innov_3d[:, 0, 0:3]
        innov_mag = torch.norm(m_innov_pos, dim=-1, keepdim=True)  # [B, 1]

        # 🌟 修改 1：将机动触发门限从 0.05(7500米) 降到 0.015(2250米)
        is_high_dyn = (innov_mag > 0.015) | (step <= 10)

        dynamic_gate_pos = torch.where(is_high_dyn, 0.1, 0.02)
        # 🌟 修改 2：稳态下的速度修正限幅从 0.2 放宽到 0.4 (允许网络每帧最大改变 200m/s 的速度)
        dynamic_gate_vel = torch.where(is_high_dyn, 1.0, 0.4)
        dynamic_gate_acc = torch.where(is_high_dyn, 1.0, 0.2)

        # 向量化安全限幅 (Clamp)
        corr_pos = torch.max(torch.min(correction[:, [0, 3, 6]], dynamic_gate_pos), -dynamic_gate_pos)
        corr_vel = torch.max(torch.min(correction[:, [1, 4, 7]], dynamic_gate_vel), -dynamic_gate_vel)
        corr_acc = torch.max(torch.min(correction[:, [2, 5, 8]], dynamic_gate_acc), -dynamic_gate_acc)

        correction_gated = torch.empty_like(correction)
        correction_gated[:, [0, 3, 6]] = corr_pos
        correction_gated[:, [1, 4, 7]] = corr_vel
        correction_gated[:, [2, 5, 8]] = corr_acc

        x_out = x_minus + correction_gated
        return x_out, hx, attn_weights