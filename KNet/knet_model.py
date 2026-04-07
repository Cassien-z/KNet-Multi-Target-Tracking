import torch
import torch.nn as nn
import math


class KNet_Tracker(nn.Module):
    # 🌟 修改：num_platforms=3，总测量维度变成 3*9 = 27
    def __init__(self, state_dim=6, num_platforms=3, meas_dim_per_platform=9, hidden_size=256):
        super().__init__()
        self.state_dim = state_dim
        self.num_platforms = num_platforms
        self.total_meas_dim = num_platforms * meas_dim_per_platform  # 27

        # 输入：27维残差 + 27维Mask + 1维dt + 6维自身状态预测 + 12维基线向量 = 73维
        input_dim = self.total_meas_dim * 2 + 1 + state_dim + 12

        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            # 输出 6x27 的卡尔曼增益
            nn.Linear(128, state_dim * self.total_meas_dim)
        )

        self.register_buffer('norm_ang', torch.tensor(math.pi))
        self.register_buffer('norm_r', torch.tensor(150000.0))

    def f_predict(self, x, dt):
        """纯 CV 预测"""
        B = x.shape[0]
        if dt.dim() == 1: dt = dt.unsqueeze(-1)
        dt_scaled = dt.squeeze(-1) * (500.0 / 150000.0)

        F = torch.eye(6, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        F[:, 0, 1] = dt_scaled
        F[:, 2, 3] = dt_scaled
        F[:, 4, 5] = dt_scaled
        return torch.bmm(F, x.unsqueeze(-1)).squeeze(-1)

    def _wrap_angle(self, angle):
        two_pi = 2.0 * math.pi
        return angle - two_pi * torch.round(angle / two_pi)

    def _get_platform_y_pred(self, x_rel):
        """将笛卡尔相对坐标转化为 9 维雷达/光电观测"""
        y_phys = torch.zeros(x_rel.shape[0], 9, device=x_rel.device)

        r_xy = torch.sqrt(x_rel[:, 0] ** 2 + x_rel[:, 2] ** 2 + 1e-9)
        az = torch.atan2(x_rel[:, 2], x_rel[:, 0])
        el = torch.atan2(x_rel[:, 4], r_xy)
        r = torch.sqrt(x_rel[:, 0] ** 2 + x_rel[:, 2] ** 2 + x_rel[:, 4] ** 2 + 1e-9)

        az_norm = az / self.norm_ang
        el_norm = el / self.norm_ang

        # Sensor 0 (Radar): az, el, r
        y_phys[:, 0], y_phys[:, 1], y_phys[:, 2] = az_norm, el_norm, r
        # Sensor 1 (EO): az, el, 0
        y_phys[:, 3], y_phys[:, 4], y_phys[:, 5] = az_norm, el_norm, 0.0
        # Sensor 2 (ESM): az, 0, 0
        y_phys[:, 6], y_phys[:, 7], y_phys[:, 8] = az_norm, 0.0, 0.0

        return y_phys

    # 🌟 新增参数 baselines (12维)
    def forward(self, meas, mask, dt, baselines, x_prev=None, hx=None):
        batch_size = meas.shape[0]
        if x_prev is None:
            x_prev = torch.zeros(batch_size, self.state_dim, device=meas.device)

        # 1. 预测相对于长机的物理状态
        x_minus = self.f_predict(x_prev, dt)

        # 2. 🌟 动态计算 3 个平台的物理观测投影
        # 长机投影 (基线为0)
        y_pred_m0 = self._get_platform_y_pred(x_minus)
        # 僚机1投影 (相对长机状态 - 僚机1相对于长机的基线)
        y_pred_w1 = self._get_platform_y_pred(x_minus - baselines[:, 0:6])
        # 僚机2投影
        y_pred_w2 = self._get_platform_y_pred(x_minus - baselines[:, 6:12])

        # 拼接 27 维预测
        y_pred = torch.cat([y_pred_m0, y_pred_w1, y_pred_w2], dim=1)

        # 3. 计算残差与去畸变
        innov = meas - y_pred
        ang_idx = [0, 1, 3, 4, 6, 9, 10, 12, 13, 15, 18, 19, 21, 22, 24]
        for idx in ang_idx:
            innov[:, idx] = self._wrap_angle(innov[:, idx] * self.norm_ang.item()) / self.norm_ang.item()

        # 4. 掩码
        innov = innov * mask

        # 5. RNN 融合 (🌟 输入追加编队几何结构 baselines)
        dt_input = dt.unsqueeze(-1) if dt.dim() == 1 else dt
        rnn_in = torch.cat([innov, mask, dt_input, x_minus, baselines], dim=1).unsqueeze(1)
        gru_out, hx_new = self.gru(rnn_in, hx)
        gru_out = self.ln(gru_out)

        # 6. 计算增益
        K = self.fc(gru_out[:, -1, :]).view(-1, self.state_dim, self.total_meas_dim)
        K = torch.clamp(K, min=-3.0, max=3.0)  # 保留上一版的机动性修改

        correction = torch.bmm(K, innov.unsqueeze(-1)).squeeze(-1)
        x_new = torch.clamp(x_minus + correction, min=-5.0, max=5.0)

        return x_new, hx_new