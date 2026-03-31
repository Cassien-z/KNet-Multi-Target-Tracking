import torch
import torch.nn as nn
import math


class KNet_Tracker(nn.Module):
    def __init__(self, state_dim=6, num_sensors=3, meas_dim_per_sensor=3, hidden_size=128):
        super().__init__()
        self.state_dim = state_dim
        self.num_sensors = num_sensors
        self.total_meas_dim = num_sensors * meas_dim_per_sensor  # 9

        input_dim = self.total_meas_dim + self.total_meas_dim + 1 + state_dim

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
        """ 恢复为最稳定的大道至简：纯 CV (匀速直线) 物理预测 """
        B = x.shape[0]
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)

        dt_scaled = dt.squeeze(-1) * (500.0 / 150000.0)

        F = torch.eye(6, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        F[:, 0, 1] = dt_scaled
        F[:, 2, 3] = dt_scaled
        F[:, 4, 5] = dt_scaled

        x_pred = torch.bmm(F, x.unsqueeze(-1)).squeeze(-1)
        return x_pred

    def _wrap_angle(self, angle):
        two_pi = 2.0 * math.pi
        return angle - two_pi * torch.round(angle / two_pi)

    def forward(self, meas, mask, dt, x_prev=None, hx=None):
        batch_size = meas.shape[0]

        if x_prev is None:
            x_prev = torch.zeros(batch_size, self.state_dim, device=meas.device)

        # 1. 物理模型预测 (不再需要 v_last)
        x_minus = self.f_predict(x_prev, dt)

        # 2. 提取物理观测 [az, el, r]
        y_pred_phys = torch.zeros(batch_size, 3, device=meas.device)
        y_pred_phys[:, 0] = torch.atan2(x_minus[:, 2], x_minus[:, 0])
        r_xy = torch.sqrt(x_minus[:, 0] ** 2 + x_minus[:, 2] ** 2 + 1e-9)
        y_pred_phys[:, 1] = torch.atan2(x_minus[:, 4], r_xy)
        y_pred_phys[:, 2] = torch.sqrt(x_minus[:, 0] ** 2 + x_minus[:, 2] ** 2 + x_minus[:, 4] ** 2 + 1e-9)

        # 归一化预测值
        y_pred = y_pred_phys.clone()
        y_pred[:, 0] = y_pred_phys[:, 0] / self.norm_ang
        y_pred[:, 1] = y_pred_phys[:, 1] / self.norm_ang
        y_pred[:, 2] = y_pred_phys[:, 2]

        # 3. 计算残差
        meas_3d = meas.view(batch_size, self.num_sensors, 3)
        y_pred_3d = y_pred.unsqueeze(1).expand(-1, self.num_sensors, -1)
        innov_3d = meas_3d - y_pred_3d
        innov_3d[:, :, 0] = self._wrap_angle(innov_3d[:, :, 0] * self.norm_ang.item()) / self.norm_ang.item()
        innov_3d[:, :, 1] = self._wrap_angle(innov_3d[:, :, 1] * self.norm_ang.item()) / self.norm_ang.item()
        innov = innov_3d.view(batch_size, self.total_meas_dim)

        # 4. 掩码
        innov = innov * mask

        # 5. RNN 状态更新
        dt_input = dt.unsqueeze(-1) if dt.dim() == 1 else dt
        rnn_in = torch.cat([innov, mask, dt_input, x_minus], dim=1).unsqueeze(1)
        gru_out, hx_new = self.gru(rnn_in, hx)
        gru_out = self.ln(gru_out)

        # 6. 提取增益 (🌟 保留这个能救命的 Tanh 修复)
        K = self.fc(gru_out[:, -1, :]).view(-1, self.state_dim, self.total_meas_dim)

        # K = torch.tanh(K) * 0.2
        K = torch.clamp(K, min=-3.0, max=3.0)

        innov_col = innov.unsqueeze(-1)
        correction = torch.bmm(K, innov_col).squeeze(-1)

        x_new = x_minus + correction
        x_new = torch.clamp(x_new, min=-5.0, max=5.0)

        return x_new, hx_new