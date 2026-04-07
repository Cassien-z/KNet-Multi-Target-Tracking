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
        """ 物理预测模型: Constant Velocity (CV) """
        B = x.shape[0]
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)

        # 🌟 核心修复 1：归一化空间的物理缩放匹配 (Velocity to Position)
        # NORM_VEL / NORM_POS = 500 / 150000 = 1/300
        # 将速度时间步缩小 300 倍，匹配位置的尺度
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

        # 1. 物理模型预测
        x_minus = self.f_predict(x_prev, dt)

        # 2. 提取物理观测 [az, el, r]
        y_pred_phys = torch.zeros(batch_size, 3, device=meas.device)
        y_pred_phys[:, 0] = torch.atan2(x_minus[:, 2], x_minus[:, 0])
        # 🌟 修复 2：加上 1e-9 防止全为 0 时产生 NaN 梯度
        r_xy = torch.sqrt(x_minus[:, 0] ** 2 + x_minus[:, 2] ** 2 + 1e-9)
        y_pred_phys[:, 1] = torch.atan2(x_minus[:, 4], r_xy)
        y_pred_phys[:, 2] = torch.sqrt(x_minus[:, 0] ** 2 + x_minus[:, 2] ** 2 + x_minus[:, 4] ** 2 + 1e-9)

        # 归一化预测值
        y_pred = y_pred_phys.clone()
        y_pred[:, 0] = y_pred_phys[:, 0] / self.norm_ang
        y_pred[:, 1] = y_pred_phys[:, 1] / self.norm_ang
        # 🌟 核心修复 3：x_minus 本身是归一化的，算出来的 r 也是归一化的
        # 绝对不能再除以 self.norm_r！
        y_pred[:, 2] = y_pred_phys[:, 2]

        # 3. 利用张量广播计算所有传感器的残差
        meas_3d = meas.view(batch_size, self.num_sensors, 3)
        y_pred_3d = y_pred.unsqueeze(1).expand(-1, self.num_sensors, -1)

        innov_3d = meas_3d - y_pred_3d

        # 还原弧度去畸变后再归一化
        innov_3d[:, :, 0] = self._wrap_angle(innov_3d[:, :, 0] * self.norm_ang.item()) / self.norm_ang.item()
        innov_3d[:, :, 1] = self._wrap_angle(innov_3d[:, :, 1] * self.norm_ang.item()) / self.norm_ang.item()

        # 压平回 9 维
        innov = innov_3d.view(batch_size, self.total_meas_dim)

        # 4. 掩码清除缺失传感器的伪残差
        innov = innov * mask

        # 5. RNN 状态更新
        dt_input = dt.unsqueeze(-1) if dt.dim() == 1 else dt
        rnn_in = torch.cat([innov, mask, dt_input, x_minus], dim=1).unsqueeze(1)
        gru_out, hx_new = self.gru(rnn_in, hx)
        gru_out = self.ln(gru_out)

        # 6. 提取并应用 6x9 卡尔曼增益
        K = self.fc(gru_out[:, -1, :]).view(-1, self.state_dim, self.total_meas_dim)
        innov_col = innov.unsqueeze(-1)
        correction = torch.bmm(K, innov_col).squeeze(-1)

        x_new = x_minus + correction

        # 🌟 修复方案：强行截断，防止累积爆炸（因为已归一化，范围可以设为 -10 到 10）
        x_new = torch.clamp(x_new, min=-10.0, max=10.0)

        return x_new, hx_new