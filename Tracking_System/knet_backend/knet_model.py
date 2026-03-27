import torch
import torch.nn as nn
import math


class KNet_Tracker(nn.Module):
    def __init__(self, state_dim=6, meas_dim=3, hidden_size=128):
        super().__init__()
        self.state_dim = state_dim  # x, vx, y, vy, z, vz
        self.meas_dim = meas_dim  # az, el, r

        # --- 1. 神经网络部分 (计算卡尔曼增益 K) ---
        # 输入维度: 3 (Innovation) + 3 (Mask) + 1 (dt) + 6 (State) = 13
        input_dim = meas_dim + meas_dim + 1 + state_dim

        # 🟢 改进 1: 增加隐藏层维度从 64 -> 128 (甚至 256)
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)

        # 🟢 改进 2: 增加网络深度和复杂度，更好地拟合非线性增益
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim * meas_dim)  # 输出 K 矩阵 (6x3 = 18)
        )

    def f_predict(self, x, dt):
        """ 物理预测模型: Constant Velocity (CV) """
        B = x.shape[0]
        # 确保 dt 是 [B, 1] 形状以便广播
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)

        F = torch.eye(6, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        F[:, 0, 1] = dt.squeeze(-1)
        F[:, 2, 3] = dt.squeeze(-1)
        F[:, 4, 5] = dt.squeeze(-1)

        x_pred = torch.bmm(F, x.unsqueeze(-1)).squeeze(-1)
        return x_pred

    def h_measure(self, x):
        """
        物理观测模型: Cartesian -> Spherical
        🟢 改进 3: 增加数值稳定性保护
        """
        px, py, pz = x[:, 0], x[:, 2], x[:, 4]

        # 防止 r=0 导致除零
        r_sq = px ** 2 + py ** 2 + pz ** 2
        r = torch.sqrt(r_sq + 1e-9)

        # 计算方位角 (-pi, pi)
        az = torch.atan2(py, px)

        # 计算俯仰角，并裁剪输入防止 asin 报错 (防止浮点误差导致 >1)
        sin_el = pz / (r + 1e-9)
        sin_el = torch.clamp(sin_el, -1.0, 1.0)
        el = torch.asin(sin_el)

        return torch.stack([az, el, r], dim=1)

    def _wrap_angle(self, angle):
        """
        🟢 改进 4: 鲁棒的角度 Wrap 函数
        将角度强制映射到 (-pi, pi]
        """
        # 使用 torch.remainder 或者手动处理，确保梯度流畅
        # 方法：angle - 2*pi * round(angle / 2*pi)
        two_pi = 2.0 * math.pi
        return angle - two_pi * torch.round(angle / two_pi)

    def forward(self, meas, mask, dt, prev_state, hx=None):
        """
        单步前向传播
        meas: [batch, 3] (归一化后的测量: az/pi, el/pi, r/150km)
        mask: [batch, 3]
        dt:   [batch]
        prev_state: [batch, 6] (归一化后的状态)
        hx:   GRU 隐状态
        """
        # 1. 物理预测 (Predict)
        x_minus = self.f_predict(prev_state, dt)

        # 2. 观测预测 (从归一化状态反推归一化观测)
        # 注意：h_measure 输出的是物理值 (弧度, 米)，需要归一化以匹配 meas
        y_pred_phys = self.h_measure(x_minus)

        # 🟢 关键：归一化预测值，使其与输入 meas 量纲一致
        # 假设归一化系数：角度除以 pi, 距离除以 150000 (需与 inference/data_loader 严格一致)
        # 这里硬编码了归一化系数，建议作为参数传入或在外部统一处理
        # 为了通用性，我们假设输入 meas 已经是归一化的，所以 y_pred 也要归一化
        y_pred = y_pred_phys.clone()
        y_pred[:, 0] = y_pred_phys[:, 0] / math.pi  # Az
        y_pred[:, 1] = y_pred_phys[:, 1] / math.pi  # El
        y_pred[:, 2] = y_pred_phys[:, 2] / 150000.0  # Range (需与 NORM_CFG['r'] 一致)

        # 3. 计算 Innovation (残差)
        innov = meas - y_pred

        # 🟢 关键：对角度残差进行 Wrap 处理
        # 只处理前两个维度 (az, el)
        innov[:, 0] = self._wrap_angle(innov[:, 0])  # 此时 innov 已是归一化的 (-1, 1) 对应 (-pi, pi)
        innov[:, 1] = self._wrap_angle(innov[:, 1])

        # 注意：上面的 _wrap_angle 是针对弧度设计的。
        # 如果 innov 已经是归一化值 (-1 ~ 1)，则应该 wrap 到 (-1, 1)
        # 修正：因为 y_pred 和 meas 都是除以 pi 后的值，范围是 (-1, 1)
        # 所以残差范围理论上是 (-2, 2)。我们需要将其 wrap 回 (-1, 1) 对应的区间吗？
        # 不，残差应该是弧度差除以 pi。
        # 正确的做法：先还原成弧度，wrap，再归一化。

        # --- 重新修正残差计算逻辑 (更稳妥) ---
        # 1. 还原为物理弧度
        innov_az_rad = (meas[:, 0] - y_pred[:, 0]) * math.pi
        innov_el_rad = (meas[:, 1] - y_pred[:, 1]) * math.pi
        innov_r = meas[:, 2] - y_pred[:, 2]  # 距离残差 (归一化后)

        # 2. Wrap 角度残差
        innov_az_rad = self._wrap_angle(innov_az_rad)
        innov_el_rad = self._wrap_angle(innov_el_rad)

        # 3. 重新归一化残差，作为 RNN 输入
        innov[:, 0] = innov_az_rad / math.pi
        innov[:, 1] = innov_el_rad / math.pi
        innov[:, 2] = innov_r

        # 4. 神经网络计算增益 K
        # 拼接特征: [Innovation, Mask, dt, x_minus]
        # 确保 dt 维度正确 [B, 1]
        if dt.dim() == 1:
            dt_input = dt.unsqueeze(-1)
        else:
            dt_input = dt

        rnn_in = torch.cat([innov, mask, dt_input, x_minus], dim=1).unsqueeze(1)

        gru_out, hx_new = self.gru(rnn_in, hx)

        # 生成 K 矩阵 [batch, 6, 3]
        K = self.fc(gru_out[:, -1, :]).view(-1, 6, 3)

        # 5. 混合修正 (Update)
        # 应用 Mask (如果某个传感器无效，强制该维度残差为 0)
        innov_masked = innov * mask

        correction = torch.bmm(K, innov_masked.unsqueeze(-1)).squeeze(-1)
        x_new = x_minus + correction

        return x_new, hx_new