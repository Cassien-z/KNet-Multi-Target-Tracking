import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def solve_3x3_onnx_safe(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Solve batched 3x3 systems using an ONNX-exportable adjugate formula.

    ``torch.linalg.solve`` is not supported by the PyTorch 2.1 ONNX exporter
    used by the delivery environment.  The information matrices passed here
    are symmetric positive definite after regularization, so their determinant
    is positive.  Keeping this fixed-size implementation in the reference
    model gives Python, MATLAB, and C the same inference graph.
    """

    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 0, 2]
    d = matrix[..., 1, 0]
    e = matrix[..., 1, 1]
    f = matrix[..., 1, 2]
    g = matrix[..., 2, 0]
    h = matrix[..., 2, 1]
    i = matrix[..., 2, 2]

    c00 = e * i - f * h
    c01 = c * h - b * i
    c02 = b * f - c * e
    c10 = f * g - d * i
    c11 = a * i - c * g
    c12 = c * d - a * f
    c20 = d * h - e * g
    c21 = b * g - a * h
    c22 = a * e - b * d
    determinant = (a * c00 + b * c10 + c * c20).clamp_min(1e-24)

    rhs = vector
    x0 = (c00 * rhs[..., 0] + c01 * rhs[..., 1] + c02 * rhs[..., 2]) / determinant
    x1 = (c10 * rhs[..., 0] + c11 * rhs[..., 1] + c12 * rhs[..., 2]) / determinant
    x2 = (c20 * rhs[..., 0] + c21 * rhs[..., 1] + c22 * rhs[..., 2]) / determinant
    return torch.stack((x0, x1, x2), dim=-1)


def sinusoidal_position_encoding(length: int, dim: int, device):
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def radar_body_to_world_xyz(radar: torch.Tensor, uav_pos: torch.Tensor, rotmat_b2e: torch.Tensor) -> torch.Tensor:
    """
    Convert body-frame radar spherical measurements to world-frame target xyz.

    radar:     [..., 3] = [az, el, r]  (body frame)
    uav_pos:   [..., 3] = [x_uav, y_uav, z_uav]
    euler_rpy: [..., 3] = [roll, pitch, yaw]

    return:
        target_world_xyz: [..., 3]
    """
    az = radar[..., 0:1]
    el = radar[..., 1:2]
    r = radar[..., 2:3]

    # body-frame relative xyz
    rel_body = torch.cat(
        [
            r * torch.cos(el) * torch.cos(az),
            r * torch.cos(el) * torch.sin(az),
            r * torch.sin(el),
        ],
        dim=-1,
    )  # [..., 3]

    rel_world = torch.matmul(rotmat_b2e, rel_body.unsqueeze(-1)).squeeze(-1)

    return rel_world + uav_pos


class ChannelMaskEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x, mask):
        feat = self.projector(x)
        m = mask.unsqueeze(-1)
        return feat * m + self.mask_token * (1.0 - m)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query_stream, context_stream):
        attn_out, _ = self.attn(query=query_stream, key=context_stream, value=context_stream)
        x = self.norm1(query_stream + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        pe = sinusoidal_position_encoding(x.size(1), x.size(2), x.device).unsqueeze(0)
        return self.encoder(x + pe)


class CollaborativeGAT(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x, adj_mask=None):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(x.size(-1))
        if adj_mask is not None:
            score = score.masked_fill(adj_mask <= 0, -1e9)
        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)
        agg = torch.matmul(attn, v)
        return self.out_norm(x + agg)


class LearnedKFrontEnd(nn.Module):
    """
    Learn a diagonal Kalman-like gain K and perform:
        x_post = x_pred + K ⊙ (z_obs - x_pred)

    State format:
        [px, vx, py, vy, pz, vz]

    Important:
        Radar angles are body-frame measurements.
        We convert them back to world-frame using UAV attitude.
    """

    def __init__(self, feature_dim: int = 64):
        super().__init__()

        # radar(3) + eo(2) + esm(2) + imu(6) + masks(3) + esm_mode(1)
        # + ages(2) + link_meas(6) + link_update(2) + pos(3) = 30
        self.obs_encoder = nn.Sequential(
            nn.Linear(30, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        # [x_pred(6), z_obs(6), innovation(6), pooled_obs(feature_dim)]
        self.k_net = nn.Sequential(
            nn.Linear(6 + 6 + 6 + feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 6),
            nn.Sigmoid(),
        )

    @staticmethod
    def _build_F(dt: torch.Tensor, device, dtype):
        """
        dt: [B]
        return F: [B, 6, 6]
        """
        bsz = dt.shape[0]
        Fm = torch.eye(6, device=device, dtype=dtype).unsqueeze(0).repeat(bsz, 1, 1)
        Fm[:, 0, 1] = dt
        Fm[:, 2, 3] = dt
        Fm[:, 4, 5] = dt
        return Fm

    def _build_obs_state(
        self,
        radar: torch.Tensor,
        radar_mask: torch.Tensor,
        fast_link: torch.Tensor,
        slow_link: torch.Tensor,
        fast_update: torch.Tensor,
        slow_update: torch.Tensor,
        fast_age: torch.Tensor,
        slow_age: torch.Tensor,
        uav_pos: torch.Tensor,
        rotmat_b2e: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build coarse observed state z_obs from radar + async links.
        radar:      [B,T,N,3]  body-frame [az,el,r]
        fast_link:  [B,T,N,3]
        slow_link:  [B,T,N,3]
        euler_rpy:  [B,T,N,3]  [roll,pitch,yaw]
        z_obs:      [B,T,6]
        """
        radar_xyz = radar_body_to_world_xyz(radar, uav_pos, rotmat_b2e)  # [B,T,N,3]
        fast_xyz = fast_link
        slow_xyz = slow_link

        radar_w = radar_mask
        fast_valid = ((fast_update > 0.5) | (fast_age < 2.0)).to(radar.dtype)
        slow_valid = ((slow_update > 0.5) | (slow_age < 5.0)).to(radar.dtype)
        fast_fresh = torch.exp(-fast_age.clamp_min(0.0) / 0.3)
        slow_fresh = torch.exp(-slow_age.clamp_min(0.0) / 2.0)
        fast_w = fast_valid * fast_fresh
        slow_w = slow_valid * slow_fresh

        # 构建基于精度矩阵（Information Matrix）的几何交叉定位融合
        # 雷达在径向（距离）方向精度高（48m），切向精度随距离增加而下降（r * 0.002）
        sigma_r = 48.0
        sigma_theta = 0.002
        
        r = torch.clamp(radar[..., 2:3], min=1.0)  # [B,T,N,1]
        var_r = sigma_r ** 2
        var_t = (r * sigma_theta) ** 2 + 1.0       # [B,T,N,1]

        # 视线单位向量
        rel_world = radar_xyz - uav_pos
        dir_vec = torch.nn.functional.normalize(rel_world, dim=-1)  # [B,T,N,3]
        dir_out = torch.einsum('...i,...j->...ij', dir_vec, dir_vec) # [B,T,N,3,3]
        I_3x3 = torch.eye(3, device=radar.device, dtype=radar.dtype).view(1, 1, 1, 3, 3)
        
        # 组装每个雷达量测的精度矩阵 W = (1/var_r)*dir_out + (1/var_t)*(I - dir_out)
        W_radar = (1.0 / var_r) * dir_out + (1.0 / var_t.unsqueeze(-1)) * (I_3x3 - dir_out) # [B,T,N,3,3]
        
        # 数据链的各向同性方差（假设 60m 噪声）
        var_link = 60.0 ** 2
        W_link = I_3x3 * (1.0 / var_link)  # [1,1,1,3,3]

        # 计算总信息矩阵和信息向量
        w_radar_exp = radar_w.unsqueeze(-1).unsqueeze(-1)
        wP = (W_radar * w_radar_exp).sum(dim=2)  # [B,T,3,3]
        wPp = (torch.einsum('...ij,...j->...i', W_radar, radar_xyz) * radar_w.unsqueeze(-1)).sum(dim=2)  # [B,T,3]

        wP += (W_link.expand_as(W_radar) * fast_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)
        wPp += (fast_xyz * fast_w.unsqueeze(-1)).sum(dim=2) / var_link

        wP += (W_link.expand_as(W_radar) * slow_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)
        wPp += (slow_xyz * slow_w.unsqueeze(-1)).sum(dim=2) / var_link

        # 添加正则项避免奇异矩阵（当无观测时，或者只有共线观测时）
        wP += torch.eye(3, device=radar.device, dtype=radar.dtype).view(1, 1, 3, 3) * 1e-6
        
        # 求解融合位置: z = W^{-1} (W*z)
        z_pos_meas = solve_3x3_onnx_safe(wP, wPp)  # [B,T,3]


        # If no valid source at some frame, carry previous estimate to avoid zeros.
        z_pos = torch.zeros_like(z_pos_meas)
        z_pos[:, 0] = z_pos_meas[:, 0]
        valid_frame = ((radar_w + fast_w + slow_w).sum(dim=2) > 0).to(radar.dtype)
        for t in range(1, radar.shape[1]):
            cur = z_pos_meas[:, t]
            keep_prev = (valid_frame[:, t] <= 0).unsqueeze(-1)
            z_pos[:, t] = torch.where(keep_prev, z_pos[:, t - 1], cur)

        dt_view = dt.view(-1, 1, 1).clamp_min(1e-6)
        z_vel = torch.zeros_like(z_pos)
        z_vel[:, 1:] = (z_pos[:, 1:] - z_pos[:, :-1]) / dt_view
        # first frame does not have reliable velocity observation
        z_vel[:, 0] = 0.0
        # avoid extreme spikes due to noisy frame-to-frame difference
        z_vel = z_vel.clamp(min=-400.0, max=400.0)

        z_obs = torch.zeros(radar.shape[0], radar.shape[1], 6, device=radar.device, dtype=radar.dtype)
        z_obs[:, :, 0] = z_pos[:, :, 0]
        z_obs[:, :, 1] = z_vel[:, :, 0]
        z_obs[:, :, 2] = z_pos[:, :, 1]
        z_obs[:, :, 3] = z_vel[:, :, 1]
        z_obs[:, :, 4] = z_pos[:, :, 2]
        z_obs[:, :, 5] = z_vel[:, :, 2]
        return z_obs

    def forward(
        self,
        radar: torch.Tensor,
        eo: torch.Tensor,
        esm: torch.Tensor,
        radar_mask: torch.Tensor,
        eo_mask: torch.Tensor,
        esm_mask: torch.Tensor,
        uav_pos: torch.Tensor,
        rotmat_b2e: torch.Tensor,
        fast_link: Optional[torch.Tensor] = None,
        slow_link: Optional[torch.Tensor] = None,
        fast_update: Optional[torch.Tensor] = None,
        slow_update: Optional[torch.Tensor] = None,
        imu: Optional[torch.Tensor] = None,
        fast_age: Optional[torch.Tensor] = None,
        slow_age: Optional[torch.Tensor] = None,
        esm_mode: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_post: [B, T, 6]
            z_obs:  [B, T, 6]
        """
        bsz, t_steps, n_agents, _ = radar.shape
        device = radar.device
        dtype = radar.dtype

        if imu is None:
            imu = torch.zeros(bsz, t_steps, n_agents, 6, device=device, dtype=dtype)
        if fast_link is None:
            fast_link = torch.zeros(bsz, t_steps, n_agents, 3, device=device, dtype=dtype)
        if slow_link is None:
            slow_link = torch.zeros(bsz, t_steps, n_agents, 3, device=device, dtype=dtype)
        if fast_update is None:
            fast_update = torch.zeros(bsz, t_steps, n_agents, device=device, dtype=dtype)
        if slow_update is None:
            slow_update = torch.zeros(bsz, t_steps, n_agents, device=device, dtype=dtype)
        if fast_age is None:
            fast_age = torch.zeros(bsz, t_steps, n_agents, device=device, dtype=dtype)
        if slow_age is None:
            slow_age = torch.zeros(bsz, t_steps, n_agents, device=device, dtype=dtype)
        if esm_mode is None:
            esm_mode = torch.zeros(bsz, t_steps, n_agents, device=device, dtype=dtype)
        if dt is None:
            dt = torch.full((bsz,), 0.1, device=device, dtype=dtype)

        # Correct radar body-frame geometry back to world frame
        z_obs = self._build_obs_state(
            radar=radar,
            radar_mask=radar_mask,
            fast_link=fast_link,
            slow_link=slow_link,
            fast_update=fast_update,
            slow_update=slow_update,
            fast_age=fast_age,
            slow_age=slow_age,
            uav_pos=uav_pos,
            rotmat_b2e=rotmat_b2e,
            dt=dt,
        )  # [B,T,6]

        raw_feat = torch.cat(
            [
                radar,                                # 3
                eo,                                   # 2
                esm,                                  # 2
                imu,                                  # 6
                radar_mask.unsqueeze(-1),             # 1
                eo_mask.unsqueeze(-1),                # 1
                esm_mask.unsqueeze(-1),               # 1
                esm_mode.unsqueeze(-1),               # 1
                fast_age.unsqueeze(-1),               # 1
                slow_age.unsqueeze(-1),               # 1
                fast_link,                            # 3
                slow_link,                            # 3
                fast_update.unsqueeze(-1),            # 1
                slow_update.unsqueeze(-1),            # 1
                uav_pos,                              # 3
            ],
            dim=-1,
        )  # [B,T,N,30]

        raw_enc = self.obs_encoder(raw_feat)  # [B,T,N,F]

        valid_sensor = ((radar_mask + eo_mask + esm_mask) > 0).float().unsqueeze(-1)
        pooled_obs = (raw_enc * valid_sensor).sum(dim=2) / (valid_sensor.sum(dim=2) + 1e-6)  # [B,T,F]

        Fm = self._build_F(dt, device, dtype)  # [B,6,6]

        x_post_list = []
        x_prev = z_obs[:, 0]  # [B,6]

        for t in range(t_steps):
            if t == 0:
                x_pred = x_prev
            else:
                x_pred = torch.bmm(Fm, x_prev.unsqueeze(-1)).squeeze(-1)

            z_t = z_obs[:, t]
            innovation = z_t - x_pred

            k_in = torch.cat([x_pred, z_t, innovation, pooled_obs[:, t]], dim=-1)
            k_diag = self.k_net(k_in)  # [B,6]

            x_post = x_pred + k_diag * innovation
            x_post_list.append(x_post)
            x_prev = x_post

        x_post = torch.stack(x_post_list, dim=1)  # [B,T,6]
        return x_post, z_obs


class CoopTrackerModel(nn.Module):
    """
    Learned-K front-end + dual-stream fusion + temporal modeling + multi-agent graph interaction.
    Radar is the active main stream; EO/ESM are fused as passive auxiliary stream.

    Important:
        Radar angles are body-frame measurements.
        Geometry inversion must use UAV attitude.
    """

    def __init__(self, d_model: int = 96):
        super().__init__()

        self.k_front = LearnedKFrontEnd(feature_dim=d_model)

        # radar: [az,el,r] + [x,y,z](corrected world geo) + [k_state(6)]
        self.radar_enc = ChannelMaskEncoder(input_dim=12, d_model=d_model)
        self.eo_enc = ChannelMaskEncoder(input_dim=5, d_model=d_model)
        self.esm_enc = ChannelMaskEncoder(input_dim=5, d_model=d_model)

        self.passive_conf_eo = nn.Linear(d_model, 1)
        self.passive_conf_esm = nn.Linear(d_model, 1)
        self.passive_missing = nn.Parameter(torch.zeros(1, 1, d_model))

        self.align_gamma = nn.Parameter(torch.ones(1, 1, d_model))
        self.align_beta = nn.Parameter(torch.zeros(1, 1, d_model))

        self.cross_radar_to_passive = CrossAttentionBlock(d_model=d_model, nhead=4)
        self.cross_passive_to_radar = CrossAttentionBlock(d_model=d_model, nhead=4)
        self.fusion_gate = nn.Linear(d_model * 2, 1)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        self.ego_pose_proj = nn.Linear(6, d_model)
        self.ego_pose_gate = nn.Sigmoid()

        self.temporal_encoder = TemporalTransformer(d_model=d_model, nhead=4, num_layers=2)
        self.coop_gat = CollaborativeGAT(d_model=d_model)

        self.pos_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        self.vel_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )

    def _adaptive_align(self, feat):
        mu = feat.mean(dim=1, keepdim=True)
        std = feat.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return self.align_gamma * (feat - mu) / std + self.align_beta

    def _passive_fusion(self, eo_feat, esm_feat, eo_mask, esm_mask):
        eo_valid = eo_mask.unsqueeze(-1)
        esm_valid = esm_mask.unsqueeze(-1)

        eo_conf = torch.sigmoid(self.passive_conf_eo(eo_feat)) * eo_valid
        esm_conf = torch.sigmoid(self.passive_conf_esm(esm_feat)) * esm_valid

        conf_sum = eo_conf + esm_conf
        fused = (eo_feat * eo_conf + esm_feat * esm_conf) / (conf_sum + 1e-6)

        passive_valid = ((eo_mask + esm_mask) > 0).float().unsqueeze(-1)
        fused = fused * passive_valid + self.passive_missing * (1.0 - passive_valid)
        return fused, passive_valid.squeeze(-1)

    def forward(
        self,
        radar,
        eo,
        esm,
        radar_mask,
        eo_mask,
        esm_mask,
        ref_pos,
        rotmat_b2e,
        uav_pos=None,
        ego_pose=None,
        fast_link=None,
        slow_link=None,
        fast_update=None,
        slow_update=None,
        imu=None,
        fast_age=None,
        slow_age=None,
        esm_mode=None,
        dt=None,
        use_gat=True,
        use_temporal=True,
    ):
        # radar/eo/esm: [B,T,N,C]
        bsz, t_steps, n_agents, _ = radar.shape
        device = radar.device
        dtype = radar.dtype

        if imu is None and ego_pose is not None:
            imu = ego_pose
        if imu is None:
            imu = torch.zeros(bsz, t_steps, n_agents, 6, device=device, dtype=dtype)

        if fast_link is None:
            fast_link = torch.zeros(bsz, t_steps, n_agents, 3, device=device, dtype=dtype)
        if slow_link is None:
            slow_link = torch.zeros(bsz, t_steps, n_agents, 3, device=device, dtype=dtype)

        # Convert absolute ECEF to relative coordinates based on ref_pos
        ref_pos_view = ref_pos.view(bsz, 1, 1, 3)
        uav_pos_rel = uav_pos

        # 判断绝对坐标是否为0，为0说明是空数据
        valid_fast = (fast_link.abs().sum(dim=-1) > 0).to(dtype)
        fast_link_rel = fast_link * valid_fast.unsqueeze(-1)

        fast_age_fixed = torch.where(valid_fast > 0.5, fast_age, torch.full_like(fast_age, 999.0))
        fast_update_fixed = fast_update * valid_fast

        valid_slow = (slow_link.abs().sum(dim=-1) > 0).to(dtype)
        slow_link_rel = slow_link * valid_slow.unsqueeze(-1)

        # 同理修复 SlowLink
        slow_age_fixed = torch.where(valid_slow > 0.5, slow_age, torch.full_like(slow_age, 999.0))
        slow_update_fixed = slow_update * valid_slow

        # ------------------------------------------------------------------
        # 1) Learned-K front-end
        # ------------------------------------------------------------------
        k_state, z_obs = self.k_front(
            radar=radar,
            eo=eo,
            esm=esm,
            radar_mask=radar_mask,
            eo_mask=eo_mask,
            esm_mask=esm_mask,
            uav_pos=uav_pos_rel,
            rotmat_b2e=rotmat_b2e,
            fast_link=fast_link_rel,
            slow_link=slow_link_rel,
            fast_update=fast_update_fixed,  # 传入修复后的变量
            slow_update=slow_update_fixed,  # 传入修复后的变量
            imu=imu,
            fast_age=fast_age_fixed,  # 传入修复后的变量
            slow_age=slow_age_fixed,  # 传入修复后的变量
            esm_mode=esm_mode,
            dt=dt,
        )

        k_state_agent = k_state.unsqueeze(2).expand(-1, -1, n_agents, -1)  # [B,T,N,6]

        # ------------------------------------------------------------------
        # 2) Multimodal backbone
        # ------------------------------------------------------------------
        radar_bt = radar.permute(0, 2, 1, 3).reshape(bsz * n_agents, t_steps, -1)
        eo_bt = eo.permute(0, 2, 1, 3).reshape(bsz * n_agents, t_steps, -1)
        esm_bt = esm.permute(0, 2, 1, 3).reshape(bsz * n_agents, t_steps, -1)
        rotmat_bt = rotmat_b2e.permute(0, 2, 1, 3, 4).reshape(bsz * n_agents, t_steps, 3, 3)

        rm_bt = radar_mask.permute(0, 2, 1).reshape(bsz * n_agents, t_steps)
        em_bt = eo_mask.permute(0, 2, 1).reshape(bsz * n_agents, t_steps)
        esm_m_bt = esm_mask.permute(0, 2, 1).reshape(bsz * n_agents, t_steps)

        pos_bt = uav_pos_rel.permute(0, 2, 1, 3).reshape(bsz * n_agents, t_steps, 3)
        imu_bt = imu.permute(0, 2, 1, 3).reshape(bsz * n_agents, t_steps, 6)
        k_state_bt = k_state_agent.permute(0, 2, 1, 3).reshape(bsz * n_agents, t_steps, 6)

        # Correct radar body-frame spherical -> world xyz using Euler attitude
        radar_geo = radar_body_to_world_xyz(radar_bt, pos_bt, rotmat_bt)  # [B*N,T,3]
        radar_in = torch.cat([radar_bt, radar_geo, k_state_bt], dim=-1)

        radar_geo_btn = radar_geo.reshape(bsz, n_agents, t_steps, 3).permute(0, 2, 1, 3).contiguous()

        # EO / ESM remain as body-frame angles + body direction; IMU will help the network align them
        eo_az = eo_bt[..., 0:1]
        eo_el = eo_bt[..., 1:2]
        eo_dir_body = torch.cat(
            [
                torch.cos(eo_el) * torch.cos(eo_az),
                torch.cos(eo_el) * torch.sin(eo_az),
                torch.sin(eo_el),
            ],
            dim=-1,
        )  # 这是机体坐标系下的方向
        # 乘上矩阵，变成 ECEF 世界坐标系下的方向
        eo_dir_world = torch.matmul(rotmat_bt, eo_dir_body.unsqueeze(-1)).squeeze(-1)
        eo_in = torch.cat([eo_bt, eo_dir_world], dim=-1)

        esm_az = esm_bt[..., 0:1]
        esm_el = esm_bt[..., 1:2]
        esm_dir_body = torch.cat(
            [
                torch.cos(esm_el) * torch.cos(esm_az),
                torch.cos(esm_el) * torch.sin(esm_az),
                torch.sin(esm_el),
            ],
            dim=-1,
        )
        esm_dir_world = torch.matmul(rotmat_bt, esm_dir_body.unsqueeze(-1)).squeeze(-1)
        esm_in = torch.cat([esm_bt, esm_dir_world], dim=-1)

        radar_feat = self.radar_enc(radar_in, rm_bt)
        eo_feat = self.eo_enc(eo_in, em_bt)
        esm_feat = self.esm_enc(esm_in, esm_m_bt)

        passive_feat, passive_valid = self._passive_fusion(eo_feat, esm_feat, em_bt, esm_m_bt)

        radar_feat = self._adaptive_align(radar_feat)
        passive_feat = self._adaptive_align(passive_feat)

        pose_gate = self.ego_pose_gate(self.ego_pose_proj(imu_bt))
        radar_feat = radar_feat * (1.0 + pose_gate)
        passive_feat = passive_feat * (1.0 + pose_gate)

        radar_ctx = self.cross_radar_to_passive(radar_feat, passive_feat)
        passive_ctx = self.cross_passive_to_radar(passive_feat, radar_feat)

        alpha = torch.sigmoid(self.fusion_gate(torch.cat([radar_ctx, passive_ctx], dim=-1)))
        fused = alpha * radar_ctx + (1.0 - alpha) * passive_ctx
        fused = self.fusion_proj(torch.cat([fused, radar_ctx], dim=-1))

        if use_temporal:
            temporal = self.temporal_encoder(fused)  # [B*N,T,D]
        else:
            temporal = fused

        temporal_btn = temporal.reshape(bsz, n_agents, t_steps, -1).permute(0, 2, 1, 3).contiguous()

        # adjacency
        valid_sensor = ((radar_mask + eo_mask + esm_mask) > 0).float()  # [B,T,N]
        adj = valid_sensor.reshape(bsz * t_steps, n_agents, 1) * valid_sensor.reshape(bsz * t_steps, 1, n_agents)

        pos_t = uav_pos_rel.reshape(bsz * t_steps, n_agents, 3)
        dist = torch.cdist(pos_t, pos_t)

        # formation spacing is ~10km, use 20km neighborhood
        dist_neighbor = (dist <= 20000.0).float()
        adj = adj * dist_neighbor

        eye = torch.eye(n_agents, device=device).unsqueeze(0)
        adj = torch.maximum(adj, eye)

        graph_in = temporal_btn.reshape(bsz * t_steps, n_agents, -1)
        if use_gat:
            graph_out = self.coop_gat(graph_in, adj_mask=adj)
        else:
            graph_out = graph_in
        graph_out = graph_out.reshape(bsz, t_steps, n_agents, -1)

        node_weight = valid_sensor.unsqueeze(-1)
        global_feat = (graph_out * node_weight).sum(dim=2) / (node_weight.sum(dim=2) + 1e-6)

        pos_res = self.pos_head(global_feat)  # [B,T,3]
        vel_res = self.vel_head(global_feat)  # [B,T,3]
        
        # Interleave position and velocity to match [px, vx, py, vy, pz, vz] format
        residual = torch.empty(bsz, t_steps, 6, device=device, dtype=dtype)
        residual[:, :, 0] = pos_res[:, :, 0] # px
        residual[:, :, 1] = vel_res[:, :, 0] # vx
        residual[:, :, 2] = pos_res[:, :, 1] # py
        residual[:, :, 3] = vel_res[:, :, 1] # vy
        residual[:, :, 4] = pos_res[:, :, 2] # pz
        residual[:, :, 5] = vel_res[:, :, 2] # vz


        # final prediction = learned-K state + deep residual refinement
        pred_rel = k_state + residual

        # Add ref_pos back to position components (0, 2, 4) to output absolute coordinates
        pred_abs = pred_rel.clone()
        pred_abs[:, :, 0] += ref_pos[:, 0].unsqueeze(1)
        pred_abs[:, :, 2] += ref_pos[:, 1].unsqueeze(1)
        pred_abs[:, :, 4] += ref_pos[:, 2].unsqueeze(1)

        _ = passive_valid
        _ = radar_geo_btn
        _ = z_obs
        return pred_abs
