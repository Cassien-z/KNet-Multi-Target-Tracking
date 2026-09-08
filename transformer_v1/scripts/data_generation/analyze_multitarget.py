"""
analyze_multitarget.py
可视化 script_multitarget.py 生成的 .mat 格式数据集（多目标）

Data_Label  : [T, num_targets, 6]  — [px, vx, py, vy, pz, vz]
Data_Input  : [T,]    — 每帧含 GlobalTime 和 UAV 列表
  UAV[u].Pose.PositionWorld : [3]   平台自身 ECEF 位置
  UAV[u].Radar.Meas/Mask    : 雷达测量 [num_targets, 3] + Mask [num_targets]
  UAV[u].EO.Meas/Mask       : EO  测量 [num_targets, 2] + Mask [num_targets]
  UAV[u].ESM.Meas/Mask      : ESM 测量 [num_targets, 2] + Mask [num_targets]
  UAV[u].FastLink.Meas      : 僚机数据链目标 ECEF XYZ [num_targets, 3]
  UAV[u].FastLink.IsUpdate  : [num_targets] (1.0 = 本帧有效数据链包)
"""

import os
import sys
import glob
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# 参数
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(SCRIPT_DIR, "train_data_multitarget")
FILE_IDX  = 0                   # 可视化第几条轨迹（0-based）
RAY_LEN   = 5000.0              # EO/ESM 方向射线显示长度 (m)

UAV_COLORS  = ["red", "limegreen", "dodgerblue", "darkorange", "purple"]
UAV_LABELS  = ["Master-0 (All)", "Wingman-1 (R)", "Wingman-2 (R+ESM)", "Wingman-3 (R+EO)", "Wingman-4 (R)"]

# ============================================================
# 加载文件
# ============================================================
mat_files = sorted(glob.glob(os.path.join(DATA_DIR, "track_data_*.mat")))
if not mat_files:
    print(f"[ERROR] No .mat files found in {DATA_DIR}/. Please run script_multitarget.py first.")
    exit()

target_file = mat_files[FILE_IDX % len(mat_files)]
print(f"[INFO] Loading: {target_file}")

mat = scipy.io.loadmat(target_file, simplify_cells=True)
Data_Input = mat["Data_Input"]    # shape (T,), dtype=object
Data_Label = mat["Data_Label"]    # shape (T, num_targets, 6)
DT         = float(mat.get("T", 0.1))
T          = len(Data_Input)
N_UAV      = 5

num_targets = Data_Label.shape[1] if Data_Label.ndim == 3 else 1
print(f"   T={T}, dt={DT}s, N_UAV={N_UAV}, num_targets={num_targets}")

# ============================================================
# 解析数据
# ============================================================
time_stamps = np.arange(T) * DT

# 目标真值
if Data_Label.ndim == 3:
    truth_pos = Data_Label[:, :, [0, 2, 4]].astype(np.float32)   # [T, num_targets, 3]
    truth_vel = Data_Label[:, :, [1, 3, 5]].astype(np.float32)   # [T, num_targets, 3]
else:
    truth_pos = Data_Label[:, [0, 2, 4]].astype(np.float32)[:, None, :]
    truth_vel = Data_Label[:, [1, 3, 5]].astype(np.float32)[:, None, :]

# 每架 UAV 的轨迹、雷达/EO/ESM 测量、数据链 ECEF 坐标
uav_pos     = [[] for _ in range(N_UAV)]   # 自身位置轨迹
radar_pts   = [[] for _ in range(N_UAV)]   # 雷达重建点
eo_rays     = [[] for _ in range(N_UAV)]   # EO 方向射线 (origin, dir)
esm_rays    = [[] for _ in range(N_UAV)]   # ESM 方向射线
dl_pts      = [[] for _ in range(N_UAV)]   # 数据链目标 ECEF 点（僚机）

# 对于 Mask 时序，我们统计每一帧内成功探测到的目标数量 (0~num_targets)
radar_mask_ts  = [[] for _ in range(N_UAV)]
eo_mask_ts     = [[] for _ in range(N_UAV)]
esm_mask_ts    = [[] for _ in range(N_UAV)]


def _to_arr_2d(v, dim):
    """提取二维数组 (N, dim)"""
    try:
        a = np.asarray(v, dtype=float)
        if a.ndim == 1 and dim == 1:
            return a
        elif a.ndim == 2 and a.shape[1] >= dim:
            return a[:, :dim]
    except Exception:
        pass
    return None


def _to_arr_1d(v):
    """提取一维数组 (N,)"""
    try:
        return np.asarray(v, dtype=float).flatten()
    except Exception:
        pass
    return None


def azel_to_unit(az, el):
    return np.array([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)])

def euler_to_rot(roll, pitch, yaw):
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


for t in range(T):
    frame = Data_Input[t]
    if not isinstance(frame, dict):
        continue

    uavs = frame.get("UAV", None)
    if not isinstance(uavs, (list, np.ndarray)):
        continue

    for u in range(min(N_UAV, len(uavs))):
        ud = uavs[u]
        if not isinstance(ud, dict):
            continue

        # --- 平台自身位置与姿态 ---
        pos = _to_arr_1d(ud.get("Pose", {}).get("PositionWorld", None))
        if pos is None: pos = np.zeros(3)
        uav_pos[u].append(pos)
        
        euler = _to_arr_1d(ud.get("Pose", {}).get("EulerRPY", None))
        if euler is None: euler = np.zeros(3)
        R_b2w = euler_to_rot(euler[0], euler[1], euler[2])

        # --- Radar ---
        r_meas = _to_arr_2d(ud.get("Radar", {}).get("Meas", None), 3)
        r_mask = _to_arr_1d(ud.get("Radar", {}).get("Mask", None))
        if r_meas is not None and r_mask is not None:
            radar_mask_ts[u].append(np.sum(r_mask > 0.5))
            for i in range(len(r_mask)):
                if r_mask[i] > 0.5:
                    az, el, r = r_meas[i]
                    vec_body = np.array([r * np.cos(el) * np.cos(az),
                                         r * np.cos(el) * np.sin(az),
                                         r * np.sin(el)])
                    radar_pts[u].append(pos + R_b2w @ vec_body)
        else:
            radar_mask_ts[u].append(0)

        # --- EO ---
        eo_meas = _to_arr_2d(ud.get("EO", {}).get("Meas", None), 2)
        eo_mask = _to_arr_1d(ud.get("EO", {}).get("Mask", None))
        if eo_meas is not None and eo_mask is not None:
            eo_mask_ts[u].append(np.sum(eo_mask > 0.5))
            for i in range(len(eo_mask)):
                if eo_mask[i] > 0.5:
                    az, el = eo_meas[i]
                    dir_body = azel_to_unit(az, el)
                    eo_rays[u].append((pos.copy(), R_b2w @ dir_body))
        else:
            eo_mask_ts[u].append(0)

        # --- ESM ---
        esm_meas = _to_arr_2d(ud.get("ESM", {}).get("Meas", None), 2)
        esm_mask = _to_arr_1d(ud.get("ESM", {}).get("Mask", None))
        if esm_meas is not None and esm_mask is not None:
            esm_mask_ts[u].append(np.sum(esm_mask > 0.5))
            for i in range(len(esm_mask)):
                if esm_mask[i] > 0.5:
                    az = esm_meas[i][0]
                    dir_body = azel_to_unit(az, 0.0)
                    esm_rays[u].append((pos.copy(), R_b2w @ dir_body))
        else:
            esm_mask_ts[u].append(0)

        # --- FastLink（僚机数据链目标 ECEF）---
        if u > 0:  # 只有僚机有数据链
            fl      = ud.get("FastLink", {})
            fl_meas = _to_arr_2d(fl.get("Meas", None), 3)
            fl_upd  = _to_arr_1d(fl.get("IsUpdate", None))
            if fl_meas is not None and fl_upd is not None:
                for i in range(len(fl_upd)):
                    if fl_upd[i] > 0.5 and np.linalg.norm(fl_meas[i]) > 1.0:
                        dl_pts[u].append(fl_meas[i].copy())


# 转 numpy
uav_pos_arr = [np.array(x) if x else np.empty((0, 3)) for x in uav_pos]
radar_arr   = [np.array(x) if x else np.empty((0, 3)) for x in radar_pts]
dl_arr      = [np.array(x) if x else np.empty((0, 3)) for x in dl_pts]

# mask 时间序列
radar_mask_arr = [np.array(x) for x in radar_mask_ts]
eo_mask_arr    = [np.array(x) for x in eo_mask_ts]
esm_mask_arr   = [np.array(x) for x in esm_mask_ts]

# ============================================================
# 绘图布局: 2列
#   左列 (行1-3): 3D 轨迹
#   右上: 速度曲线
#   右中: 检测到的目标数量时序图
#   右下: 僚机数据链路累计观测质量
# ============================================================
fig = plt.figure(figsize=(20, 12))
fig.suptitle(f"Multi-Target MAT Dataset Visualization — {os.path.basename(target_file)}", fontsize=15, fontweight="bold")

ax3d = fig.add_subplot(1, 2, 1, projection="3d")

# ---- 3D 轨迹 (多目标) ----
for i in range(num_targets):
    ax3d.plot(truth_pos[:, i, 0], truth_pos[:, i, 1], truth_pos[:, i, 2],
              color="black", linewidth=1.5, zorder=5, alpha=0.7)
ax3d.plot([], [], [], color="black", linewidth=1.5, label=f"Target Truths ({num_targets})")

# 平台轨迹
for u in range(N_UAV):
    if uav_pos_arr[u].shape[0] > 0:
        ax3d.plot(uav_pos_arr[u][:, 0], uav_pos_arr[u][:, 1], uav_pos_arr[u][:, 2],
                  linestyle="--", color=UAV_COLORS[u], linewidth=1.5,
                  label=UAV_LABELS[u], alpha=0.8)
        # 起始标记
        ax3d.scatter(*uav_pos_arr[u][0], color=UAV_COLORS[u], s=60, marker="^", zorder=6)

# 雷达点云（主机）
if radar_arr[0].shape[0] > 0:
    ax3d.scatter(radar_arr[0][:, 0], radar_arr[0][:, 1], radar_arr[0][:, 2],
                 color="magenta", s=4, alpha=0.3, label="Radar Meas (Master)", marker="o")

# EO 射线（主机，随机采样 40 条防遮挡）
if eo_rays[0]:
    idxs = np.random.choice(len(eo_rays[0]), min(40, len(eo_rays[0])), replace=False)
    for i in idxs:
        o, d = eo_rays[0][i]
        end  = o + d * RAY_LEN
        ax3d.plot([o[0], end[0]], [o[1], end[1]], [o[2], end[2]],
                  color="cyan", alpha=0.2, linewidth=0.5)
    ax3d.plot([], [], color="cyan", linewidth=1.0, label="EO Ray (Master, sample)")

# 数据链 ECEF 点（僚机）
for u in range(1, 5):
    if dl_arr[u].shape[0] > 0:
        ax3d.scatter(dl_arr[u][:, 0], dl_arr[u][:, 1], dl_arr[u][:, 2],
                     color=UAV_COLORS[u], s=6, alpha=0.3,
                     label=f"Datalink ECEF (Wingman {u})", marker="x")

ax3d.set_xlabel("X (m)")
ax3d.set_ylabel("Y (m)")
ax3d.set_zlabel("Z (m)")
ax3d.set_title(f"3D Trajectories & Sensor Observations ({num_targets} Targets)")
ax3d.legend(fontsize=7, loc="upper left")

# ---- 右侧子图 ----
ax_vel  = fig.add_subplot(3, 2, 2)
ax_mask = fig.add_subplot(3, 2, 4)
ax_dl   = fig.add_subplot(3, 2, 6)

# 速度曲线 (多目标叠画)
for i in range(num_targets):
    ax_vel.plot(time_stamps, truth_vel[:, i, 0], color="red",   alpha=0.2, linewidth=0.8)
    ax_vel.plot(time_stamps, truth_vel[:, i, 1], color="green", alpha=0.2, linewidth=0.8)
    ax_vel.plot(time_stamps, truth_vel[:, i, 2], color="blue",  alpha=0.2, linewidth=0.8)

ax_vel.plot([], [], color="red", label="Vx")
ax_vel.plot([], [], color="green", label="Vy")
ax_vel.plot([], [], color="blue", label="Vz")

ax_vel.set_ylabel("Velocity (m/s)")
ax_vel.set_title("Target Velocities")
ax_vel.legend(fontsize=8)
ax_vel.grid(True, linestyle="--", alpha=0.5)

# Sensor Detected Target Count time series (Master UAV-0)
if len(radar_mask_arr[0]) > 0:
    ts_mask = time_stamps[:len(radar_mask_arr[0])]
    ax_mask.plot(ts_mask, radar_mask_arr[0], color="magenta", label="Radar Detected Targets", linewidth=1.2, alpha=0.8)
    ax_mask.plot(ts_mask, eo_mask_arr[0],    color="cyan",    label="EO Detected Targets",    linewidth=1.2, alpha=0.8)
    ax_mask.plot(ts_mask, esm_mask_arr[0],   color="orange",  label="ESM Detected Targets",   linewidth=1.2, alpha=0.8)

ax_mask.set_ylabel("Number of Detected Targets")
ax_mask.set_title(f"Master Sensor Detections (Max: {num_targets})")
ax_mask.set_ylim(-0.5, num_targets + 0.5)
ax_mask.legend(fontsize=8)
ax_mask.grid(True, linestyle="--", alpha=0.5)

# Datalink reception quality (wingmen cumulative detected targets)
for u in range(1, 5):
    if len(radar_mask_arr[u]) > 0:
        ts_u  = time_stamps[:len(radar_mask_arr[u])]
        valid = np.array(radar_mask_arr[u])
        cum   = np.cumsum(valid)
        ax_dl.plot(ts_u, cum, color=UAV_COLORS[u], label=f"Wingman {u} Radar Detections (cumsum)")

ax_dl.set_xlabel("Time (s)")
ax_dl.set_ylabel("Cumulative Target Measurements")
ax_dl.set_title("Wingman Radar Coverage (Cumulative Target Points)")
ax_dl.legend(fontsize=8)
ax_dl.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
print("[DONE] Multi-Target Visualization complete")
