"""
analyze_singletarget.py
可视化四个数据集生成脚本生成的 .mat 格式数据（ECEF 地心坐标系）

Data_Label  : [T, 6]  — [X, Vx, Y, Vy, Z, Vz]  (ECEF 绝对坐标)
Data_Input  : [T,]    — 每帧含 GlobalTime 和 UAV 列表 (1 长机 + 4 僚机)
  UAV[0] 长机:
    Pose.PositionWorld  — ECEF 绝对位置
    Pose.EulerRPY       — [roll, pitch, yaw] (ENU 参考系航向/俯仰)
    Radar/EO/ESM        — 机体系量测 [az, el, r] / [az, el] / [az, el]
  UAV[1..4] 僚机:
    Pose.PositionWorld  — ECEF 绝对位置
    FastLink.Meas       — 僚机雷达重建的目标 ECEF XYZ
    FastLink.IsUpdate   — 1.0 = 本帧有有效数据链包

编队配置：
  WM1 (u=1): 雷达
  WM2 (u=2): 雷达 + EO
  WM3 (u=3): 雷达 + HGESM
  WM4 (u=4): 雷达
"""

import os
import glob
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────── 用户参数 ────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 修改此变量以切换数据集目录和文件
DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "data", "test_data")
FILE_PAT  = r"D:\workstation\End-to-End-Tracking-System\transformer\data\test_data\test_data_cssw\track_data_cssw_*.mat"
FILE_IDX  = 12         # 第几条轨迹（0-based）

RAY_LEN   = 4000.0    # EO/ESM 方向射线显示长度 (m)

# 5个 UAV 的颜色 / 标签
UAV_COLORS = ["red", "limegreen", "dodgerblue", "orange", "violet"]
UAV_LABELS = [
    "长机-0 (雷达+光电+电子侦察)",
    "僚机-1 (雷达)",
    "僚机-2 (雷达+光电)",
    "僚机-3 (雷达+高精电子侦察)",
    "僚机-4 (雷达)",
]
N_UAV = 5

# ─────────────────────── 加载文件 ────────────────────────────────────
mat_files = sorted(glob.glob(os.path.join(DATA_DIR, FILE_PAT)))
if not mat_files:
    print(f"[ERROR] No .mat files in {DATA_DIR}")
    raise SystemExit(1)

target_file = mat_files[FILE_IDX % len(mat_files)]
print(f"[INFO] Loading: {target_file}")

mat        = scipy.io.loadmat(target_file, simplify_cells=True)
Data_Input = mat["Data_Input"]
Data_Label = mat["Data_Label"]
DT         = float(mat.get("T", 0.1))
T          = len(Data_Input)

print(f"   T={T}, dt={DT}s, N_UAV={N_UAV}")

# ─────────────────────── ECEF 工具 ───────────────────────────────────
def _get_body_frame_axes_np(vel_ecef, pos_ecef):
    """
    在 ECEF 中定义机体坐标系轴（与 ecef_utils.get_body_frame_axes 一致）：
      fwd   — 沿速度方向
      right — 当地水平面内垂直于前进方向（使用当地地理"上"定义）
      up    — 垂直于前进方向，朝向当地天顶
    """
    local_up = pos_ecef / (np.linalg.norm(pos_ecef) + 1e-12)
    speed = np.linalg.norm(vel_ecef)
    unit_fwd = vel_ecef / speed if speed > 1e-6 else np.array([1., 0., 0.])

    unit_right = np.cross(local_up, unit_fwd)
    nr = np.linalg.norm(unit_right)
    if nr < 1e-6:
        arb = np.array([1., 0., 0.]) if abs(local_up[0]) < 0.9 else np.array([0., 1., 0.])
        unit_right = np.cross(local_up, arb)
        unit_right /= np.linalg.norm(unit_right)
    else:
        unit_right /= nr

    unit_up = np.cross(unit_fwd, unit_right)
    unit_up /= np.linalg.norm(unit_up)
    return unit_fwd, unit_right, unit_up


def body_azel_r_to_ecef(az, el, r, platform_pos, platform_vel):
    """
    将机体系量测 (az, el, r) + 平台 ECEF 状态 → 目标 ECEF 坐标。
    az=0 朝前, az>0 向右; el=0 水平, el>0 向上。
    """
    fwd, rgt, up = _get_body_frame_axes_np(platform_vel, platform_pos)
    dx = r * np.cos(el) * np.cos(az)  # 前向
    dy = r * np.cos(el) * np.sin(az)  # 右向
    dz = r * np.sin(el)               # 上向
    return platform_pos + dx * fwd + dy * rgt + dz * up


def body_azel_to_direction(az, el, platform_pos, platform_vel):
    """EO/ESM 角度量测 → ECEF 方向单位矢量"""
    fwd, rgt, up = _get_body_frame_axes_np(platform_vel, platform_pos)
    dx = np.cos(el) * np.cos(az)
    dy = np.cos(el) * np.sin(az)
    dz = np.sin(el)
    return dx * fwd + dy * rgt + dz * up


# ─────────────────────── 辅助函数 ────────────────────────────────────
def _to_arr(v, dim):
    try:
        a = np.asarray(v, dtype=float).flatten()
        if a.size >= dim:
            return a[:dim]
    except Exception:
        pass
    return np.zeros(dim)

def _scalar(v, default=0.0):
    try:
        return float(np.asarray(v).flatten()[0])
    except Exception:
        return default


# ─────────────────────── 解析数据 ────────────────────────────────────
time_stamps = np.arange(T) * DT

truth_pos = Data_Label[:, [0, 2, 4]].astype(float)   # [T, 3]
truth_vel = Data_Label[:, [1, 3, 5]].astype(float)   # [T, 3]

uav_pos      = [[] for _ in range(N_UAV)]
uav_vel      = [[] for _ in range(N_UAV)]   # ECEF 速度（从 EulerRPY + PositionWorld 派生）
radar_pts    = [[] for _ in range(N_UAV)]   # 雷达重建的目标 ECEF 点
eo_rays      = [[] for _ in range(N_UAV)]   # EO 方向射线
esm_rays     = [[] for _ in range(N_UAV)]   # ESM 方向射线
dl_pts       = [[] for _ in range(N_UAV)]   # 数据链 ECEF 点（僚机）

radar_mask_ts = [[] for _ in range(N_UAV)]
eo_mask_ts    = [[] for _ in range(N_UAV)]
esm_mask_ts   = [[] for _ in range(N_UAV)]
dl_update_ts  = [[] for _ in range(N_UAV)]


def _euler_to_ecef_vel(roll, pitch, yaw, speed, pos_ecef):
    """
    从 EulerRPY (ENU 参考系的航向/俯仰) 重建 ECEF 速度矢量。
    yaw   = 航向角（ENU，从北顺时针，rad）
    pitch = 俯仰角（rad）
    speed = 飞行速度量级（m/s），若未知可传 None
    """
    # ENU 速度
    vE = np.cos(pitch) * np.sin(yaw)
    vN = np.cos(pitch) * np.cos(yaw)
    vU = np.sin(pitch)
    vel_enu = np.array([vE, vN, vU])

    # ENU → ECEF 旋转（用位置近似纬经度）
    r = np.linalg.norm(pos_ecef)
    lat = np.arcsin(pos_ecef[2] / r)
    lon = np.arctan2(pos_ecef[1], pos_ecef[0])
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    R_n2e = np.array([
        [-slon,          clon,          0.    ],
        [-slat*clon,    -slat*slon,     clat  ],
        [ clat*clon,     clat*slon,     slat  ],
    ]).T   # ENU→ECEF
    return R_n2e @ vel_enu  # 单位向量，若 speed 已知则乘以 speed


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

        pos   = _to_arr(ud.get("Pose", {}).get("PositionWorld", None), 3)
        euler = _to_arr(ud.get("Pose", {}).get("EulerRPY", None), 3)
        uav_pos[u].append(pos.copy())

        # 用 EulerRPY 重建 ECEF 速度方向（用于机体系量测还原）
        roll, pitch, yaw = euler
        vel_dir = _euler_to_ecef_vel(roll, pitch, yaw, 1.0, pos)
        # 用 250 m/s 作为速度量级估计（不影响方向）
        vel_est = vel_dir * 250.0
        uav_vel[u].append(vel_est.copy())

        # ── Radar ──────────────────────────────────────────────
        r_meas = _to_arr(ud.get("Radar", {}).get("Meas", None), 3)
        r_mask = _scalar(ud.get("Radar", {}).get("Mask", 0.0))
        radar_mask_ts[u].append(r_mask)
        if r_mask > 0.5:
            az, el, r = r_meas
            # 正确：用 ECEF 机体系还原目标位置
            pt = body_azel_r_to_ecef(az, el, r, pos, vel_est)
            radar_pts[u].append(pt)

        # ── EO ─────────────────────────────────────────────────
        eo_meas = _to_arr(ud.get("EO", {}).get("Meas", None), 2)
        eo_mask = _scalar(ud.get("EO", {}).get("Mask", 0.0))
        eo_mask_ts[u].append(eo_mask)
        if eo_mask > 0.5:
            az, el = eo_meas
            d = body_azel_to_direction(az, el, pos, vel_est)
            eo_rays[u].append((pos.copy(), d))

        # ── ESM ────────────────────────────────────────────────
        esm_meas = _to_arr(ud.get("ESM", {}).get("Meas", None), 2)
        esm_mask = _scalar(ud.get("ESM", {}).get("Mask", 0.0))
        esm_mask_ts[u].append(esm_mask)
        if esm_mask > 0.5:
            az = esm_meas[0]
            d = body_azel_to_direction(az, 0.0, pos, vel_est)
            esm_rays[u].append((pos.copy(), d))

        # ── FastLink（仅僚机） ──────────────────────────────────
        fl      = ud.get("FastLink", {})
        fl_meas = _to_arr(fl.get("Meas", None), 3)
        fl_upd  = _scalar(fl.get("IsUpdate", 0.0))
        dl_update_ts[u].append(fl_upd)
        if u > 0 and fl_upd > 0.5 and np.linalg.norm(fl_meas) > 1e5:
            dl_pts[u].append(fl_meas.copy())

# 转 numpy
uav_pos_arr = [np.array(x) if x else np.empty((0, 3)) for x in uav_pos]
radar_arr   = [np.array(x) if x else np.empty((0, 3)) for x in radar_pts]
dl_arr      = [np.array(x) if x else np.empty((0, 3)) for x in dl_pts]

radar_mask_arr  = [np.array(x) for x in radar_mask_ts]
eo_mask_arr     = [np.array(x) for x in eo_mask_ts]
esm_mask_arr    = [np.array(x) for x in esm_mask_ts]
dl_update_arr   = [np.array(x) for x in dl_update_ts]

# ─────────────────────── 绘图 ────────────────────────────────────────
# 将 ECEF 坐标相对化（以真值轨迹中心为原点），方便阅读
ref_center = truth_pos.mean(axis=0)

def rel(pts):
    if pts.shape[0] == 0:
        return pts
    return pts - ref_center[None, :]


fig = plt.figure(figsize=(22, 12))
fig.suptitle(
    f"多对一协同打击末端引导场景三维态势与目标规避机动分析 — {os.path.basename(target_file)}",
    fontsize=18, fontweight="bold"
)

# ───── 3D 轨迹 (左列) ───────────────────────────────────────────────
ax3d = fig.add_subplot(1, 2, 1, projection="3d")

tp = rel(truth_pos)
ax3d.plot(tp[:, 0], tp[:, 1], tp[:, 2],
          color="black", linewidth=2.5, label="目标真实轨迹", zorder=5)
ax3d.scatter(*tp[0], color="black", s=80, marker="*", zorder=6)

# 平台轨迹
for u in range(N_UAV):
    if uav_pos_arr[u].shape[0] > 0:
        pp = rel(uav_pos_arr[u])
        ax3d.plot(pp[:, 0], pp[:, 1], pp[:, 2],
                  linestyle="--", color=UAV_COLORS[u], linewidth=1.5,
                  label=UAV_LABELS[u], alpha=0.85)
        ax3d.scatter(*pp[0], color=UAV_COLORS[u], s=60, marker="^", zorder=6)

# 雷达点云（主机 u=0，用 ECEF 机体系重建）
if radar_arr[0].shape[0] > 0:
    rp = rel(radar_arr[0])
    ax3d.scatter(rp[:, 0], rp[:, 1], rp[:, 2],
                 color="magenta", s=8, alpha=0.35,
                 label="长机雷达量测", marker="o")

# EO 射线（主机，随机采样 15 条）
if eo_rays[0]:
    sample_idx = np.random.choice(len(eo_rays[0]), min(15, len(eo_rays[0])), replace=False)
    for i in sample_idx:
        o, d = eo_rays[0][i]
        end  = o + d * RAY_LEN
        o2, end2 = o - ref_center, end - ref_center
        ax3d.plot([o2[0], end2[0]], [o2[1], end2[1]], [o2[2], end2[2]],
                  color="cyan", alpha=0.35, linewidth=0.8)
    ax3d.plot([], [], color="cyan", linewidth=1.2, label="长机光电测向射线(采样)")

# 数据链 ECEF 点（4 架僚机 u=1..4）
dl_colors = ["limegreen", "dodgerblue", "orange", "violet"]
for u in range(1, N_UAV):
    if dl_arr[u].shape[0] > 0:
        dp = rel(dl_arr[u])
        ax3d.scatter(dp[:, 0], dp[:, 1], dp[:, 2],
                     color=dl_colors[u - 1], s=10, alpha=0.45,
                     label=f"数据链融合(僚机-{u})", marker="x")

ax3d.set_xlabel("X轴相对坐标 (m)")
ax3d.set_ylabel("Y轴相对坐标 (m)")
ax3d.set_zlabel("Z轴相对坐标 (m)")
ax3d.set_title("三维空间相对运动轨迹与态势")
ax3d.legend(fontsize=9, loc="upper left")

# ───── 右侧子图 3 行 (规避机动与末端引导) ───────────────────────────────
ax_vel  = fig.add_subplot(3, 2, 2)
ax_acc  = fig.add_subplot(3, 2, 4)
ax_dist = fig.add_subplot(3, 2, 6)

# 1. 速度曲线
ax_vel.plot(time_stamps, truth_vel[:, 0], color="red",   label="X向速度", lw=1.2)
ax_vel.plot(time_stamps, truth_vel[:, 1], color="green", label="Y向速度", lw=1.2)
ax_vel.plot(time_stamps, truth_vel[:, 2], color="blue",  label="Z向速度", lw=1.2)
ax_vel.set_ylabel("速度 (m/s)")
ax_vel.set_title("目标三维速度矢量分析")
ax_vel.legend(fontsize=9); ax_vel.grid(True, linestyle="--", alpha=0.5)

# 2. 加速度曲线 (突出规避机动)
acc_diff = np.diff(truth_vel, axis=0) / DT
acc_diff = np.vstack([acc_diff, acc_diff[-1:]])  # 补齐长度
acc_norm = np.linalg.norm(acc_diff, axis=1)

ax_acc.plot(time_stamps, acc_norm, color="purple", label="总加速度幅值", lw=2.0)
ax_acc.plot(time_stamps, acc_diff[:, 0], color="red",   label="X向加速度", lw=1.0, alpha=0.5, linestyle=":")
ax_acc.plot(time_stamps, acc_diff[:, 1], color="green", label="Y向加速度", lw=1.0, alpha=0.5, linestyle=":")
ax_acc.plot(time_stamps, acc_diff[:, 2], color="blue",  label="Z向加速度", lw=1.0, alpha=0.5, linestyle=":")
ax_acc.set_ylabel("加速度 (m/s²)")
ax_acc.set_title("目标规避机动加速度分析")
ax_acc.legend(fontsize=9, loc="upper right"); ax_acc.grid(True, linestyle="--", alpha=0.5)

# 3. 协同平台与目标的相对距离 (突出末端引导)
for u in range(N_UAV):
    if uav_pos_arr[u].shape[0] > 0:
        pos_u = uav_pos_arr[u]
        min_len = min(len(pos_u), len(truth_pos))
        dist = np.linalg.norm(pos_u[:min_len] - truth_pos[:min_len], axis=1)
        ax_dist.plot(time_stamps[:min_len], dist, color=UAV_COLORS[u], label=UAV_LABELS[u], lw=1.5)

ax_dist.set_xlabel("时间 (s)")
ax_dist.set_ylabel("相对距离 (m)")
ax_dist.set_title("协同打击平台末端引导收敛距离分析")
ax_dist.legend(fontsize=9, loc="upper right"); ax_dist.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "analyze_output.png"), dpi=120)
plt.show()
print("[DONE] Visualization complete. Saved analyze_output.png")
