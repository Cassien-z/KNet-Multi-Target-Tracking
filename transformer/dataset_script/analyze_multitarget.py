"""
analyze_multitarget_ecef.py
可视化 script_multitarget_ecef.py 生成的 .mat 格式数据集（多目标）
自动将 ECEF 坐标重投影至局部 ENU (东北天) 坐标系以便于三维观察。
"""

import os
import glob
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# 字体配置（防止中文显示为方块）
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei']   # Windows 使用黑体
plt.rcParams['axes.unicode_minus'] = False       # 正常显示负号

# 引入我们自己定义的 ecef_utils 工具
from ecef_utils import (
    ecef_to_geodetic, ecef_to_enu_matrix, get_body_frame_axes
)

# ============================================================
# 参数
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 指向新的 ECEF 数据集目录
DATA_DIR  = os.path.join(SCRIPT_DIR, "train_data_multitarget_1ms")
FILE_IDX  = 0                   # 可视化第几条轨迹（0-based）
RAY_LEN   = 5000.0              # EO/ESM 方向射线显示长度 (m)

UAV_COLORS  = ["red", "limegreen", "dodgerblue", "darkorange", "purple"]
UAV_LABELS  = ["主机-0（全传感器）", "僚机-1（雷达）", "僚机-2（雷达+ESM）", "僚机-3（雷达+EO）", "僚机-4（雷达）"]

# ============================================================
# 加载文件
# ============================================================
mat_files = sorted(glob.glob(os.path.join(DATA_DIR, "track_data_highfi_0000.mat")))
if not mat_files:
    print(f"[错误] 在 {DATA_DIR}/ 中未找到 .mat 文件，请先运行 script_multitarget_ecef.py。")
    exit()

target_file = mat_files[FILE_IDX % len(mat_files)]
print(f"[信息] 正在加载: {target_file}")

mat = scipy.io.loadmat(target_file, simplify_cells=True)
Data_Input = mat["Data_Input"]    # shape (T,), dtype=object
Data_Label = mat["Data_Label"]    # shape (T, num_targets, 6)
DT         = float(mat.get("T", 0.025))
T          = len(Data_Input)
N_UAV      = 5

num_targets = Data_Label.shape[1] if Data_Label.ndim == 3 else 1
print(f"   T={T}, dt={DT}s, N_UAV={N_UAV}, num_targets={num_targets}")

# ============================================================
# 坐标系准备: 以 Master 第 0 帧的位置作为 ENU 坐标原点
# ============================================================
frame0 = Data_Input[0]
origin_ecef = np.array(frame0["UAV"][0]["Pose"]["PositionWorld"], dtype=float)
lat0, lon0, alt0 = ecef_to_geodetic(*origin_ecef)
R_ecef2enu = ecef_to_enu_matrix(lat0, lon0)  # ECEF 到 ENU 的旋转矩阵

# ============================================================
# 解析目标真值并转至 ENU
# ============================================================
time_stamps = np.arange(T) * DT

if Data_Label.ndim == 3:
    truth_pos_ecef = Data_Label[:, :, [0, 2, 4]].astype(np.float32)
    truth_vel_ecef = Data_Label[:, :, [1, 3, 5]].astype(np.float32)
else:
    truth_pos_ecef = Data_Label[:, [0, 2, 4]].astype(np.float32)[:, None, :]
    truth_vel_ecef = Data_Label[:, [1, 3, 5]].astype(np.float32)[:, None, :]

truth_pos_enu = np.zeros_like(truth_pos_ecef)
truth_vel_enu = np.zeros_like(truth_vel_ecef)

for t in range(T):
    for i in range(num_targets):
        truth_pos_enu[t, i] = R_ecef2enu @ (truth_pos_ecef[t, i] - origin_ecef)
        truth_vel_enu[t, i] = R_ecef2enu @ truth_vel_ecef[t, i]

# ============================================================
# 解析观测数据
# ============================================================
uav_pos_enu = [[] for _ in range(N_UAV)]
radar_pts   = [[] for _ in range(N_UAV)]
eo_rays     = [[] for _ in range(N_UAV)]
esm_rays    = [[] for _ in range(N_UAV)]
dl_pts      = [[] for _ in range(N_UAV)]

radar_mask_ts = [[] for _ in range(N_UAV)]
eo_mask_ts    = [[] for _ in range(N_UAV)]
esm_mask_ts   = [[] for _ in range(N_UAV)]

def _to_arr_2d(v, dim):
    try:
        a = np.asarray(v, dtype=float)
        if a.ndim == 1 and dim == 1: return a
        elif a.ndim == 2 and a.shape[1] >= dim: return a[:, :dim]
    except Exception: pass
    return None

def _to_arr_1d(v):
    try: return np.asarray(v, dtype=float).flatten()
    except Exception: pass
    return None

last_uav_pos_ecef = [None] * N_UAV

for t in range(T):
    frame = Data_Input[t]
    if not isinstance(frame, dict): continue
    uavs = frame.get("UAV", None)
    if not isinstance(uavs, (list, np.ndarray)): continue

    for u in range(min(N_UAV, len(uavs))):
        ud = uavs[u]
        if not isinstance(ud, dict): continue

        # --- 平台自身位置 (ECEF) ---
        pos_ecef = _to_arr_1d(ud.get("Pose", {}).get("PositionWorld", None))
        if pos_ecef is None: pos_ecef = np.zeros(3)

        # 通过数值差分计算 ECEF 速度，从而获取正确的机体轴
        if last_uav_pos_ecef[u] is None:
            if t + 1 < T:
                pos_next = np.array(Data_Input[t+1]["UAV"][u]["Pose"]["PositionWorld"], dtype=float)
                vel_ecef = (pos_next - pos_ecef) / DT
            else:
                vel_ecef = np.array([1.0, 0.0, 0.0])
        else:
            vel_ecef = (pos_ecef - last_uav_pos_ecef[u]) / DT
        last_uav_pos_ecef[u] = pos_ecef

        # 保存 ENU 位置用于画图
        pos_enu = R_ecef2enu @ (pos_ecef - origin_ecef)
        uav_pos_enu[u].append(pos_enu)

        # 获取机体轴 (ECEF 中)
        fwd, right, up = get_body_frame_axes(vel_ecef, pos_ecef)
        R_b2e = np.column_stack((fwd, right, up))

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
                    pt_ecef = pos_ecef + R_b2e @ vec_body
                    radar_pts[u].append(R_ecef2enu @ (pt_ecef - origin_ecef))
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
                    dir_body = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
                    dir_ecef = R_b2e @ dir_body
                    dir_enu = R_ecef2enu @ dir_ecef
                    eo_rays[u].append((pos_enu.copy(), dir_enu))
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
                    dir_body = np.array([np.cos(az), np.sin(az), 0.0])
                    dir_ecef = R_b2e @ dir_body
                    dir_enu = R_ecef2enu @ dir_ecef
                    esm_rays[u].append((pos_enu.copy(), dir_enu))
        else:
            esm_mask_ts[u].append(0)

        # --- FastLink (ECEF) ---
        if u > 0:
            fl_meas = _to_arr_2d(ud.get("FastLink", {}).get("Meas", None), 3)
            fl_upd  = _to_arr_1d(ud.get("FastLink", {}).get("IsUpdate", None))
            if fl_meas is not None and fl_upd is not None:
                for i in range(len(fl_upd)):
                    if fl_upd[i] > 0.5 and np.linalg.norm(fl_meas[i]) > 1.0:
                        dl_enu = R_ecef2enu @ (fl_meas[i] - origin_ecef)
                        dl_pts[u].append(dl_enu)

# 数据转换
uav_pos_arr = [np.array(x) if x else np.empty((0, 3)) for x in uav_pos_enu]
radar_arr   = [np.array(x) if x else np.empty((0, 3)) for x in radar_pts]
dl_arr      = [np.array(x) if x else np.empty((0, 3)) for x in dl_pts]

radar_mask_arr = [np.array(x) for x in radar_mask_ts]
eo_mask_arr    = [np.array(x) for x in eo_mask_ts]
esm_mask_arr   = [np.array(x) for x in esm_mask_ts]

# ============================================================
# 图1：三维轨迹图（ENU 坐标系）
# ============================================================
fig1 = plt.figure(figsize=(12, 9))
fig1.suptitle(f"三维轨迹图（ENU 局部坐标系）— {os.path.basename(target_file)}", fontsize=14, fontweight="bold")
ax3d = fig1.add_subplot(111, projection="3d")

# 绘制多目标真实轨迹 (ENU)
for i in range(num_targets):
    ax3d.plot(truth_pos_enu[:, i, 0], truth_pos_enu[:, i, 1], truth_pos_enu[:, i, 2],
              color="black", linewidth=1.5, zorder=5, alpha=0.7)
ax3d.plot([], [], [], color="black", linewidth=1.5, label=f"目标真实轨迹（共 {num_targets} 个）")

# 绘制无人机平台轨迹 (ENU)
for u in range(N_UAV):
    if uav_pos_arr[u].shape[0] > 0:
        ax3d.plot(uav_pos_arr[u][:, 0], uav_pos_arr[u][:, 1], uav_pos_arr[u][:, 2],
                  linestyle="--", color=UAV_COLORS[u], linewidth=1.5,
                  label=UAV_LABELS[u], alpha=0.8)
        ax3d.scatter(*uav_pos_arr[u][0], color=UAV_COLORS[u], s=60, marker="^", zorder=6)

# 主机雷达点云
if radar_arr[0].shape[0] > 0:
    ax3d.scatter(radar_arr[0][:, 0], radar_arr[0][:, 1], radar_arr[0][:, 2],
                 color="magenta", s=4, alpha=0.3, label="主机雷达量测", marker="o")

# 主机 EO 射线 (抽样显示防遮挡)
if eo_rays[0]:
    idxs = np.random.choice(len(eo_rays[0]), min(40, len(eo_rays[0])), replace=False)
    for i in idxs:
        o, d = eo_rays[0][i]
        end  = o + d * RAY_LEN
        ax3d.plot([o[0], end[0]], [o[1], end[1]], [o[2], end[2]], color="cyan", alpha=0.2, linewidth=0.5)
    ax3d.plot([], [], color="cyan", linewidth=1.0, label="主机 EO 射线（抽样）")

# 僚机数据链点云 (ENU)
for u in range(1, 5):
    if dl_arr[u].shape[0] > 0:
        ax3d.scatter(dl_arr[u][:, 0], dl_arr[u][:, 1], dl_arr[u][:, 2],
                     color=UAV_COLORS[u], s=6, alpha=0.3,
                     label=f"数据链量测（僚机 {u}）", marker="x")

ax3d.set_xlabel("东向 (m)")
ax3d.set_ylabel("北向 (m)")
ax3d.set_zlabel("天向 (m)")
ax3d.set_title(f"ENU 局部坐标系三维轨迹（{num_targets} 个目标）")
ax3d.legend(fontsize=7, loc="upper left")
plt.tight_layout()
plt.show()

# ============================================================
# 图2：目标速度曲线（ENU 分量）
# ============================================================
fig2, ax_vel = plt.subplots(figsize=(10, 5))
fig2.suptitle("目标速度时序曲线（ENU 坐标系）", fontsize=14, fontweight="bold")

for i in range(num_targets):
    ax_vel.plot(time_stamps, truth_vel_enu[:, i, 0], color="red",   alpha=0.2, linewidth=0.8)
    ax_vel.plot(time_stamps, truth_vel_enu[:, i, 1], color="green", alpha=0.2, linewidth=0.8)
    ax_vel.plot(time_stamps, truth_vel_enu[:, i, 2], color="blue",  alpha=0.2, linewidth=0.8)

ax_vel.plot([], [], color="red",   label="东向速度 $V_E$")
ax_vel.plot([], [], color="green", label="北向速度 $V_N$")
ax_vel.plot([], [], color="blue",  label="天向速度 $V_U$")
ax_vel.set_xlabel("时间 (s)")
ax_vel.set_ylabel("速度 (m/s)")
ax_vel.set_title("目标速度分量（ENU 坐标系）")
ax_vel.legend(fontsize=8)
ax_vel.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ============================================================
# 图3：主机传感器探测目标数量
# ============================================================
fig3, ax_mask = plt.subplots(figsize=(10, 5))
fig3.suptitle("主机传感器探测目标数量统计", fontsize=14, fontweight="bold")

if len(radar_mask_arr[0]) > 0:
    ts_mask = time_stamps[:len(radar_mask_arr[0])]
    ax_mask.plot(ts_mask, radar_mask_arr[0], color="magenta", label="雷达探测目标数", linewidth=1.2, alpha=0.8)
    ax_mask.plot(ts_mask, eo_mask_arr[0],    color="cyan",    label="EO 探测目标数",  linewidth=1.2, alpha=0.8)
    ax_mask.plot(ts_mask, esm_mask_arr[0],   color="orange",  label="ESM 探测目标数", linewidth=1.2, alpha=0.8)

ax_mask.set_xlabel("时间 (s)")
ax_mask.set_ylabel("探测目标数量")
ax_mask.set_title(f"主机传感器探测统计（目标总数: {num_targets}）")
ax_mask.set_ylim(-0.5, num_targets + 0.5)
ax_mask.legend(fontsize=8)
ax_mask.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ============================================================
# 图4：僚机雷达累计探测量
# ============================================================
fig4, ax_dl = plt.subplots(figsize=(10, 5))
fig4.suptitle("僚机雷达累计探测目标量", fontsize=14, fontweight="bold")

for u in range(1, 5):
    if len(radar_mask_arr[u]) > 0:
        ts_u  = time_stamps[:len(radar_mask_arr[u])]
        valid = np.array(radar_mask_arr[u])
        cum   = np.cumsum(valid)
        ax_dl.plot(ts_u, cum, color=UAV_COLORS[u], label=f"僚机 {u} 雷达（累计）")

ax_dl.set_xlabel("时间 (s)")
ax_dl.set_ylabel("累计探测目标量测数")
ax_dl.set_title("僚机雷达覆盖情况（累计目标量测点数）")
ax_dl.legend(fontsize=8)
ax_dl.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

print("[完成] 多目标数据集可视化完毕")