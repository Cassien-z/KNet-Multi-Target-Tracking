import os
import sys
import glob
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_FILE = PROJECT_ROOT / 'data' / 'val_data_v5' / 'track_cssw_0000.mat'
SAVE_PATH = PROJECT_ROOT / 'artifacts' / 'figures' / 'v5_dataset_visualization.png'
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

if not os.path.exists(TARGET_FILE):
    print(f"Error: {TARGET_FILE} not found.")
    sys.exit(1)

mat = scipy.io.loadmat(TARGET_FILE, simplify_cells=True)
Data_Input = mat["Data_Input"]
Data_Label = mat["Data_Label"]
DT = float(mat.get("T", 0.1))
T = len(Data_Input)
N_UAV = 5

num_targets = Data_Label.shape[1] if Data_Label.ndim == 3 else 1

time_stamps = np.arange(T) * DT

if Data_Label.ndim == 3:
    truth_pos = Data_Label[:, :, [0, 2, 4]].astype(np.float32)
    truth_vel = Data_Label[:, :, [1, 3, 5]].astype(np.float32)
else:
    truth_pos = Data_Label[:, [0, 2, 4]].astype(np.float32)[:, None, :]
    truth_vel = Data_Label[:, [1, 3, 5]].astype(np.float32)[:, None, :]

uav_pos = [[] for _ in range(N_UAV)]
radar_pts = [[] for _ in range(N_UAV)]
dl_pts = [[] for _ in range(N_UAV)]
radar_mask_ts = [[] for _ in range(N_UAV)]
eo_mask_ts = [[] for _ in range(N_UAV)]
esm_mask_ts = [[] for _ in range(N_UAV)]

def _to_arr_2d(v, dim):
    try:
        a = np.asarray(v, dtype=float)
        if a.ndim == 1 and dim == 1: return a
        elif a.ndim == 2 and a.shape[1] >= dim: return a[:, :dim]
    except: pass
    return None

def _to_arr_1d(v):
    try: return np.asarray(v, dtype=float).flatten()
    except: pass
    return None

def euler_to_rot(roll, pitch, yaw):
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def get_body_frame_axes(vel_ecef, pos_ecef):
    world_up = np.array([0.0, 0.0, 1.0])
    speed = np.linalg.norm(vel_ecef)
    unit_fwd = np.array([1.0, 0.0, 0.0]) if speed < 1e-6 else vel_ecef / speed
    unit_right = np.cross(world_up, unit_fwd)
    norm_r = np.linalg.norm(unit_right)
    unit_right = np.array([0.0, 1.0, 0.0]) if norm_r < 1e-6 else unit_right / norm_r
    unit_up = np.cross(unit_fwd, unit_right)
    unit_up /= np.linalg.norm(unit_up)
    return unit_fwd, unit_right, unit_up

for t in range(T):
    frame = Data_Input[t]
    if not isinstance(frame, dict): continue
    uavs = frame.get("UAV", None)
    if not isinstance(uavs, (list, np.ndarray)): continue

    for u in range(min(N_UAV, len(uavs))):
        ud = uavs[u]
        if not isinstance(ud, dict): continue
        pos = _to_arr_1d(ud.get("Pose", {}).get("PositionWorld", None))
        vel = _to_arr_1d(ud.get("Pose", {}).get("VelocityWorld", None))
        if pos is None: pos = np.zeros(3)
        if vel is None: vel = np.zeros(3)
        uav_pos[u].append(pos)
        
        unit_fwd, unit_right, unit_up = get_body_frame_axes(vel, pos)

        r_meas = _to_arr_2d(ud.get("Radar", {}).get("Meas", None), 3)
        r_mask = _to_arr_1d(ud.get("Radar", {}).get("Mask", None))
        if r_meas is not None and r_mask is not None:
            radar_mask_ts[u].append(np.sum(r_mask > 0.5))
            for i in range(len(r_mask)):
                if r_mask[i] > 0.5:
                    az, el, r = r_meas[i]
                    dx_b = r * np.cos(el) * np.cos(az)
                    dy_b = r * np.cos(el) * np.sin(az)
                    dz_b = r * np.sin(el)
                    radar_pts[u].append(pos + dx_b * unit_fwd + dy_b * unit_right + dz_b * unit_up)
        else: radar_mask_ts[u].append(0)

        eo_mask = _to_arr_1d(ud.get("EO", {}).get("Mask", None))
        if eo_mask is not None: eo_mask_ts[u].append(np.sum(eo_mask > 0.5))
        else: eo_mask_ts[u].append(0)

        esm_mask = _to_arr_1d(ud.get("ESM", {}).get("Mask", None))
        if esm_mask is not None: esm_mask_ts[u].append(np.sum(esm_mask > 0.5))
        else: esm_mask_ts[u].append(0)

        if u > 0:
            fl = ud.get("FastLink", {})
            fl_meas = _to_arr_2d(fl.get("Meas", None), 3)
            fl_upd  = _to_arr_1d(fl.get("IsUpdate", None))
            if fl_meas is not None and fl_upd is not None:
                for i in range(len(fl_upd)):
                    if fl_upd[i] > 0.5 and np.linalg.norm(fl_meas[i]) > 1.0:
                        dl_pts[u].append(fl_meas[i].copy())

uav_pos_arr = [np.array(x) if x else np.empty((0, 3)) for x in uav_pos]
radar_arr   = [np.array(x) if x else np.empty((0, 3)) for x in radar_pts]
dl_arr      = [np.array(x) if x else np.empty((0, 3)) for x in dl_pts]
radar_mask_arr = [np.array(x) for x in radar_mask_ts]
eo_mask_arr    = [np.array(x) for x in eo_mask_ts]
esm_mask_arr   = [np.array(x) for x in esm_mask_ts]

fig = plt.figure(figsize=(18, 10))
fig.suptitle("V5 Dataset Validation Scenario Visualization (CS-SW Mode)", fontsize=16, fontweight="bold", y=0.95)

ax3d = fig.add_subplot(1, 2, 1, projection="3d")
for i in range(num_targets):
    ax3d.plot(truth_pos[:, i, 0], truth_pos[:, i, 1], truth_pos[:, i, 2], color="black", linewidth=1.5, zorder=5)
    ax3d.scatter(*truth_pos[0, i], color="green", s=30, marker="o", zorder=6)   # Start
    ax3d.scatter(*truth_pos[-1, i], color="red", s=50, marker="*", zorder=6)    # End
ax3d.plot([], [], [], color="black", linewidth=1.5, label=f"Target Truths ({num_targets})")
ax3d.scatter([], [], [], color="green", marker="o", label="Trajectory Start")
ax3d.scatter([], [], [], color="red", marker="*", label="Trajectory End")

UAV_COLORS  = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
UAV_LABELS  = ["Master UAV (All Sensors + Datalink Rx)", "Wingman 1 (Radar Only)", "Wingman 2 (Radar+ESM)", "Wingman 3 (Radar+EO)", "Wingman 4 (Radar Only)"]

for u in range(N_UAV):
    if uav_pos_arr[u].shape[0] > 0:
        ax3d.plot(uav_pos_arr[u][:, 0], uav_pos_arr[u][:, 1], uav_pos_arr[u][:, 2], linestyle="--", color=UAV_COLORS[u], linewidth=1.5, label=UAV_LABELS[u], alpha=0.8)
        # Mark Start (o) and End (X)
        ax3d.scatter(*uav_pos_arr[u][0], color=UAV_COLORS[u], s=30, marker="o", zorder=6)
        ax3d.scatter(*uav_pos_arr[u][-1], color=UAV_COLORS[u], s=40, marker="X", zorder=6)

if radar_arr[0].shape[0] > 0:
    ax3d.scatter(radar_arr[0][:, 0], radar_arr[0][:, 1], radar_arr[0][:, 2], color="magenta", s=6, alpha=0.4, label="Radar Meas (Master)", marker="o")

for u in range(1, 5):
    if dl_arr[u].shape[0] > 0:
        ax3d.scatter(dl_arr[u][:, 0], dl_arr[u][:, 1], dl_arr[u][:, 2], color=UAV_COLORS[u], s=8, alpha=0.3, label=f"Datalink to Master (Wingman {u})", marker="x")

ax3d.set_xlabel("X (m)")
ax3d.set_ylabel("Y (m)")
ax3d.set_zlabel("Z (m)")
ax3d.set_title(f"3D Formation and Target Trajectories", fontsize=14, pad=10)
ax3d.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.0))

ax_vel  = fig.add_subplot(3, 2, 2)
ax_mask = fig.add_subplot(3, 2, 4)
ax_dl   = fig.add_subplot(3, 2, 6)

for i in range(num_targets):
    ax_vel.plot(time_stamps, truth_vel[:, i, 0], color="red",   alpha=0.3, linewidth=1.2)
    ax_vel.plot(time_stamps, truth_vel[:, i, 1], color="green", alpha=0.3, linewidth=1.2)
    ax_vel.plot(time_stamps, truth_vel[:, i, 2], color="blue",  alpha=0.3, linewidth=1.2)

ax_vel.plot([], [], color="red", label="Vx")
ax_vel.plot([], [], color="green", label="Vy")
ax_vel.plot([], [], color="blue", label="Vz")
ax_vel.set_ylabel("Velocity (m/s)")
ax_vel.set_title("V5 CS-SW Target Velocities (3 Targets)", fontsize=12)
ax_vel.legend(fontsize=9, loc='upper right')
ax_vel.grid(True, linestyle="--", alpha=0.5)

if len(radar_mask_arr[0]) > 0:
    ts_mask = time_stamps[:len(radar_mask_arr[0])]
    ax_mask.plot(ts_mask, radar_mask_arr[0], color="magenta", label="Radar Detected (10Hz, Lossy)", linewidth=1.5, alpha=0.8)
    ax_mask.plot(ts_mask, eo_mask_arr[0],    color="cyan",    label="EO Detected (10Hz)",    linewidth=1.5, alpha=0.8)
    ax_mask.plot(ts_mask, esm_mask_arr[0],   color="orange",  label="ESM Detected (5Hz)",   linewidth=1.5, alpha=0.8)

ax_mask.set_ylabel("Targets Detected")
ax_mask.set_title(f"Master Sensor Detection Count (15% Packet Loss injected)", fontsize=12)
ax_mask.set_ylim(-0.5, num_targets + 0.5)
ax_mask.legend(fontsize=9, loc='lower right')
ax_mask.grid(True, linestyle="--", alpha=0.5)

for u in range(1, 5):
    if len(radar_mask_arr[u]) > 0:
        ts_u  = time_stamps[:len(radar_mask_arr[u])]
        valid = np.array(radar_mask_arr[u])
        cum   = np.cumsum(valid)
        ax_dl.plot(ts_u, cum, color=UAV_COLORS[u], label=f"Wingman {u} Datalink (cumsum)", linewidth=2, alpha=0.8)

ax_dl.set_xlabel("Time (s)")
ax_dl.set_ylabel("Cumulative Tx Measurements")
ax_dl.set_title("Wingman Radar Coverage Sent via 0.1s Delayed Datalink", fontsize=12)
ax_dl.legend(fontsize=9, loc='upper left')
ax_dl.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(SAVE_PATH, dpi=300)
print(f"Plot saved successfully to {SAVE_PATH}")
