import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =========================
# 参数
# =========================
RAY_LEN_EO = 100000.0  # EO 方向射线长度
RAY_LEN_ESM = 100000.0  # ESM 方向射线长度
# 🟢 注意：这里文件名改为了刚生成的异步协同数据集
DATA_FILE = "dataset_final_stonesoup_v1_4_asynchronous.json"

# =========================
# 读取数据
# =========================
try:
    with open(DATA_FILE, "r") as f:
        dataset = json.load(f)
    print(f"成功加载 {DATA_FILE}，共 {len(dataset)} 帧")
except FileNotFoundError:
    print(f"❌ 找不到文件 {DATA_FILE}，请先运行 batch_generate.py 或 script.py")
    exit()

# =========================
# 数据容器
# =========================
truth_pts = []
# 🟢 适应多平台：使用字典按 ID 存储轨迹
platform_pts_dict = {
    "master_0": [],
    "wingman_1": [],
    "wingman_2": []
}

radar_pts = []
eo_rays = []  # [(P_pos, dir_world)]
esm_rays = []


# =========================
# 工具函数
# =========================
def azel_to_unit_vec(az, el):
    """根据方位角和俯仰角生成单位向量"""
    return np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el)
    ])


def get_body_to_world_matrix(vx, vy, vz):
    """计算从机载坐标系到世界坐标系的旋转矩阵 (R_b2w)"""
    v = np.array([vx, vy, vz])
    speed = np.linalg.norm(v)
    if speed < 1e-6:
        return np.eye(3)

    unit_x = v / speed
    unit_y = np.cross([0, 0, 1], unit_x)
    if np.linalg.norm(unit_y) < 1e-6:
        unit_y = np.array([0, 1, 0])
    else:
        unit_y /= np.linalg.norm(unit_y)
    unit_z = np.cross(unit_x, unit_y)

    return np.column_stack((unit_x, unit_y, unit_z))


# =========================
# 遍历帧
# =========================
for frame in dataset:
    # 提取目标真值
    truth = frame["ground_truth_state"]
    Tx, Ty, Tz = truth[0], truth[2], truth[4]
    truth_pts.append([Tx, Ty, Tz])

    platforms_data = frame.get("platforms", {})

    # 🟢 遍历该帧下的所有平台
    for p_id, p_data in platforms_data.items():
        platform_state = p_data.get("platform_state")

        # ⚠️ 安全检查：慢数据链在开局可能由于延迟导致状态为空
        if platform_state is None:
            continue

        Px, Py, Pz = platform_state[0], platform_state[2], platform_state[4]
        Pvx, Pvy, Pvz = platform_state[1], platform_state[3], platform_state[5]
        P_pos = np.array([Px, Py, Pz])

        # 记录该平台轨迹
        if p_id in platform_pts_dict:
            platform_pts_dict[p_id].append([Px, Py, Pz])

        # 获取该平台的 R_b2w 旋转矩阵
        R_b2w = get_body_to_world_matrix(Pvx, Pvy, Pvz)

        sensors = p_data.get("sensors", {})

        # ---------- Radar (Sensor 0) ----------
        if "sensor_0" in sensors and sensors["sensor_0"]["observations"]:
            for obs in sensors["sensor_0"]["observations"]:
                az, el, r = obs["meas"][0], obs["meas"][1], obs["meas"][2]
                vec_body = np.array([
                    r * np.cos(el) * np.cos(az),
                    r * np.cos(el) * np.sin(az),
                    r * np.sin(el)
                ])
                vec_world = R_b2w @ vec_body
                radar_pts.append(P_pos + vec_world)

        # ---------- EO (Sensor 1) ----------
        if "sensor_1" in sensors and sensors["sensor_1"]["observations"]:
            for obs in sensors["sensor_1"]["observations"]:
                az, el = obs["meas"][0], obs["meas"][1]
                dir_body = azel_to_unit_vec(az, el)
                dir_world = R_b2w @ dir_body
                eo_rays.append((P_pos, dir_world))

        # ---------- ESM (Sensor 2) ----------
        if "sensor_2" in sensors and sensors["sensor_2"]["observations"]:
            for obs in sensors["sensor_2"]["observations"]:
                az = obs["meas"][0]
                el = 0.0
                dir_body = azel_to_unit_vec(az, el)
                dir_world = R_b2w @ dir_body
                esm_rays.append((P_pos, dir_world))

# =========================
# 转数组
# =========================
truth_pts = np.array(truth_pts)
radar_pts = np.array(radar_pts) if len(radar_pts) > 0 else np.empty((0, 3))

# =========================
# 绘图
# =========================
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection="3d")

# 1. 真值轨迹
ax.plot(truth_pts[:, 0], truth_pts[:, 1], truth_pts[:, 2],
        color="black", linewidth=3, label="Target Truth")

# 2. 绘制各平台轨迹 (使用不同颜色区分)
colors = {"master_0": "red", "wingman_1": "green", "wingman_2": "blue"}
# 🌟 修改图例标签
labels = {"master_0": "Master (No Delay)", "wingman_1": "Wingman 1 (Fast Link)", "wingman_2": "Wingman 2 (Fast Link)"}

for p_id, pts in platform_pts_dict.items():
    if len(pts) > 0:
        pts_arr = np.array(pts)
        ax.plot(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2],
                linestyle="--", color=colors.get(p_id, "gray"), linewidth=2, label=labels.get(p_id, p_id))

# 3. 雷达点集
if len(radar_pts) > 0:
    ax.scatter(radar_pts[:, 0], radar_pts[:, 1], radar_pts[:, 2],
               s=5, c="blue", alpha=0.5, label="Radar Detections (Multi-platform)")

# 4. 射线投影 (为了性能和画面清晰度，每 20 根画一根)
stride = 20

for P, d in eo_rays[::stride]:
    end = P + d * RAY_LEN_EO
    ax.plot([P[0], end[0]], [P[1], end[1]], [P[2], end[2]],
            color="orange", alpha=0.2, linewidth=1)

for P, d in esm_rays[::stride]:
    end = P + d * RAY_LEN_ESM
    ax.plot([P[0], end[0]], [P[1], end[1]], [P[2], end[2]],
            color="purple", alpha=0.2, linewidth=1)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Multi-Platform Cooperative Tracking Simulation\n(Master + 2 Wingmen with Fast Datalink)")
ax.legend()
plt.show()