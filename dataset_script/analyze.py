import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =========================
# 参数
# =========================
RAY_LEN_EO = 100000.0  # EO 方向射线长度
RAY_LEN_ESM = 100000.0  # ESM 方向射线长度

# =========================
# 读取数据
# =========================
with open("track_data_003.json", "r") as f:
    dataset = json.load(f)

# =========================
# 数据容器
# =========================
truth_pts = []
platform_pts = []
radar_pts = []

eo_rays = []  # [(P, dir)]
esm_rays = []


# =========================
# 工具函数
# =========================
def azel_to_unit_vec(az, el):
    """根据方位角和俯仰角生成单位向量 (默认在当前参考系下)"""
    return np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el)
    ])


def get_body_to_world_matrix(vx, vy, vz):
    """
    计算从机载坐标系到世界坐标系的旋转矩阵 (R_b2w)
    机载坐标系定义：X轴指向速度方向，Z轴朝上，Y轴由右手法则决定 (指向侧方)
    """
    v = np.array([vx, vy, vz])
    speed = np.linalg.norm(v)

    # 如果平台静止，退化为世界坐标系
    if speed < 1e-6:
        return np.eye(3)

    unit_x = v / speed

    # 侧向单位向量 (Y)
    unit_y = np.cross([0, 0, 1], unit_x)
    if np.linalg.norm(unit_y) < 1e-6:  # 垂直飞行特例
        unit_y = np.array([0, 1, 0])
    else:
        unit_y /= np.linalg.norm(unit_y)

    # 上方单位向量 (Z)
    unit_z = np.cross(unit_x, unit_y)

    # 旋转矩阵的列就是机载坐标系的三个基向量在世界坐标系下的表示
    return np.column_stack((unit_x, unit_y, unit_z))


# =========================
# 遍历帧
# =========================
for frame in dataset:
    truth = frame["ground_truth_state"]
    platform = frame["platform_state"]
    sensors = frame["sensors"]

    # 目标真实坐标
    Tx, Ty, Tz = truth[0], truth[2], truth[4]

    # 平台真实坐标与速度
    Px, Py, Pz = platform[0], platform[2], platform[4]
    Pvx, Pvy, Pvz = platform[1], platform[3], platform[5]

    P_pos = np.array([Px, Py, Pz])

    truth_pts.append([Tx, Ty, Tz])
    platform_pts.append([Px, Py, Pz])

    # 核心：计算当前帧的 Body 到 World 的旋转矩阵
    R_b2w = get_body_to_world_matrix(Pvx, Pvy, Pvz)

    # ---------- Radar (Sensor 0) ----------
    if "sensor_0" in sensors and sensors["sensor_0"]["observations"]:
        for obs in sensors["sensor_0"]["observations"]:
            az, el, r = obs["meas"][0], obs["meas"][1], obs["meas"][2]

            # 1. 计算在机载坐标系下的相对笛卡尔向量
            vec_body = np.array([
                r * np.cos(el) * np.cos(az),
                r * np.cos(el) * np.sin(az),
                r * np.sin(el)
            ])

            # 2. 旋转到世界坐标系
            vec_world = R_b2w @ vec_body

            # 3. 叠加平台的世界坐标
            radar_pts.append(P_pos + vec_world)

    # ---------- EO (Sensor 1) ----------
    if "sensor_1" in sensors and sensors["sensor_1"]["observations"]:
        for obs in sensors["sensor_1"]["observations"]:
            az, el = obs["meas"][0], obs["meas"][1]

            # 1. 生成机载坐标系下的方向射线
            dir_body = azel_to_unit_vec(az, el)

            # 2. 旋转到世界坐标系
            dir_world = R_b2w @ dir_body
            eo_rays.append((P_pos, dir_world))

    # ---------- ESM (Sensor 2) ----------
    if "sensor_2" in sensors and sensors["sensor_2"]["observations"]:
        for obs in sensors["sensor_2"]["observations"]:
            az = obs["meas"][0]
            el = 0.0

            # 1. 生成机载坐标系下的方向射线
            dir_body = azel_to_unit_vec(az, el)

            # 2. 旋转到世界坐标系
            dir_world = R_b2w @ dir_body
            esm_rays.append((P_pos, dir_world))

# =========================
# 转数组
# =========================
truth_pts = np.array(truth_pts)
platform_pts = np.array(platform_pts)
radar_pts = np.array(radar_pts) if len(radar_pts) > 0 else np.empty((0, 3))

# =========================
# 绘图
# =========================
fig = plt.figure(figsize=(13, 10))
ax = fig.add_subplot(111, projection="3d")

# 真值轨迹
ax.plot(truth_pts[:, 0], truth_pts[:, 1], truth_pts[:, 2],
        color="black", linewidth=3, label="Target Truth")

# 平台轨迹
ax.plot(platform_pts[:, 0], platform_pts[:, 1], platform_pts[:, 2],
        linestyle="--", color="gray", linewidth=2, label="Platform")

# 雷达点
if len(radar_pts) > 0:
    ax.scatter(radar_pts[:, 0], radar_pts[:, 1], radar_pts[:, 2],
               s=8, c="blue", label="Radar Detections")

# EO 方向射线
for P, d in eo_rays[::10]:
    end = P + d * RAY_LEN_EO
    ax.plot([P[0], end[0]], [P[1], end[1]], [P[2], end[2]],
            color="orange", alpha=0.3, linewidth=1)

# ESM 方向射线
for P, d in esm_rays[::10]:
    end = P + d * RAY_LEN_ESM
    ax.plot([P[0], end[0]], [P[1], end[1]], [P[2], end[2]],
            color="purple", alpha=0.3, linewidth=1)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("StoneSoup Dataset Analysis (Body-Frame Corrected)")
ax.legend()
plt.show()