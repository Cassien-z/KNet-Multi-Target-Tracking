import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# 导入您提供的推理引擎
from inference import KNet_Engine


def generate_master_pov_animation(json_path, weights_path, output_filename="master_pov_tracking.gif"):
    print(f"🔄 初始化 KNet 推理引擎...")
    engine = KNet_Engine(weights_path)

    print(f"📂 加载数据: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 存储转换后的相对坐标
    target_rel_traj = []
    pred_rel_traj = []

    print("🚀 开始边推理边进行坐标转换 (World -> Master Body Frame)...")
    for frame in data:
        t = frame['time_s']
        # 1. 获取目标真值世界坐标
        gt_w = np.array(frame['ground_truth_state'])
        t_pos_w = gt_w[[0, 3, 6]] if len(gt_w) == 9 else gt_w[[0, 2, 4]]

        # 2. 获取长机世界坐标和速度
        platforms = frame.get('platforms', {})
        master_state = platforms.get('master_0', {}).get('platform_state')
        if master_state is None: continue

        m_pos_w = np.array([master_state[0], master_state[3], master_state[6]])
        m_vel_w = np.array([master_state[1], master_state[4], master_state[7]])

        # 3. KNet 在线推理获取预测轨迹 (世界坐标系)
        pred_abs_state = engine.predict_step(platforms, t)
        pred_pos_w = pred_abs_state[[0, 2, 4]]

        # 4. 构建旋转矩阵 R_w2b (参考您的 data_loader.py 逻辑)
        speed = np.linalg.norm(m_vel_w)
        if speed > 1e-6:
            unit_x = m_vel_w / speed
            unit_y = np.cross([0, 0, 1], unit_x)
            if np.linalg.norm(unit_y) < 1e-6: unit_y = np.array([0, 1, 0])
            unit_y /= np.linalg.norm(unit_y)
            unit_z = np.cross(unit_x, unit_y)
            R_w2b = np.vstack((unit_x, unit_y, unit_z))
        else:
            R_w2b = np.eye(3)

        # 5. 转换到主机坐标系
        target_rel_traj.append(R_w2b @ (t_pos_w - m_pos_w))
        pred_rel_traj.append(R_w2b @ (pred_pos_w - m_pos_w))

    target_rel_traj = np.array(target_rel_traj)
    pred_rel_traj = np.array(pred_rel_traj)

    # 🎨 动画渲染部分
    print("🎨 正在渲染 3D 动图...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 设置固定范围（以主机为中心）
    limit = 60000
    ax.set_xlim([0, limit])
    ax.set_ylim([-limit / 2, limit / 2])
    ax.set_zlim([-limit / 4, limit / 4])
    ax.set_xlabel('Forward X (m)')
    ax.set_ylabel('Right Y (m)')
    ax.set_zlabel('Down Z (m)')

    # 初始化图形对象
    gt_line, = ax.plot([], [], [], color='black', linewidth=2, label='Target Truth')
    pred_line, = ax.plot([], [], [], color='blue', linestyle='--', linewidth=1.5, label='KNet Predict')
    master_dot = ax.scatter([0], [0], [0], color='red', s=100, marker='^', label='Master (Origin)')

    # 绘制雷达探测方框 (FOV)
    fov_az, fov_el = np.radians(60), np.radians(3)
    p1 = [limit, -limit * np.tan(fov_az), -limit * np.tan(fov_el)]
    p2 = [limit, limit * np.tan(fov_az), -limit * np.tan(fov_el)]
    p3 = [limit, limit * np.tan(fov_az), limit * np.tan(fov_el)]
    p4 = [limit, -limit * np.tan(fov_az), limit * np.tan(fov_el)]

    # 静态 FOV 线条
    for p in [p1, p2, p3, p4]:
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color='lime', alpha=0.3, linestyle=':')

    # 远端探测平面（方框）
    fov_rect = Poly3DCollection([[p1, p2, p3, p4]], facecolors='lime', alpha=0.05, edgecolors='lime')
    ax.add_collection3d(fov_rect)

    ax.legend(loc='upper right')

    def update(n):
        gt_line.set_data(target_rel_traj[:n, 0], target_rel_traj[:n, 1])
        gt_line.set_3d_properties(target_rel_traj[:n, 2])

        pred_line.set_data(pred_rel_traj[:n, 0], pred_rel_traj[:n, 1])
        pred_line.set_3d_properties(pred_rel_traj[:n, 2])

        # 视角微调
        ax.view_init(elev=-15, azim=195)
        return gt_line, pred_line

    ani = animation.FuncAnimation(fig, update, frames=len(target_rel_traj), interval=50, blit=False)

    print(f"💾 正在保存为 {output_filename}...")
    ani.save(output_filename, writer='pillow', fps=20)
    print("✅ 保存成功！")


if __name__ == "__main__":
    generate_master_pov_animation("knet_verify_data/track_data_105.json", "knet_weights_finetuned.pth")