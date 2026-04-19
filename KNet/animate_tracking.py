import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# 导入你提供的推理引擎和坐标转换函数
from inference import KNet_Engine, convert_body_to_world_meas


def generate_3d_animation(json_path, weights_path, output_filename="tracking_3d.gif"):
    print(f"🔄 初始化 KNet 推理引擎...")
    engine = KNet_Engine(weights_path)

    print(f"📂 加载测试数据: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 准备存储航迹的列表
    gt_trajectory = []
    pred_trajectory = []
    meas_points = []  # 存储长机的雷达量测(散点)，用于展示噪声

    print("🚀 开始边推理边提取数据...")
    for frame in data:
        t = frame['time_s']
        gt_w = np.array(frame['ground_truth_state'])
        platforms = frame.get('platforms', {})
        master_state = platforms.get('master_0', {}).get('platform_state')

        if master_state is None:
            continue

        # 1. 提取真实轨迹 Ground Truth (兼容 6D 和 9D)
        if len(gt_w) == 9:
            gt_abs_pos = gt_w[[0, 3, 6]]
        else:
            gt_abs_pos = gt_w[[0, 2, 4]]
        gt_trajectory.append(gt_abs_pos)

        # 2. 调用你的 KNet 进行网络预测
        pred_abs_state = engine.predict_step(platforms, t)
        pred_pos = pred_abs_state[[0, 2, 4]]  # KNet 输出的是 [X, Vx, Y, Vy, Z, Vz]
        pred_trajectory.append(pred_pos)

        # 3. 提取长机主雷达的原始量测数据 (只取第一条用于可视化噪声)
        try:
            radar_obs = platforms['master_0']['sensors']['sensor_0']['observations'][0]['meas']
            az, el, r = radar_obs
            vx, vy, vz = master_state[1], master_state[4], master_state[7]  # 长机速度
            # 还原到世界坐标系
            m_az, m_el, m_r = convert_body_to_world_meas(az, el, r, vx, vy, vz)
            # 极坐标转直角坐标
            mx = master_state[0] + m_r * np.cos(m_el) * np.cos(m_az)
            my = master_state[3] + m_r * np.cos(m_el) * np.sin(m_az)
            mz = master_state[6] + m_r * np.sin(m_el)
            meas_points.append([mx, my, mz])
        except Exception:
            meas_points.append([np.nan, np.nan, np.nan])  # 如果这帧没数据就算了

    gt_trajectory = np.array(gt_trajectory)
    pred_trajectory = np.array(pred_trajectory)
    meas_points = np.array(meas_points)

    print("🎨 开始渲染 3D 动画...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 计算视角边界
    all_points = np.vstack((gt_trajectory, pred_trajectory))
    ax.set_xlim([np.min(all_points[:, 0]) - 2000, np.max(all_points[:, 0]) + 2000])
    ax.set_ylim([np.min(all_points[:, 1]) - 2000, np.max(all_points[:, 1]) + 2000])
    ax.set_zlim([np.min(all_points[:, 2]) - 2000, np.max(all_points[:, 2]) + 2000])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('KNet Multi-Platform Cooperative Tracking (3D)')

    # 初始化线和散点
    gt_line, = ax.plot([], [], [], color='black', linewidth=3, label='Ground Truth (Target)')
    pred_line, = ax.plot([], [], [], color='blue', linewidth=2, linestyle='--', label='KNet Tracker')
    meas_scatter, = ax.plot([], [], [], 'o', color='red', markersize=2, alpha=0.3, label='Radar Meas (Noisy)')

    ax.legend(loc='upper right')

    def init():
        gt_line.set_data([], [])
        gt_line.set_3d_properties([])
        pred_line.set_data([], [])
        pred_line.set_3d_properties([])
        meas_scatter.set_data([], [])
        meas_scatter.set_3d_properties([])
        return gt_line, pred_line, meas_scatter

    def update(frame_idx):
        # 每次更新画出前 frame_idx 个点
        current_gt = gt_trajectory[:frame_idx]
        current_pred = pred_trajectory[:frame_idx]
        current_meas = meas_points[:frame_idx]

        gt_line.set_data(current_gt[:, 0], current_gt[:, 1])
        gt_line.set_3d_properties(current_gt[:, 2])

        pred_line.set_data(current_pred[:, 0], current_pred[:, 1])
        pred_line.set_3d_properties(current_pred[:, 2])

        # 过滤掉 NaN 的量测数据
        valid_meas = current_meas[~np.isnan(current_meas[:, 0])]
        if len(valid_meas) > 0:
            # 散点图用 plot 画点比 scatter 在动画里更高效
            meas_scatter.set_data(valid_meas[:, 0], valid_meas[:, 1])
            meas_scatter.set_3d_properties(valid_meas[:, 2])

        # 旋转视角，增加 3D 动感
        ax.view_init(elev=20., azim=frame_idx * 0.5)

        return gt_line, pred_line, meas_scatter

    # 创建动画，每帧间隔 50 毫秒
    ani = animation.FuncAnimation(fig, update, frames=len(gt_trajectory),
                                  init_func=init, blit=False, interval=50)

    print(f"💾 正在保存动画到 {output_filename} (这可能需要一两分钟)...")
    ani.save(output_filename, writer='pillow', fps=20)
    print("✅ 动画保存完成！")


if __name__ == "__main__":
    # 请修改这里的 JSON 路径为你实际的测试数据路径
    JSON_DATA_PATH = "knet_verify_data/track_data_109.json"
    WEIGHTS_PATH = "knet_weights_finetuned.pth"

    generate_3d_animation(JSON_DATA_PATH, WEIGHTS_PATH, "knet_tracking_result.gif")