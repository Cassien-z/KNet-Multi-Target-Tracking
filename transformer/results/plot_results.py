import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 0. 字体配置 (防止图表中的中文显示为方块)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 请使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def evaluate_dynamic_convergence(true_pos, pred_pos, P_history=None, threshold=150.0, window_size=10, dt=0.1,
                                 warmup_frames=15):
    """
    截取冷启动阶段并计算 RMSE。

    参数:
        warmup_frames (int): > 0 时直接将前 warmup_frames 帧视为冷启动阶段（固定截断），
                             = 0 时退回动态阈值检测模式。
    """
    # 计算每一帧的真实位置误差 (欧氏距离)
    errors = np.linalg.norm(true_pos - pred_pos, axis=1)

    # ---- 固定帧数模式 ----
    if warmup_frames > 0:
        conv_idx = min(warmup_frames, len(true_pos) - 1)
        print(f"⏱️  固定冷启动模式：前 {conv_idx} 帧视为冷启动阶段（约 {conv_idx * dt:.1f} 秒）")
        metric = errors  # 仍返回误差曲线供绘图使用

    # ---- 动态阈值检测模式 ----
    else:
        conv_idx = 0
        is_converged = False

        if P_history is not None:
            print("🔍 检测到协方差矩阵，使用【协方差迹】进行动态收敛判断...")
            if P_history.ndim == 3:
                metric = np.trace(P_history[:, 0:3, 0:3], axis1=1, axis2=2)
            else:
                metric = np.sum(P_history[:, 0:3], axis=1)
        else:
            print("🔍 未提供协方差矩阵，使用【位置误差】进行动态收敛判断...")
            metric = errors

        # 滑动窗口寻找连续低于阈值的点
        converged_mask = metric < threshold
        for i in range(len(converged_mask) - window_size + 1):
            if np.all(converged_mask[i: i + window_size]):
                conv_idx = i
                is_converged = True
                break

        if not is_converged:
            print(f"⚠️ 警告: 评价指标未连续 {window_size} 帧降至阈值 {threshold} 以下！将默认截断前 20 帧。")
            conv_idx = min(20, len(true_pos) // 2)
        else:
            print(f"✅ 动态截取成功：模型在第 {conv_idx} 帧 (约 {conv_idx * dt:.1f} 秒) 达到稳态。")

    # 计算 RMSE
    global_rmse = np.sqrt(np.mean(errors ** 2))
    steady_rmse = np.sqrt(np.mean(errors[conv_idx:] ** 2))

    print("-" * 40)
    print(f"📊 全局 RMSE (含冷启动, 共 {len(errors)} 帧):   {global_rmse:.2f} m")
    print(f"📈 稳态 RMSE (去冷启动, 共 {len(errors) - conv_idx} 帧): {steady_rmse:.2f} m")
    print(f"📉 误差降幅: {(global_rmse - steady_rmse) / global_rmse * 100:.1f}%")
    print("-" * 40)

    return conv_idx, metric


def plot_tracking_results(pred_file="model_predictions.npy", truth_file="ground_truth.npy", p_file="P_history.npy",
                          dt=0.1, warmup_frames=15):
    """
    绘制跟踪结果。

    参数:
        warmup_frames (int): 固定冷启动帧数，默认 15 帧。
                             设为 0 则启用动态阈值检测。
    """
    # ==========================================
    # 1. 加载数据
    # ==========================================
    try:
        preds = np.load(pred_file)
        truth = np.load(truth_file)
    except FileNotFoundError:
        print(f"❌ 找不到预测或真实轨迹文件！请检查 {pred_file} 和 {truth_file}")
        return

    # 尝试加载协方差矩阵 (如果有)
    try:
        P_history = np.load(p_file)
    except FileNotFoundError:
        P_history = None

    T = preds.shape[0]
    time_stamps = np.arange(T) * dt

    # 提取位置 (索引 0, 2, 4 分别对应 X, Y, Z)
    pred_px, pred_py, pred_pz = preds[:, 0], preds[:, 2], preds[:, 4]
    true_px, true_py, true_pz = truth[:, 0], truth[:, 2], truth[:, 4]

    true_pos = np.column_stack((true_px, true_py, true_pz))
    pred_pos = np.column_stack((pred_px, pred_py, pred_pz))

    # ==========================================
    # 2. 动态收敛评估
    # ==========================================
    # 这里的 threshold 是误差的阈值 (如果是位置误差，单位是米)
    # 如果你加载了 P_history，请把阈值调大 (比如 1000)
    threshold_val = 100.0 if P_history is None else 1000.0
    conv_idx, metric = evaluate_dynamic_convergence(true_pos, pred_pos, P_history, threshold=threshold_val, dt=dt,
                                                     warmup_frames=warmup_frames)
    conv_time = conv_idx * dt

    # ==========================================
    # 3. 绘制分开的独立轨迹图与误差图
    # ==========================================
    # 数据单位转换为 km，并将起点平移至原点 (计算相对位移 Delta)
    start_x, start_y, start_z = true_px[0], true_py[0], true_pz[0]
    true_px_k = (true_px - start_x) / 1000.0
    true_py_k = (true_py - start_y) / 1000.0
    true_pz_k = (true_pz - start_z) / 1000.0
    pred_px_k = (pred_px - start_x) / 1000.0
    pred_py_k = (pred_py - start_y) / 1000.0
    pred_pz_k = (pred_pz - start_z) / 1000.0

    pos_err = np.linalg.norm(true_pos - pred_pos, axis=1)

    # ---- 辅助绘图函数 ----
    def create_3d_plot(t_start, t_end, title, tight_limits=False):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(true_px_k[t_start:t_end], true_py_k[t_start:t_end], true_pz_k[t_start:t_end], 
                color='red', linestyle='-', linewidth=2, label='真实目标轨迹')
        ax.plot(pred_px_k[t_start:t_end], pred_py_k[t_start:t_end], pred_pz_k[t_start:t_end], 
                color='blue', linestyle='--', linewidth=2, label='预测轨迹')
        
        ax.scatter(true_px_k[t_start], true_py_k[t_start], true_pz_k[t_start], color='red', marker='o', s=60, label='真实起点', zorder=5)
        ax.scatter(true_px_k[t_end-1], true_py_k[t_end-1], true_pz_k[t_end-1], color='red', marker='*', s=100, label='真实终点', zorder=5)
        ax.scatter(pred_px_k[t_start], pred_py_k[t_start], pred_pz_k[t_start], color='blue', marker='o', s=60, label='预测起点', zorder=5)
        ax.scatter(pred_px_k[t_end-1], pred_py_k[t_end-1], pred_pz_k[t_end-1], color='blue', marker='*', s=100, label='预测终点', zorder=5)

        ax.set_title(title)
        ax.set_xlabel(r"$\Delta X$ (km)")
        ax.set_ylabel(r"$\Delta Y$ (km)")
        ax.set_zlabel(r"$\Delta Z$ (km)")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if tight_limits:
            safe_pred_px_k = pred_px_k[t_start:t_end] if t_end <= conv_idx else pred_px_k[max(conv_idx, t_start):t_end]
            safe_pred_py_k = pred_py_k[t_start:t_end] if t_end <= conv_idx else pred_py_k[max(conv_idx, t_start):t_end]
            safe_pred_pz_k = pred_pz_k[t_start:t_end] if t_end <= conv_idx else pred_pz_k[max(conv_idx, t_start):t_end]
            
            # 若安全片段为空，退回到全图
            if len(safe_pred_px_k) == 0:
                safe_pred_px_k = pred_px_k[t_start:t_end]
                safe_pred_py_k = pred_py_k[t_start:t_end]
                safe_pred_pz_k = pred_pz_k[t_start:t_end]

            min_x = min(true_px_k[t_start:t_end].min(), safe_pred_px_k.min())
            max_x = max(true_px_k[t_start:t_end].max(), safe_pred_px_k.max())
            min_y = min(true_py_k[t_start:t_end].min(), safe_pred_py_k.min())
            max_y = max(true_py_k[t_start:t_end].max(), safe_pred_py_k.max())
            min_z = min(true_pz_k[t_start:t_end].min(), safe_pred_pz_k.min())
            max_z = max(true_pz_k[t_start:t_end].max(), safe_pred_pz_k.max())
            
            margin_x = max(0.01, (max_x - min_x) * 0.02)
            margin_y = max(0.01, (max_y - min_y) * 0.02)
            margin_z = max(0.01, (max_z - min_z) * 0.02)
            
            ax.set_xlim(min_x - margin_x, max_x + margin_x)
            ax.set_ylim(min_y - margin_y, max_y + margin_y)
            ax.set_zlim(min_z - margin_z, max_z + margin_z)

        fig.legend(loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        return fig, ax

    def create_2d_plot(t_start, t_end, title, axis1, axis2, true_v1, true_v2, pred_v1, pred_v2, tight_limits=False):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(true_v1[t_start:t_end], true_v2[t_start:t_end], color='red', linestyle='-', linewidth=2, label='真实目标轨迹')
        ax.plot(pred_v1[t_start:t_end], pred_v2[t_start:t_end], color='blue', linestyle='--', linewidth=2, label='预测轨迹')
        
        ax.scatter(true_v1[t_start], true_v2[t_start], color='red', marker='o', s=60, label='真实起点', zorder=5)
        ax.scatter(true_v1[t_end-1], true_v2[t_end-1], color='red', marker='*', s=100, label='真实终点', zorder=5)
        ax.scatter(pred_v1[t_start], pred_v2[t_start], color='blue', marker='o', s=60, label='预测起点', zorder=5)
        ax.scatter(pred_v1[t_end-1], pred_v2[t_end-1], color='blue', marker='*', s=100, label='预测终点', zorder=5)

        ax.set_title(title)
        ax.set_xlabel(axis1)
        ax.set_ylabel(axis2)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if tight_limits:
            safe_pred_v1 = pred_v1[t_start:t_end] if t_end <= conv_idx else pred_v1[max(conv_idx, t_start):t_end]
            safe_pred_v2 = pred_v2[t_start:t_end] if t_end <= conv_idx else pred_v2[max(conv_idx, t_start):t_end]
            
            if len(safe_pred_v1) == 0:
                safe_pred_v1 = pred_v1[t_start:t_end]
                safe_pred_v2 = pred_v2[t_start:t_end]
                
            min_1 = min(true_v1[t_start:t_end].min(), safe_pred_v1.min())
            max_1 = max(true_v1[t_start:t_end].max(), safe_pred_v1.max())
            min_2 = min(true_v2[t_start:t_end].min(), safe_pred_v2.min())
            max_2 = max(true_v2[t_start:t_end].max(), safe_pred_v2.max())
            
            margin_1 = max(0.01, (max_1 - min_1) * 0.02)
            margin_2 = max(0.01, (max_2 - min_2) * 0.02)
            
            ax.set_xlim(min_1 - margin_1, max_1 + margin_1)
            ax.set_ylim(min_2 - margin_2, max_2 + margin_2)

        fig.legend(loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        return fig, ax

    def create_error_plot(t_start, t_end, title, show_conv_idx=False, show_rmse=False):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        current_err = pos_err[t_start:t_end]
        ax.plot(np.arange(t_start, t_end), current_err, color='blue', linewidth=2.0, marker='.', label="预测误差")
        
        if show_conv_idx and conv_idx > 0:
            ax.axvspan(0, conv_idx, color='gray', alpha=0.2, label='冷启动阶段')
            ax.axvline(x=conv_idx, color='green', linestyle=':', linewidth=2, label='稳态起点')
            
        if show_rmse:
            full_rmse = np.sqrt(np.mean(current_err**2))
            ax.axhline(y=full_rmse, color='purple', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.text(t_end * 0.98, full_rmse, f'全段 RMSE: {full_rmse:.2f}m', color='purple', 
                    va='bottom', ha='right', fontsize=11, fontweight='bold')
            
            if conv_idx > 0 and conv_idx < len(current_err):
                steady_err = current_err[conv_idx:]
                steady_rmse = np.sqrt(np.mean(steady_err**2))
                ax.axhline(y=steady_rmse, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
                ax.text(t_end * 0.98, steady_rmse, f'去除冷启动 RMSE: {steady_rmse:.2f}m', color='orange', 
                        va='top', ha='right', fontsize=11, fontweight='bold')
            
        ax.set_xlabel("时间步 (帧)")
        ax.set_ylabel("三维位置误差 (m)")
        ax.set_title(title)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        fig.legend(loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        return fig, ax

    # ---- 全段图形 ----
    len_tot = len(true_pos)
    # create_3d_plot(0, len_tot, "3D跟踪轨迹图 (全段)", tight_limits=True)
    # create_2d_plot(0, len_tot, "X-Y 平面轨迹图 (全段)", r"$\Delta X$ (km)", r"$\Delta Y$ (km)", true_px_k, true_py_k, pred_px_k, pred_py_k, tight_limits=True)
    # create_2d_plot(0, len_tot, "X-Z 平面轨迹图 (全段)", r"$\Delta X$ (km)", r"$\Delta Z$ (km)", true_px_k, true_pz_k, pred_px_k, pred_pz_k, tight_limits=True)
    create_error_plot(0, len_tot, "全段逐帧误差对比", show_conv_idx=True, show_rmse=True)

    # ---- 局部放大图形 (0-30帧) ----
    zoom_end = min(len_tot, 30)
    # create_3d_plot(0, zoom_end, f"3D博弈轨迹局部放大 (帧 0 - {zoom_end - 1})", tight_limits=False)
    # create_2d_plot(0, zoom_end, f"X-Y 平面轨迹局部放大 (帧 0 - {zoom_end - 1})", r"$\Delta X$ (km)", r"$\Delta Y$ (km)", true_px_k, true_py_k, pred_px_k, pred_py_k, tight_limits=False)
    # create_2d_plot(0, zoom_end, f"X-Z 平面轨迹局部放大 (帧 0 - {zoom_end - 1})", r"$\Delta X$ (km)", r"$\Delta Z$ (km)", true_px_k, true_pz_k, pred_px_k, pred_pz_k, tight_limits=False)
    # create_error_plot(0, zoom_end, f"局部片段逐帧误差对比 (帧 0 - {zoom_end - 1})", show_conv_idx=False)

    plt.show()


if __name__ == "__main__":
    # warmup_frames=15 表示固定前 15 帧为冷启动阶段
    # 可调整为 10~20，或设为 0 启用动态阈值检测
    plot_tracking_results(warmup_frames=15)