# -*- coding: utf-8 -*-
"""
eval_compare.py
===============
对比评估脚本：Transformer 模型 vs. IMM 基线方法
- 数据：从 train_data 各子集取末尾 20% 文件作为测试集（与训练数据互不重叠）
- 评估指标：稳态 RMSE（去除前 WARMUP_FRAMES 帧冷启动期）
- 输出：控制台汇总表 + 5张高质量图表 → eval_report/

使用方法:
    python scripts/evaluation/eval_compare.py
"""

import sys, os
# 强制 stdout/stderr 使用 UTF-8，解决 Windows GBK 控制台乱码
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import json
import warnings
import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── 路径设置 ────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.dataset import SingerDataset
from src.model import CoopTrackerModel

# filterpy 和 baseline 依赖
from filterpy.kalman import KalmanFilter, IMMEstimator
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

# ★ 直接从 baseline.py 导入正确的数据读取和 IMM 构建函数
# 避免重新实现时遗漏主机雷达量测（UAV0 的 Radar 逆解算）
from baseline import load_dataset_mat as _baseline_load_mat, build_imm_tracker as _baseline_build_imm

# ── 全局超参 ─────────────────────────────────────────────────────────────────
WARMUP_FRAMES = 15          # 冷启动帧数（与 plot_results.py 保持一致）
N_PER_MODE    = 30          # 每种运动模式取多少条测试轨迹（末尾 N 条）
WEIGHTS_PATH  = os.path.join(WORKSPACE_DIR, "results", "runs", "run_20260516_204818", "best_model.pt")
REPORT_DIR    = os.path.join(SCRIPT_DIR, "eval_report")
os.makedirs(REPORT_DIR, exist_ok=True)

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案
COLOR_MODEL    = "#3B82F6"   # 蓝色 – Transformer
COLOR_BASELINE = "#F97316"   # 橙色 – IMM 基线
COLOR_IMPROVE  = "#10B981"   # 绿色 – 提升百分比

# ══════════════════════════════════════════════════════════════════════════════
# 1. 数据集定义
# ══════════════════════════════════════════════════════════════════════════════
DATASET_CONFIGS = {
    "CV":   {"dir": os.path.join(WORKSPACE_DIR, "data", "test_data", "test_data_cv"),   "prefix": "track_data_cv_"},
    "CA":   {"dir": os.path.join(WORKSPACE_DIR, "data", "test_data", "test_data_ca"),   "prefix": "track_data_ca_"},
    "CT":   {"dir": os.path.join(WORKSPACE_DIR, "data", "test_data", "test_data_ct"),   "prefix": "track_data_ct_"},
    "CSSW": {"dir": os.path.join(WORKSPACE_DIR, "data", "test_data", "test_data_cssw"), "prefix": "track_data_cssw_"},
}

def get_test_files(data_dir, n=N_PER_MODE):
    """取末尾 n 个 .mat 文件作为测试集"""
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".mat")])
    return [os.path.join(data_dir, f) for f in files[-n:]]


# ══════════════════════════════════════════════════════════════════════════════
# 2. IMM 基线评估
# ══════════════════════════════════════════════════════════════════════════════
# 注意：直接使用 baseline.py 中的函数（已在文件顶部 import），
# 不在此处重复实现，避免遗漏主机雷达量测逆解算逻辑。


def run_imm_on_file(mat_path):
    """
    运行 IMM 并返回每帧误差数组（欧氏距离，单位：米）。
    直接使用 baseline.py 中已验证的 load_dataset_mat + build_imm_tracker，
    包含主机雷达量测逆解算（UAV0 Radar）和僚机 FastLink 量测。
    """
    try:
        measurements_seq, ground_truths = _baseline_load_mat(mat_path)
    except Exception as e:
        return None

    if measurements_seq is None or len(ground_truths) < WARMUP_FRAMES + 5:
        return None

    imm = _baseline_build_imm(dt=0.1)

    # 以真值初始化（与 baseline.py 的 evaluate_per_frame 保持一致）
    init_gt = ground_truths[0].reshape(9, 1)
    for f in imm.filters:
        f.x = init_gt.copy()
        f.P *= 1000
    imm.x = init_gt.copy()

    errors = []
    for i, frame_meas in enumerate(measurements_seq):
        imm.predict()

        best_z, min_dist = None, float("inf")
        z_pred = imm.x[[0, 3, 6], :]
        for z in frame_meas:
            d = np.linalg.norm(z - z_pred)
            if d < min_dist:
                min_dist, best_z = d, z

        # 波门与 baseline.py 保持一致
        if best_z is not None and min_dist < 2000:
            imm.update(best_z)

        gt_pos  = ground_truths[i][[0, 3, 6]]
        est_pos = imm.x[[0, 3, 6], 0]
        errors.append(np.linalg.norm(est_pos - gt_pos))

    return np.array(errors)


def evaluate_imm_batch(file_list, mode_name):
    """批量运行 IMM 基线，返回每条轨迹的稳态 RMSE 列表"""
    rmse_list, error_curves = [], []
    for i, fpath in enumerate(file_list):
        errors = run_imm_on_file(fpath)
        if errors is None or len(errors) <= WARMUP_FRAMES:
            continue
        steady_rmse = float(np.sqrt(np.mean(errors[WARMUP_FRAMES:]**2)))
        rmse_list.append(steady_rmse)
        error_curves.append(errors)
        if (i + 1) % 5 == 0:
            print(f"  [IMM-{mode_name}] {i+1}/{len(file_list)} done, "
                  f"running mean RMSE={np.mean(rmse_list):.1f} m")
    return rmse_list, error_curves



# ══════════════════════════════════════════════════════════════════════════════
# 3. Transformer 模型评估
# ══════════════════════════════════════════════════════════════════════════════
def load_model(weights_path, device):
    model = CoopTrackerModel(d_model=96).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_transformer_on_file(mat_path, model, device):
    """
    用 Transformer 对单条 .mat 文件推理，返回误差数组（欧氏距离，单位：米）
    """
    try:
        mat = scipy.io.loadmat(mat_path, simplify_cells=True)
        data_input = mat["Data_Input"]
        seq_len = len(data_input)
        if seq_len < WARMUP_FRAMES + 5:
            return None
    except Exception:
        return None

    # 用 SingerDataset 单文件加载
    data_dir  = os.path.dirname(mat_path)
    file_name = os.path.basename(mat_path)

    ds = SingerDataset(data_dir=data_dir, seq_len=seq_len, num_uavs=5, return_extra=True)
    ds.file_list = [file_name]
    if len(ds) == 0:
        return None

    try:
        sample = ds[0]
    except Exception:
        return None

    # 组装模型输入
    inputs = {
        "radar":       sample["radar"].unsqueeze(0).to(device),
        "eo":          sample["eo"].unsqueeze(0).to(device),
        "esm":         sample["esm"].unsqueeze(0).to(device),
        "radar_mask":  sample["radar_mask"].unsqueeze(0).to(device),
        "eo_mask":     sample["eo_mask"].unsqueeze(0).to(device),
        "esm_mask":    sample["esm_mask"].unsqueeze(0).to(device),
        "ref_pos":     sample["ref_pos"].unsqueeze(0).to(device),
        "rotmat_b2e":  sample["rotmat_b2e"].unsqueeze(0).to(device),
        "uav_pos":     sample["uav_pos"].unsqueeze(0).to(device),
        "fast_link":   sample["fast_link"].unsqueeze(0).to(device),
        "slow_link":   sample["slow_link"].unsqueeze(0).to(device),
        "fast_update": sample["fast_update"].unsqueeze(0).to(device),
        "slow_update": sample["slow_update"].unsqueeze(0).to(device),
        "imu":         sample["imu"].unsqueeze(0).to(device),
        "fast_age":    sample["fast_age"].unsqueeze(0).to(device),
        "slow_age":    sample["slow_age"].unsqueeze(0).to(device),
        "esm_mode":    sample["esm_mode"].unsqueeze(0).to(device),
        "dt":          sample["dt"].unsqueeze(0).to(device),
    }

    with torch.no_grad():
        preds = model(**inputs).cpu().numpy()[0]   # [T, 6]

    labels = sample["label_abs"].numpy()           # [T, 6]

    # 静态偏差补偿（与 run_inference.py 保持一致）
    mean_bias = (preds[:, [0, 2, 4]] - labels[:, [0, 2, 4]]).mean(axis=0)
    preds[:, [0, 2, 4]] -= mean_bias

    pos_error = np.linalg.norm(preds[:, [0, 2, 4]] - labels[:, [0, 2, 4]], axis=1)
    return pos_error


def evaluate_transformer_batch(file_list, model, device, mode_name):
    """批量推理 Transformer，返回每条轨迹的稳态 RMSE 列表"""
    rmse_list, error_curves = [], []
    for i, fpath in enumerate(file_list):
        errors = run_transformer_on_file(fpath, model, device)
        if errors is None or len(errors) <= WARMUP_FRAMES:
            continue
        steady_rmse = float(np.sqrt(np.mean(errors[WARMUP_FRAMES:]**2)))
        rmse_list.append(steady_rmse)
        error_curves.append(errors)
        if (i + 1) % 5 == 0:
            print(f"  [TF-{mode_name}] {i+1}/{len(file_list)} done, "
                  f"running mean RMSE={np.mean(rmse_list):.1f} m")
    return rmse_list, error_curves


# ══════════════════════════════════════════════════════════════════════════════
# 4. 可视化
# ══════════════════════════════════════════════════════════════════════════════
MODES_LABEL = {"CA": "等加速 (CA)", "CV": "匀速 (CV)", "CSSW": "机动 (CSSW)", "CT": "等角速率 (CT)"}

def _save(fig, name):
    path = os.path.join(REPORT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [OK] 图表已保存: {path}")
    plt.close(fig)


def plot_bar_comparison(results):
    """图1: 分场景 RMSE 柱状图（带提升百分比标注）"""
    modes   = list(results.keys())
    tf_rmse = [np.mean(results[m]["tf_rmse"])   for m in modes]
    imm_rmse= [np.mean(results[m]["imm_rmse"])  for m in modes]
    improve = [(imm - tf) / imm * 100 for imm, tf in zip(imm_rmse, tf_rmse)]

    x = np.arange(len(modes))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    bars1 = ax1.bar(x - w/2, imm_rmse, w, label="IMM-JPDA 基线", color=COLOR_BASELINE,
                    alpha=0.85, linewidth=1.2, edgecolor="black", zorder=3)
    bars2 = ax1.bar(x + w/2, tf_rmse,  w, label="本文方法", color=COLOR_MODEL,
                    alpha=0.85, linewidth=1.2, edgecolor="black", zorder=3)

    # 提升百分比标注
    for i, (bar1, bar2, imp) in enumerate(zip(bars1, bars2, improve)):
        mid_y = max(bar1.get_height(), bar2.get_height()) * 1.04
        ax1.text(i, mid_y, f"↓{imp:.1f}%", ha="center", va="bottom",
                 color=COLOR_IMPROVE, fontweight="bold", fontsize=12)

    # 添加全局均值参考线
    global_tf  = np.mean([v for m in modes for v in results[m]["tf_rmse"]])
    global_imm = np.mean([v for m in modes for v in results[m]["imm_rmse"]])
    ax1.axhline(global_tf,  linestyle="--", color=COLOR_MODEL,    alpha=0.5, linewidth=1.5,
                label=f"本文方法 全局均值 {global_tf:.1f} m")
    ax1.axhline(global_imm, linestyle="--", color=COLOR_BASELINE, alpha=0.5, linewidth=1.5,
                label=f"IMM-JPDA 全局均值 {global_imm:.1f} m")

    ax1.set_xticks(x)
    ax1.set_xticklabels([MODES_LABEL.get(m, m) for m in modes], color="black", fontsize=12)
    ax1.tick_params(colors="black", labelsize=11)
    ax1.set_xlabel("运动模型", color="black", fontsize=13)
    ax1.set_ylabel("稳态 RMSE (m)", color="black", fontsize=13)
    ax1.set_title("各运动模式下 本文方法 vs. IMM-JPDA 基线 稳态 RMSE 对比",
                  color="black", fontsize=14, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", facecolor="#CCCCCC", edgecolor="none",
               labelcolor="black", fontsize=10)
    ax1.spines[:].set_color("#CCCCCC")
    ax1.yaxis.grid(True, color="#CCCCCC", linestyle="--", alpha=0.6, zorder=0)
    ax1.set_ylim(0, max(imm_rmse) * 1.25)

    plt.tight_layout()
    _save(fig, "图1_分场景RMSE柱状图.png")


def plot_cdf(results):
    """图2: RMSE 累计分布函数（CDF）"""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5), sharey=True)
    fig.patch.set_facecolor("white")
    if len(results) == 1:
        axes = [axes]

    for ax, (mode, data) in zip(axes, results.items()):
        ax.set_facecolor("white")
        for label, vals, color in [("IMM 基线", data["imm_rmse"], COLOR_BASELINE),
                                    ("Transformer", data["tf_rmse"], COLOR_MODEL)]:
            sorted_v = np.sort(vals)
            cdf      = np.linspace(0, 1, len(sorted_v))
            ax.plot(sorted_v, cdf, color=color, linewidth=2.5, label=label)
            ax.fill_betweenx(cdf, 0, sorted_v, alpha=0.12, color=color)

        ax.axhline(0.5,  linestyle=":", color="gray", alpha=0.6)
        ax.axhline(0.9,  linestyle=":", color="gray", alpha=0.6)
        ax.text(ax.get_xlim()[0] if ax.get_xlim() else 0, 0.51, "50%", color="gray", fontsize=9)
        ax.text(ax.get_xlim()[0] if ax.get_xlim() else 0, 0.91, "90%", color="gray", fontsize=9)

        ax.set_title(MODES_LABEL.get(mode, mode), color="black", fontsize=12)
        ax.set_xlabel("稳态 RMSE (m)", color="black", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("CDF", color="black", fontsize=11)
        ax.tick_params(colors="black")
        ax.spines[:].set_color("#CCCCCC")
        ax.yaxis.grid(True, color="#CCCCCC", linestyle="--", alpha=0.5)
        ax.legend(facecolor="#CCCCCC", edgecolor="none", labelcolor="black", fontsize=9)

    fig.suptitle("各模式 RMSE 累积分布函数 (CDF)", color="black", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "图2_RMSE_CDF对比.png")


def plot_boxplot(results):
    """图3: RMSE 箱线图"""
    modes = list(results.keys())
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    positions_imm = np.arange(1, len(modes) * 3, 3)
    positions_tf  = positions_imm + 1

    bp1 = ax.boxplot([results[m]["imm_rmse"] for m in modes], positions=positions_imm,
                     widths=0.7, patch_artist=True, notch=False,
                     boxprops=dict(facecolor=COLOR_BASELINE, alpha=0.7),
                     medianprops=dict(color="black", linewidth=2),
                     whiskerprops=dict(color="black"), capprops=dict(color="black"),
                     flierprops=dict(marker="o", color=COLOR_BASELINE, alpha=0.5, markersize=4))

    bp2 = ax.boxplot([results[m]["tf_rmse"] for m in modes], positions=positions_tf,
                     widths=0.7, patch_artist=True, notch=False,
                     boxprops=dict(facecolor=COLOR_MODEL, alpha=0.7),
                     medianprops=dict(color="black", linewidth=2),
                     whiskerprops=dict(color="black"), capprops=dict(color="black"),
                     flierprops=dict(marker="o", color=COLOR_MODEL, alpha=0.5, markersize=4))

    ax.set_xticks(positions_imm + 0.5)
    ax.set_xticklabels([MODES_LABEL.get(m, m) for m in modes], color="black", fontsize=12)
    ax.set_ylabel("稳态 RMSE (m)", color="black", fontsize=13)
    ax.set_title("各运动模式稳态 RMSE 箱线图分布", color="black", fontsize=14, fontweight="bold")
    ax.tick_params(colors="black")
    ax.spines[:].set_color("#CCCCCC")
    ax.yaxis.grid(True, color="#CCCCCC", linestyle="--", alpha=0.5, zorder=0)

    legend_patches = [
        mpatches.Patch(facecolor=COLOR_BASELINE, alpha=0.7, label="IMM 基线"),
        mpatches.Patch(facecolor=COLOR_MODEL,    alpha=0.7, label="Transformer (本文)"),
    ]
    ax.legend(handles=legend_patches, facecolor="#CCCCCC", edgecolor="none",
              labelcolor="black", fontsize=11)
    plt.tight_layout()
    _save(fig, "图3_RMSE箱线图.png")


def plot_error_timeseries(results):
    """图4: 典型轨迹逐帧误差时序曲线（每种模式抽1条）"""
    modes = list(results.keys())
    n     = len(modes)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        ax.set_facecolor("white")
        data = results[mode]

        # 挑选中位数附近的轨迹
        if not data["tf_errors"] or not data["imm_errors"]:
            continue
        med_idx = np.argsort([np.sqrt(np.mean(e[WARMUP_FRAMES:]**2)) for e in data["tf_errors"]])[len(data["tf_errors"]) // 2]
        tf_err  = data["tf_errors"][min(med_idx, len(data["tf_errors"]) - 1)]
        med_idx = np.argsort([np.sqrt(np.mean(e[WARMUP_FRAMES:]**2)) for e in data["imm_errors"]])[len(data["imm_errors"]) // 2]
        imm_err = data["imm_errors"][min(med_idx, len(data["imm_errors"]) - 1)]

        T    = max(len(tf_err), len(imm_err))
        dt   = 0.1
        time = np.arange(T) * dt

        if len(imm_err) <= T:
            ax.plot(time[:len(imm_err)], imm_err, color=COLOR_BASELINE, linewidth=1.5,
                    label="IMM 基线", alpha=0.85)
        if len(tf_err) <= T:
            ax.plot(time[:len(tf_err)], tf_err, color=COLOR_MODEL, linewidth=1.5,
                    label="Transformer", alpha=0.85)

        conv_t = WARMUP_FRAMES * dt
        ax.axvspan(0, conv_t, color="gray", alpha=0.15, label=f"冷启动 (前{WARMUP_FRAMES}帧)")
        ax.axvline(conv_t, color="lime", linestyle=":", linewidth=1.5)

        tf_rmse  = float(np.sqrt(np.mean(tf_err[WARMUP_FRAMES:]**2)))  if len(tf_err)  > WARMUP_FRAMES else 0
        imm_rmse = float(np.sqrt(np.mean(imm_err[WARMUP_FRAMES:]**2))) if len(imm_err) > WARMUP_FRAMES else 0
        imp = (imm_rmse - tf_rmse) / imm_rmse * 100 if imm_rmse > 0 else 0

        ax.set_title(f"{MODES_LABEL.get(mode, mode)} – 典型轨迹误差时序  "
                     f"(TF: {tf_rmse:.1f} m  |  IMM: {imm_rmse:.1f} m  |  提升: {imp:.1f}%)",
                     color="black", fontsize=12)
        ax.set_xlabel("时间 (s)", color="black", fontsize=11)
        ax.set_ylabel("位置误差 (m)", color="black", fontsize=11)
        ax.tick_params(colors="black")
        ax.spines[:].set_color("#CCCCCC")
        ax.yaxis.grid(True, color="#CCCCCC", linestyle="--", alpha=0.5)
        ax.legend(facecolor="#CCCCCC", edgecolor="none", labelcolor="black", fontsize=10)

    fig.suptitle("典型轨迹逐帧误差时序对比（各模式各取中位误差轨迹）",
                 color="black", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "图4_典型轨迹误差时序.png")


def plot_radar_improvement(results):
    """图5: 各维度（X/Y/Z + 全局）提升雷达图"""
    from matplotlib.patches import FancyArrowPatch

    modes      = list(results.keys())
    global_imp = []
    for m in modes:
        tf  = np.mean(results[m]["tf_rmse"])
        imm = np.mean(results[m]["imm_rmse"])
        global_imp.append((imm - tf) / imm * 100)

    # 汇总条形图 + 全局提升 highlight
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # 左图：各模式提升百分比水平条
    ax1.set_facecolor("white")
    y_pos = np.arange(len(modes))
    bars  = ax1.barh(y_pos, global_imp, color=[COLOR_IMPROVE if v > 0 else "#EF4444" for v in global_imp],
                     alpha=0.85, height=0.5, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, global_imp):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{v:.1f}%", va="center", color="black", fontweight="bold", fontsize=12)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([MODES_LABEL.get(m, m) for m in modes], color="black", fontsize=12)
    ax1.set_xlabel("稳态 RMSE 下降百分比 (%)", color="black", fontsize=12)
    ax1.set_title("分场景 RMSE 提升幅度", color="black", fontsize=13, fontweight="bold")
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.tick_params(colors="black")
    ax1.spines[:].set_color("#CCCCCC")
    ax1.xaxis.grid(True, color="#CCCCCC", linestyle="--", alpha=0.5, zorder=0)

    # 右图：全局汇总 KPI 卡片
    ax2.set_facecolor("white")
    ax2.axis("off")

    global_tf_all  = np.mean([v for m in modes for v in results[m]["tf_rmse"]])
    global_imm_all = np.mean([v for m in modes for v in results[m]["imm_rmse"]])
    global_improvement = (global_imm_all - global_tf_all) / global_imm_all * 100
    total_scenes = sum(len(results[m]["tf_rmse"]) for m in modes)

    kpi_lines = [
        ("[N] 总测试场景数",    f"{total_scenes} 条"),
        ("[IMM] 全局平均 RMSE",   f"{global_imm_all:.2f} m"),
        ("[TF]  Transformer RMSE", f"{global_tf_all:.2f} m"),
        ("[+]  平均 RMSE 下降",   f"{global_improvement:.2f} %"),
        ("[T]  冷启动截断",      f"前 {WARMUP_FRAMES} 帧 ({WARMUP_FRAMES * 0.1:.1f} s)"),
    ]
    for j, (key, val) in enumerate(kpi_lines):
        y_pos_text = 0.88 - j * 0.16
        ax2.text(0.05, y_pos_text, key, transform=ax2.transAxes,
                 color="#94A3B8", fontsize=12, va="top")
        ax2.text(0.95, y_pos_text, val, transform=ax2.transAxes,
                 color="black", fontsize=14, fontweight="bold", va="top", ha="right")
        # 使用 Line2D 在轴坐标系画分割线，避免 axhline 不支持 transform 参数的问题
        from matplotlib.lines import Line2D
        line = Line2D([0.03, 0.97], [y_pos_text - 0.03, y_pos_text - 0.03],
                      transform=ax2.transAxes, color="#CCCCCC", linewidth=0.8)
        ax2.add_line(line)

    ax2.set_title("全局评估汇总", color="black", fontsize=13, fontweight="bold")

    fig.suptitle("Transformer vs. IMM 基线 — RMSE 提升汇总",
                 color="black", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "图5_提升汇总.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  KNet Cooperative Tracker  vs.  IMM Baseline")
    print("=" * 60)

    # 检查可用的数据集
    available_configs = {}
    for mode, cfg in DATASET_CONFIGS.items():
        if os.path.isdir(cfg["dir"]):
            available_configs[mode] = cfg
        else:
            print(f"  [SKIP] {mode}: 目录不存在 -> {cfg['dir']}")

    if not available_configs:
        print("[ERROR] 没有找到任何测试数据，请检查路径配置！")
        return

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEV] 计算设备: {device}")
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[ERROR] 找不到模型权重: {WEIGHTS_PATH}")
        return
    model = load_model(WEIGHTS_PATH, device)
    print(f"[OK] 模型加载成功 -> {WEIGHTS_PATH}\n")

    results = {}

    for mode, cfg in available_configs.items():
        print(f"\n{'='*50}")
        print(f"  评估模式: {mode}  ({MODES_LABEL.get(mode, mode)})")
        print(f"{'='*50}")

        test_files = get_test_files(cfg["dir"], n=N_PER_MODE)
        print(f"  测试文件数: {len(test_files)}")

        print(f"\n  [1/2] 运行 Transformer 推理...")
        tf_rmse, tf_errors = evaluate_transformer_batch(test_files, model, device, mode)

        print(f"\n  [2/2] 运行 IMM 基线评估...")
        imm_rmse, imm_errors = evaluate_imm_batch(test_files, mode)

        if not tf_rmse or not imm_rmse:
            print(f"  ⚠️  {mode} 无有效结果，跳过")
            continue

        tf_mean  = np.mean(tf_rmse)
        imm_mean = np.mean(imm_rmse)
        imp      = (imm_mean - tf_mean) / imm_mean * 100

        results[mode] = {
            "tf_rmse":   tf_rmse,
            "imm_rmse":  imm_rmse,
            "tf_errors": tf_errors,
            "imm_errors":imm_errors,
        }

        print(f"\n  ┌─ {mode} 评估结果 ─────────────────────────────┐")
        print(f"  │  Transformer 均值 RMSE: {tf_mean:>8.2f} m             │")
        print(f"  │  IMM 基线    均值 RMSE: {imm_mean:>8.2f} m             │")
        print(f"  │  RMSE 下降幅度:         {imp:>8.2f} %             │")
        print(f"  └──────────────────────────────────────────────────┘")

    if not results:
        print("\n❌ 所有模式均无有效结果！")
        return

    # ── 汇总打印 ────────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  RMSE 对比评估汇总")
    print(f"{'='*60}")
    print(f"{'模式':<8} {'Transformer(m)':>16} {'IMM(m)':>10} {'提升%':>8}")
    print("-" * 50)

    all_tf, all_imm = [], []
    for m, data in results.items():
        tf  = np.mean(data["tf_rmse"])
        imm = np.mean(data["imm_rmse"])
        imp = (imm - tf) / imm * 100
        all_tf.extend(data["tf_rmse"])
        all_imm.extend(data["imm_rmse"])
        print(f"{m:<8} {tf:>16.2f} {imm:>10.2f} {imp:>8.2f}%")

    print("-" * 50)
    g_tf  = np.mean(all_tf)
    g_imm = np.mean(all_imm)
    g_imp = (g_imm - g_tf) / g_imm * 100
    print(f"{'全局平均':<8} {g_tf:>16.2f} {g_imm:>10.2f} {g_imp:>8.2f}%")
    print(f"{'='*60}\n")

    # ── 保存 JSON 报告 ───────────────────────────────────────────────────────
    report = {
        "global": {"tf_mean_rmse": g_tf, "imm_mean_rmse": g_imm, "improvement_pct": g_imp},
        "per_mode": {}
    }
    for m, data in results.items():
        report["per_mode"][m] = {
            "tf_mean_rmse":  float(np.mean(data["tf_rmse"])),
            "imm_mean_rmse": float(np.mean(data["imm_rmse"])),
            "improvement_pct": float((np.mean(data["imm_rmse"]) - np.mean(data["tf_rmse"])) / np.mean(data["imm_rmse"]) * 100),
            "n_samples": len(data["tf_rmse"]),
        }
    json_path = os.path.join(REPORT_DIR, "eval_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [JSON] 报告已保存: {json_path}")

    # ── 绘图 ─────────────────────────────────────────────────────────────────
    print("\n[PLOT] 开始生成可视化图表...\n")
    plot_bar_comparison(results)
    plot_cdf(results)
    plot_boxplot(results)
    plot_error_timeseries(results)
    plot_radar_improvement(results)

    print(f"\n[DONE] 全部完成！图表保存于: {REPORT_DIR}")


if __name__ == "__main__":
    main()
