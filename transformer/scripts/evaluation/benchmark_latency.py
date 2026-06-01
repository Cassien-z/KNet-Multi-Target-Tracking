import sys
import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")

# 尝试导入 seaborn，如果没有则回退到 matplotlib 默认样式
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ── 路径设置 ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.dataset import SingerDataset
from src.model import CoopTrackerModel

# ── 全局配置 ─────────────────────────────────────────────────────────────────
WARMUP_ITERATIONS = 50      # GPU 预热迭代次数
TEST_ITERATIONS   = 500     # 正式测试迭代次数
SEQ_LEN           = 1       # 测试单帧推理 (取时间维度 T=1)

WEIGHTS_PATH = os.path.join(WORKSPACE_DIR, "results", "runs", "run_20260516_204818", "best_model.pt")
REPORT_DIR   = os.path.join(SCRIPT_DIR, "latency_report")
os.makedirs(REPORT_DIR, exist_ok=True)

# 字体与配色方案 (高端现代感)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
if HAS_SEABORN:
    sns.set_theme(style="whitegrid", font="SimHei")

COLORS = {
    "full": "#3B82F6",         # 蓝色
}
LABELS = {
    "full": "本文模型 (Ours)",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. 准确的单帧推理耗时测量核心函数
# ══════════════════════════════════════════════════════════════════════════════
def measure_inference_time(model, sample_batch, device, mode="full", num_warmup=50, num_test=500):
    """
    使用 torch.cuda.Event 精确测量 GPU 单次前向推理时延。
    """
    model.eval()
    
    # 构造单帧输入数据 (截取时间维度 T=1)
    single_frame_batch = {}
    for k, v in sample_batch.items():
        if isinstance(v, torch.Tensor):
            if k == "ref_pos":
                single_frame_batch[k] = v.to(device)
            else:
                # 假设除了 ref_pos 外，其他所有的 tensor 第一维(B)之后是时间维度(T)
                if v.ndim >= 2:
                    single_frame_batch[k] = v[:, :SEQ_LEN, ...].to(device)
                else:
                    single_frame_batch[k] = v.to(device)

    # 准备输入字典
    def run_forward():
        inputs = {
            "radar": single_frame_batch["radar"],
            "eo": single_frame_batch["eo"],
            "esm": single_frame_batch["esm"],
            "radar_mask": single_frame_batch["radar_mask"],
            "eo_mask": single_frame_batch["eo_mask"],
            "esm_mask": single_frame_batch["esm_mask"],
            "ref_pos": single_frame_batch["ref_pos"],
            "rotmat_b2e": single_frame_batch["rotmat_b2e"],
            "uav_pos": single_frame_batch["uav_pos"],
            "fast_link": single_frame_batch["fast_link"],
            "slow_link": single_frame_batch["slow_link"],
            "fast_update": single_frame_batch["fast_update"],
            "slow_update": single_frame_batch["slow_update"],
            "imu": single_frame_batch["imu"],
            "fast_age": single_frame_batch["fast_age"],
            "slow_age": single_frame_batch["slow_age"],
            "esm_mode": single_frame_batch["esm_mode"],
            "dt": single_frame_batch["dt"],
        }
        # 如果模型有 mode 参数可以指定消融实验模式，这里简单模拟
        # 实际代码中可能是 kwargs 传入或者改变模型结构
        with torch.no_grad():
            out = model(**inputs)
        return out

    print(f"  [{LABELS[mode]}] 正在进行 GPU 预热 ({num_warmup} 次)...")
    for _ in range(num_warmup):
        run_forward()
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"  [{LABELS[mode]}] 正在进行正式测试 ({num_test} 次)...")
    latencies = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(num_test):
        if device.type == "cuda":
            start_event.record()
            run_forward()
            end_event.record()
            torch.cuda.synchronize()
            # elapsed_time 返回毫秒(ms)
            latency = start_event.elapsed_time(end_event)
        else:
            t0 = time.perf_counter()
            run_forward()
            latency = (time.perf_counter() - t0) * 1000.0
        
        latencies.append(latency)

    lat_array = np.array(latencies)
    print(f"  -> Mean: {lat_array.mean():.2f} ms, P95: {np.percentile(lat_array, 95):.2f} ms\n")
    return lat_array


# ══════════════════════════════════════════════════════════════════════════════
# 2. 优美的图表绘制函数
# ══════════════════════════════════════════════════════════════════════════════
def plot_beautiful_latency_stats(latency_dict):
    """
    用两张高颜值图表展示单模型的推理时延
    """
    m = "full"
    data = latency_dict[m]
    label = LABELS[m]
    color = COLORS[m]

    # ---------------------------------------------------------
    # 图 1: 核密度估计与直方图 (KDE + Histogram)
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6), facecolor="#F8FAFC")
    ax1.set_facecolor("#F8FAFC")
    
    mean_val = np.mean(data)
    p95_val = np.percentile(data, 95)
    
    if HAS_SEABORN:
        sns.histplot(data, bins=30, kde=True, color=color, ax=ax1, edgecolor="white", alpha=0.6)
    else:
        ax1.hist(data, bins=30, color=color, edgecolor="white", alpha=0.6, density=True)
    
    # 标注 Mean 和 P95
    ax1.axvline(mean_val, color="#F59E0B", linestyle="--", linewidth=2.5, label=f"平均时延: {mean_val:.2f} ms")
    ax1.axvline(p95_val, color="#EF4444", linestyle=":", linewidth=2.5, label=f"P95 时延: {p95_val:.2f} ms")
    
    ax1.set_title("模型单帧推理时延分布", fontsize=16, fontweight="bold", pad=15)
    ax1.set_xlabel("单帧推理耗时 (ms)", fontsize=13)
    ax1.set_ylabel("频次 / 密度", fontsize=13)
    ax1.legend(loc="upper right", frameon=True, fontsize=12, facecolor="white", edgecolor="#E2E8F0")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    path1 = os.path.join(REPORT_DIR, "时延图1_分布直方图.png")
    plt.tight_layout()
    fig1.savefig(path1, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # ---------------------------------------------------------
    # 图 2: 累积分布函数 (CDF)
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6), facecolor="#F8FAFC")
    ax2.set_facecolor("#F8FAFC")

    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    ax2.plot(sorted_data, yvals, label=label, color=color, linewidth=3)
    
    # 标记 P95
    ax2.scatter(p95_val, 0.95, color="#EF4444", zorder=5, s=80, label=f"P95: {p95_val:.2f} ms")
    ax2.vlines(x=p95_val, ymin=0, ymax=0.95, color="#EF4444", linestyle=":", linewidth=2, alpha=0.8)
    ax2.hlines(y=0.95, xmin=sorted_data[0], xmax=p95_val, color="#EF4444", linestyle=":", linewidth=2, alpha=0.8)

    ax2.set_title("单帧推理时延累积分布函数 (CDF)", fontsize=16, fontweight="bold", pad=15)
    ax2.set_xlabel("单帧推理耗时 (ms)", fontsize=13)
    ax2.set_ylabel("累积概率 (CDF)", fontsize=13)
    ax2.legend(loc="lower right", frameon=True, fontsize=12, facecolor="white", edgecolor="#E2E8F0")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    path2 = os.path.join(REPORT_DIR, "时延图2_CDF曲线图.png")
    plt.tight_layout()
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"\n[OK] 优美的时延图表已生成至目录: {REPORT_DIR}")
    print(f"  - {os.path.basename(path1)}\n  - {os.path.basename(path2)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*60)
    print("  Transformer 单帧前向推理耗时基准测试 (Benchmark)")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] 当前使用设备: {device}")

    # 1. 寻找可用测试数据
    data_dir = os.path.join(WORKSPACE_DIR, "data", "test_data", "test_data_cv")
    if not os.path.exists(data_dir):
        # 备选训练数据路径
        data_dir = os.path.join(WORKSPACE_DIR, "data", "train_data", "train_data_cv")
    if not os.path.exists(data_dir):
        print("[ERROR] 找不到任何数据文件，无法获取输入特征 Shape")
        return

    print(f"[Data] 加载数据集从: {data_dir}")
    # 随便加载一个即可获取 tensor 字典
    ds = SingerDataset(data_dir=data_dir, seq_len=50, num_uavs=5, return_extra=True)
    ds.file_list = ds.file_list[:1] # 只需1个文件
    loader = DataLoader(ds, batch_size=1)
    sample_batch = next(iter(loader))

    # 2. 实例化模型并加载权重
    print("[Model] 实例化 CoopTrackerModel...")
    model = CoopTrackerModel(d_model=96).to(device)
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("  - 权重加载成功")
    else:
        print(f"  - 警告: 找不到权重文件 {WEIGHTS_PATH}，将使用随机权重进行时间测试")
    
    # 3. 测量不同模式的单帧推理时延
    print(f"\n[Benchmark] 开始性能测试 (Warmup: {WARMUP_ITERATIONS}, Tests: {TEST_ITERATIONS})")
    
    # 注：由于原代码中模型没有显式的 mode 参数切换结构，
    # 若你的代码中可以通过参数切换 (如 no_gat)，请在 run_forward 中调整。
    # 这里为了图表演示，我们将同一个模型运行3次，人为模拟少许扰动以区分分布，
    # 实际应用中你应该替换为分别开启/关闭不同模块。
    
    latency_dict = {}
    
    # 真实测试 "full" 模式
    lat_full = measure_inference_time(model, sample_batch, device, mode="full", 
                                      num_warmup=WARMUP_ITERATIONS, num_test=TEST_ITERATIONS)
    latency_dict["full"] = lat_full
    
    # 4. 生成美观的图表
    print("\n[Plot] 正在生成高颜值图表...")
    plot_beautiful_latency_stats(latency_dict)
    
if __name__ == "__main__":
    main()
