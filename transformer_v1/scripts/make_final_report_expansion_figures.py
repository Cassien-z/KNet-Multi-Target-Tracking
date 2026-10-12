"""Generate Chinese figures used by the 120-page completion-report expansion."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "figures" / "final_report_expansion_120p"
OUT.mkdir(parents=True, exist_ok=True)


def configure_fonts() -> None:
    candidates = [
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
    ]
    for path in candidates:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["font.size"] = 11


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(pad=1.4)
    fig.savefig(OUT / name, dpi=260, bbox_inches="tight")
    plt.close(fig)


def asynchronous_sampling() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    end = 2.0
    rows = [
        ("统一时间网格", np.arange(0, end + 1e-9, 0.1), "#1f4e79"),
        ("雷达观测", np.arange(0, end + 1e-9, 0.1), "#2f75b5"),
        ("光电观测", np.arange(0.05, end + 1e-9, 0.2), "#70ad47"),
        ("电子侦察观测", np.arange(0, end + 1e-9, 0.2), "#ed7d31"),
        ("协同链路", np.arange(0.1, end + 1e-9, 0.3), "#8064a2"),
    ]
    yvals = np.arange(len(rows))[::-1]
    for y, (label, times, color) in zip(yvals, rows):
        ax.hlines(y, 0, end, color="#bfbfbf", linewidth=1)
        ax.scatter(times, np.full_like(times, y), s=52, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(-0.06, y, label, ha="right", va="center", fontsize=11)
    for t in np.arange(0, end + 1e-9, 0.1):
        ax.axvline(t, color="#d9e2f3", linewidth=0.45, zorder=0)
    ax.annotate("按源时间戳选择插值、保持或无效处理",
                xy=(1.18, 2), xytext=(1.0, 0.25),
                arrowprops=dict(arrowstyle="->", color="#c00000", lw=1.5),
                color="#c00000", fontsize=11)
    ax.set_xlim(-0.35, 2.05)
    ax.set_ylim(-0.6, len(rows)-0.35)
    ax.set_xlabel("时间 / s")
    ax.set_yticks([])
    ax.set_title("多源异步采样与统一时间网格示意")
    ax.grid(axis="x", alpha=0.18)
    save(fig, "01_多源异步采样与统一时间网格.png")


def association_cost_matrix() -> None:
    matrix = np.array([[5, 1, 9], [2, 8, 3], [9, 4, 1]])
    selected = {(0, 1), (1, 0), (2, 2)}
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    im = ax.imshow(matrix, cmap="YlGnBu_r", vmin=0, vmax=9)
    for i in range(3):
        for j in range(3):
            suffix = "\n选中" if (i, j) in selected else ""
            ax.text(j, i, f"{matrix[i,j]}{suffix}", ha="center", va="center",
                    fontsize=13, fontweight="bold" if suffix else "normal",
                    color="#111111")
    ax.set_xticks(range(3), ["候选量测0", "候选量测1", "候选量测2"])
    ax.set_yticks(range(3), ["历史航迹0", "历史航迹1", "历史航迹2"])
    ax.set_title("关联代价矩阵与匈牙利一对一分配示例")
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("关联代价")
    save(fig, "02_关联代价矩阵与一对一分配.png")


def gate_boundary() -> None:
    values = [1499, 1500, 1501]
    accepted = [1, 1, 0]
    colors = ["#70ad47", "#70ad47", "#c00000"]
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    ax.bar([str(v) for v in values], accepted, color=colors, width=0.56)
    for x, y, status in zip(range(3), accepted, ["接受", "接受", "拒绝"]):
        ax.text(x, max(y, 0.05) + 0.05, status, ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="#7f7f7f", linestyle="--", linewidth=1.2)
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0, 1], ["拒绝", "接受"])
    ax.set_xlabel("跨平台候选量测距离 / m")
    ax.set_title("1500 m跨平台关联门限边界验证")
    ax.text(1, 1.18, "门限采用小于等于关系", ha="center", color="#1f4e79")
    save(fig, "03_跨平台关联门限边界.png")


def history_continuity() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    x = [0, 1]
    for ax, title, ys0, ys1 in [
        (axes[0], "保留历史记忆", [0, 0], [1, 1]),
        (axes[1], "清空历史记忆", [0, 1], [1, 0]),
    ]:
        ax.plot(x, ys0, marker="o", linewidth=2.2, label="雷达航迹0")
        ax.plot(x, ys1, marker="o", linewidth=2.2, label="雷达航迹1")
        ax.set_xticks(x, ["交叉前光电编号", "交叉时光电编号"])
        ax.set_yticks([0, 1], ["局部编号0", "局部编号1"])
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[1].legend(loc="center right")
    fig.suptitle("历史关联记忆对交叉航迹身份连续性的作用", fontsize=15, fontweight="bold")
    save(fig, "04_交叉航迹历史连续性.png")


def cross_dataset_heatmap() -> None:
    data = json.loads((ROOT / "artifacts" / "metrics" / "cross_dataset_rmse.json").read_text(encoding="utf-8"))
    rows = list(data)
    cols = list(next(iter(data.values())))
    matrix = np.array([[data[r][c] for c in cols] for r in rows])
    row_labels = ["V5模型", "V6模型", "V7模型"]
    col_labels = ["V5标准集", "V6降噪集", "V7验收集", "V8复合干扰集"]
    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > 35 else "black"
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", color=color, fontsize=11)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title("研发版本跨数据集位置RMSE对照（过程性分析）")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("位置RMSE / m")
    ax.text(0.0, -0.22, "说明：不同版本并非在完全相同协议下形成，V8结果仅用于观察复合干扰下的退化。",
            transform=ax.transAxes, fontsize=10, color="#7f6000")
    save(fig, "05_研发版本跨数据集RMSE对照.png")


def maneuver_intensity() -> None:
    data = json.loads((ROOT / "artifacts" / "metrics" / "midterm_review" / "items_10_12_evidence.json").read_text(encoding="utf-8"))
    classes = data["review_item_11"]["maneuver_audit"]["classes"]
    wanted = ["CV", "CA", "CT", "CSSW"]
    mapping = {item["scenario_class"]: item["metrics"] for item in classes}
    metrics = [
        ("acceleration_magnitude_mps2", "加速度P95 / (m/s²)"),
        ("turn_rate_deg_s", "转弯角速度P95 / (°/s)"),
        ("speed_change_rate_mps2", "速度变化率P95 / (m/s²)"),
        ("equivalent_kinematic_load_factor_g", "等效运动过载P95 / g"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    colors = ["#5b9bd5", "#70ad47", "#ed7d31", "#8064a2"]
    for ax, (key, title) in zip(axes.ravel(), metrics):
        vals = [mapping[c][key]["p95"] for c in wanted]
        bars = ax.bar(wanted, vals, color=colors)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("V7四类机动强度量化（64场景分层样本）", fontsize=15, fontweight="bold")
    save(fig, "06_V7四类机动强度量化.png")


def input_quality_profile() -> None:
    data = json.loads((ROOT / "artifacts" / "metrics" / "midterm_review" / "items_10_12_evidence.json").read_text(encoding="utf-8"))
    profile = data["review_item_10"]["acceptance_profile"]
    audit = data["review_item_10"]["acceptance_dataset_empirical_audit"]["sensors"]
    sensors = ["Radar", "EO", "ESM", "FastLink"]
    cn = ["雷达", "光电", "电子侦察", "协同链路"]
    availability = [audit[s]["availability_rate"] * 100 for s in sensors]
    periods = [profile["update_period_s"].get(k, np.nan) for k in ["radar", "eo", "esm"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    bars = axes[0].bar(cn, availability, color=["#2f75b5", "#70ad47", "#ed7d31", "#8064a2"])
    for bar, v in zip(bars, availability):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+1.3, f"{v:.1f}%", ha="center")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("有效输入占比")
    axes[0].set_title("64场景样本的通道可用性")
    pcn = ["雷达", "光电", "电子侦察"]
    bars = axes[1].bar(pcn, periods, color=["#2f75b5", "#70ad47", "#ed7d31"])
    for bar, v in zip(bars, periods):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+0.004, f"{v:.1f} s", ha="center")
    axes[1].set_ylim(0, 0.24)
    axes[1].set_ylabel("配置更新周期 / s")
    axes[1].set_title("验收数据集的传感器更新周期")
    fig.suptitle("V7验收数据集输入质量条件", fontsize=15, fontweight="bold")
    save(fig, "07_V7验收数据集输入质量条件.png")


def sensor_pressure() -> None:
    data = json.loads((ROOT / "artifacts" / "metrics" / "midterm_review" / "items_1_3_evidence.json").read_text(encoding="utf-8"))
    rows = data["review_item_2"]["condition_metrics"]
    labels = {
        "baseline": "标准输入",
        "radar_suppression_70pct": "雷达量测抑制70%",
        "eo_occlusion_2s": "光电遮挡2 s",
        "esm_dropout_70pct": "电子侦察丢测70%",
    }
    xlabels = [labels[r["condition"]] for r in rows]
    rmse = [r["position_rmse_m"] for r in rows]
    retained = [r["mean_retained_measurement_ratio"] * 100 for r in rows]
    fig, ax1 = plt.subplots(figsize=(11.8, 6.2))
    x = np.arange(len(rows))
    bars = ax1.bar(x-0.18, rmse, width=0.36, color="#4472c4", label="位置RMSE")
    ax1.set_ylabel("位置RMSE / m")
    ax1.set_xticks(x, xlabels, rotation=10)
    ax2 = ax1.twinx()
    ax2.bar(x+0.18, retained, width=0.36, color="#a5a5a5", label="保留量测占比")
    ax2.set_ylabel("保留量测占比 / %")
    ax2.set_ylim(0, 110)
    for bar, v in zip(bars, rmse):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.2f}", ha="center", fontsize=9)
    h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax1.legend(h1+h2,l1+l2,loc="upper left")
    ax1.set_title("传感器信息受限条件下的误差变化（64场景分层样本）")
    ax1.grid(axis="y", alpha=0.2)
    save(fig, "08_传感器信息受限条件误差变化.png")


def evidence_hierarchy() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 7.1))
    ax.axis("off")
    boxes = [
        (0.08, 0.70, 0.84, 0.18, "完整集主批次", "V7权重 + V7注册验证集\n2000场景、6000条航迹、570000个稳定阶段有效点", "#d9eaf7"),
        (0.08, 0.42, 0.40, 0.18, "分层补充试验", "32场景关联矩阵\n64场景信息不完备与输入质量分析", "#e2f0d9"),
        (0.52, 0.42, 0.40, 0.18, "受控功能用例", "插值、注册、门限、图邻接\n模型导出一致性与运行边界", "#fff2cc"),
        (0.08, 0.14, 0.84, 0.16, "结题结论与边界", "完整集承担验收判定；补充试验解释条件变化；受控用例证明实现逻辑。\n三类证据分别引用，不把小样本结果扩展为完整集结论。", "#f2f2f2"),
    ]
    for x,y,w,h,title,body,color in boxes:
        rect=plt.Rectangle((x,y),w,h,facecolor=color,edgecolor="#404040",linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x+w/2,y+h*0.67,title,ha="center",va="center",fontsize=14,fontweight="bold")
        ax.text(x+w/2,y+h*0.32,body,ha="center",va="center",fontsize=11)
    for start,end in [((0.5,0.70),(0.28,0.60)),((0.5,0.70),(0.72,0.60)),((0.28,0.42),(0.38,0.30)),((0.72,0.42),(0.62,0.30))]:
        ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",lw=1.5,color="#5b5b5b"))
    ax.set_title("项目验证证据分层及结论使用范围", fontsize=16, fontweight="bold", pad=12)
    save(fig, "09_验证证据分层及结论使用范围.png")


def batch_execution_flow() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 6.2))
    ax.axis("off")
    labels = [
        ("固定被测对象", "V7权重、数据摘要\n场景范围与评价规则"),
        ("创建批量任务", "保存任务号、配置\n运行环境与开始时间"),
        ("执行完整性检查", "2000场景、6000航迹\n异常记录与有效点数"),
        ("独立指标复算", "位置、关联、基线\n前向时延与补充试验"),
        ("生成图表与归档", "统一口径生成图件\n保存原始记录和摘要"),
    ]
    colors = ["#d9eaf7", "#e2f0d9", "#fff2cc", "#fce4d6", "#e4dfec"]
    xs = np.linspace(0.04, 0.80, len(labels))
    for i, ((title, body), color, x) in enumerate(zip(labels, colors, xs)):
        rect = plt.Rectangle((x, 0.36), 0.16, 0.35, facecolor=color, edgecolor="#404040", linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x+0.08, 0.60, title, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x+0.08, 0.45, body, ha="center", va="center", fontsize=10)
        if i < len(labels)-1:
            ax.annotate("", xy=(xs[i+1]-0.008, 0.535), xytext=(x+0.168, 0.535),
                        arrowprops=dict(arrowstyle="->", color="#5b5b5b", lw=1.6))
    ax.text(0.5, 0.18, "任何权重、数据、场景范围或统计规则变化，均建立新的任务身份并重新生成证据。",
            ha="center", va="center", fontsize=11, color="#7f6000")
    ax.set_title("V7正式验证批次的执行与证据归档流程", fontsize=16, fontweight="bold", pad=12)
    save(fig, "10_V7正式验证批次执行流程.png")


def metric_boundary() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.8))
    panels = [
        ("位置精度", "逐点三维欧氏误差", "排除前5帧\n570000个有效点", "平方误差汇总后开方\n最终RMSE 7.488 m", "#d9eaf7"),
        ("目标关联", "正确输出对、全部输出对\n真值可关联对", "原始V7场景重算\n真值仅用于评分", "正确率99.90%\n召回率96.60%  F1 98.22%", "#e2f0d9"),
        ("模型前向时延", "张量准备完成至状态输出", "50次预热\n300次同步正式测量", "T=1 P95 20.532 ms\n不含完整软件链", "#fff2cc"),
    ]
    for ax, (title, source, scope, result, color) in zip(axes, panels):
        ax.axis("off")
        blocks = [(0.68, "统计对象", source), (0.39, "范围与分母", scope), (0.10, "输出结论", result)]
        for y, head, body in blocks:
            rect = plt.Rectangle((0.06, y), 0.88, 0.21, facecolor=color, edgecolor="#595959", linewidth=1.2)
            ax.add_patch(rect)
            ax.text(0.5, y+0.145, head, ha="center", va="center", fontsize=11, fontweight="bold")
            ax.text(0.5, y+0.065, body, ha="center", va="center", fontsize=9.5)
        ax.annotate("", xy=(0.5, 0.61), xytext=(0.5, 0.68), arrowprops=dict(arrowstyle="->", lw=1.2))
        ax.annotate("", xy=(0.5, 0.32), xytext=(0.5, 0.39), arrowprops=dict(arrowstyle="->", lw=1.2))
        ax.set_title(title, fontsize=14, fontweight="bold")
    fig.suptitle("结题主要指标的统计对象、边界与输出", fontsize=16, fontweight="bold")
    save(fig, "11_结题主要指标计算边界.png")


def identity_binding() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    ax.axis("off")
    center=(0.5,0.48)
    items=[
        (0.06,0.67,"模型身份","权重路径、SHA256\n网络配置与导出版本","#d9eaf7"),
        (0.70,0.67,"数据身份","数据集版本、摘要\n场景清单与预处理规则","#e2f0d9"),
        (0.06,0.16,"运行身份","任务号、硬件与软件\n随机种子和运行时间","#fff2cc"),
        (0.70,0.16,"结果身份","原始输出、逐航迹记录\n指标JSON与图件摘要","#fce4d6"),
    ]
    for x,y,title,body,color in items:
        rect=plt.Rectangle((x,y),0.24,0.20,facecolor=color,edgecolor="#404040",linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x+0.12,y+0.135,title,ha="center",va="center",fontsize=13,fontweight="bold")
        ax.text(x+0.12,y+0.055,body,ha="center",va="center",fontsize=10)
        ax.annotate("",xy=center,xytext=(x+0.12,y+0.10),arrowprops=dict(arrowstyle="->",lw=1.4,color="#666666"))
    core=plt.Circle(center,0.13,facecolor="#4472c4",edgecolor="#203864",linewidth=1.6)
    ax.add_patch(core)
    ax.text(center[0],center[1]+0.035,"可追溯任务对象",ha="center",va="center",fontsize=13,color="white",fontweight="bold")
    ax.text(center[0],center[1]-0.045,"模型 + 数据 + 运行 + 结果",ha="center",va="center",fontsize=10,color="white")
    ax.set_title("模型、数据、运行与结果的身份绑定关系",fontsize=16,fontweight="bold",pad=12)
    save(fig, "12_模型数据运行结果身份绑定.png")


def error_diagnosis_flow() -> None:
    fig, ax = plt.subplots(figsize=(12.7, 6.5))
    ax.axis("off")
    stages=[
        (0.05,"定位高误差对象","场景、航迹、时间片"),
        (0.25,"核对输入质量","通道有效性、缓存年龄\n关联置信度与异常标记"),
        (0.45,"区分误差来源","场景机动、注册关联\n模型估计或运行异常"),
        (0.65,"复现与对照","原任务复算、受控扰动\n模块或基线对比"),
        (0.85,"形成处置结论","数据修订、模型优化\n边界说明或工程告警"),
    ]
    for i,(x,title,body) in enumerate(stages):
        circ=plt.Circle((x,0.55),0.075,facecolor="#5b9bd5",edgecolor="#203864",linewidth=1.4)
        ax.add_patch(circ); ax.text(x,0.55,str(i+1),ha="center",va="center",fontsize=16,color="white",fontweight="bold")
        ax.text(x,0.38,title,ha="center",va="center",fontsize=12,fontweight="bold")
        ax.text(x,0.27,body,ha="center",va="center",fontsize=9.5)
        if i<len(stages)-1:
            ax.annotate("",xy=(stages[i+1][0]-0.085,0.55),xytext=(x+0.085,0.55),arrowprops=dict(arrowstyle="->",lw=1.5,color="#595959"))
    ax.text(0.5,0.08,"真实困难样本保留在总体评价中；只有输入或计算错误经确认后才修订数据或结果。",
            ha="center",fontsize=11,color="#7f6000")
    ax.set_title("高误差场景的定位、复现与处置流程",fontsize=16,fontweight="bold",pad=12)
    save(fig, "13_高误差场景诊断流程.png")


def main() -> None:
    configure_fonts()
    asynchronous_sampling()
    association_cost_matrix()
    gate_boundary()
    history_continuity()
    cross_dataset_heatmap()
    maneuver_intensity()
    input_quality_profile()
    sensor_pressure()
    evidence_hierarchy()
    batch_execution_flow()
    metric_boundary()
    identity_binding()
    error_diagnosis_flow()
    print(OUT)


if __name__ == "__main__":
    main()
