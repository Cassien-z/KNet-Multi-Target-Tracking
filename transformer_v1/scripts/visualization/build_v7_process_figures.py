"""Draw five Chinese process diagrams for the V7 completion-report draft.

The diagrams are structural illustrations, not plots of measured acceptance results.
Each is exported as a high-resolution PNG for Word and an editable SVG.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "figures" / "v7_process"
OUT.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS")
INSTALLED = {font.name for font in font_manager.fontManager.ttflist}
FONT = next((name for name in FONT_CANDIDATES if name in INSTALLED), "DejaVu Sans")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)

NAVY = "#183B56"
BLUE = "#2F6C9B"
TEAL = "#188978"
AMBER = "#BE8228"
PURPLE = "#7465A7"
INK = "#172F42"
MUTED = "#5D7282"
LINE = "#B9CBD4"
PALE_BLUE = "#EBF4FA"
PALE_TEAL = "#EAF7F3"
PALE_AMBER = "#FFF5E4"
PALE_PURPLE = "#F2EFF9"
PALE_GRAY = "#F2F6F7"


def canvas(title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    ax.text(4, 52.5, title, fontsize=25, color=NAVY, weight="bold", va="center")
    ax.text(4, 48.7, subtitle, fontsize=14.5, color=MUTED, va="center")
    ax.plot([4, 96], [46.1, 46.1], color=LINE, lw=1.1)
    return fig, ax


def box(ax, x, y, w, h, title, body="", *, fill=PALE_BLUE, edge=BLUE,
        title_size=18, body_size=13.6, radius=1.6, title_y=0.63):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.65, edgecolor=edge, facecolor=fill, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * title_y, title, fontsize=title_size,
            weight="bold", color=INK, ha="center", va="center", zorder=4)
    if body:
        ax.text(x + w / 2, y + h * 0.28, body, fontsize=body_size,
                color=MUTED, ha="center", va="center", linespacing=1.42, zorder=4)
    return patch


def arrow(ax, x1, y1, x2, y2, *, color=NAVY, lw=2.1, style="-|>",
          mutation=17, curve=0, dash=None, z=3):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=mutation,
        linewidth=lw, color=color, connectionstyle=f"arc3,rad={curve}",
        linestyle=dash or "solid", zorder=z,
    )
    ax.add_patch(patch)
    return patch


def pill(ax, x, y, w, text, *, color=BLUE, fill=PALE_BLUE, size=13.6):
    p = FancyBboxPatch((x, y), w, 3.4, boxstyle="round,pad=0.01,rounding_size=1.65",
                       linewidth=0, facecolor=fill, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + 1.7, text, ha="center", va="center",
            color=color, weight="bold", fontsize=size, zorder=3)


def footer(ax, text):
    ax.plot([4, 96], [5.5, 5.5], color=LINE, lw=0.8)
    ax.text(4, 2.9, text, fontsize=11.8, color=MUTED, va="center")


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def figure_1_route():
    fig, ax = canvas("项目总体技术路线与端到端范围", "以 V7 最终模型和注册验证集为主线，区分软件全流程与模型联合学习范围")
    stages = [
        (4, "仿真场景与观测", "多平台、多目标\n雷达／光电／电侦", PALE_GRAY, MUTED),
        (22.8, "关联与注册", "身份匹配、时空对齐\n按目标组织序列", PALE_TEAL, TEAL),
        (41.6, "融合跟踪模型", "Learned-K、双流交互\n时序建模与图协同", PALE_BLUE, BLUE),
        (60.4, "连续状态输出", "三维位置与速度\n逐目标航迹", PALE_BLUE, BLUE),
        (79.2, "验证与归档", "完整 V7 批测\n指标、图表与证据", PALE_AMBER, AMBER),
    ]
    for x, title, body, fill, edge in stages:
        box(ax, x, 21.5, 16.8, 17.5, title, body, fill=fill, edge=edge,
            title_size=15.6, body_size=11.7)
    for x in (20.9, 39.7, 58.5, 77.3):
        arrow(ax, x, 30.2, x + 1.7, 30.2, color=BLUE, mutation=13)
    pill(ax, 4.5, 40.9, 33.0, "软件全流程：结构化观测 → 航迹结果", color=TEAL, fill=PALE_TEAL, size=13)
    ax.plot([41.6, 77.2], [17.5, 17.5], color=BLUE, lw=3.0)
    ax.plot([41.6, 41.6], [17.5, 20.6], color=BLUE, lw=2.0)
    ax.plot([77.2, 77.2], [17.5, 20.6], color=BLUE, lw=2.0)
    ax.text(59.4, 14.1, "端到端模型范围：注册序列 → 状态估计", fontsize=16,
            color=BLUE, ha="center", weight="bold")
    ax.text(50, 9.8, "目标关联采用确定性处理；真值仅用于离线评估，不进入推理输入。",
            fontsize=13.5, ha="center", color=MUTED)
    footer(ax, "对应：技术总结 3.1；依据 docs/end_to_end_scope.md 与 V7 推理、注册代码绘制。")
    save(fig, "01_V7项目总体技术路线")


def _platform(ax, x, y, w, name, sensors, *, master=False):
    fill = PALE_BLUE if master else PALE_TEAL
    edge = BLUE if master else TEAL
    box(ax, x, y, w, 10.1, name, sensors, fill=fill, edge=edge,
        title_size=16.8, body_size=11.3, title_y=0.68)


def figure_2_scenario():
    fig, ax = canvas("V7 多平台协同观测场景", "单场景含 1 架长机、4 架僚机与 3 个目标；图示平台和传感器组合，不按空间比例绘制")
    ax.text(7, 41.8, "目标群", color=NAVY, fontsize=17, weight="bold")
    targets = [(39, "目标 1"), (57, "目标 2"), (75, "目标 3")]
    for x, name in targets:
        p = Circle((x, 40), radius=3.2, edgecolor=AMBER, facecolor=PALE_AMBER, lw=2.0, zorder=3)
        ax.add_patch(p)
        ax.text(x, 40, name, ha="center", va="center", fontsize=15, color=INK, weight="bold", zorder=4)
    placements = [
        (5, "僚机 1", "雷达＋光电＋电侦"),
        (23, "僚机 2", "雷达"),
        (41, "长机", "雷达＋光电＋电侦"),
        (59, "僚机 3", "雷达＋光电"),
        (77, "僚机 4", "雷达＋电侦"),
    ]
    for x, name, sensors in placements:
        _platform(ax, x, 19.1, 16.2, name, sensors, master=(name == "长机"))
    for x, tx in [(13.1, 39), (31.1, 39), (49.1, 57), (67.1, 75), (85.1, 75)]:
        arrow(ax, x, 29.5, tx, 36.8, color=AMBER, lw=1.5, mutation=11, dash=(0, (3, 3)), z=1)
    ax.plot([13.1, 85.1], [14.1, 14.1], color=TEAL, lw=2.0)
    for x in (13.1, 31.1, 67.1, 85.1):
        arrow(ax, x, 18.8, x, 14.3, color=TEAL, lw=1.6, mutation=12)
    arrow(ax, 49.1, 14.1, 49.1, 18.7, color=TEAL, lw=2.0)
    pill(ax, 36.6, 8.6, 25.0, "僚机协同信息 → 长机", color=TEAL, fill=PALE_TEAL, size=13.5)
    footer(ax, "依据 V7 数据生成脚本绘制；雷达测距测角，光电和电子侦察提供角度约束；箭头不表示实际覆盖范围。")
    save(fig, "02_V7多平台协同观测场景")


def figure_3_registration():
    fig, ax = canvas("多源观测关联与注册流程", "保留原始源时间戳，先确定目标身份，再形成统一时间网格上的模型输入")
    labels = ["雷达", "光电", "电侦"]
    colors = [BLUE, TEAL, PURPLE]
    xs = [[18, 24, 31, 36, 43], [20, 27, 35, 42], [17, 29, 39]]
    for i, (label, color, row) in enumerate(zip(labels, colors, xs)):
        y = 40.0 - i * 4.5
        ax.text(6.5, y, label, color=color, fontsize=15.5, va="center", weight="bold")
        ax.plot([17, 44], [y, y], color=LINE, lw=1.2)
        for x in row:
            ax.add_patch(Circle((x, y), 0.45, facecolor=color, edgecolor="white", lw=0.4))
    ax.text(48, 35.6, "源时间戳不齐\n局部观测缺失", fontsize=14, color=MUTED,
            va="center", linespacing=1.35)
    pill(ax, 67, 35.2, 24, "统一时间网格 · 10 Hz", color=BLUE, fill=PALE_BLUE, size=14)
    for y in (40.0, 35.5, 31.0):
        ax.plot([67, 91], [y - 3.3, y - 3.3], color=LINE, lw=0.8)
        for x in (69, 74, 79, 84, 89):
            ax.add_patch(Circle((x, y - 3.3), 0.28, facecolor=BLUE, edgecolor="none"))
    stages = [
        (4, "原始观测", "平台位姿、量测\n源时间戳与有效标志", PALE_GRAY, MUTED),
        (27, "目标关联", "同平台量测匹配\n跨平台雷达映射", PALE_TEAL, TEAL),
        (50, "时序注册", "按源时间戳对齐\n雷达野值处理", PALE_BLUE, BLUE),
        (73, "按目标成序列", "多源量测＋掩码\n位姿与数据链信息", PALE_PURPLE, PURPLE),
    ]
    for x, title, body, fill, edge in stages:
        box(ax, x, 13.7, 19.2, 13.2, title, body, fill=fill, edge=edge,
            title_size=17.6, body_size=12.4)
    for x in (23.4, 46.4, 69.4):
        arrow(ax, x, 20.2, x + 3.4, 20.2, color=BLUE, mutation=15)
    ax.text(50, 9.7, "独立审计：真值只用于关联正确率和位置误差计分，不参与目标匹配与模型推理。",
            fontsize=13.3, color=MUTED, ha="center")
    footer(ax, "对应：技术总结 3.3；依据 src/data_association/pipeline_core.py 和 track_association.py 绘制。")
    save(fig, "03_V7多源观测关联注册流程")


def figure_4_model():
    fig, ax = canvas("V7 融合跟踪模型总体架构", "输入是已关联、已注册的多源时序观测；输出为目标连续三维位置与速度")
    top = [
        (4, "注册序列输入", "雷达／光电／电侦\n掩码、位姿、数据链", PALE_GRAY, MUTED),
        (28, "Learned-K 前端", "几何与物理先验\n形成粗状态", PALE_AMBER, AMBER),
        (52, "双流特征编码", "主动雷达流\n被动光电＋电侦流", PALE_TEAL, TEAL),
        (76, "跨模态融合", "双向交叉注意力\n置信门控融合", PALE_TEAL, TEAL),
    ]
    for x, title, body, fill, edge in top:
        box(ax, x, 30.7, 19.3, 13.2, title, body, fill=fill, edge=edge,
            title_size=16.8, body_size=12.6)
    for x in (23.5, 47.5, 71.5):
        arrow(ax, x, 37.3, x + 4.3, 37.3, color=BLUE, mutation=15)
    bottom = [
        (76, "时序建模", "Transformer 编码\n使用完整时间窗口", PALE_BLUE, BLUE),
        (52, "协同 GAT", "有效平台邻接\n跨平台特征交互", PALE_PURPLE, PURPLE),
        (28, "状态残差回归", "位置残差＋速度残差", PALE_BLUE, BLUE),
        (4, "最终状态", "粗状态＋学习残差\n三维位置与速度", PALE_AMBER, AMBER),
    ]
    for x, title, body, fill, edge in bottom:
        box(ax, x, 11.8, 19.3, 13.2, title, body, fill=fill, edge=edge,
            title_size=16.6, body_size=12.4)
    arrow(ax, 85.6, 30.4, 85.6, 25.3, color=BLUE, mutation=15)
    for x in (75.7, 51.7, 27.7):
        arrow(ax, x, 18.4, x - 4.2, 18.4, color=BLUE, mutation=15)
    # Coarse-state bypass for the final residual addition.
    ax.plot([37.6, 37.6, 13.5, 13.5], [30.5, 27.5, 27.5, 25.3],
            color=AMBER, lw=1.7, linestyle=(0, (5, 3)), zorder=1)
    ax.text(26.5, 27.8, "粗状态旁路", color=AMBER, fontsize=12.8,
            va="bottom", ha="center", weight="bold")
    ax.text(50, 8.6, "说明：Transformer 使用完整序列；本图不表示模型已具备因果在线逐帧能力。",
            fontsize=13.2, color=MUTED, ha="center")
    footer(ax, "对应：技术总结 3.4；依据 src/model.py 中 CoopTrackerModel.forward 绘制。")
    save(fig, "04_V7融合跟踪模型总体架构")


def figure_5_evidence():
    fig, ax = canvas("原型软件与验证证据闭环", "把最终 V7 权重、完整验证集、批测任务、原始结果和报告图表串成可复核链条")
    steps = [
        (4, "版本锁定", "V7 权重\nV7 注册验证集", PALE_BLUE, BLUE),
        (23, "批量推理", "2000 场景\n任务 ID 与运行记录", PALE_TEAL, TEAL),
        (42, "指标计算", "位置误差／关联\n时延与补充试验", PALE_TEAL, TEAL),
        (61, "结果可视化", "中文图表\n条件与口径说明", PALE_PURPLE, PURPLE),
        (80, "结题报告", "工作与技术总结\n结论与适用边界", PALE_AMBER, AMBER),
    ]
    for x, title, body, fill, edge in steps:
        box(ax, x, 28.6, 15.7, 15.0, title, body, fill=fill, edge=edge,
            title_size=16.8, body_size=12.2)
    for x in (19.9, 38.9, 57.9, 76.9):
        arrow(ax, x, 36.1, x + 2.8, 36.1, color=BLUE, mutation=14)
    ax.text(4, 23.8, "软件执行", fontsize=16.2, weight="bold", color=NAVY)
    pill(ax, 17, 21.9, 66.0, "前端选择与展示  →  后端任务调度  →  模型推理与评估服务", color=BLUE,
         fill=PALE_BLUE, size=15)
    ax.text(4, 16.2, "证据归档", fontsize=16.2, weight="bold", color=NAVY)
    evidence = [
        (17, "模型／数据摘要"),
        (38, "逐航迹结果与日志"),
        (59, "独立指标证据"),
        (80, "图表及引用索引"),
    ]
    for x, label in evidence:
        pill(ax, x, 13.9, 18.2, label, color=TEAL, fill=PALE_TEAL, size=12.5)
    for x in (11.85, 30.85, 49.85, 68.85, 87.85):
        arrow(ax, x, 28.4, x, 25.6, color=LINE, lw=1.3, mutation=10)
    ax.text(50, 9.3, "复核时以任务 ID 回溯模型、数据、统计口径和图表来源；真值仅用于离线计分。",
            fontsize=13.5, color=MUTED, ha="center")
    footer(ax, "对应：技术总结 3.5—3.6；依据原型软件批测、评估与报告服务绘制。")
    save(fig, "05_V7原型软件验证证据闭环")


def _local_motion(path: Path):
    """Return target-0 truth in initial-heading coordinates and speed change."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    track = payload["RegisteredTrack"][0]
    position = np.asarray(track["target_pos"], dtype=float)
    velocity = np.asarray(track["target_vel"], dtype=float)
    origin = position[0]
    lon = np.arctan2(origin[1], origin[0])
    lat = np.arctan2(origin[2], np.hypot(origin[0], origin[1]))
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    displacement = position - origin
    horizontal = np.column_stack((displacement @ east, displacement @ north))
    heading = horizontal[5] - horizontal[0]
    heading /= max(float(np.linalg.norm(heading)), 1e-12)
    along = horizontal @ heading
    across = horizontal @ np.array([-heading[1], heading[0]])
    speed = np.linalg.norm(velocity, axis=1)
    t_grid = np.asarray(payload["t_grid"], dtype=float)
    time_s = t_grid - t_grid[0]
    return along, across, speed - speed[0], time_s


def figure_6_maneuvers():
    samples = [
        ("CV 匀速", "track_cv_0000.pkl", BLUE),
        ("CA 加速", "track_ca_0004.pkl", TEAL),
        ("CT 转弯", "track_ct_0000.pkl", AMBER),
        ("CSSW 复合机动", "track_cssw_0002.pkl", PURPLE),
    ]
    fig = plt.figure(figsize=(13.2, 7.4), facecolor="white")
    fig.text(0.045, 0.945, "V7 四类目标机动样本", fontsize=25, color=NAVY, weight="bold", va="top")
    fig.text(0.045, 0.865, "从注册验证集读取目标真值；上排为水平轨迹，下排为相对初速度变化。各子图独立缩放。",
             fontsize=14.5, color=MUTED, va="top")
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.815, 0.815], transform=fig.transFigure, color=LINE, lw=1.1))
    for idx, (label, filename, color) in enumerate(samples):
        source = ROOT / "data" / "val_data_v7_registered" / filename
        along, across, delta_speed, time_s = _local_motion(source)
        x = 0.06 + idx * 0.235
        trajectory = fig.add_axes([x, 0.43, 0.205, 0.30])
        trajectory.plot(along / 1000, across / 1000, color=color, lw=2.5)
        trajectory.scatter([along[0] / 1000], [across[0] / 1000], s=42, color=TEAL, zorder=5)
        trajectory.scatter([along[-1] / 1000], [across[-1] / 1000], s=54, marker="^", color=color, zorder=5)
        trajectory.set_title(label, fontsize=17.2, color=INK, weight="bold", pad=10)
        trajectory.set_xlabel("前向位移 / 千米", fontsize=12.5, color=MUTED)
        if idx == 0:
            trajectory.set_ylabel("侧向位移 / 千米", fontsize=12.5, color=MUTED)
        trajectory.grid(color="#DCE7EC", lw=0.7)
        trajectory.tick_params(labelsize=11.6, colors=MUTED)
        trajectory.spines[["top", "right"]].set_visible(False)
        extent = float(np.ptp(across / 1000))
        if extent < 0.04:
            trajectory.set_ylim(-0.025, 0.025)
        else:
            margin = max(extent * 0.14, 0.015)
            trajectory.set_ylim(float(np.min(across / 1000) - margin), float(np.max(across / 1000) + margin))
        speed_ax = fig.add_axes([x, 0.17, 0.205, 0.17])
        speed_ax.plot(time_s, delta_speed, color=color, lw=2.4)
        speed_ax.axhline(0, color=LINE, lw=0.85)
        speed_ax.set_xlim(float(time_s[0]), float(time_s[-1]))
        speed_ax.set_xlabel("时间 / 秒", fontsize=12.5, color=MUTED)
        if idx == 0:
            speed_ax.set_ylabel("速度变化 / 米每秒", fontsize=12.0, color=MUTED)
        speed_ax.grid(color="#E4ECEF", lw=0.7)
        speed_ax.tick_params(labelsize=11.6, colors=MUTED)
        speed_ax.spines[["top", "right"]].set_visible(False)
        speed_range = float(np.ptp(delta_speed))
        if speed_range < 1:
            speed_ax.set_ylim(-5, 5)
        else:
            margin = max(speed_range * 0.16, 3)
            speed_ax.set_ylim(float(np.min(delta_speed) - margin), float(np.max(delta_speed) + margin))
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.095, 0.095], transform=fig.transFigure, color=LINE, lw=0.8))
    fig.text(0.045, 0.055, "样本：CV 0000、CA 0004、CT 0000、CSSW 0002；仅说明机动形态，不用于跟踪精度评价。",
             fontsize=11.8, color=MUTED, va="center")
    save(fig, "06_V7四类目标机动样本")


def figure_7_dual_stream():
    fig, ax = canvas("主动与被动观测的双流交互", "V7 模型将雷达约束与光电、电子侦察角度信息分别编码，再做跨模态融合")
    ax.text(4, 39.4, "主动流", fontsize=16, color=BLUE, weight="bold", va="center")
    ax.text(4, 21.0, "被动流", fontsize=16, color=TEAL, weight="bold", va="center")
    upper = [
        (15, "雷达输入", "距离＋方位／俯仰\n几何位置＋粗状态", PALE_BLUE, BLUE),
        (36, "掩码编码", "无效量测抑制\n生成雷达特征", PALE_BLUE, BLUE),
        (57, "自适应对齐", "特征归一化\n姿态信息调制", PALE_BLUE, BLUE),
    ]
    lower = [
        (15, "光电／电侦", "角度与世界方向\n分别编码、保留掩码", PALE_TEAL, TEAL),
        (36, "置信融合", "有效源动态加权\n缺失时使用占位特征", PALE_TEAL, TEAL),
        (57, "自适应对齐", "特征归一化\n姿态信息调制", PALE_TEAL, TEAL),
    ]
    for x, title, body, fill, edge in upper:
        box(ax, x, 32.0, 17.5, 13.0, title, body, fill=fill, edge=edge,
            title_size=16.2, body_size=11.5)
    for x, title, body, fill, edge in lower:
        box(ax, x, 13.5, 17.5, 13.0, title, body, fill=fill, edge=edge,
            title_size=16.2, body_size=11.5)
    for x in (32.7, 53.7):
        arrow(ax, x, 38.5, x + 3.1, 38.5, color=BLUE, mutation=14)
        arrow(ax, x, 20.0, x + 3.1, 20.0, color=TEAL, mutation=14)
    box(ax, 79, 23.0, 17.0, 13.0, "交叉注意力", "双向查询上下文\n融合门控动态加权",
        fill=PALE_PURPLE, edge=PURPLE, title_size=15.3, body_size=11.1)
    arrow(ax, 74.6, 38.5, 78.8, 31.8, color=BLUE, mutation=14)
    arrow(ax, 74.6, 20.0, 78.8, 27.0, color=TEAL, mutation=14)
    pill(ax, 74.2, 14.5, 22.2, "融合特征 → 时序模块", color=PURPLE, fill=PALE_PURPLE, size=12.8)
    arrow(ax, 87.5, 22.8, 87.5, 18.2, color=PURPLE, mutation=14)
    ax.text(50, 9.2, "两路交叉注意力分别以雷达和被动特征为查询；融合门控生成随输入变化的权重。",
            fontsize=13.0, color=MUTED, ha="center")
    footer(ax, "对应：技术总结 3.4；依据 src/model.py 的通道掩码编码、被动源置信融合与双向交叉注意力绘制。")
    save(fig, "07_V7主动被动双流交互")


def figure_8_temporal_gat():
    fig, ax = canvas("时序建模与跨平台图协同", "先在每个平台的完整时间窗口内提取机动特征，再于同一时刻交换有效平台信息")
    box(ax, 4, 34.1, 40.0, 8.4, "阶段一：逐平台时序编码", "",
        fill=PALE_BLUE, edge=BLUE, title_size=18, title_y=0.5)
    box(ax, 56, 34.1, 40.0, 8.4, "阶段二：同一时刻图协同", "",
        fill=PALE_PURPLE, edge=PURPLE, title_size=18, title_y=0.5)
    # A matrix of source-valid feature tokens: rows are platforms, columns are time steps.
    for i in range(5):
        y = 30.5 - i * 4.2
        ax.text(6.0, y, f"平台 {i + 1}", fontsize=12.5, color=MUTED, va="center")
        for j in range(6):
            x = 17.7 + j * 4.3
            missing = (i, j) in {(1, 2), (2, 4), (4, 1)}
            fill = "#DCE5E8" if missing else BLUE
            rect = FancyBboxPatch((x, y - 1.25), 3.0, 2.5,
                                  boxstyle="round,pad=0.0,rounding_size=0.4",
                                  linewidth=0, facecolor=fill, zorder=2)
            ax.add_patch(rect)
    pill(ax, 4.7, 5.8, 39.0, "Transformer：完整窗口编码", color=BLUE, fill=PALE_BLUE, size=14.0)
    arrow(ax, 45, 21.3, 55, 21.3, color=NAVY, lw=2.4, mutation=18)
    nodes = [(76, 29.5), (64, 23.0), (69, 13.3), (84, 13.3), (89, 23.0)]
    edges = [(0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4), (0, 3)]
    for a, b in edges:
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        ax.plot([x1, x2], [y1, y2], color="#B2A8CC", lw=1.9, zorder=1)
    for i, (x, y) in enumerate(nodes):
        circle = Circle((x, y), 2.55, facecolor=PALE_PURPLE if i else PALE_BLUE,
                        edgecolor=PURPLE if i else BLUE, linewidth=2.0, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, "长机" if i == 0 else f"僚{i}", color=INK, fontsize=12.6,
                weight="bold", ha="center", va="center", zorder=4)
    pill(ax, 55.5, 5.8, 40.5, "有效源＋近邻关系 → GAT → 聚合", color=PURPLE,
         fill=PALE_PURPLE, size=13.4)
    ax.text(50, 2.2, "邻接条件：传感器有效、平台间距不超过 20 千米，并保留自环；完整窗口编码不等于因果在线推理。",
            fontsize=11.5, color=MUTED, ha="center")
    save(fig, "08_V7时序建模与图协同")


def figure_9_scene_3d():
    """Plot an actual V7 registered scene in the spirit of midterm Figure 19."""
    scene = ROOT / "data" / "val_data_v7_registered" / "track_cssw_0002.pkl"
    with scene.open("rb") as handle:
        payload = pickle.load(handle)
    platforms = np.asarray(payload["UAV_Pos"], dtype=float)
    targets = np.stack(
        [np.asarray(track["target_pos"], dtype=float) for track in payload["RegisteredTrack"]],
        axis=1,
    )
    origin = platforms[0, 0]
    lon = np.arctan2(origin[1], origin[0])
    lat = np.arctan2(origin[2], np.hypot(origin[0], origin[1]))
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    basis = np.stack((east, north, up), axis=0)
    platform_enu = (platforms - origin) @ basis.T / 1000.0
    target_enu = (targets - origin) @ basis.T / 1000.0

    fig = plt.figure(figsize=(13.2, 7.4), facecolor="white")
    fig.text(0.045, 0.945, "V7 多平台多目标三维场景示例", fontsize=25,
             color=NAVY, weight="bold", va="top")
    fig.text(0.045, 0.865, "由 V7 注册验证集 CSSW 0002 场景重建（100 帧、10 Hz）；彩色实线为目标真值，虚线为平台轨迹。",
             fontsize=14.2, color=MUTED, va="top")
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.815, 0.815], transform=fig.transFigure,
                              color=LINE, lw=1.1))
    ax = fig.add_axes([0.035, 0.13, 0.72, 0.68], projection="3d")
    target_colors = [BLUE, AMBER, PURPLE]
    for i, color in enumerate(target_colors):
        path = target_enu[:, i, :]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, lw=2.7, label=f"目标 {i + 1}")
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], color=color, s=23, marker="o")
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color=color, s=43, marker="^")
    platform_colors = [NAVY, "#6E8798", "#8297A5", "#9CAEB8", "#B4C3CA"]
    for i, color in enumerate(platform_colors):
        path = platform_enu[:, i, :]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color,
                lw=2.0 if i == 0 else 1.35, ls="--", alpha=0.96 if i == 0 else 0.83,
                label="长机" if i == 0 else f"僚机 {i}")
    ax.set_xlabel("东向位移 / 千米", labelpad=13, fontsize=11.6, color=INK)
    ax.set_ylabel("北向位移 / 千米", labelpad=13, fontsize=11.6, color=INK)
    ax.set_zlabel("高度差 / 千米", labelpad=11, fontsize=11.6, color=INK)
    ax.tick_params(labelsize=10.8, colors=MUTED, pad=1)
    ax.view_init(elev=23, azim=-57)
    ax.set_box_aspect((1.0, 1.35, 0.56))
    ax.grid(True, color="#DCE7EC", alpha=0.7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#F7FAFB")
        axis.pane.set_edgecolor("#E2E9ED")

    legend_ax = fig.add_axes([0.76, 0.21, 0.19, 0.54])
    legend_ax.axis("off")
    legend_ax.text(0.0, 0.98, "轨迹图例", fontsize=17, color=NAVY,
                   weight="bold", va="top", transform=legend_ax.transAxes)
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(-0.02, 0.87),
                     frameon=False, fontsize=12.5, handlelength=2.8, labelspacing=0.72)
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.095, 0.095], transform=fig.transFigure,
                              color=LINE, lw=0.8))
    fig.text(0.045, 0.055, "图为真实 V7 仿真场景几何关系，起点统一到长机初始位置；高度轴为增强辨识而调整显示比例。",
             fontsize=11.7, color=MUTED, va="center")
    save(fig, "09_V7多平台多目标三维场景")


def main():
    figure_1_route()
    figure_2_scenario()
    figure_3_registration()
    figure_4_model()
    figure_5_evidence()
    figure_6_maneuvers()
    figure_7_dual_stream()
    figure_8_temporal_gat()
    figure_9_scene_3d()
    print(f"Created nine PNG and nine SVG figures in {OUT}")
    print(f"Chinese font: {FONT}")


if __name__ == "__main__":
    main()
