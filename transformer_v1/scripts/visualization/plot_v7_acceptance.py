"""Draw Chinese, source-checked acceptance figures from one complete V7 batch."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.acceptance_common import job_signature, load_batch_report


MODES = ("CV", "CA", "CT", "CSSW")
MODE_NAMES = {"CV": "匀速 CV", "CA": "匀加速 CA", "CT": "协调转弯 CT", "CSSW": "复合机动 CSSW"}
SENSOR_NAMES = {"Radar": "雷达", "EO": "光电", "ESM": "电子侦察", "FastLink": "快速链路", "SlowLink": "慢速链路"}
COLORS = ("#3274A1", "#5B9D85", "#C68938", "#9A6AB0")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _font() -> None:
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = name
            break
    else:
        raise RuntimeError("未找到中文字体；无法生成中文验收图")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["savefig.facecolor"] = "white"


def _mode(name: str) -> str:
    match = re.search(r"track[_-](cv|ca|ct|cssw)(?:[_-]|\.)", name, re.I)
    if not match:
        raise ValueError(f"无法识别机动类别：{name}")
    return match.group(1).upper()


def _weighted_rmse(records: list[dict]) -> float:
    count = sum(int(item["valid_frames"]) for item in records)
    if count <= 0:
        raise ValueError("无有效评估帧")
    squared = sum(float(item["rmse_m"]) ** 2 * int(item["valid_frames"]) for item in records)
    return float(np.sqrt(squared / count))


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _check(report: dict, association: dict, baseline: dict, latency: dict) -> None:
    config = report["config"]
    metrics = report["result"]["metrics"]
    manifest = report["result"].get("scenario_manifest", {})
    if config.get("model_id") != "v7" or config.get("dataset_id") != "v7":
        raise ValueError("验收图仅支持V7模型与V7数据集")
    if config.get("max_scenarios") is not None or config.get("warmup_frames") != 5:
        raise ValueError("验收图要求V7完整集且每条航迹排除前5帧")
    if metrics.get("evaluated_scenarios") != 2000 or manifest.get("class_counts") != dict.fromkeys(MODES, 500):
        raise ValueError("验收批次没有覆盖四类机动各500个场景")
    signature = job_signature(report)
    for name, evidence in (("关联", association), ("基线", baseline), ("时延", latency)):
        if evidence.get("job_id") != report["id"] or evidence.get("job_signature") != signature:
            raise ValueError(f"{name}证据与批测任务签名不一致")
    if latency.get("model_sha256") != report["result"]["provenance"]["model_sha256"]:
        raise ValueError("时延测试权重与批测权重不同")
    for field, sequence_length in (("one_frame_input", 1), ("full_sequence_input", 100)):
        measurement = latency.get(field, {})
        samples = np.asarray(measurement.get("samples_ms", []), dtype=float)
        if (
            measurement.get("sequence_length") != sequence_length
            or samples.size != measurement.get("test_iterations")
            or samples.size == 0
            or not np.all(np.isfinite(samples))
            or np.any(samples <= 0)
            or not np.isclose(np.percentile(samples, 95), measurement.get("p95_ms", np.nan), atol=1e-6)
        ):
            raise ValueError(f"{field}原始计时样本与统计值不一致")
    if not association.get("valid_for_assessment") or not baseline.get("valid_for_assessment"):
        raise ValueError("关联或基线证据不可用于指标评估")


def build(report_path: Path, output_dir: Path) -> dict:
    report = load_batch_report(report_path)
    job_id = str(report["id"])
    evidence_dir = PROJECT_ROOT / "artifacts" / "metrics" / "acceptance" / job_id
    association = _load(evidence_dir / "association_accuracy.json")
    baseline = _load(evidence_dir / "rmse_improvement_pct.json")
    latency = _load(evidence_dir / "forward_latency.json")
    _check(report, association, baseline, latency)
    _font()
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = report["result"]["tracks"]
    groups = {mode: [item for item in tracks if _mode(item["track"]) == mode] for mode in MODES}
    overall = _weighted_rmse(tracks)
    if not np.isclose(overall, report["result"]["metrics"]["position_rmse_m"], atol=1e-5):
        raise ValueError("逐航迹RMSE无法重建批测总体RMSE")
    group_rmse = [_weighted_rmse(groups[mode]) for mode in MODES]
    outputs = []

    # 1. Metric that the contract actually specifies: point-weighted 3-D RMSE.
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    labels = [MODE_NAMES[mode] for mode in MODES] + ["V7 总体"]
    values = group_rmse + [overall]
    bars = ax.bar(labels, values, color=[*COLORS, "#245C83"], width=0.62)
    ax.axhline(8.0, color="#B4413E", linestyle="--", linewidth=1.8, label="合同上限 8 米")
    ax.set_ylabel("三维位置 RMSE（米）")
    ax.set_title("V7 完整验证集跟踪精度")
    ax.set_ylim(0, max(9.7, max(values) * 1.22))
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.3f}", ha="center")
    ax.legend(frameon=False)
    fig.text(0.5, -0.01, "2000个场景、6000条航迹；每条排除前5帧；合同8米指标针对总体RMSE，四类机动各500场景", ha="center", fontsize=9, color="#555")
    path = output_dir / "01_V7分机动与总体位置精度.png"; _save(fig, path); outputs.append(path.name)

    # 2. Track distribution is explanatory; the contract is based on overall point RMSE.
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    data = [[float(item["rmse_m"]) for item in groups[mode]] for mode in MODES]
    box = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.53,
                     medianprops={"color": "#263238", "linewidth": 1.8})
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.72)
    ax.set_xticks(range(1, 5), [f"{MODE_NAMES[mode]}\n{len(groups[mode])}条航迹" for mode in MODES])
    ax.set_ylabel("逐航迹 RMSE（米）")
    ax.set_title("不同机动类型的航迹误差分布")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    fig.text(0.5, -0.01, "箱体为四分位范围；须用上一图的总体有效点RMSE判定8米指标", ha="center", fontsize=9, color="#555")
    path = output_dir / "02_V7逐航迹误差分布.png"; _save(fig, path); outputs.append(path.name)

    # 3. Rebuild the baseline comparison by maneuver from the same file manifest.
    baseline_files = baseline["per_file"]
    if len(baseline_files) != 2000 or int(baseline["evaluated_points"]) != int(report["result"]["metrics"]["evaluated_points"]):
        raise ValueError("IMM-JPDA基线与V7完整批次的文件数或有效点数不一致")
    baseline_groups = {mode: [item for item in baseline_files if _mode(item["file"]) == mode] for mode in MODES}
    baseline_rmse = []
    for mode in MODES:
        items = baseline_groups[mode]
        count = sum(int(x["evaluated_points"]) for x in items)
        baseline_rmse.append(float(np.sqrt(sum(float(x["rmse_m"])**2 * int(x["evaluated_points"]) for x in items) / count)))
    overall_baseline = float(baseline["imm_jpda_rmse_m"])
    if not np.isclose(overall_baseline, np.sqrt(sum(float(x["rmse_m"])**2 * int(x["evaluated_points"]) for x in baseline_files) / baseline["evaluated_points"]), atol=1e-5):
        raise ValueError("逐场景基线RMSE无法重建总体基线RMSE")
    gains = [(base - model) / base * 100 for model, base in zip(group_rmse + [overall], baseline_rmse + [overall_baseline])]
    fig, ax = plt.subplots(figsize=(9.3, 5.1))
    bars = ax.bar(labels, gains, color=[*COLORS, "#245C83"], width=0.63)
    ax.axhline(15.0, color="#B4413E", linestyle="--", linewidth=1.8)
    ax.set_ylim(0, max(100, max(gains) * 1.1))
    ax.set_ylabel("相对 IMM-JPDA 的 RMSE 降幅（%）")
    ax.set_title("V7 模型与 IMM-JPDA 参考实现对比")
    ax.grid(axis="y", alpha=0.18); ax.set_axisbelow(True)
    for bar, gain, model, base in zip(bars, gains, group_rmse + [overall], baseline_rmse + [overall_baseline]):
        ax.text(bar.get_x() + bar.get_width()/2, gain + 1.4, f"{gain:.1f}%", ha="center")
        ax.text(bar.get_x() + bar.get_width()/2, 3.0, f"{model:.1f}/{base:.1f}米", ha="center", color="white", fontsize=8)
    fig.text(0.5, -0.02,
             "红虚线为合同下限15%；柱内为V7/基线RMSE；双方场景和统计范围相同\n"
             "基线仅用雷达与快链路、以真值初始化，传感器输入不完全相同",
             ha="center", fontsize=9, color="#555")
    path = output_dir / "03_V7同场景基线对比.png"; _save(fig, path); outputs.append(path.name)

    # 4. Association overall and per sensor; only overall is the contract criterion.
    summary = association["association"]
    sensor = summary["per_sensor"]
    sensor_rows = [(name, item) for name, item in sensor.items() if item.get("accuracy") is not None]
    sensor_rows.sort(key=lambda pair: pair[0])
    assoc_labels = ["总体"] + [SENSOR_NAMES.get(name, name) for name, _ in sensor_rows]
    assoc_values = [float(association["value"]) * 100] + [float(item["accuracy"]) * 100 for _, item in sensor_rows]
    assoc_counts = [int(summary["total"])] + [int(item["total"]) for _, item in sensor_rows]
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    bars = ax.bar(assoc_labels, assoc_values, color=["#245C83"] + ["#5B9D85"] * len(sensor_rows), width=0.62)
    ax.axhline(85, color="#B4413E", linestyle="--", linewidth=1.8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("正确关联率（%）")
    ax.set_title("V7 完整验证集目标关联审计（合同总体阈值85%）")
    ax.grid(axis="y", alpha=0.18); ax.set_axisbelow(True)
    for bar, value, count in zip(bars, assoc_values, assoc_counts):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.6, f"{value:.2f}%\n（{count}次）", ha="center", fontsize=9)
    fig.text(0.5, -0.01, "真值仅用于关联完成后的审计；合同85%阈值针对总体关联率，分传感器数据用于解释", ha="center", fontsize=9, color="#555")
    path = output_dir / "04_V7目标关联正确率.png"; _save(fig, path); outputs.append(path.name)

    # 5. Two separate measured forward shapes, both with raw timing samples.
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    timing_cases = (("单帧输入 T=1", latency["one_frame_input"], "#3274A1"),
                    ("完整序列输入 T=100", latency["full_sequence_input"], "#9A6AB0"))
    for ax, (title, item, color) in zip(axes, timing_cases):
        values = np.asarray(item["samples_ms"], dtype=float)
        ax.hist(values, bins=25, color=color, alpha=0.82, edgecolor="white")
        ax.axvline(item["p95_ms"], color="#B4413E", linewidth=2, label=f"P95 {item['p95_ms']:.1f} 毫秒")
        if item["sequence_length"] == 100:
            ax.axvline(100, color="#333", linestyle="--", linewidth=1.5, label="100 毫秒")
        ax.set_title(title)
        ax.set_xlabel("模型前向耗时（毫秒）")
        ax.set_xlim(0, max(35 if item["sequence_length"] == 1 else 110, float(np.max(values)) * 1.1))
        ax.grid(axis="y", alpha=0.15); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("测试次数")
    fig.suptitle("V7 模型前向时间分布", fontsize=14)
    fig.text(0.5, -0.02, "同一V7权重和注册样本；每种输入先预热50次，再计时300次；不含读取、目标关联、界面与通信", ha="center", fontsize=9, color="#555")
    path = output_dir / "05_V7模型前向时间.png"; _save(fig, path); outputs.append(path.name)

    # 6. Supplemental stress evidence with explicit limits, not part of 8 m acceptance.
    stress = _load(PROJECT_ROOT / "artifacts" / "metrics" / "midterm_review" / "items_1_3_evidence.json")
    if stress["model"]["sha256"] != report["result"]["provenance"]["model_sha256"]:
        raise ValueError("扰动实验使用了不同权重")
    conditions = stress["review_item_3"]["conditions"]
    names = {"baseline": "原始V7样本", "eo_channel_missing": "光电缺失", "random_measurement_drop_20pct": "随机丢测20%",
             "all_source_occlusion_0_5s": "全源遮挡0.5秒", "datalink_delay_0_2s": "链路追加延迟0.2秒"}
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    stress_labels = [names.get(item["condition"], item["condition"]) for item in conditions]
    stress_values = [float(item["position_rmse_m"]) for item in conditions]
    bars = ax.barh(stress_labels, stress_values, color=["#3274A1"] + ["#C68938"] * (len(conditions)-1))
    ax.invert_yaxis(); ax.set_xlim(0, max(stress_values) * 1.28)
    ax.set_xlabel("三维位置 RMSE（米）")
    ax.set_title("V7 模型在信息不完备条件下的误差变化")
    ax.grid(axis="x", alpha=0.18); ax.set_axisbelow(True)
    for bar, value in zip(bars, stress_values):
        ax.text(value + 0.16, bar.get_y() + bar.get_height()/2, f"{value:.2f}", va="center")
    fig.text(0.5, -0.01, "补充扰动实验：64个分层场景；模型权重不变；该图不属于V7标准质量条件下的8米验收判定", ha="center", fontsize=9, color="#555")
    path = output_dir / "06_V7信息不完备扰动实验.png"; _save(fig, path); outputs.append(path.name)

    # 7. Stratified association audit under deterministic V7 input perturbations.
    interference = _load(PROJECT_ROOT / "artifacts" / "metrics" / "midterm_review" / "items_4_6_evidence.json")
    audit = interference["review_item_6"]
    if audit["scenario_count"] != 32 or not audit["association_matrix"]["all_cells_passed"]:
        raise ValueError("关联扰动审计不完整")
    condition_names = {
        "baseline": "原始条件", "radar_blinking": "雷达闪烁", "radar_soft_suppression_70pct": "雷达软压制",
        "radar_range_smearing_100m": "雷达距离抖动", "composite_interference": "复合干扰",
    }
    combination_names = {"radar_only": "仅雷达", "radar_eo": "雷达＋光电",
                         "radar_esm": "雷达＋电子侦察", "radar_eo_esm": "雷达＋光电＋电子侦察"}
    cells = {(item["scenario_class"], item["condition"]): item["sensor_combinations"]
             for item in audit["association_matrix"]["cells"]}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.15, top=0.90, hspace=0.55, wspace=0.25)
    for ax, (combo, title) in zip(axes.flat, combination_names.items()):
        matrix = np.array([[100 * cells[(mode, condition)][combo]["accuracy"]
                            for condition in audit["conditions"]] for mode in MODES])
        picture = ax.imshow(matrix, cmap="YlGn", vmin=85, vmax=100, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(5), [condition_names[name] for name in audit["conditions"]], rotation=20, ha="right")
        ax.set_yticks(range(4), [MODE_NAMES[mode] for mode in MODES])
        ax.tick_params(labelsize=9)
        for row in range(4):
            for column in range(5):
                ax.text(column, row, f"{matrix[row,column]:.1f}%", ha="center", va="center", fontsize=8,
                        color="white" if matrix[row,column] >= 96 else "#1E3A2A")
    colorbar_axis = fig.add_axes([0.905, 0.21, 0.018, 0.62])
    fig.colorbar(picture, cax=colorbar_axis, label="正确关联率（%）")
    fig.suptitle("V7场景扰动下的分层关联正确率", fontsize=15)
    fig.text(0.5, 0.02, "32个分层场景 × 5种输入条件 × 4种传感器组合，共80个判定单元；内存扰动，不改变原始V7数据", ha="center", fontsize=9, color="#555")
    path = output_dir / "07_V7干扰与传感器组合关联矩阵.png"; _save(fig, path); outputs.append(path.name)

    # 8. V7-only component ablation tied to the same complete batch.
    ablation = _load(evidence_dir / "ablation_v7.json")
    if ablation.get("job_signature") != job_signature(report) or ablation.get("model_sha256") != report["result"]["provenance"]["model_sha256"]:
        raise ValueError("V7消融结果与最终批次不匹配")
    ablation_modes = ("full", "dual_no_gat", "radar_only", "no_temporal")
    ablation_names = ("完整模型", "关闭图协同", "仅雷达主流", "关闭时序模块")
    ablation_values = [float(ablation["results"][mode]["position_rmse_m"]) for mode in ablation_modes]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    bars = ax.bar(ablation_names, ablation_values, color=["#245C83", "#C68938", "#5B9D85", "#9A6AB0"], width=0.6)
    ax.axhline(8.0, color="#B4413E", linestyle="--", linewidth=1.5, label="完整模型总体指标上限 8 米")
    ax.set_ylim(0, max(ablation_values) * 1.15)
    ax.set_ylabel("三维位置 RMSE（米）")
    ax.set_title("V7完整验证集模型模块消融")
    ax.grid(axis="y", alpha=0.18); ax.set_axisbelow(True)
    for bar, value in zip(bars, ablation_values):
        ax.text(bar.get_x()+bar.get_width()/2, value + max(ablation_values)*0.015, f"{value:.2f}", ha="center")
    ax.legend(frameon=False)
    fig.text(0.5, -0.01, "同一V7权重和2000个场景；推理时关闭模块或输入，不重新训练消融模型", ha="center", fontsize=9, color="#555")
    path = output_dir / "08_V7模型模块消融.png"; _save(fig, path); outputs.append(path.name)

    summary_path = output_dir / "验证图数据摘要.json"
    summary_path.write_text(json.dumps({
        "job_id": job_id, "job_signature": job_signature(report),
        "model_sha256": report["result"]["provenance"]["model_sha256"],
        "dataset_sha256": report["result"]["provenance"]["dataset_sha256"],
        "scenarios": report["result"]["metrics"]["evaluated_scenarios"],
        "tracks": report["result"]["metrics"]["evaluated_tracks"],
        "evaluated_points": report["result"]["metrics"]["evaluated_points"],
        "position_rmse_m": overall, "position_rmse_by_mode_m": dict(zip(MODES, group_rmse)),
        "association_accuracy": association["value"],
        "imm_jpda_rmse_m": overall_baseline, "rmse_improvement_pct": baseline["value"],
        "baseline_measurement_sources": baseline.get("settings", {}).get("measurement_sources"),
        "baseline_truth_initialization": True,
        "one_frame_p95_ms": latency["one_frame_input"]["p95_ms"],
        "full_sequence_p95_ms": latency["full_sequence_input"]["p95_ms"],
        "latency_load_condition": latency.get("load_condition"),
        "minimum_perturbed_association_cell_accuracy": audit["association_matrix"]["minimum_cell_accuracy"],
        "ablation_rmse_m": dict(zip(ablation_modes, ablation_values)),
        "figures": outputs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir), "figures": outputs, "summary": str(summary_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-report", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    job = load_batch_report(args.batch_report)
    output = args.output_dir or PROJECT_ROOT / "artifacts" / "reports" / "v7_final_validation" / job["id"]
    print(json.dumps(build(args.batch_report, output), ensure_ascii=False, indent=2))
