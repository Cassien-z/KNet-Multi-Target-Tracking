"""Recompute supplementary report figures from the locked V7 evidence."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
JOB = "2ea4f9eb0f2845b599fcf4837d71a018"
REPORT = ROOT / "artifacts" / "reports" / "batch_tests" / f"{JOB}.json"
METRICS = ROOT / "artifacts" / "metrics" / "acceptance" / JOB
OUTPUT = ROOT / "artifacts" / "figures" / "v7_report_expansion"
MODEL_SHA256 = "d22427274a8b680c477f2e7074d91efd03a89a5347f33862a6c6de7ce35dc51a"
ORDER = ("CV", "CA", "CT", "CSSW")
LABELS = {"CV": "CV 匀速", "CA": "CA 加速", "CT": "CT 转弯", "CSSW": "CSSW 复合机动"}
COLORS = {"CV": "#247BA0", "CA": "#2A9D8F", "CT": "#E9A23B", "CSSW": "#C05746"}


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    batch = json.loads(REPORT.read_text(encoding="utf-8"))
    result = batch["result"]
    assert batch["id"] == JOB and result["provenance"]["model_sha256"] == MODEL_SHA256
    assert result["metrics"]["evaluated_scenarios"] == 2000
    assert result["metrics"]["evaluated_tracks"] == 6000
    assert result["metrics"]["evaluated_points"] == 570000
    assert result["scenario_manifest"]["class_counts"] == {"CA": 500, "CSSW": 500, "CT": 500, "CV": 500}

    by_scene: dict[str, list[float]] = defaultdict(list)
    by_class: dict[str, list[float]] = defaultdict(list)
    for track in result["tracks"]:
        key = track["track"].split(".pkl_target_")[0]
        cls = key.split("_")[1].upper()
        assert cls in ORDER and track["valid_frames"] == 95
        value = float(track["rmse_m"])
        assert math.isfinite(value)
        by_scene[key].append(value)
        by_class[cls].append(value)
    assert len(by_scene) == 2000 and all(len(values) == 3 for values in by_scene.values())
    scenes = {
        cls: np.array([
            math.sqrt(np.mean(np.square(values)))
            for key, values in by_scene.items() if key.split("_")[1].upper() == cls
        ])
        for cls in ORDER
    }
    assert all(len(values) == 500 for values in scenes.values())
    all_track_rmse = np.array([float(item["rmse_m"]) for item in result["tracks"]])
    assert abs(math.sqrt(np.mean(all_track_rmse**2)) - result["metrics"]["position_rmse_m"]) < 1e-4

    fig, ax = plt.subplots(figsize=(10.8, 5.4), layout="constrained")
    for cls in ORDER:
        x = np.sort(scenes[cls])
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y * 100, where="post", lw=2.3, color=COLORS[cls], label=LABELS[cls])
    ax.axvline(8, color="#4B5563", ls="--", lw=1.5, label="8 米参照线")
    ax.set_xlim(3, 17)
    ax.set_ylim(0, 100)
    ax.set_xlabel("单场景三目标有效点合并 RMSE（米）")
    ax.set_ylabel("累计场景比例（%）")
    ax.set_title("V7 完整验证集场景级误差累计分布", fontsize=15, pad=14)
    ax.grid(axis="both", alpha=0.18)
    ax.legend(ncol=3, frameon=False, loc="lower right")
    fig.text(0.01, -0.02, "数据：V7 主批次 2000 个场景；每场景 3 条航迹、每条 95 个有效点。场景级统计用于分层分析，不替代总体点级 RMSE。", fontsize=9, color="#586270")
    _save(fig, "17_V7场景级误差累计分布")

    fig, ax = plt.subplots(figsize=(10.8, 5.4), layout="constrained")
    x = np.arange(len(ORDER))
    tail_share = []
    for cls in ORDER:
        squared = np.sort(np.square(np.array(by_class[cls])))[::-1]
        top_n = len(squared) // 10
        tail_share.append(float(np.sum(squared[:top_n]) / np.sum(squared) * 100))
    bars = ax.bar(x, tail_share, width=0.57, color=[COLORS[c] for c in ORDER])
    for bar, value in zip(bars, tail_share):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.1f}%", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x, [LABELS[c] for c in ORDER])
    ax.set_ylim(0, max(tail_share) + 10)
    ax.set_ylabel("类别内平方误差贡献占比（%）")
    ax.set_title("逐航迹高误差尾部对类别 RMSE 的贡献", fontsize=15, pad=14)
    ax.grid(axis="y", alpha=0.18)
    fig.text(0.01, -0.02, "每类 1500 条航迹；按逐航迹 RMSE 从高到低取前 150 条，计算其平方误差占该类全部航迹平方误差之比。各航迹均为 95 个有效点。", fontsize=9, color="#586270")
    _save(fig, "18_V7高误差航迹尾部贡献")

    idle = json.loads((METRICS / "forward_latency.json").read_text(encoding="utf-8"))
    busy = json.loads((METRICS / "forward_latency_concurrent.json").read_text(encoding="utf-8"))
    assert idle["job_id"] == busy["job_id"] == JOB
    assert idle["model_sha256"] == busy["model_sha256"] == MODEL_SHA256
    latencies = [
        ("T=1 空闲", idle["one_frame_input"]["samples_ms"], "#247BA0"),
        ("T=100 空闲", idle["full_sequence_input"]["samples_ms"], "#2A9D8F"),
        ("T=100 高负载", busy["full_sequence_input"]["samples_ms"], "#C05746"),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.4), layout="constrained")
    for label, raw, color in latencies:
        values = np.sort(np.array(raw, dtype=float))
        assert len(values) == 300 and np.all(np.isfinite(values))
        y = np.arange(1, len(values) + 1) / len(values) * 100
        ax.step(values, y, where="post", lw=2.3, label=f"{label}（P95 {np.percentile(values, 95):.1f} ms）", color=color)
    ax.axvline(100, color="#4B5563", ls="--", lw=1.5, label="100 毫秒参照线")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)
    ax.set_xlabel("单次模型前向耗时（毫秒）")
    ax.set_ylabel("累计测量比例（%）")
    ax.set_title("V7 模型前向原始计时样本分布", fontsize=15, pad=14)
    ax.grid(axis="both", alpha=0.18)
    ax.legend(frameon=False, loc="lower right")
    fig.text(0.01, -0.02, "同一 V7 权重；各条件先预热 50 次，再同步计时 300 次。T=1 与 T=100 输入长度不同，高负载为独立运行条件。", fontsize=9, color="#586270")
    _save(fig, "19_V7前向原始计时分布")

    baseline = json.loads((METRICS / "rmse_improvement_pct.json").read_text(encoding="utf-8"))
    assert baseline["job_id"] == JOB and baseline["evaluated_scenarios"] == 2000
    pairs = []
    for item in baseline["per_file"]:
        key = item["file"].removesuffix(".mat")
        cls = key.split("_")[1].upper()
        own_rmse = math.sqrt(np.mean(np.square(by_scene[key])))
        pairs.append((cls, own_rmse, float(item["rmse_m"])))
    assert len(pairs) == 2000
    base_values = np.array([item[2] for item in pairs])
    base_sq_sorted = np.sort(base_values**2)[::-1]
    baseline_top_decile_share = float(np.sum(base_sq_sorted[:200]) / np.sum(base_sq_sorted) * 100)
    fig, ax = plt.subplots(figsize=(10.8, 5.4), layout="constrained")
    for cls in ORDER:
        selected = [(own, ref) for c, own, ref in pairs if c == cls]
        ax.scatter([p[1] for p in selected], [p[0] for p in selected], s=14, alpha=0.42, color=COLORS[cls], label=LABELS[cls], linewidths=0)
    ax.axhline(8, color="#4B5563", ls="--", lw=1.4, label="8 米参照线")
    ax.set_xscale("log")
    ax.set_xlim(20, 1300)
    ax.set_ylim(2, 18)
    ax.set_xlabel("IMM-JPDA 参考实现单场景 RMSE（米，对数坐标）")
    ax.set_ylabel("V7 模型单场景 RMSE（米）")
    ax.set_title("V7 与参考实现的 2000 场景配对结果", fontsize=15, pad=14)
    ax.grid(alpha=0.18)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    fig.text(0.01, -0.02, "双方场景与有效点相同；参考实现只用长机雷达和僚机快链路，且以首帧真值初始化。散点差异不能单独归因于算法结构。", fontsize=9, color="#586270")
    _save(fig, "20_V7同场景配对基线分布")

    quality = json.loads((ROOT / "artifacts" / "metrics" / "midterm_review" / "items_10_12_evidence.json").read_text(encoding="utf-8"))
    audit = quality["review_item_10"]["acceptance_dataset_empirical_audit"]
    assert audit["scenario_count"] == 64 and audit["track_count"] == 192
    sensors = [("Radar", "雷达", "#247BA0"), ("EO", "光电", "#2A9D8F"), ("ESM", "电子侦察", "#E9A23B"), ("FastLink", "快速链路", "#C05746")]
    fig, ax = plt.subplots(figsize=(10.8, 5.0), layout="constrained")
    for i, (key, label, color) in enumerate(sensors):
        item = audit["sensors"][key]
        assert item["total_input_slots"] == 96000
        valid = 100 * item["availability_rate"]
        ax.barh(i, valid, height=0.58, color=color)
        ax.barh(i, 100 - valid, left=valid, height=0.58, color="#E5E9ED")
        ax.text(valid / 2, i, f"有效 {valid:.1f}%", ha="center", va="center", color="white", fontsize=11, weight="bold")
        ax.text(valid + (100 - valid) / 2, i, f"缺失 {100-valid:.1f}%", ha="center", va="center", color="#46505B", fontsize=10)
    ax.set_yticks(range(4), [label for _, label, _ in sensors])
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("输入槽比例（%）")
    ax.set_title("V7 分层样本多源输入可用性审计", fontsize=15, pad=14)
    ax.grid(axis="x", alpha=0.16)
    fig.text(0.01, -0.02, "数据：64 个 V7 分层场景、192 条航迹；各来源统计 96000 个输入槽。可用性比例不等于关联正确率，也不代表完整 2000 场景的精确比例。", fontsize=9, color="#586270")
    _save(fig, "21_V7多源输入可用性")

    summary = {
        "job_id": JOB,
        "model_sha256": MODEL_SHA256,
        "source_report": str(REPORT.relative_to(ROOT)),
        "scene_level": {
            cls: {
                "n_scenes": len(scenes[cls]),
                "median_rmse_m": float(np.median(scenes[cls])),
                "p90_rmse_m": float(np.percentile(scenes[cls], 90)),
                "scenes_rmse_le_8": int(np.sum(scenes[cls] <= 8)),
                "top_10pct_track_squared_error_share_pct": tail_share[i],
            }
            for i, cls in enumerate(ORDER)
        },
        "latency_p95_ms": {label: float(np.percentile(raw, 95)) for label, raw, _ in latencies},
        "baseline_scene_median_rmse_m": float(np.median(base_values)),
        "baseline_scene_p90_rmse_m": float(np.percentile(base_values, 90)),
        "baseline_top_10pct_scene_squared_error_share_pct": baseline_top_decile_share,
        "input_availability_64_scenes": {key: audit["sensors"][key]["availability_rate"] for key, _, _ in sensors},
    }
    (OUTPUT / "新增图数据摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build()
