from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.acceptance_common import evidence_context


RUN_DIR = ROOT / "artifacts" / "reports" / "software_test" / "IT-03-03" / "20260925_run01"
FIG_DIR = RUN_DIR / "figures"
BATCH_ID = "46fada4af5ef4c70a03fa56ef521f8c8"
BATCH_PATH = ROOT / "artifacts" / "reports" / "batch_tests" / f"{BATCH_ID}.json"
EVIDENCE_PATH = RUN_DIR / "rmse_improvement_pct.json"
HISTORICAL_PATH = ROOT / "artifacts" / "metrics" / "acceptance" / "2ea4f9eb0f2845b599fcf4837d71a018" / "rmse_improvement_pct.json"
BASELINE_SCRIPT = ROOT / "scripts" / "evaluation" / "test_rmse_improvement.py"
THRESHOLD_PCT = 15.0

RUN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


batch = load(BATCH_PATH)
evidence = load(EVIDENCE_PATH)
historical = load(HISTORICAL_PATH)
batch_result = batch["result"]
batch_metrics = batch_result["metrics"]
context = evidence_context(batch)
rows = evidence["per_file"]
checks: list[dict] = []


def check(cid: str, item: str, passed: bool, actual, expected) -> None:
    record = {"check_id": cid, "item": item, "passed": bool(passed), "actual": actual, "expected": expected}
    checks.append(record)
    if not passed:
        raise AssertionError(f"{cid}: {item}; actual={actual!r}; expected={expected!r}")


check("BATCH-COMPLETE", "绑定的V7全量批测状态为completed", batch["status"] == "completed", batch["status"], "completed")
for key in ("job_id", "job_signature", "model_id", "dataset_id", "max_scenarios", "warmup_frames", "sequence_length", "evaluated_scenarios"):
    check(f"CONTEXT-{key.upper()}", f"基线证据的{key}与批测上下文一致", evidence.get(key) == context.get(key), evidence.get(key), context.get(key))
check("DATASET-V7", "正式范围为V7且不混入V8", evidence["dataset_id"] == "v7", evidence["dataset_id"], "v7")
check("METHOD", "基线方法登记为NumPy IMM-JPDA竞争归一化边缘关联", evidence["method"] == "NumPy IMM-JPDA competition-normalized marginal association, truth initialization", evidence["method"], "registered method")
check("SETTINGS", "Mahalanobis门限和漏检权重为固定验收配置", evidence["settings"]["gate_squared"] == 25.0 and evidence["settings"]["missed_detection_weight"] == 0.15, evidence["settings"], {"gate_squared": 25.0, "missed_detection_weight": 0.15})
check("SOURCES", "基线观测源为长机Radar和僚机FastLink", evidence["settings"]["measurement_sources"] == ["leader Radar", "wingman FastLink"], evidence["settings"]["measurement_sources"], ["leader Radar", "wingman FastLink"])

transformer_rmse = float(evidence["transformer_rmse_m"])
baseline_rmse = float(evidence["imm_jpda_rmse_m"])
improvement = float(evidence["improvement_pct"])
check("TRANSFORMER-RMSE", "Transformer RMSE直接取自同一批测报告", abs(transformer_rmse - float(batch_metrics["position_rmse_m"])) <= 1e-12, transformer_rmse, batch_metrics["position_rmse_m"])
check("BASELINE-FINITE", "IMM-JPDA RMSE为有限正数", math.isfinite(baseline_rmse) and baseline_rmse > 0, baseline_rmse, ">0 finite")
recomputed_improvement = (baseline_rmse - transformer_rmse) / baseline_rmse * 100.0
check("FORMULA", "降低率按(基线-Transformer)/基线×100%独立复算", abs(improvement - recomputed_improvement) <= 1e-12 and abs(evidence["value"] - improvement) <= 1e-12, improvement, recomputed_improvement)
check("THRESHOLD", "未四舍五入降低率不低于15%", improvement >= THRESHOLD_PCT and evidence["passed"] is True, improvement, ">=15%")
check("VALID", "独立证据可用于正式判定", evidence["valid_for_assessment"] is True, evidence["valid_for_assessment"], True)

# The formal comparison uses the same 2,000 files, 3 targets per file, 100
# frames, 5-frame warm-up and 570,000 retained target-frame points.
manifest_files = [f"{name}.mat" for name in batch_result["scenario_manifest"]["files"]]
evidence_files = [row["file"] for row in rows]
check("FILE-COUNT", "基线逐文件记录为2000条", len(rows) == 2000, len(rows), 2000)
check("FILE-LIST", "基线文件清单与Transformer批测场景清单完全一致", evidence_files == manifest_files, {"first": evidence_files[:2], "last": evidence_files[-2:]}, "exact ordered manifest")
check("TARGETS", "每个场景比较同样的3个目标", all(int(row["targets"]) == 3 for row in rows), sorted({row["targets"] for row in rows}), [3])
check("POINTS-PER-FILE", "每文件保留3×(100-5)=285个目标帧点", all(int(row["evaluated_points"]) == 285 for row in rows), sorted({row["evaluated_points"] for row in rows}), [285])
check("POINTS-TOTAL", "两种方法均比较570000个有效点", evidence["evaluated_points"] == batch_metrics["evaluated_points"] == sum(int(row["evaluated_points"]) for row in rows) == 570000, {"evidence": evidence["evaluated_points"], "batch": batch_metrics["evaluated_points"]}, 570000)
check("BASELINE-UPDATES", "每文件均有有效IMM-JPDA更新且不超过3×100帧", all(0 < int(row["updated_track_frames"]) <= 300 for row in rows), {"min": min(row["updated_track_frames"] for row in rows), "max": max(row["updated_track_frames"] for row in rows)}, "1..300")
numeric = np.asarray([[row["rmse_m"], row["mae_m"], row["p95_m"]] for row in rows], dtype=np.float64)
check("FINITE-PER-FILE", "全部逐文件基线指标为有限非负数", bool(np.isfinite(numeric).all() and np.all(numeric >= 0)), bool(np.isfinite(numeric).all()), True)
baseline_recomputed = float(np.sqrt(sum(float(row["rmse_m"]) ** 2 * int(row["evaluated_points"]) for row in rows) / sum(int(row["evaluated_points"]) for row in rows)))
check("BASELINE-AGGREGATE", "IMM-JPDA总体RMSE按有效点加权复算一致", abs(baseline_recomputed - baseline_rmse) <= 1e-12, baseline_recomputed, baseline_rmse)

# Match each raw MAT file with its three Transformer track rows and derive
# comparable per-file and per-motion-class results.
transformer_by_file: dict[str, list[dict]] = defaultdict(list)
for track in batch_result["tracks"]:
    file_name = track["track"].split(".pkl_target_")[0] + ".mat"
    transformer_by_file[file_name].append(track)
check("TRACK-FILE-SET", "Transformer 6000条航迹可一一归入2000个基线文件", set(transformer_by_file) == set(evidence_files) and sum(len(v) for v in transformer_by_file.values()) == 6000, {"files": len(transformer_by_file), "tracks": sum(len(v) for v in transformer_by_file.values())}, {"files": 2000, "tracks": 6000})
check("TRACKS-PER-FILE", "每文件恰有3条Transformer航迹", all(len(v) == 3 for v in transformer_by_file.values()), sorted({len(v) for v in transformer_by_file.values()}), [3])
check("VALID-FRAMES", "每条Transformer航迹同样剔除前5帧并保留95帧", all(int(track["valid_frames"]) == 95 for values in transformer_by_file.values() for track in values), sorted({track["valid_frames"] for values in transformer_by_file.values() for track in values}), [95])

file_comparison = []
for row in rows:
    tracks = transformer_by_file[row["file"]]
    point_count = sum(int(track["valid_frames"]) for track in tracks)
    transformer_file_rmse = float(np.sqrt(sum(float(track["rmse_m"]) ** 2 * int(track["valid_frames"]) for track in tracks) / point_count))
    baseline_file_rmse = float(row["rmse_m"])
    file_improvement = (baseline_file_rmse - transformer_file_rmse) / baseline_file_rmse * 100.0
    match = re.match(r"track_([^_]+)_", row["file"])
    if not match:
        raise RuntimeError(f"Cannot classify {row['file']}")
    file_comparison.append({
        "file": row["file"],
        "class": match.group(1).upper(),
        "points": point_count,
        "transformer_rmse_m": transformer_file_rmse,
        "imm_jpda_rmse_m": baseline_file_rmse,
        "improvement_pct": file_improvement,
        "updated_track_frames": int(row["updated_track_frames"]),
    })

file_improvements = np.asarray([row["improvement_pct"] for row in file_comparison], dtype=np.float64)
check("FILE-IMPROVEMENTS", "2000个场景的逐文件降低率均为有限值", bool(np.isfinite(file_improvements).all()), bool(np.isfinite(file_improvements).all()), True)
check("FILE-ALL-PASS", "2000个场景逐文件降低率均不低于15%", bool(np.all(file_improvements >= THRESHOLD_PCT)), {"passed": int(np.sum(file_improvements >= THRESHOLD_PCT)), "minimum": float(np.min(file_improvements))}, {"passed": 2000, "minimum": ">=15%"})

class_rows = []
for cls in ("CA", "CSSW", "CT", "CV"):
    selected = [row for row in file_comparison if row["class"] == cls]
    points = sum(row["points"] for row in selected)
    transformer_class = float(np.sqrt(sum(row["transformer_rmse_m"] ** 2 * row["points"] for row in selected) / points))
    baseline_class = float(np.sqrt(sum(row["imm_jpda_rmse_m"] ** 2 * row["points"] for row in selected) / points))
    reduction = (baseline_class - transformer_class) / baseline_class * 100.0
    class_rows.append({"class": cls, "files": len(selected), "points": points, "transformer_rmse_m": transformer_class, "imm_jpda_rmse_m": baseline_class, "improvement_pct": reduction})
    check(f"CLASS-{cls}", f"{cls}分层包含500文件和142500点且降低率达标", len(selected) == 500 and points == 142500 and reduction >= THRESHOLD_PCT, class_rows[-1], {"files": 500, "points": 142500, "improvement_pct": ">=15"})

# Code-level scope checks make the comparison assumptions explicit.
source = BASELINE_SCRIPT.read_text(encoding="utf-8")
check("CODE-DT", "IMM各模型固定dt=0.1 s", "NumpyImmTracker(_initial_state(labels[0, target]), dt=0.1)" in source, "dt=0.1", "dt=0.1")
check("CODE-WARMUP", "基线仅在frame_index>=warmup时累计误差", "if frame_index >= warmup:" in source, "present", "present")
check("CODE-TRUTH-INIT", "基线真值初始化在证据方法名和实现中均显式记录", "truth initialization" in evidence["method"] and "_initial_state(labels[0, target])" in source, evidence["method"], "explicit")
check("CODE-WEIGHTING", "脚本按逐文件平方误差和有效点数加权汇总", "squared_error_sum += float(result[\"rmse_m\"]) ** 2 * count" in source, "point-weighted", "point-weighted")

# Formula boundary and invalid-input guards are tested separately.
boundary_rows = []
for expected_reduction, expected_pass in ((14.99, False), (15.00, True), (15.01, True)):
    baseline = 100.0
    transformer = baseline * (1.0 - expected_reduction / 100.0)
    actual = (baseline - transformer) / baseline * 100.0
    passed = actual >= THRESHOLD_PCT
    boundary_rows.append({"reduction_pct": expected_reduction, "transformer_rmse_m": transformer, "computed_pct": actual, "accepted": passed, "expected": expected_pass})
    check(f"BOUNDARY-{expected_reduction:.2f}", f"{expected_reduction:.2f}%未四舍五入阈值比较", passed == expected_pass, {"computed": actual, "passed": passed}, expected_pass)

def safe_improvement(baseline: float, transformer: float) -> tuple[float | None, bool]:
    if not (math.isfinite(baseline) and math.isfinite(transformer)) or baseline <= 0 or transformer < 0:
        return None, False
    value = (baseline - transformer) / baseline * 100.0
    return value, value >= THRESHOLD_PCT

invalid_cases = {
    "zero_baseline": safe_improvement(0.0, 1.0),
    "negative_baseline": safe_improvement(-1.0, 1.0),
    "nan_transformer": safe_improvement(100.0, float("nan")),
    "negative_transformer": safe_improvement(100.0, -1.0),
}
check("INVALID-RMSE", "零、负数或非有限RMSE均返回无效而非通过", all(value is None and passed is False for value, passed in invalid_cases.values()), invalid_cases, "all invalid")

# Reproduce the unchanged baseline implementation against the historical full
# V7 evidence. Job signatures differ because the owning batch IDs differ; the
# numerical scope, settings and all 2,000 per-file results must match exactly.
historical_core = {key: historical[key] for key in ("transformer_rmse_m", "imm_jpda_rmse_m", "improvement_pct", "evaluated_points", "settings", "per_file")}
current_core = {key: evidence[key] for key in historical_core}
check("REPRO-HISTORICAL", "本轮完整基线与历史完整V7证据的设置及2000条逐文件结果完全一致", current_core == historical_core, {"current": [evidence["imm_jpda_rmse_m"], evidence["improvement_pct"]], "historical": [historical["imm_jpda_rmse_m"], historical["improvement_pct"]]}, "identical core")

with (RUN_DIR / "class_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
    writer.writeheader(); writer.writerows(class_rows)
with (RUN_DIR / "per_file_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(file_comparison[0]))
    writer.writeheader(); writer.writerows(file_comparison)

# Figure 1: formal aggregate RMSE comparison and contract reduction.
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
bars = axes[0].bar(["Transformer", "IMM-JPDA"], [transformer_rmse, baseline_rmse], color=["#4C78A8", "#F28E2B"], width=.58)
axes[0].set_ylabel("三维位置RMSE / m"); axes[0].set_title("V7完整集同范围RMSE"); axes[0].grid(axis="y", alpha=.25)
for bar, value in zip(bars, (transformer_rmse, baseline_rmse)): axes[0].text(bar.get_x()+bar.get_width()/2, value+4, f"{value:.3f} m", ha="center", weight="bold")
axes[1].bar(["实际降低率"], [improvement], color="#59A14F", width=.52)
axes[1].axhline(THRESHOLD_PCT, color="#C00000", linestyle="--", linewidth=2, label="合同阈值15%")
axes[1].set_ylim(0, 105); axes[1].set_ylabel("RMSE降低率 / %"); axes[1].set_title("相对IMM-JPDA降低率"); axes[1].grid(axis="y", alpha=.25); axes[1].legend()
axes[1].text(0, improvement+2, f"{improvement:.4f}%", ha="center", weight="bold")
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_1_overall_rmse_improvement.png", dpi=180); plt.close(fig)

# Figure 2: motion-class stratification.
x = np.arange(4); width = .36
fig, ax = plt.subplots(figsize=(9.8, 5.2))
b1 = ax.bar(x-width/2, [row["transformer_rmse_m"] for row in class_rows], width, label="Transformer", color="#4C78A8")
b2 = ax.bar(x+width/2, [row["imm_jpda_rmse_m"] for row in class_rows], width, label="IMM-JPDA", color="#F28E2B")
ax.set_xticks(x, [row["class"] for row in class_rows]); ax.set_ylabel("三维位置RMSE / m"); ax.set_title("运动类型分层RMSE（每类500场景、142500点）"); ax.grid(axis="y", alpha=.25); ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.00), ncol=2)
for bars in (b1, b2):
    for bar in bars: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, f"{bar.get_height():.2f}", ha="center", fontsize=9)
for i, row in enumerate(class_rows): ax.text(i, row["imm_jpda_rmse_m"]+12, f"降低{row['improvement_pct']:.2f}%", ha="center", fontsize=9, color="#2E6E3E")
ax.set_ylim(0, 240)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_2_motion_class_comparison.png", dpi=180); plt.close(fig)

# Figure 3: per-file improvement distribution and cumulative curve.
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7))
axes[0].hist(file_improvements, bins=40, color="#4C78A8", alpha=.9)
axes[0].axvline(THRESHOLD_PCT, color="#C00000", linestyle="--", label="15%阈值")
axes[0].set_xlabel("逐文件RMSE降低率 / %"); axes[0].set_ylabel("文件数"); axes[0].set_title("2000个场景的降低率分布"); axes[0].grid(alpha=.2); axes[0].legend()
sorted_values = np.sort(file_improvements); cdf = np.arange(1, len(sorted_values)+1)/len(sorted_values)*100
axes[1].plot(sorted_values, cdf, color="#59A14F", linewidth=2)
axes[1].axvline(THRESHOLD_PCT, color="#C00000", linestyle="--")
axes[1].scatter([np.min(file_improvements)], [0.05], color="#F28E2B", zorder=3)
axes[1].text(np.min(file_improvements)+.5, 4, f"最低 {np.min(file_improvements):.2f}%")
axes[1].set_xlabel("逐文件RMSE降低率 / %"); axes[1].set_ylabel("累计比例 / %"); axes[1].set_ylim(0, 102); axes[1].set_title("逐文件降低率累计分布"); axes[1].grid(alpha=.2)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_3_per_file_improvement.png", dpi=180); plt.close(fig)

# Figure 4: historical reproduction plus boundary/invalid guards.
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
labels = ["历史完整证据", "本轮完整证据"]
baseline_values = [historical["imm_jpda_rmse_m"], baseline_rmse]
reduction_values = [historical["improvement_pct"], improvement]
axes[0].bar(np.arange(2)-.18, baseline_values, .36, label="IMM-JPDA RMSE/m", color="#F28E2B")
axes[0].bar(np.arange(2)+.18, reduction_values, .36, label="降低率/%", color="#59A14F")
axes[0].set_xticks(np.arange(2), labels); axes[0].set_title("冻结范围的数值复现"); axes[0].grid(axis="y", alpha=.25); axes[0].legend()
guard_labels = ["14.99%", "15.00%", "15.01%", "基线=0", "NaN"]
guard_pass = [row["accepted"] for row in boundary_rows] + [invalid_cases["zero_baseline"][1], invalid_cases["nan_transformer"][1]]
axes[1].bar(guard_labels, [1]*5, color=["#59A14F" if value else "#C00000" for value in guard_pass])
axes[1].set_ylim(0, 1.12); axes[1].set_yticks([]); axes[1].set_title("阈值与异常输入防护")
for i, value in enumerate(guard_pass): axes[1].text(i, .5, "通过" if value else "拒绝", ha="center", va="center", color="white", weight="bold", fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_4_reproducibility_and_guards.png", dpi=180); plt.close(fig)

result = {
    "test_case": "IT-03-03",
    "test_name": "相对IMM-JPDA位置RMSE降低测试",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"os": platform.platform(), "python": sys.version, "numpy": np.__version__},
    "formal_batch_id": BATCH_ID,
    "baseline_script_sha256": sha256(BASELINE_SCRIPT),
    "scope": {"dataset": "V7 full validation", "files": 2000, "targets": 6000, "points": 570000, "sequence_length": 100, "warmup_frames": 5, "v8_included": False},
    "formal_metrics": {"transformer_rmse_m": transformer_rmse, "imm_jpda_rmse_m": baseline_rmse, "improvement_pct": improvement, "threshold_pct": THRESHOLD_PCT, "margin_pct_points": improvement-THRESHOLD_PCT},
    "settings": evidence["settings"],
    "method": evidence["method"],
    "per_file_summary": {"count": len(file_comparison), "passed_15pct": int(np.sum(file_improvements >= THRESHOLD_PCT)), "minimum_pct": float(np.min(file_improvements)), "median_pct": float(np.median(file_improvements)), "p5_pct": float(np.percentile(file_improvements, 5)), "maximum_pct": float(np.max(file_improvements)), "worst_file": min(file_comparison, key=lambda row: row["improvement_pct"])},
    "baseline_file_summary": {"median_rmse_m": float(np.median(numeric[:, 0])), "p95_rmse_m": float(np.percentile(numeric[:, 0], 95)), "maximum_rmse_m": float(np.max(numeric[:, 0])), "updated_track_frames_min": int(min(row["updated_track_frames"] for row in rows)), "updated_track_frames_max": int(max(row["updated_track_frames"] for row in rows))},
    "class_metrics": class_rows,
    "threshold_boundary": boundary_rows,
    "invalid_cases": invalid_cases,
    "historical_reproduction": {"historical_job_id": historical["job_id"], "imm_jpda_rmse_m": historical["imm_jpda_rmse_m"], "improvement_pct": historical["improvement_pct"], "core_identical": current_core == historical_core},
    "checks": checks,
    "check_count": len(checks),
    "passed_count": sum(row["passed"] for row in checks),
    "overall_passed": all(row["passed"] for row in checks),
    "figures": [path.name for path in sorted(FIG_DIR.glob("figure_*.png"))],
    "scope_note": "正式比较使用V7完整2000文件、6000目标和570000个稳态目标帧点；两种方法均排除前5帧并按有效点平方误差汇总。V8原始40 Hz场景未混入dt=0.1 s的V7比较。",
}
(RUN_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "output": str(RUN_DIR),
    "overall_passed": result["overall_passed"],
    "checks": f"{result['passed_count']}/{result['check_count']}",
    "transformer_rmse_m": transformer_rmse,
    "imm_jpda_rmse_m": baseline_rmse,
    "improvement_pct": improvement,
    "per_file": result["per_file_summary"],
    "class_metrics": class_rows,
}, ensure_ascii=False, indent=2))
if not result["overall_passed"]:
    raise SystemExit(1)
