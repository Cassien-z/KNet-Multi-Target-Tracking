"""Recompute and cross-check factual claims used by the V7 final report.

This audit does not replace customer acceptance.  It verifies that the report's
quantitative claims and stated evidence boundaries agree with artifacts in the
current workspace.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[2]
JOB_ID = "2ea4f9eb0f2845b599fcf4837d71a018"
REPORT = ROOT / "artifacts" / "reports" / "batch_tests" / f"{JOB_ID}.json"
ACCEPTANCE = ROOT / "artifacts" / "metrics" / "acceptance" / JOB_ID
REVIEW = ROOT / "artifacts" / "metrics" / "midterm_review"
EXPANSION = ROOT / "artifacts" / "figures" / "v7_report_expansion" / "新增图数据摘要.json"
FINAL_TEXT = ROOT / "docs" / "基于多源异构数据融合的端对端适变跟踪方法研究_结题报告初稿.md"
OUTPUT = ROOT / "artifacts" / "reports" / "fact_audit" / "结题报告事实复核_20260921.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


checks: list[dict] = []


def check(name: str, ok: bool, actual, expected, source: str) -> None:
    checks.append(
        {
            "name": name,
            "status": "passed" if ok else "failed",
            "actual": actual,
            "expected": expected,
            "source": source,
        }
    )


def close(actual: float, expected: float, tol: float = 1e-9) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol)


batch = load(REPORT)
result = batch["result"]
metrics = result["metrics"]
tracks = result["tracks"]
association = load(ACCEPTANCE / "association_accuracy.json")
baseline = load(ACCEPTANCE / "rmse_improvement_pct.json")
latency_idle = load(ACCEPTANCE / "forward_latency.json")
latency_busy = load(ACCEPTANCE / "forward_latency_concurrent.json")
ablation = load(ACCEPTANCE / "ablation_v7.json")
review_13 = load(REVIEW / "items_1_3_evidence.json")
review_46 = load(REVIEW / "items_4_6_evidence.json")
review_1012 = load(REVIEW / "items_10_12_evidence.json")
expansion = load(EXPANSION)
report_text = FINAL_TEXT.read_text(encoding="utf-8")

report_source = str(REPORT.relative_to(ROOT))
check("主批次任务号", batch["id"] == JOB_ID, batch["id"], JOB_ID, report_source)
check("主批次状态", batch["status"] == "completed", batch["status"], "completed", report_source)
check("模型标识", result["model_id"] == "v7", result["model_id"], "v7", report_source)
check("数据集标识", result["dataset_id"] == "v7", result["dataset_id"], "v7", report_source)
check("预热帧数", batch["config"]["warmup_frames"] == 5, batch["config"]["warmup_frames"], 5, report_source)
check("序列长度", batch["config"]["sequence_length"] == 100, batch["config"]["sequence_length"], 100, report_source)
check("完整场景数", metrics["evaluated_scenarios"] == 2000, metrics["evaluated_scenarios"], 2000, report_source)
check("完整航迹数", metrics["evaluated_tracks"] == 6000, metrics["evaluated_tracks"], 6000, report_source)
check("有效目标点数", metrics["evaluated_points"] == 570000, metrics["evaluated_points"], 570000, report_source)
check("总体位置RMSE", close(metrics["position_rmse_m"], 7.488174720926246), metrics["position_rmse_m"], 7.488174720926246, report_source)
check("总体位置MAE", close(metrics["position_mae_m"], 6.337683879564049), metrics["position_mae_m"], 6.337683879564049, report_source)
check("点级P95", close(metrics["position_p95_m"], 13.6862154006958), metrics["position_p95_m"], 13.6862154006958, report_source)

model_path = Path(result["model_path"])
model_hash = sha256(model_path)
check("V7权重文件存在", model_path.is_file(), str(model_path), "existing file", report_source)
check("V7权重SHA-256", model_hash == result["provenance"]["model_sha256"], model_hash, result["provenance"]["model_sha256"], report_source)

raw_count = sum(1 for item in (ROOT / "data" / "val_data_v7").iterdir() if item.is_file())
registered_count = sum(1 for item in (ROOT / "data" / "val_data_v7_registered").iterdir() if item.is_file())
check("V7原始场景文件数", raw_count == 2000, raw_count, 2000, "data/val_data_v7")
check("V7注册场景文件数", registered_count == 2000, registered_count, 2000, "data/val_data_v7_registered")

track_counts = Counter()
class_squared = defaultdict(float)
class_points = Counter()
scene_squared = defaultdict(float)
scene_points = Counter()
track_rmse = []
valid_frames = []
for track in tracks:
    key = track["track"].split(".pkl_target_")[0]
    maneuver = key.split("_")[1].upper()
    points = int(track["valid_frames"])
    value = float(track["rmse_m"])
    track_counts[maneuver] += 1
    class_squared[maneuver] += value * value * points
    class_points[maneuver] += points
    scene_squared[key] += value * value * points
    scene_points[key] += points
    track_rmse.append(value)
    valid_frames.append(points)

expected_class_counts = {"CV": 1500, "CA": 1500, "CT": 1500, "CSSW": 1500}
check("四类航迹均衡覆盖", dict(track_counts) == expected_class_counts, dict(track_counts), expected_class_counts, report_source)
check("逐航迹均为95个有效帧", set(valid_frames) == {95}, sorted(set(valid_frames)), [95], report_source)
recomputed_rmse = math.sqrt(sum(v * v * p for v, p in zip(track_rmse, valid_frames)) / sum(valid_frames))
check("逐航迹加权重算总体RMSE", close(recomputed_rmse, metrics["position_rmse_m"]), recomputed_rmse, metrics["position_rmse_m"], report_source)

class_rmse = {k: math.sqrt(class_squared[k] / class_points[k]) for k in sorted(class_squared)}
expected_class_rmse = {"CA": 7.04087408000285, "CSSW": 8.515877149073338, "CT": 7.281348704291699, "CV": 7.012769226376944}
check("分机动RMSE", all(close(class_rmse[k], v) for k, v in expected_class_rmse.items()), class_rmse, expected_class_rmse, str((ACCEPTANCE / "ablation_v7.json").relative_to(ROOT)))
check("逐航迹RMSE中位数", close(median(track_rmse), 6.601447703527198), median(track_rmse), 6.601447703527198, report_source)
check("8米内航迹数", sum(v <= 8 for v in track_rmse) == 4309, sum(v <= 8 for v in track_rmse), 4309, report_source)

detection_rmse = math.sqrt((class_squared["CV"] + class_squared["CT"]) / (class_points["CV"] + class_points["CT"]))
strike_rmse = math.sqrt((class_squared["CA"] + class_squared["CSSW"]) / (class_points["CA"] + class_points["CSSW"]))
check("协同探测组派生RMSE", close(detection_rmse, 7.148320, tol=1e-6), detection_rmse, 7.148320, report_source)
check("协同打击组派生RMSE", close(strike_rmse, 7.813260, tol=1e-6), strike_rmse, 7.813260, report_source)

association_source = str((ACCEPTANCE / "association_accuracy.json").relative_to(ROOT))
assoc = association["association"]
check("关联正确数", assoc["correct"] == 4142948, assoc["correct"], 4142948, association_source)
check("关联有效总数", assoc["total"] == 4162065, assoc["total"], 4162065, association_source)
check("关联正确率", close(assoc["accuracy"], 0.9954068473221827), assoc["accuracy"], 0.9954068473221827, association_source)
check("关联分源计数可加和", sum(v["correct"] for v in assoc["per_sensor"].values()) == assoc["correct"] and sum(v["total"] for v in assoc["per_sensor"].values()) == assoc["total"], {"correct": sum(v["correct"] for v in assoc["per_sensor"].values()), "total": sum(v["total"] for v in assoc["per_sensor"].values())}, {"correct": assoc["correct"], "total": assoc["total"]}, association_source)
check("关联分组未使用真值", assoc["truth_used_for_observation_grouping"] is False, assoc["truth_used_for_observation_grouping"], False, association_source)

baseline_source = str((ACCEPTANCE / "rmse_improvement_pct.json").relative_to(ROOT))
check("参考实现总体RMSE", close(baseline["imm_jpda_rmse_m"], 189.12752510136), baseline["imm_jpda_rmse_m"], 189.12752510136, baseline_source)
improvement = (baseline["imm_jpda_rmse_m"] - baseline["transformer_rmse_m"]) / baseline["imm_jpda_rmse_m"] * 100
check("参考实现相对降幅", close(improvement, 96.04067429269585), improvement, 96.04067429269585, baseline_source)
check("参考实现输入来源边界", baseline["settings"]["measurement_sources"] == ["leader Radar", "wingman FastLink"], baseline["settings"]["measurement_sources"], ["leader Radar", "wingman FastLink"], baseline_source)
check("参考实现首帧真值初始化边界", "truth initialization" in baseline["method"], baseline["method"], "contains truth initialization", baseline_source)

scene_rmse = {key: math.sqrt(scene_squared[key] / scene_points[key]) for key in scene_squared}
scene_summary = {}
tail_summary = {}
for maneuver in ["CV", "CA", "CT", "CSSW"]:
    maneuver_scenes = [value for key, value in scene_rmse.items() if key.split("_")[1].upper() == maneuver]
    maneuver_tracks = [
        float(track["rmse_m"])
        for track in tracks
        if track["track"].split("_")[1].upper() == maneuver
    ]
    tail_count = len(maneuver_tracks) // 10
    scene_summary[maneuver] = {
        "n_scenes": len(maneuver_scenes),
        "median_rmse_m": median(maneuver_scenes),
        "p90_rmse_m": percentile(maneuver_scenes, 0.90),
        "scenes_rmse_le_8": sum(value <= 8 for value in maneuver_scenes),
    }
    tail_summary[maneuver] = (
        sum(value * value for value in sorted(maneuver_tracks, reverse=True)[:tail_count])
        / sum(value * value for value in maneuver_tracks)
        * 100
    )
scene_ok = True
tail_ok = True
for maneuver, expected in expansion["scene_level"].items():
    actual = scene_summary[maneuver]
    scene_ok = scene_ok and actual["n_scenes"] == expected["n_scenes"]
    scene_ok = scene_ok and close(actual["median_rmse_m"], expected["median_rmse_m"])
    scene_ok = scene_ok and close(actual["p90_rmse_m"], expected["p90_rmse_m"])
    scene_ok = scene_ok and actual["scenes_rmse_le_8"] == expected["scenes_rmse_le_8"]
    tail_ok = tail_ok and close(tail_summary[maneuver], expected["top_10pct_track_squared_error_share_pct"])
check("四类场景级分布", scene_ok, scene_summary, expansion["scene_level"], str(EXPANSION.relative_to(ROOT)))
check("四类高误差前10%航迹平方误差贡献", tail_ok, tail_summary, {k: v["top_10pct_track_squared_error_share_pct"] for k, v in expansion["scene_level"].items()}, str(EXPANSION.relative_to(ROOT)))

baseline_pairs = {Path(item["file"]).stem: float(item["rmse_m"]) for item in baseline["per_file"]}
paired_keys = sorted(set(scene_rmse) & set(baseline_pairs))
check("同场景配对数", len(paired_keys) == 2000, len(paired_keys), 2000, baseline_source)
check("参考实现逐场景RMSE均高于V7", all(baseline_pairs[k] > scene_rmse[k] for k in paired_keys), sum(baseline_pairs[k] > scene_rmse[k] for k in paired_keys), 2000, baseline_source)
baseline_values = list(baseline_pairs.values())
check("参考实现场景中位数", close(median(baseline_values), 62.051780967794656), median(baseline_values), 62.051780967794656, baseline_source)
check("参考实现场景P90", close(percentile(baseline_values, 0.90), 330.93686952631066), percentile(baseline_values, 0.90), 330.93686952631066, baseline_source)
top_n = len(baseline_values) // 10
top_share = sum(v * v for v in sorted(baseline_values, reverse=True)[:top_n]) / sum(v * v for v in baseline_values) * 100
check("参考实现前10%场景平方误差贡献", close(top_share, 75.00145069691347), top_share, 75.00145069691347, baseline_source)

for label, evidence, section, expected in (
    ("T=1空闲P95", latency_idle, "one_frame_input", 20.532025000602516),
    ("T=100空闲P95", latency_idle, "full_sequence_input", 87.31399499974941),
    ("T=100高负载P95", latency_busy, "full_sequence_input", 108.74213500001133),
):
    samples = [float(v) for v in evidence[section]["samples_ms"]]
    actual = percentile(samples, 0.95)
    source = str((ACCEPTANCE / ("forward_latency_concurrent.json" if evidence is latency_busy else "forward_latency.json")).relative_to(ROOT))
    check(f"{label}样本数", len(samples) == 300, len(samples), 300, source)
    check(f"{label}原始样本重算", close(actual, expected), actual, expected, source)

ablation_source = str((ACCEPTANCE / "ablation_v7.json").relative_to(ROOT))
ablation_expected = {"full": 7.488174720926246, "dual_no_gat": 8.853243887952592, "radar_only": 8.22165575704485, "no_temporal": 29.623392413907933}
ablation_actual = {key: value["position_rmse_m"] for key, value in ablation["results"].items()}
check("推理时消融RMSE", all(close(ablation_actual[k], v) for k, v in ablation_expected.items()), ablation_actual, ablation_expected, ablation_source)
check("消融未重新训练边界", ablation["interpretation"] == "Inference-time component masking; no ablated model was retrained.", ablation["interpretation"], "Inference-time component masking; no ablated model was retrained.", ablation_source)

matrix = review_46["review_item_6"]["association_matrix"]
cell_count = sum(len(cell["sensor_combinations"]) for cell in matrix["cells"])
check("关联扰动场景数", review_46["review_item_6"]["scenario_count"] == 32, review_46["review_item_6"]["scenario_count"], 32, "artifacts/metrics/midterm_review/items_4_6_evidence.json")
check("关联扰动判定单元数", cell_count == 80, cell_count, 80, "artifacts/metrics/midterm_review/items_4_6_evidence.json")
check("关联扰动最低单元", close(matrix["minimum_cell_accuracy"], 0.9096345514950166), matrix["minimum_cell_accuracy"], 0.9096345514950166, "artifacts/metrics/midterm_review/items_4_6_evidence.json")
check("关联扰动全部单元超过85%", matrix["all_cells_passed"] is True, matrix["all_cells_passed"], True, "artifacts/metrics/midterm_review/items_4_6_evidence.json")

gate = review_13["review_item_2"]["observed_weight_ranges"]["active_stream_gate_mean"]
check("64场景双流门控范围", close(gate["minimum"], 0.6546018823222383) and close(gate["maximum"], 0.7965175794275086), gate, {"minimum": 0.6546018823222383, "maximum": 0.7965175794275086}, "artifacts/metrics/midterm_review/items_1_3_evidence.json")
pressure = {item["condition"]: item for item in review_13["review_item_3"]["conditions"]}
pressure_expected = {"baseline": 7.4505775929909195, "eo_channel_missing": 7.633507255294867, "random_measurement_drop_20pct": 8.986472185189687, "all_source_occlusion_0_5s": 11.970814291061531, "datalink_delay_0_2s": 8.964644138387955}
check("64场景信息不完备RMSE", all(close(pressure[k]["position_rmse_m"], v) for k, v in pressure_expected.items()), {k: pressure[k]["position_rmse_m"] for k in pressure_expected}, pressure_expected, "artifacts/metrics/midterm_review/items_1_3_evidence.json")
check("64场景各压力条件均为有限输出", all(close(item["finite_output_ratio"], 1.0) for item in pressure.values()), {k: v["finite_output_ratio"] for k, v in pressure.items()}, "all 1.0", "artifacts/metrics/midterm_review/items_1_3_evidence.json")

profile = review_1012["review_item_10"]["acceptance_profile"]
check("V7质量配置场景结构", profile["duration_s"] == 10.0 and profile["frame_period_s"] == 0.1 and profile["targets_per_scenario"] == 3 and profile["platforms"] == 5, {k: profile[k] for k in ["duration_s", "frame_period_s", "targets_per_scenario", "platforms"]}, {"duration_s": 10.0, "frame_period_s": 0.1, "targets_per_scenario": 3, "platforms": 5}, "artifacts/metrics/midterm_review/items_10_12_evidence.json")
availability = {k: v["availability_rate"] for k, v in review_1012["review_item_10"]["acceptance_dataset_empirical_audit"]["sensors"].items()}
check("64场景输入可用率", all(close(availability[k], expansion["input_availability_64_scenes"][k]) for k in availability), availability, expansion["input_availability_64_scenes"], str(EXPANSION.relative_to(ROOT)))

maneuvers = {item["scenario_class"]: item for item in review_1012["review_item_11"]["maneuver_audit"]["classes"]}
check("V7机动审计每类48条航迹", all(maneuvers[k]["track_count"] == 48 for k in ["CV", "CA", "CT", "CSSW"]), {k: maneuvers[k]["track_count"] for k in ["CV", "CA", "CT", "CSSW"]}, "all 48", "artifacts/metrics/midterm_review/items_10_12_evidence.json")
check("V8机动审计5条航迹", maneuvers["COMPOSITE-JAMMING"]["track_count"] == 5, maneuvers["COMPOSITE-JAMMING"]["track_count"], 5, "artifacts/metrics/midterm_review/items_10_12_evidence.json")

check("扩展图绑定同一任务号", expansion["job_id"] == JOB_ID, expansion["job_id"], JOB_ID, str(EXPANSION.relative_to(ROOT)))
check("扩展图绑定同一权重摘要", expansion["model_sha256"] == model_hash, expansion["model_sha256"], model_hash, str(EXPANSION.relative_to(ROOT)))

required_phrases = [
    "事实复核：** 2026 年 9 月 21 日",
    "中期报告内部并不一致",
    "现有生成代码实际通过距离跳变实现",
    "不据此断言其他每个版本在相同条件下均不达标",
    "不代替甲方的验收结论",
]
check("正文包含关键事实边界", all(item in report_text for item in required_phrases), [item for item in required_phrases if item in report_text], required_phrases, str(FINAL_TEXT.relative_to(ROOT)))
for forbidden in ("唯一达到当前精度要求", "距离或相位欺骗"):
    check(f"正文已移除无充分证据表述：{forbidden}", forbidden not in report_text, report_text.count(forbidden), 0, str(FINAL_TEXT.relative_to(ROOT)))

failed = [item for item in checks if item["status"] == "failed"]
payload = {
    "audit_date": str(date(2026, 9, 21)),
    "scope": "当前工作区可复核的结题报告事实、数值及证据边界",
    "job_id": JOB_ID,
    "summary": {"total": len(checks), "passed": len(checks) - len(failed), "failed": len(failed)},
    "conclusion": "当前工作区可复核项目全部通过" if not failed else "存在未通过复核项",
    "limitations": [
        "不替代甲方验收、现场测试、正式签收或论文专利外部状态凭证。",
        "V7被确定为结题最终主批次属于项目组口径；当前审计未对所有历史版本执行同协议完整复测。",
        "IMM-JPDA参考实现与V7输入模态不完全相同，96.04%不能解释为严格同输入的纯算法净增益。",
        "T=1前向时间只覆盖注册张量到模型状态输出，不代表完整在线软件链时延。",
    ],
    "checks": checks,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload["summary"], ensure_ascii=False))
print(OUTPUT)
if failed:
    for item in failed:
        print("FAILED", item["name"], item["actual"], item["expected"])
    raise SystemExit(1)
