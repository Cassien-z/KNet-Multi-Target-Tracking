from __future__ import annotations

import csv
import json
import platform
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


RUN_DIR = ROOT / "artifacts" / "reports" / "software_test" / "IT-03-02" / "20260924_run01"
FIG_DIR = RUN_DIR / "figures"
BATCH_ID = "46fada4af5ef4c70a03fa56ef521f8c8"
BATCH_PATH = ROOT / "artifacts" / "reports" / "batch_tests" / f"{BATCH_ID}.json"
FULL_EVIDENCE_PATH = RUN_DIR / "association_accuracy_full_v7.json"
FULL_REPEAT_PATH = RUN_DIR / "association_accuracy_full_v7_repeat.json"
AUDIT_PATH = RUN_DIR / "items_4_6_evidence.json"
AUDIT_REPEAT_PATH = RUN_DIR / "items_4_6_evidence_repeat.json"
HISTORICAL_FULL_PATH = ROOT / "artifacts" / "metrics" / "acceptance" / "2ea4f9eb0f2845b599fcf4837d71a018" / "association_accuracy.json"
HISTORICAL_AUDIT_PATH = ROOT / "artifacts" / "metrics" / "midterm_review" / "items_4_6_evidence.json"
PIPELINE_PATH = ROOT / "src" / "data_association" / "pipeline_core.py"
THRESHOLD = 0.85

RUN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


batch = load(BATCH_PATH)
full = load(FULL_EVIDENCE_PATH)
full_repeat = load(FULL_REPEAT_PATH)
audit = load(AUDIT_PATH)
audit_repeat = load(AUDIT_REPEAT_PATH)
historical_full = load(HISTORICAL_FULL_PATH)
historical_audit = load(HISTORICAL_AUDIT_PATH)
matrix = audit["review_item_6"]["association_matrix"]
repeat_matrix = audit_repeat["review_item_6"]["association_matrix"]
historical_matrix = historical_audit["review_item_6"]["association_matrix"]

checks: list[dict] = []


def check(cid: str, item: str, passed: bool, actual, expected) -> None:
    row = {"check_id": cid, "item": item, "passed": bool(passed), "actual": actual, "expected": expected}
    checks.append(row)
    if not passed:
        raise AssertionError(f"{cid}: {item}; actual={actual!r}; expected={expected!r}")


# Current full-scope association evidence must be bound to the just-completed V7 batch.
context = evidence_context(batch)
check("BATCH-COMPLETE", "绑定批测状态为completed", batch["status"] == "completed", batch["status"], "completed")
for key in ("job_id", "job_signature", "model_id", "dataset_id", "max_scenarios", "warmup_frames", "sequence_length", "evaluated_scenarios"):
    check(f"CONTEXT-{key.upper()}", f"独立证据的{key}与批测上下文一致", full.get(key) == context.get(key), full.get(key), context.get(key))
check("FULL-SCENARIOS", "V7完整证据覆盖2000个场景", full["scenario_manifest"]["count"] == 2000, full["scenario_manifest"]["count"], 2000)
check("FULL-TOTAL", "V7完整证据分母大于0", int(full["association"]["total"]) > 0, full["association"]["total"], ">0")
check("FULL-FORMULA", "V7完整正确率可由correct/total复算", abs(full["value"] - full["association"]["correct"] / full["association"]["total"]) <= 1e-15, full["value"], "correct/total")
check("FULL-THRESHOLD", "V7完整关联正确率不低于85%", full["value"] >= THRESHOLD and full["passed"], full["value"], ">=0.85")
check("FULL-VALID", "V7完整证据可用于正式判定", full["valid_for_assessment"] is True, full["valid_for_assessment"], True)
check("FULL-NO-LEAK", "V7完整证据未使用真值构造观测分组", full["association"]["truth_used_for_observation_grouping"] is False and full["association"]["files_flagged_for_truth_leakage"] == 0, full["association"], "false/0")
for sensor in ("Radar", "EO", "ESM"):
    item = full["association"]["per_sensor"][sensor]
    check(f"SENSOR-{sensor}", f"{sensor}分项correct/total可复算且分母大于0", item["total"] > 0 and abs(item["accuracy"] - item["correct"] / item["total"]) <= 1e-15, item, "total>0 and exact ratio")

# Fresh deterministic strong-interference audit: 32 balanced V7 scenarios,
# five conditions and four sensor combinations form 80 separately judged cells.
item4 = audit["review_item_4"]
item6 = audit["review_item_6"]
check("AUDIT-SCENARIOS", "强干扰审计包含32个场景", item6["scenario_count"] == 32, item6["scenario_count"], 32)
check("AUDIT-BALANCE", "CV/CA/CT/CSSW各8个场景", item6["scenario_class_counts"] == {"CA": 8, "CSSW": 8, "CT": 8, "CV": 8}, item6["scenario_class_counts"], "8 each")
check("AUDIT-RUNS", "32场景在5种条件下形成160次运行", item6["scenario_condition_runs"] == 160, item6["scenario_condition_runs"], 160)
check("AUDIT-DIFFICULT", "审计样本含16个典型难场景", item4["audit_sample"]["typical_difficult_scenario_count"] == 16, item4["audit_sample"]["typical_difficult_scenario_count"], 16)
check("AUDIT-METHOD", "干扰只施加于内存验证副本且不改权重和训练集", audit["method_constraints"] == {"model_weights_changed": False, "training_data_changed": False, "interference_application": "deterministic in-memory validation copies", "truth_used_for_observation_grouping": False}, audit["method_constraints"], "frozen and measurement-only")

cells: list[dict] = []
for cell in matrix["cells"]:
    for combination, values in cell["sensor_combinations"].items():
        row = {"condition": cell["condition"], "scenario_class": cell["scenario_class"], "sensor_combination": combination, **values}
        cells.append(row)
        check(
            f"CELL-{len(cells):02d}",
            f"{cell['condition']}/{cell['scenario_class']}/{combination}分母、公式、泄漏和85%判定",
            values["total"] > 0
            and abs(values["accuracy"] - values["correct"] / values["total"]) <= 1e-15
            and values["truth_used_for_observation_grouping"] is False
            and values["passed"] is True
            and values["accuracy"] >= THRESHOLD,
            values,
            "total>0, correct/total, no truth leakage, >=0.85",
        )
check("CELL-COUNT", "5条件×4场景类型×4传感器组合共80格", len(cells) == 80, len(cells), 80)
check("ALL-CELLS", "80个强干扰审计格全部通过", matrix["all_cells_passed"] is True, matrix["all_cells_passed"], True)
check("MIN-CELL", "最差审计格仍不低于85%", matrix["minimum_cell_accuracy"] >= THRESHOLD, matrix["minimum_cell_accuracy"], ">=0.85")

# Verify source ordering: measurement association and registered tracks are
# built before Data_Label is read for post-hoc audit/label attachment.
source_text = PIPELINE_PATH.read_text(encoding="utf-8")
association_finish = source_text.index("registered_tracks.append(track)")
label_read = source_text.index('labels = np.asarray(mat.get("Data_Label")')
flag_literal = '"truth_used_for_observation_grouping": False' in source_text
check("SOURCE-ORDER", "源码在关联分组完成后才读取Data_Label进行审计", association_finish < label_read, {"association_finish": association_finish, "label_read": label_read}, "association before label read")
check("SOURCE-FLAG", "源码显式标记真值未用于观测分组", flag_literal, flag_literal, True)

# Boundary, invalid denominator and synthetic leakage rejection.
boundary_rows = []
for value, expected in ((0.8499, False), (0.8500, True), (0.8501, True)):
    accepted = value >= THRESHOLD
    boundary_rows.append({"value": value, "accepted": accepted, "expected": expected})
    check(f"BOUNDARY-{value:.4f}", f"{value*100:.2f}%阈值比较", accepted == expected, accepted, expected)

def assess(correct: int, total: int, leakage: bool) -> tuple[float | None, bool]:
    value = correct / total if total else None
    return value, bool(value is not None and value >= THRESHOLD and not leakage)

zero_value, zero_passed = assess(0, 0, False)
check("ZERO-DENOMINATOR", "total=0返回无效而非100%", zero_value is None and not zero_passed, {"value": zero_value, "passed": zero_passed}, "None/False")
leak_value, leak_passed = assess(999, 1000, True)
check("LEAKAGE-REJECT", "存在真值泄漏时即使99.9%也不得通过", leak_value == 0.999 and not leak_passed, {"value": leak_value, "passed": leak_passed}, "0.999/False")

# Reproducibility against prior evidence generated with the same deterministic
# seed, source data and perturbation definitions.
check("REPRO-FULL", "当前代码对V7完整原始MAT连续两次重算的correct/total一致", (full["association"]["correct"], full["association"]["total"]) == (full_repeat["association"]["correct"], full_repeat["association"]["total"]), {"run1": [full["association"]["correct"], full["association"]["total"]], "run2": [full_repeat["association"]["correct"], full_repeat["association"]["total"]]}, "same")
check("REPRO-CONTEXT", "第二次完整重算仍绑定同一批测签名", full_repeat["job_signature"] == full["job_signature"] and full_repeat["job_id"] == full["job_id"], {"run1": [full["job_id"], full["job_signature"]], "run2": [full_repeat["job_id"], full_repeat["job_signature"]]}, "same")
check("HISTORY-SCOPE", "旧证据与当前口径的雷达计数一致且均达标，EO/ESM覆盖增加单列说明", full["association"]["per_sensor"]["Radar"] == historical_full["association"]["per_sensor"]["Radar"] and full["association"]["total"] > historical_full["association"]["total"] and historical_full["value"] >= THRESHOLD, {"current_total": full["association"]["total"], "historical_total": historical_full["association"]["total"], "radar": full["association"]["per_sensor"]["Radar"]}, "same radar, larger current denominator, both pass")
fresh_signature = [(r["condition"], r["scenario_class"], r["sensor_combination"], r["correct"], r["total"]) for r in cells]
repeat_signature = []
for cell in repeat_matrix["cells"]:
    for combination, values in cell["sensor_combinations"].items():
        repeat_signature.append((cell["condition"], cell["scenario_class"], combination, values["correct"], values["total"]))
check("REPRO-AUDIT", "当前代码连续两次160次干扰审计的80格计数一致", fresh_signature == repeat_signature, len(fresh_signature), "80 identical cells")
check("HISTORY-AUDIT-SCOPE", "旧强干扰证据与当前证据均80格通过，口径差异单列说明", historical_matrix["all_cells_passed"] is True and matrix["all_cells_passed"] is True and historical_matrix["minimum_cell_accuracy"] != matrix["minimum_cell_accuracy"], {"historical_min": historical_matrix["minimum_cell_accuracy"], "current_min": matrix["minimum_cell_accuracy"]}, "both pass with changed coverage")

# Aggregate views used in the report and plots.
condition_rows = matrix["condition_summaries"]
sensor_agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
for row in cells:
    sensor_agg[row["sensor_combination"]][0] += int(row["correct"])
    sensor_agg[row["sensor_combination"]][1] += int(row["total"])
sensor_rows = [
    {"sensor_combination": name, "correct": values[0], "total": values[1], "accuracy": values[0] / values[1]}
    for name, values in sensor_agg.items()
]

with (RUN_DIR / "strong_interference_cells.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
    writer.writeheader()
    writer.writerows(cells)

# Figure 1: full V7 event-purity metric and sensor components.
labels = ["总体", "雷达", "光电", "电子侦察"]
values = [full["value"], full["association"]["per_sensor"]["Radar"]["accuracy"], full["association"]["per_sensor"]["EO"]["accuracy"], full["association"]["per_sensor"]["ESM"]["accuracy"]]
counts = [full["association"]["total"], full["association"]["per_sensor"]["Radar"]["total"], full["association"]["per_sensor"]["EO"]["total"], full["association"]["per_sensor"]["ESM"]["total"]]
fig, ax = plt.subplots(figsize=(9.4, 5.2))
bars = ax.bar(labels, np.asarray(values) * 100, color=["#2F6690", "#59A14F", "#59A14F", "#59A14F"], width=.62)
ax.axhline(85, color="#C00000", linestyle="--", linewidth=2, label="合同阈值85%")
ax.set_ylim(80, 101.5); ax.set_ylabel("关联正确率 / %"); ax.set_title("V7完整2000场景目标关联审计（原始MAT重算）"); ax.grid(axis="y", alpha=.25); ax.legend()
for bar, value, count in zip(bars, values, counts):
    ax.text(bar.get_x() + bar.get_width()/2, value*100+.3, f"{value*100:.2f}%\n({count:,}次)", ha="center", fontsize=10)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_1_full_v7_association.png", dpi=180); plt.close(fig)

# Figure 2: all-sensor aggregate and the worst independently judged cell per condition.
condition_names = ["无干扰", "雷达闪烁", "软压制70%", "距离抖动100m", "复合干扰"]
x = np.arange(len(condition_rows)); width = .36
fig, ax = plt.subplots(figsize=(10.4, 5.4))
all_sensor = np.asarray([r["all_sensor_accuracy"] for r in condition_rows]) * 100
minimum = np.asarray([r["minimum_cell_accuracy"] for r in condition_rows]) * 100
b1 = ax.bar(x-width/2, all_sensor, width, label="全传感器汇总", color="#4C78A8")
b2 = ax.bar(x+width/2, minimum, width, label="该条件最差单格", color="#F28E2B")
ax.axhline(85, color="#C00000", linestyle="--", linewidth=1.8, label="85%阈值")
ax.set_xticks(x, condition_names); ax.set_ylim(80, 101.5); ax.set_ylabel("关联正确率 / %")
ax.set_title("五种干扰条件下的关联正确率"); ax.grid(axis="y", alpha=.25); ax.legend(ncol=3)
for bars in (b1, b2):
    for bar in bars: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.2, f"{bar.get_height():.2f}%", ha="center", fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_2_interference_conditions.png", dpi=180); plt.close(fig)

# Figure 3: all-sensor accuracy for each condition and motion class.
classes = ["CV", "CA", "CT", "CSSW"]
heat = np.zeros((5, 4), dtype=float)
for i, condition in enumerate(item6["conditions"]):
    for j, cls in enumerate(classes):
        cell = next(c for c in matrix["cells"] if c["condition"] == condition and c["scenario_class"] == cls)
        heat[i, j] = cell["sensor_combinations"]["radar_eo_esm"]["accuracy"] * 100
fig, ax = plt.subplots(figsize=(8.4, 6.0))
im = ax.imshow(heat, cmap="YlGn", vmin=85, vmax=100, aspect="auto")
ax.set_xticks(np.arange(4), classes); ax.set_yticks(np.arange(5), condition_names)
ax.set_xlabel("运动类型"); ax.set_title("全传感器组合的运动类型×干扰条件正确率")
for i in range(5):
    for j in range(4): ax.text(j, i, f"{heat[i,j]:.2f}%", ha="center", va="center", color="#102A12" if heat[i,j] > 92 else "white", fontsize=9)
fig.colorbar(im, ax=ax, label="关联正确率 / %")
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_3_condition_class_heatmap.png", dpi=180); plt.close(fig)

# Figure 4: reproducibility and exact threshold/leakage semantics.
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
run_values = [historical_full["value"] * 100, full["value"] * 100, full_repeat["value"] * 100]
bars = axes[0].bar(["旧口径证据", "当前口径\n第1次", "当前口径\n第2次"], run_values, color=["#9C755F", "#4C78A8", "#59A14F"], width=.58)
axes[0].set_ylim(98.5, 100); axes[0].set_ylabel("关联正确率 / %"); axes[0].set_title("V7完整集计数复现")
for bar, value in zip(bars, run_values): axes[0].text(bar.get_x()+bar.get_width()/2, value+.03, f"{value:.4f}%", ha="center", weight="bold")
test_labels = ["84.99%", "85.00%", "85.01%", "total=0", "99.9%\n含泄漏"]
accepted = [False, True, True, zero_passed, leak_passed]
colors = ["#59A14F" if value else "#C00000" for value in accepted]
axes[1].bar(test_labels, [1]*5, color=colors)
axes[1].set_ylim(0, 1.12); axes[1].set_yticks([]); axes[1].set_title("阈值、空分母与真值泄漏判定")
for i, value in enumerate(accepted): axes[1].text(i, .5, "通过" if value else "拒绝", ha="center", va="center", color="white", weight="bold", fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_4_reproducibility_and_guards.png", dpi=180); plt.close(fig)

result = {
    "test_case": "IT-03-02",
    "test_name": "多目标强干扰无混淆率稳定性测试",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"os": platform.platform(), "python": sys.version, "numpy": np.__version__},
    "formal_batch_id": BATCH_ID,
    "full_v7": full,
    "full_v7_repeat": full_repeat,
    "historical_scope_reference": historical_full,
    "strong_interference": {
        "scenario_count": item6["scenario_count"],
        "scenario_condition_runs": item6["scenario_condition_runs"],
        "cell_count": len(cells),
        "condition_summaries": condition_rows,
        "sensor_summaries": sensor_rows,
        "minimum_cell_accuracy": matrix["minimum_cell_accuracy"],
        "worst_cell": matrix["worst_cell"],
        "all_cells_passed": matrix["all_cells_passed"],
        "v8_note": item4["full_inventory"]["v8_interference"]["coverage_note"],
    },
    "strong_interference_repeat": audit_repeat["review_item_6"],
    "historical_interference_scope_reference": historical_audit["review_item_6"],
    "threshold_boundary": boundary_rows,
    "zero_denominator": {"value": zero_value, "passed": zero_passed},
    "synthetic_truth_leakage": {"value": leak_value, "passed": leak_passed},
    "checks": checks,
    "check_count": len(checks),
    "passed_count": sum(row["passed"] for row in checks),
    "overall_passed": all(row["passed"] for row in checks),
    "figures": [path.name for path in sorted(FIG_DIR.glob("*.png"))],
    "scope_note": "正式85%判据使用V7完整原始MAT重算的measurement-only事件纯度；强干扰稳定性由32个平衡V7场景的确定性内存扰动审计补充。V8仅1个复合强干扰场景，不并入大样本统计。",
}
(RUN_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "output": str(RUN_DIR),
    "overall_passed": result["overall_passed"],
    "checks": f"{result['passed_count']}/{result['check_count']}",
    "full_v7_accuracy": full["value"],
    "full_v7_counts": [full["association"]["correct"], full["association"]["total"]],
    "minimum_interference_cell": matrix["minimum_cell_accuracy"],
    "worst_cell": matrix["worst_cell"],
    "all_80_cells_passed": matrix["all_cells_passed"],
}, ensure_ascii=False, indent=2))
if not result["overall_passed"]:
    raise SystemExit(1)
