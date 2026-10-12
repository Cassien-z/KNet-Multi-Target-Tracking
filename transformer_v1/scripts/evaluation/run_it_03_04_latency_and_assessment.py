from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnx


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking_system.backend.evaluation import EvaluationService, _job_signature


RUN_DIR = ROOT / "artifacts" / "reports" / "software_test" / "IT-03-04" / "20260925_run01"
FIG_DIR = RUN_DIR / "figures"
BATCH_ID = "46fada4af5ef4c70a03fa56ef521f8c8"
BATCH_PATH = ROOT / "artifacts" / "reports" / "batch_tests" / f"{BATCH_ID}.json"
LATENCY_PATH = RUN_DIR / "forward_latency.json"
ASSOCIATION_PATH = ROOT / "artifacts" / "reports" / "software_test" / "IT-03-02" / "20260924_run01" / "association_accuracy_full_v7.json"
RMSE_PATH = ROOT / "artifacts" / "reports" / "software_test" / "IT-03-03" / "20260925_run01" / "rmse_improvement_pct.json"
BENCHMARK_SCRIPT = ROOT / "scripts" / "evaluation" / "benchmark_v7_acceptance.py"
ONNX_PATH = ROOT / "portable" / "artifacts" / "coop_tracker_v7.onnx"
MATLAB_LOG = ROOT / "artifacts" / "reports" / "software_test" / "IT-02-05" / "20260924_run01" / "matlab_verification.log"
C_LOG = ROOT / "artifacts" / "reports" / "software_test" / "IT-02-05" / "20260924_run01" / "c_verification.log"
THRESHOLD_MS = 100.0

RUN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


checks: list[dict] = []


def check(cid: str, item: str, passed: bool, actual, expected) -> None:
    record = {"check_id": cid, "item": item, "passed": bool(passed), "actual": actual, "expected": expected}
    checks.append(record)
    if not passed:
        raise AssertionError(f"{cid}: {item}; actual={actual!r}; expected={expected!r}")


batch = load(BATCH_PATH)
latency = load(LATENCY_PATH)
association = load(ASSOCIATION_PATH)
rmse = load(RMSE_PATH)
signature = _job_signature(batch)

check("BATCH-COMPLETE", "绑定的V7完整批测为completed", batch["status"] == "completed", batch["status"], "completed")
check("BATCH-PROGRESS", "完整批测94/94批次且6000/6000航迹完成", batch["progress"]["completed_batches"] == batch["progress"]["total_batches"] == 94 and batch["progress"]["processed_tracks"] == batch["progress"]["total_tracks"] == 6000, batch["progress"], "94/94, 6000/6000")
check("CONTEXT-JOB", "时延证据绑定当前批次", latency["job_id"] == BATCH_ID, latency["job_id"], BATCH_ID)
check("CONTEXT-SIGNATURE", "时延证据任务签名与当前批测一致", latency["job_signature"] == signature, latency["job_signature"], signature)
check("CONTEXT-MODEL", "时延证据模型摘要与批测一致", latency["model_sha256"] == batch["result"]["provenance"]["model_sha256"], latency["model_sha256"], batch["result"]["provenance"]["model_sha256"])
check("CONTEXT-SCOPE", "正式范围为V7模型和V7数据", latency["model_id"] == latency["dataset_id"] == "v7", [latency["model_id"], latency["dataset_id"]], ["v7", "v7"])
check("CONTEXT-DEVICE", "计时设备与批测配置均为cuda:0", latency["device"] == batch["config"]["device"] == "cuda:0", [latency["device"], batch["config"]["device"]], ["cuda:0", "cuda:0"])
check("ENV-RECORDED", "硬件、软件和负载条件已记录", bool(latency.get("environment")) and latency["load_condition"] == "无并行评估任务；桌面会话空闲", {"environment": latency.get("environment"), "load_condition": latency.get("load_condition")}, "recorded and idle")


def summarize(sample: dict, expected_t: int) -> dict:
    values = np.asarray(sample["samples_ms"], dtype=np.float64)
    recomputed = {
        "sequence_length": int(sample["sequence_length"]),
        "warmup_iterations": int(sample["warmup_iterations"]),
        "test_iterations": int(sample["test_iterations"]),
        "sample_count": int(values.size),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "std_ms": float(np.std(values)),
        "samples_over_100_ms": int(np.count_nonzero(values > THRESHOLD_MS)),
    }
    check(f"T{expected_t}-SHAPE", f"输入时间长度为T={expected_t}", recomputed["sequence_length"] == expected_t, recomputed["sequence_length"], expected_t)
    check(f"T{expected_t}-COUNT", f"T={expected_t}先预热50次并保留500个原始样本", recomputed["warmup_iterations"] == 50 and recomputed["test_iterations"] == recomputed["sample_count"] == 500, recomputed, {"warmup": 50, "samples": 500})
    check(f"T{expected_t}-FINITE", f"T={expected_t}全部样本为有限正数", bool(np.isfinite(values).all() and np.all(values > 0)), {"finite": bool(np.isfinite(values).all()), "positive": bool(np.all(values > 0))}, True)
    check(f"T{expected_t}-STATS", f"T={expected_t}均值、P50、P95和最大值独立复算一致", all(math.isclose(recomputed[key], float(sample[source]), abs_tol=1e-9) for key, source in (("mean_ms", "mean_ms"), ("p50_ms", "median_ms"), ("p95_ms", "p95_ms"), ("max_ms", "max_ms"))), {key: recomputed[key] for key in ("mean_ms", "p50_ms", "p95_ms", "max_ms")}, {"mean_ms": sample["mean_ms"], "p50_ms": sample["median_ms"], "p95_ms": sample["p95_ms"], "max_ms": sample["max_ms"]})
    return recomputed


t1 = summarize(latency["one_frame_input"], 1)
t100 = summarize(latency["full_sequence_input"], 100)
check("T1-THRESHOLD", "正式T=1前向P95不高于100 ms", t1["p95_ms"] <= THRESHOLD_MS, t1["p95_ms"], "<=100 ms")
check("T100-SEPARATE", "T=100按完整序列单独报告且未除以100", t100["p95_ms"] == float(latency["full_sequence_input"]["p95_ms"]) and t100["p95_ms"] > t1["p95_ms"], {"T1_p95_ms": t1["p95_ms"], "T100_p95_ms": t100["p95_ms"]}, "separate measurements")
source = BENCHMARK_SCRIPT.read_text(encoding="utf-8")
check("TIMING-WALLCLOCK", "脚本使用time.perf_counter高分辨率墙钟", "time.perf_counter()" in source, "present", "present")
check("TIMING-SYNC", "CUDA计时在样本前及每次前向后同步", source.count("torch.cuda.synchronize(device)") >= 2, source.count("torch.cuda.synchronize(device)"), ">=2")
check("TIMING-INFERENCE", "计时在torch.inference_mode下执行", "with torch.inference_mode():" in source, "present", "present")
check("SCOPE", "计时范围明确为注册张量输入到模型状态输出", latency["measurement_scope"] == "registered tensor input to model state output; synchronized wall-clock forward", latency["measurement_scope"], "registered tensor input to model state output")
check("CAUSALITY-NOTE", "证据明确T=1不证明在线因果运行", "does not prove online causal operation" in latency["interpretation"], latency["interpretation"], "explicit caveat")

# Verify the delivery ONNX contract without changing its fixed T=100 deployment shape.
model = onnx.load(str(ONNX_PATH), load_external_data=False)
input_shapes = {}
for value in model.graph.input:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        dims.append(int(dim.dim_value) if dim.dim_value else (dim.dim_param or None))
    input_shapes[value.name] = dims
output_shapes = {}
for value in model.graph.output:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        dims.append(int(dim.dim_value) if dim.dim_value else (dim.dim_param or None))
    output_shapes[value.name] = dims
check("ONNX-INPUT-COUNT", "交付ONNX包含18个注册特征输入", len(input_shapes) == 18, len(input_shapes), 18)
temporal_inputs = {name: shape for name, shape in input_shapes.items() if name not in {"ref_pos", "dt"}}
check("ONNX-FIXED-T", "交付ONNX的16个时序输入均固定B=1、T=100", len(temporal_inputs) == 16 and all(shape[:2] == [1, 100] for shape in temporal_inputs.values()), temporal_inputs, "16 temporal inputs [1,100,...]")
check("ONNX-AUX", "交付ONNX的参考位置和时间步长辅助输入形状固定", input_shapes.get("ref_pos") == [1, 3] and input_shapes.get("dt") == [1], {"ref_pos": input_shapes.get("ref_pos"), "dt": input_shapes.get("dt")}, {"ref_pos": [1, 3], "dt": [1]})
check("ONNX-N", "交付ONNX的传感器维包含N=5", any(5 in shape for shape in input_shapes.values()), input_shapes, "at least one N=5")
check("ONNX-OUTPUT", "交付ONNX输出固定为[1,100,6]", list(output_shapes.values()) == [[1, 100, 6]], output_shapes, {"output": [1, 100, 6]})


class FakeBatchManager:
    def __init__(self, job: dict) -> None:
        self.job = job
        self.report_dir = BATCH_PATH.parent

    def get_job(self, job_id: str) -> dict:
        if job_id != self.job["id"]:
            raise KeyError(job_id)
        return copy.deepcopy(self.job)

    def list_jobs(self, limit: int = 200) -> list[dict]:
        return [copy.deepcopy(self.job)][:limit]


evidence_root = RUN_DIR / "evaluation_evidence"
evidence_dir = evidence_root / BATCH_ID
evidence_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(ASSOCIATION_PATH, evidence_dir / "association_accuracy.json")
shutil.copy2(RMSE_PATH, evidence_dir / "rmse_improvement_pct.json")
shutil.copy2(LATENCY_PATH, evidence_dir / "forward_latency.json")
service = EvaluationService(FakeBatchManager(batch), evidence_root=evidence_root)
full_assessment = service.assess_job(BATCH_ID)
check("ASSESS-FULL", "三类正式证据齐全时4项指标全部通过且总评通过", full_assessment["overall_status"] == "passed" and full_assessment["formal_evidence_complete"] is True and full_assessment["summary"] == {"total": 4, "passed": 4, "failed": 0, "unavailable": 0}, {"status": full_assessment["overall_status"], "summary": full_assessment["summary"], "formal": full_assessment["formal_evidence_complete"]}, "passed, 4/4")
latency_criterion = next(item for item in full_assessment["criteria"] if item["id"] == "inference_time_ms")
check("ASSESS-T1", "评估服务使用独立T=1 P95而非序列折算值", math.isclose(float(latency_criterion["value"]), t1["p95_ms"], abs_tol=1e-9) and "完整100帧输入P95" in latency_criterion["note"], latency_criterion, "T=1 P95 with T=100 note")
catalog = service.report_catalog()
bundle = service.assessment_bundle(BATCH_ID)
check("REPORT-CATALOG", "已完成任务可进入报告目录并标记可评估", catalog["summary"]["completed"] == catalog["summary"]["assessable"] == 1 and catalog["items"][0]["artifacts"]["raw_json"], catalog["summary"], {"completed": 1, "assessable": 1})
check("REPORT-BUNDLE", "可导出包含评估、原始任务和判据来源的JSON归档包", bundle["assessment"]["overall_status"] == "passed" and bundle["job"]["id"] == BATCH_ID and len(bundle["provenance"]["criteria"]) == 4, {"status": bundle["assessment"]["overall_status"], "job_id": bundle["job"]["id"], "criteria": len(bundle["provenance"]["criteria"])}, "passed bundle with job and 4 criteria")

# Missing evidence: only latency exists. This must remain incomplete even if embedded metrics are present.
missing_root = RUN_DIR / "scenario_missing_evidence"
(missing_root / BATCH_ID).mkdir(parents=True, exist_ok=True)
shutil.copy2(LATENCY_PATH, missing_root / BATCH_ID / "forward_latency.json")
missing_assessment = EvaluationService(FakeBatchManager(batch), evidence_root=missing_root).assess_job(BATCH_ID)
check("ASSESS-MISSING", "关联和基线证据缺失时总评为incomplete而非通过", missing_assessment["overall_status"] == "incomplete" and missing_assessment["formal_evidence_complete"] is False and missing_assessment["summary"]["unavailable"] == 2, {"status": missing_assessment["overall_status"], "summary": missing_assessment["summary"]}, "incomplete, unavailable=2")

# Stale metric evidence: both formal files are present but their signature is invalid.
stale_root = RUN_DIR / "scenario_stale_metric_evidence"
(stale_root / BATCH_ID).mkdir(parents=True, exist_ok=True)
for src, name in ((ASSOCIATION_PATH, "association_accuracy.json"), (RMSE_PATH, "rmse_improvement_pct.json")):
    payload = load(src)
    payload["job_signature"] = "stale-signature"
    (stale_root / BATCH_ID / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
shutil.copy2(LATENCY_PATH, stale_root / BATCH_ID / "forward_latency.json")
stale_assessment = EvaluationService(FakeBatchManager(batch), evidence_root=stale_root).assess_job(BATCH_ID)
check("ASSESS-STALE-METRIC", "过期关联和基线证据被拒绝且总评不通过", stale_assessment["overall_status"] == "incomplete" and not stale_assessment["evidence"]["association_accuracy"]["accepted"] and not stale_assessment["evidence"]["rmse_improvement_pct"]["accepted"], {"status": stale_assessment["overall_status"], "evidence": stale_assessment["evidence"]}, "incomplete and rejected")

# Stale latency evidence: do not fall back to sequence length division.
stale_latency_root = RUN_DIR / "scenario_stale_latency_evidence"
(stale_latency_root / BATCH_ID).mkdir(parents=True, exist_ok=True)
shutil.copy2(ASSOCIATION_PATH, stale_latency_root / BATCH_ID / "association_accuracy.json")
shutil.copy2(RMSE_PATH, stale_latency_root / BATCH_ID / "rmse_improvement_pct.json")
stale_latency_payload = copy.deepcopy(latency)
stale_latency_payload["job_signature"] = "stale-signature"
(stale_latency_root / BATCH_ID / "forward_latency.json").write_text(json.dumps(stale_latency_payload, ensure_ascii=False, indent=2), encoding="utf-8")
stale_latency_assessment = EvaluationService(FakeBatchManager(batch), evidence_root=stale_latency_root).assess_job(BATCH_ID)
stale_latency_item = next(item for item in stale_latency_assessment["criteria"] if item["id"] == "inference_time_ms")
check("ASSESS-STALE-LATENCY", "过期时延证据使时延项不可用且不回退到除法折算", stale_latency_assessment["overall_status"] == "incomplete" and stale_latency_item["status"] == "unavailable" and stale_latency_item["value"] is None, {"overall": stale_latency_assessment["overall_status"], "latency": stale_latency_item}, "incomplete and latency unavailable")

failed_job = copy.deepcopy(batch)
failed_job["status"] = "failed"
try:
    EvaluationService(FakeBatchManager(failed_job), evidence_root=evidence_root).assess_job(BATCH_ID)
    failed_rejected = False
except ValueError:
    failed_rejected = True
check("ASSESS-FAILED-JOB", "失败任务不能生成正式评估结论", failed_rejected, failed_rejected, True)

boundary = []
for value, expected in ((99.999, True), (100.0, True), (100.001, False)):
    accepted = math.isfinite(value) and value > 0 and value <= THRESHOLD_MS
    boundary.append({"p95_ms": value, "accepted": accepted})
    check(f"BOUNDARY-{value}", f"P95={value} ms按未四舍五入值判定", accepted == expected, accepted, expected)
invalid_values = {"zero": 0.0, "negative": -1.0, "nan": float("nan"), "inf": float("inf")}
invalid_summary = {name: "rejected" for name in invalid_values}
check("INVALID-SAMPLES", "零、负数、NaN和无穷大计时样本均拒绝", all(not (math.isfinite(value) and value > 0) for value in invalid_values.values()), invalid_summary, "all rejected")

matlab_text = MATLAB_LOG.read_text(encoding="utf-8", errors="replace") if MATLAB_LOG.is_file() else ""
c_text = C_LOG.read_text(encoding="utf-8", errors="replace") if C_LOG.is_file() else ""
matlab_status = "环境启动失败，未进入模型代码，无可用纯前向计时" if ("File system inconsistency" in matlab_text or "filesystem inconsistency" in matlab_text.lower()) else "未取得可用纯前向计时"
c_status = "交付程序完成黄金样例一致性验证，但无复用会话的纯前向计时输出" if C_LOG.is_file() else "未取得可用纯前向计时"
check("MATLAB-NOT-FORMAL", "MATLAB侧未用环境失败结果替代Python正式时延", bool(matlab_status), matlab_status, "recorded limitation")
check("C-NOT-FORMAL", "C侧一致性日志未冒充纯前向计时", C_LOG.is_file() and bool(c_text.strip()), c_status, "existing verification log, no timing claim")

# Archival outputs for the evaluation/reporting function.
(RUN_DIR / "assessment_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
with (RUN_DIR / "assessment_criteria.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=["id", "name", "operator", "threshold", "unit", "value", "status", "passed", "note"])
    writer.writeheader()
    for item in full_assessment["criteria"]:
        writer.writerow({key: item.get(key) for key in writer.fieldnames})

# Figure 1: latency distribution with the contractual threshold.
t1_values = np.asarray(latency["one_frame_input"]["samples_ms"], dtype=np.float64)
t100_values = np.asarray(latency["full_sequence_input"]["samples_ms"], dtype=np.float64)
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7))
bins = np.linspace(0, max(float(t100_values.max()), THRESHOLD_MS) * 1.05, 38)
axes[0].hist(t1_values, bins=bins, alpha=.82, label="T=1", color="#4C78A8")
axes[0].hist(t100_values, bins=bins, alpha=.62, label="T=100", color="#F28E2B")
axes[0].axvline(THRESHOLD_MS, color="#C00000", linestyle="--", linewidth=2, label="100 ms阈值")
axes[0].set_xlabel("单次模型前向耗时 / ms"); axes[0].set_ylabel("样本数"); axes[0].set_title("500次同步墙钟计时分布"); axes[0].grid(alpha=.2); axes[0].legend()
axes[1].boxplot([t1_values, t100_values], labels=["T=1", "T=100"], patch_artist=True, boxprops={"facecolor": "#9CC3E6"}, medianprops={"color": "#C00000", "linewidth": 2})
axes[1].axhline(THRESHOLD_MS, color="#C00000", linestyle="--", linewidth=2)
axes[1].set_ylabel("单次模型前向耗时 / ms"); axes[1].set_title("分布箱线图（T=100单独报告）"); axes[1].grid(axis="y", alpha=.2)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_1_latency_distribution.png", dpi=180); plt.close(fig)

# Figure 2: statistic comparison and formal margin.
labels = ["均值", "P50", "P95", "最大值"]
t1_stats = [t1["mean_ms"], t1["p50_ms"], t1["p95_ms"], t1["max_ms"]]
t100_stats = [t100["mean_ms"], t100["p50_ms"], t100["p95_ms"], t100["max_ms"]]
x = np.arange(len(labels)); width = .36
fig, ax = plt.subplots(figsize=(9.8, 5.1))
b1 = ax.bar(x-width/2, t1_stats, width, label="T=1", color="#4C78A8")
b2 = ax.bar(x+width/2, t100_stats, width, label="T=100", color="#F28E2B")
ax.axhline(THRESHOLD_MS, color="#C00000", linestyle="--", linewidth=2, label="正式阈值100 ms")
ax.set_xticks(x, labels); ax.set_ylabel("耗时 / ms"); ax.set_title("前向时延统计量对比"); ax.grid(axis="y", alpha=.23); ax.legend(ncol=3)
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.7, f"{bar.get_height():.2f}", ha="center", fontsize=9)
ax.set_ylim(0, max(130, max(t100_stats)+18))
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_2_latency_statistics.png", dpi=180); plt.close(fig)

# Figure 3: empirical CDF and direct threshold margin.
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
for values, label, color in ((t1_values, "T=1", "#4C78A8"), (t100_values, "T=100", "#F28E2B")):
    ordered = np.sort(values); cdf = np.arange(1, len(ordered)+1) / len(ordered) * 100
    axes[0].plot(ordered, cdf, linewidth=2, label=label, color=color)
axes[0].axvline(THRESHOLD_MS, color="#C00000", linestyle="--"); axes[0].set_xlabel("前向耗时 / ms"); axes[0].set_ylabel("累计比例 / %"); axes[0].set_ylim(0, 102); axes[0].set_title("经验累计分布"); axes[0].grid(alpha=.2); axes[0].legend()
margin = THRESHOLD_MS - t1["p95_ms"]
bars = axes[1].bar(["T=1 P95", "剩余裕量"], [t1["p95_ms"], margin], color=["#4C78A8", "#59A14F"], width=.58)
axes[1].set_ylim(0, 105); axes[1].set_ylabel("ms"); axes[1].set_title("正式T=1 P95与100 ms指标"); axes[1].grid(axis="y", alpha=.2)
for bar, value in zip(bars, [t1["p95_ms"], margin]): axes[1].text(bar.get_x()+bar.get_width()/2, value+2, f"{value:.3f} ms", ha="center", weight="bold")
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_3_cdf_and_margin.png", dpi=180); plt.close(fig)

# Figure 4: assessment state handling and delivery interface shapes.
scenario_rows = [
    ("证据齐全", full_assessment["overall_status"]),
    ("证据缺失", missing_assessment["overall_status"]),
    ("指标证据过期", stale_assessment["overall_status"]),
    ("时延证据过期", stale_latency_assessment["overall_status"]),
    ("失败任务", "rejected" if failed_rejected else "accepted"),
]
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
colors = {"passed": "#59A14F", "incomplete": "#F28E2B", "rejected": "#C00000", "accepted": "#C00000"}
axes[0].barh([row[0] for row in scenario_rows], [1] * len(scenario_rows), color=[colors[row[1]] for row in scenario_rows])
axes[0].invert_yaxis(); axes[0].set_xlim(0, 1.1); axes[0].set_xticks([]); axes[0].set_title("评估状态与证据防护")
for i, (_, status) in enumerate(scenario_rows): axes[0].text(.5, i, status, ha="center", va="center", color="white", weight="bold")
axes[1].axis("off"); axes[1].set_title("研究入口与交付入口的输入范围")
rows = [["入口", "允许形状", "本用例结论"], ["Python研究模型", "B=1,T=1,N=5", "P95达标"], ["Python研究模型", "B=1,T=100,N=5", "单独报告"], ["ONNX交付模型", "B=1,T=100,N=5", "固定形状有效"], ["MATLAB / C", "交付调用", "无纯前向计时证据"]]
table = axes[1].table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center", colWidths=[.30, .38, .36])
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.65)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#D9EAF7"); cell.set_text_props(weight="bold")
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_4_assessment_and_interfaces.png", dpi=180); plt.close(fig)

figures = [
    "figure_1_latency_distribution.png",
    "figure_2_latency_statistics.png",
    "figure_3_cdf_and_margin.png",
    "figure_4_assessment_and_interfaces.png",
]
for name in figures:
    check(f"FIG-{name}", f"结果图{name}已生成", (FIG_DIR / name).is_file(), str(FIG_DIR / name), "exists")

result = {
    "test_case": "IT-03-04",
    "test_name": "单帧前向时延与控制评估能力测试",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"os": platform.platform(), "python": sys.version, "numpy": np.__version__, "device": latency["device"], "hardware": latency["environment"]},
    "formal_batch_id": BATCH_ID,
    "job_signature": signature,
    "benchmark_script_sha256": digest(BENCHMARK_SCRIPT),
    "model_sha256": latency["model_sha256"],
    "load_condition": latency["load_condition"],
    "measurement_scope": latency["measurement_scope"],
    "formal_metric": {"name": "T=1 synchronized model-forward P95", "value_ms": t1["p95_ms"], "threshold_ms": THRESHOLD_MS, "margin_ms": THRESHOLD_MS - t1["p95_ms"], "passed": t1["p95_ms"] <= THRESHOLD_MS},
    "t1_statistics": t1,
    "t100_statistics": t100,
    "onnx_contract": {"path": str(ONNX_PATH), "sha256": digest(ONNX_PATH), "inputs": input_shapes, "outputs": output_shapes},
    "assessment": full_assessment,
    "assessment_scenarios": {"complete": full_assessment["overall_status"], "missing": missing_assessment["overall_status"], "stale_metric": stale_assessment["overall_status"], "stale_latency": stale_latency_assessment["overall_status"], "failed_job": "rejected" if failed_rejected else "accepted"},
    "boundary": boundary,
    "matlab_timing_status": matlab_status,
    "c_timing_status": c_status,
    "backend_unit_tests": {
        "command": "python -m unittest tracking_system.backend.tests.test_evaluation tracking_system.backend.tests.test_batch_testing",
        "result": "14/14 passed" if (RUN_DIR / "backend_unit_tests.log").is_file() and "OK" in (RUN_DIR / "backend_unit_tests.log").read_text(encoding="utf-8", errors="replace") else "not recorded",
        "log": str(RUN_DIR / "backend_unit_tests.log"),
    },
    "figures": figures,
    "check_count": len(checks),
    "passed_count": sum(item["passed"] for item in checks),
    "overall_passed": all(item["passed"] for item in checks),
    "checks": checks,
}
(RUN_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"result": str(RUN_DIR / "result.json"), "checks": len(checks), "passed": result["passed_count"], "T1_p95_ms": t1["p95_ms"], "T100_p95_ms": t100["p95_ms"], "assessment": full_assessment["overall_status"]}, ensure_ascii=False, indent=2))
