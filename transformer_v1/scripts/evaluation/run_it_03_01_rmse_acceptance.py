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
import torch
from torch.utils.data._utils.collate import default_collate


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_pkl import RegisteredPklDataset
from src.model import CoopTrackerModel
from tracking_system.backend.batch_testing import (
    _forward_batch,
    ensure_finite_tensor,
    stable_stage_mask,
)


RUN_DIR = ROOT / "artifacts" / "reports" / "software_test" / "IT-03-01" / "20260924_run01"
FIG_DIR = RUN_DIR / "figures"
BATCH_ID = "46fada4af5ef4c70a03fa56ef521f8c8"
BATCH_PATH = ROOT / "artifacts" / "reports" / "batch_tests" / f"{BATCH_ID}.json"
PREVIOUS_BATCH_ID = "2ea4f9eb0f2845b599fcf4837d71a018"
PREVIOUS_BATCH_PATH = ROOT / "artifacts" / "reports" / "batch_tests" / f"{PREVIOUS_BATCH_ID}.json"
MODEL_PATH = ROOT / "results" / "runs_40off(v7)" / "run_20260728_103728" / "best_model.pt"
DATA_DIR = ROOT / "data" / "val_data_v7_registered"
THRESHOLD_M = 8.0
WARMUP = 5

RUN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(20260924)
np.random.seed(20260924)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


checks: list[dict] = []


def check(cid: str, item: str, passed: bool, actual, expected) -> None:
    row = {"check_id": cid, "item": item, "passed": bool(passed), "actual": actual, "expected": expected}
    checks.append(row)
    if not passed:
        raise AssertionError(f"{cid}: {item}; actual={actual!r}; expected={expected!r}")


report = json.loads(BATCH_PATH.read_text(encoding="utf-8"))
previous = json.loads(PREVIOUS_BATCH_PATH.read_text(encoding="utf-8"))
result = report["result"]
metrics = result["metrics"]
config = report["config"]
tracks = result["tracks"]
provenance = result["provenance"]

check("JOB-COMPLETE", "正式全量批测状态为completed", report["status"] == "completed", report["status"], "completed")
check("JOB-PROGRESS", "正式批测94个批次全部完成", report["progress"]["completed_batches"] == report["progress"]["total_batches"] == 94, report["progress"], "94/94")
check("MODEL-ID", "冻结模型登记为V7", config["model_id"] == "v7", config["model_id"], "v7")
check("DATASET-ID", "验收数据集登记为V7", config["dataset_id"] == "v7", config["dataset_id"], "v7")
check("DEVICE", "正式批测使用cuda:0", config["device"] == "cuda:0", config["device"], "cuda:0")
check("SEQ", "序列长度固定为100", config["sequence_length"] == 100, config["sequence_length"], 100)
check("WARMUP", "每条航迹排除前5帧", config["warmup_frames"] == WARMUP, config["warmup_frames"], WARMUP)
check("FULL", "max_scenarios为空并覆盖全量样本", config["max_scenarios"] is None, config["max_scenarios"], None)
check("SAMPLING", "场景清单按运动类型分层登记", config["sampling_strategy"] == "stratified_by_scenario_type", config["sampling_strategy"], "stratified_by_scenario_type")
check("WEIGHT-HASH", "批测记录的冻结权重摘要与当前文件一致", provenance["model_sha256"] == sha256(MODEL_PATH), provenance["model_sha256"], sha256(MODEL_PATH))

class_counts = result["scenario_manifest"]["class_counts"]
check("SCENARIOS", "正式批测覆盖2000个场景", metrics["evaluated_scenarios"] == 2000 and len(result["scenario_manifest"]["files"]) == 2000, metrics["evaluated_scenarios"], 2000)
check("CLASS-BALANCE", "CA、CSSW、CT、CV各500个场景", class_counts == {"CA": 500, "CSSW": 500, "CT": 500, "CV": 500}, class_counts, {"CA": 500, "CSSW": 500, "CT": 500, "CV": 500})
check("TRACKS", "正式批测覆盖6000条航迹", metrics["evaluated_tracks"] == len(tracks) == 6000, len(tracks), 6000)
check("TRACK-UNIQUE", "每条正式航迹只统计一次", len({item["track"] for item in tracks}) == 6000, len({item["track"] for item in tracks}), 6000)
check("POINTS", "稳态有效点数为570000", metrics["evaluated_points"] == 570000, metrics["evaluated_points"], 570000)
check("VALID-FRAMES", "每条100帧航迹剔除5帧后保留95点", all(item["valid_frames"] == 95 for item in tracks), sorted({item["valid_frames"] for item in tracks}), [95])

numeric_fields = ["rmse_m", "mae_m", "p95_m", "max_error_m"]
numeric_matrix = np.asarray([[item[field] for field in numeric_fields] for item in tracks], dtype=np.float64)
check("FINITE", "全部逐航迹指标均为有限非负数", bool(np.isfinite(numeric_matrix).all() and np.all(numeric_matrix >= 0)), bool(np.isfinite(numeric_matrix).all()), True)

# Independently recompute both aggregation definitions from the 6000 per-track
# records. Per-track RMSE^2 multiplied by its valid-frame count recovers the
# track's squared-error sum exactly up to floating-point storage precision.
track_rmse = np.asarray([item["rmse_m"] for item in tracks], dtype=np.float64)
track_points = np.asarray([item["valid_frames"] for item in tracks], dtype=np.int64)
mean_track_rmse = float(np.mean(track_rmse))
point_weighted_rmse = float(np.sqrt(np.sum(np.square(track_rmse) * track_points) / np.sum(track_points)))
check("RECALC-MEAN", "独立复算航迹等权平均RMSE", abs(mean_track_rmse - metrics["mean_track_rmse_m"]) <= 1e-12, mean_track_rmse, metrics["mean_track_rmse_m"])
check("RECALC-POINT", "独立复算全部有效帧点加权RMSE", abs(point_weighted_rmse - metrics["position_rmse_m"]) <= 1e-12, point_weighted_rmse, metrics["position_rmse_m"])
check("FORMAL-THRESHOLD", "航迹等权平均RMSE使用未四舍五入值不大于8 m", mean_track_rmse <= THRESHOLD_M, mean_track_rmse, "<=8 m")
check("SUPPLEMENT-THRESHOLD", "全部有效帧点加权RMSE不大于8 m", point_weighted_rmse <= THRESHOLD_M, point_weighted_rmse, "<=8 m")

# Manual arithmetic fixture makes the distinction between equal-track and
# point-weighted aggregation explicit.
manual_tracks = [np.asarray([3.0, 4.0]), np.asarray([0.0, 12.0, 0.0])]
manual_track_rmse = [float(np.sqrt(np.mean(np.square(item)))) for item in manual_tracks]
manual_equal = float(np.mean(manual_track_rmse))
manual_point = float(np.sqrt(np.mean(np.square(np.concatenate(manual_tracks)))))
manual_from_summaries = float(np.sqrt(sum(len(e) * r * r for e, r in zip(manual_tracks, manual_track_rmse)) / sum(map(len, manual_tracks))))
check("MANUAL-TRACK", "手工样本逐航迹RMSE复算正确", np.allclose(manual_track_rmse, [math.sqrt(12.5), math.sqrt(48.0)]), manual_track_rmse, [math.sqrt(12.5), math.sqrt(48.0)])
check("MANUAL-POINT", "手工样本点加权RMSE可由逐航迹平方和恢复", abs(manual_point - manual_from_summaries) <= 1e-12, manual_point, manual_from_summaries)
check("MANUAL-DIFFER", "手工样本证明航迹等权与点加权口径不同", abs(manual_equal - manual_point) > 0.1, {"equal": manual_equal, "point": manual_point}, "different")

boundary_rows = []
for value, expected in [(7.999, True), (8.000, True), (8.001, False)]:
    actual = value <= THRESHOLD_M
    boundary_rows.append({"value_m": value, "accepted": actual, "expected": expected})
    check(f"BOUNDARY-{value:.3f}", f"{value:.3f} m阈值比较", actual == expected, actual, expected)

# Stable-stage masking and numerical-failure handling are tested directly.
fixture_mask = stable_stage_mask(torch.ones((2, 100), dtype=torch.bool), WARMUP)
check("MASK-FIRST", "稳定阶段掩码剔除每条航迹前5帧", not bool(fixture_mask[:, :5].any()), int(fixture_mask[:, :5].sum()), 0)
check("MASK-COUNT", "稳定阶段掩码每条保留95帧", bool(torch.all(fixture_mask.sum(dim=1) == 95)), fixture_mask.sum(dim=1).tolist(), [95, 95])
nan_rejected = False
try:
    ensure_finite_tensor("synthetic prediction", torch.tensor([1.0, float("nan")]))
except RuntimeError:
    nan_rejected = True
check("NAN-FAIL", "预测NaN会触发失败而非静默删除", nan_rejected, nan_rejected, True)

# Group the 6000 records by motion class. These are diagnostic strata; the
# contract criterion applies to the predeclared full V7 aggregate.
groups: dict[str, list[dict]] = defaultdict(list)
for item in tracks:
    match = re.match(r"track_([^_]+)_", item["track"])
    if not match:
        raise RuntimeError(f"Cannot classify track name: {item['track']}")
    groups[match.group(1).upper()].append(item)

class_rows = []
for name in ["CA", "CSSW", "CT", "CV"]:
    items = groups[name]
    values = np.asarray([item["rmse_m"] for item in items], dtype=np.float64)
    weights = np.asarray([item["valid_frames"] for item in items], dtype=np.float64)
    row = {
        "class": name,
        "scenarios": class_counts[name],
        "tracks": len(items),
        "valid_points": int(weights.sum()),
        "mean_track_rmse_m": float(np.mean(values)),
        "point_weighted_rmse_m": float(np.sqrt(np.average(np.square(values), weights=weights))),
        "track_rmse_p95_m": float(np.percentile(values, 95)),
        "track_pass_rate": float(np.mean(values <= THRESHOLD_M)),
    }
    class_rows.append(row)
    check(f"CLASS-TRACKS-{name}", f"{name}分层包含1500条航迹", row["tracks"] == 1500, row["tracks"], 1500)
    check(f"CLASS-POINTS-{name}", f"{name}分层包含142500个稳态点", row["valid_points"] == 142500, row["valid_points"], 142500)

with (RUN_DIR / "class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
    writer.writeheader(); writer.writerows(class_rows)

# The new full run must reproduce the prior frozen-artifact acceptance run.
previous_metrics = previous["result"]["metrics"]
check("REPRO-MEAN", "新旧全量批测航迹等权RMSE在10^-12 m内一致", abs(mean_track_rmse - previous_metrics["mean_track_rmse_m"]) <= 1e-12, mean_track_rmse, previous_metrics["mean_track_rmse_m"])
check("REPRO-POINT", "新旧全量批测点加权RMSE在10^-12 m内一致", abs(point_weighted_rmse - previous_metrics["position_rmse_m"]) <= 1e-12, point_weighted_rmse, previous_metrics["position_rmse_m"])
check("REPRO-SCOPE", "新旧批测模型和数据摘要一致", provenance["model_sha256"] == previous["result"]["provenance"]["model_sha256"] and provenance["dataset_sha256"] == previous["result"]["provenance"]["dataset_sha256"], {"model": provenance["model_sha256"], "dataset": provenance["dataset_sha256"]}, "same hashes")

# Repeat one representative frozen sample three times to check deterministic
# output without re-running the entire 2000-scenario set three times.
device = torch.device("cuda:0")
dataset = RegisteredPklDataset(data_dir=str(DATA_DIR), seq_len=100, num_uavs=5, return_extra=True)
sample_batch = default_collate([dataset[0]])
model = CoopTrackerModel(d_model=96).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(checkpoint, strict=True)
model.eval()
repeat_outputs = []
repeat_rmses = []
with torch.inference_mode():
    for _ in range(3):
        prediction, truth = _forward_batch(model, sample_batch, device)
        mask = stable_stage_mask(sample_batch["target_valid_mask"].to(device), WARMUP)
        distance = torch.linalg.vector_norm(prediction[..., [0, 2, 4]] - truth[..., [0, 2, 4]], dim=-1)
        selected = distance[mask]
        ensure_finite_tensor("representative repeat position error", selected)
        repeat_outputs.append(prediction.detach().cpu().numpy())
        repeat_rmses.append(float(torch.sqrt(torch.mean(selected.square())).cpu()))
repeat_max_output_delta = max(float(np.max(np.abs(item - repeat_outputs[0]))) for item in repeat_outputs[1:])
repeat_rmse_range = max(repeat_rmses) - min(repeat_rmses)
check("REPEAT-OUTPUT", "代表性样例三次输出完全一致", repeat_max_output_delta == 0.0, repeat_max_output_delta, 0.0)
check("REPEAT-RMSE", "代表性样例三次RMSE完全一致", repeat_rmse_range == 0.0, repeat_rmses, "range=0")

# Figure 1: the two required aggregate definitions.
fig, ax = plt.subplots(figsize=(8.8, 5.0))
values = [mean_track_rmse, point_weighted_rmse]
bars = ax.bar([0, 1], values, color=["#4C78A8", "#59A14F"], width=.58)
ax.axhline(THRESHOLD_M, color="#C00000", linestyle="--", linewidth=2, label="合同阈值 8 m")
ax.set_xticks([0, 1], ["航迹等权平均RMSE\n正式判据", "全部有效帧汇总RMSE\n补充报告"])
ax.set_ylabel("三维位置RMSE / m"); ax.set_ylim(0, 9.2); ax.grid(axis="y", alpha=.25); ax.legend()
ax.set_title("V7全量2000场景的两种RMSE聚合口径")
for bar, value in zip(bars, values): ax.text(bar.get_x()+bar.get_width()/2, value+.12, f"{value:.6f} m", ha="center", weight="bold")
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_1_overall_rmse.png", dpi=180); plt.close(fig)

# Figure 2: diagnostic motion-class stratification.
x = np.arange(4); width = .35
fig, ax = plt.subplots(figsize=(9.6, 5.2))
bars1 = ax.bar(x-width/2, [r["mean_track_rmse_m"] for r in class_rows], width, label="航迹等权RMSE", color="#4C78A8")
bars2 = ax.bar(x+width/2, [r["point_weighted_rmse_m"] for r in class_rows], width, label="点加权RMSE", color="#F28E2B")
ax.axhline(THRESHOLD_M, color="#C00000", linestyle="--", linewidth=1.8, label="8 m参考线")
ax.set_xticks(x, [r["class"] for r in class_rows]); ax.set_ylabel("三维位置RMSE / m"); ax.set_ylim(0, 9.7)
ax.set_title("运动类型分层结果（每类500场景、1500航迹）"); ax.grid(axis="y", alpha=.25); ax.legend(ncol=3)
for bars in [bars1,bars2]:
    for bar in bars: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.08, f"{bar.get_height():.3f}", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_2_motion_class_rmse.png", dpi=180); plt.close(fig)

# Figure 3: track-level distribution and CDF.
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
axes[0].hist(track_rmse, bins=45, color="#4C78A8", alpha=.88)
axes[0].axvline(THRESHOLD_M, color="#C00000", linestyle="--", label="8 m")
axes[0].set_xlabel("单航迹RMSE / m"); axes[0].set_ylabel("航迹数"); axes[0].set_title("6000条航迹RMSE分布"); axes[0].legend(); axes[0].grid(alpha=.2)
sorted_rmse = np.sort(track_rmse); cdf = np.arange(1, len(sorted_rmse)+1)/len(sorted_rmse)
axes[1].plot(sorted_rmse, cdf*100, color="#59A14F", linewidth=2)
axes[1].axvline(THRESHOLD_M, color="#C00000", linestyle="--")
pass_rate = float(np.mean(track_rmse <= THRESHOLD_M))
axes[1].scatter([THRESHOLD_M],[pass_rate*100],color="#C00000",zorder=3)
axes[1].text(THRESHOLD_M+.25,pass_rate*100,f"≤8 m航迹占比 {pass_rate*100:.1f}%",va="center")
axes[1].set_xlabel("单航迹RMSE / m"); axes[1].set_ylabel("累计比例 / %"); axes[1].set_ylim(0,102); axes[1].set_title("航迹RMSE累计分布"); axes[1].grid(alpha=.2)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_3_track_rmse_distribution.png", dpi=180); plt.close(fig)

# Figure 4: reproducibility and exact threshold semantics.
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
run_labels = ["2026-09-17\n历史全量", "2026-09-24\n本轮全量"]
axes[0].bar(np.arange(2)-.18, [previous_metrics["mean_track_rmse_m"], mean_track_rmse], .36, label="航迹等权", color="#4C78A8")
axes[0].bar(np.arange(2)+.18, [previous_metrics["position_rmse_m"], point_weighted_rmse], .36, label="点加权", color="#59A14F")
axes[0].set_xticks(np.arange(2),run_labels); axes[0].set_ylim(6.5,8.1); axes[0].set_ylabel("RMSE / m")
axes[0].set_title("冻结模型与数据的全量复现"); axes[0].legend(); axes[0].grid(axis="y",alpha=.25)
boundary_colors=["#59A14F" if r["accepted"] else "#C00000" for r in boundary_rows]
axes[1].bar([f"{r['value_m']:.3f}" for r in boundary_rows],[1,1,1],color=boundary_colors)
axes[1].set_ylim(0,1.15); axes[1].set_yticks([]); axes[1].set_xlabel("构造RMSE / m"); axes[1].set_title("8 m未四舍五入阈值比较")
for i,row in enumerate(boundary_rows): axes[1].text(i,.52,"通过" if row["accepted"] else "不通过",ha="center",va="center",color="white",weight="bold")
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_4_reproducibility_and_threshold.png", dpi=180); plt.close(fig)

output = {
    "test_case": "IT-03-01",
    "test_name": "航迹平均位置RMSE达标测试",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "formal_batch_id": BATCH_ID,
    "previous_batch_id": PREVIOUS_BATCH_ID,
    "environment": {"os": platform.platform(), "python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "device": torch.cuda.get_device_name(0)},
    "scope": {"dataset": "V7 full validation", "scenarios": 2000, "tracks": 6000, "points": 570000, "sequence_length": 100, "warmup_frames": 5, "max_scenarios": None, "sampling_strategy": config["sampling_strategy"]},
    "provenance": provenance,
    "official_metrics": metrics,
    "independent_recalculation": {"mean_track_rmse_m": mean_track_rmse, "point_weighted_rmse_m": point_weighted_rmse},
    "manual_fixture": {"track_errors": [item.tolist() for item in manual_tracks], "track_rmse_m": manual_track_rmse, "equal_track_rmse_m": manual_equal, "point_weighted_rmse_m": manual_point},
    "threshold_boundary": boundary_rows,
    "class_metrics": class_rows,
    "representative_repeat": {"sample": tracks[0]["track"], "rmse_runs_m": repeat_rmses, "max_output_delta": repeat_max_output_delta, "rmse_range": repeat_rmse_range},
    "checks": checks,
    "check_count": len(checks),
    "passed_count": sum(item["passed"] for item in checks),
    "overall_passed": all(item["passed"] for item in checks),
    "figures": [path.name for path in sorted(FIG_DIR.glob("*.png"))],
    "scope_note": "V8强干扰附加集未并入V7合同总体；CSSW分层高于8 m，已单列披露，不改变事前确定的V7全量航迹等权总体判据。",
}
(RUN_DIR / "result.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "output": str(RUN_DIR), "overall_passed": output["overall_passed"],
    "checks": f"{output['passed_count']}/{output['check_count']}",
    "mean_track_rmse_m": mean_track_rmse, "point_weighted_rmse_m": point_weighted_rmse,
    "class_metrics": class_rows, "repeat": output["representative_repeat"],
}, ensure_ascii=False, indent=2))
if not output["overall_passed"]:
    raise SystemExit(1)
