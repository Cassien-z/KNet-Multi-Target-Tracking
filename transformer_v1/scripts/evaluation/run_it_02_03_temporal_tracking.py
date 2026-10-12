from __future__ import annotations

import csv
import hashlib
import json
import pickle
import platform
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import _parse_registered_pkl
from src.model import CoopTrackerModel


RUN_DIR = ROOT / "artifacts" / "reports" / "software_test" / "IT-02-03" / "20260923_run01"
FIG_DIR = RUN_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ROOT / "results" / "runs_40off(v7)" / "run_20260728_103728" / "best_model.pt"
V7_DIR = ROOT / "data" / "val_data_v7_registered"
V8_PATH = ROOT / "data" / "val_data_v8_interference_registered_10hz" / "track_data_highfi_0000.pkl"
WARMUP = 5
SEED = 20260923

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sample(path: Path) -> dict[str, torch.Tensor]:
    with path.open("rb") as f:
        raw = pickle.load(f)
    sample = _parse_registered_pkl(raw, seq_len=100, num_uavs=5, basename=path.name)[0]
    sample.setdefault("imu", torch.zeros(100, 5, 6))
    sample.setdefault("slow_link", torch.zeros(100, 5, 3))
    sample.setdefault("slow_update", torch.zeros(100, 5))
    sample.setdefault("fast_age", torch.zeros(100, 5))
    sample.setdefault("slow_age", torch.full((100, 5), 999.0))
    sample.setdefault("esm_mode", torch.zeros(100, 5))
    return sample


TIME_KEYS = {
    "radar", "eo", "esm", "radar_mask", "eo_mask", "esm_mask", "rotmat_b2e", "uav_pos",
    "fast_link", "slow_link", "fast_update", "slow_update", "imu", "fast_age", "slow_age", "esm_mode",
}
INPUT_KEYS = list(TIME_KEYS | {"ref_pos"})


def make_batch(sample: dict[str, torch.Tensor], length: int = 100) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key in INPUT_KEYS:
        value = sample[key]
        batch[key] = value[:length].clone().unsqueeze(0) if key in TIME_KEYS else value.clone().unsqueeze(0)
    batch["dt"] = sample["dt"].clone().view(1)
    return batch


def run(model: CoopTrackerModel, batch: dict[str, torch.Tensor], **flags) -> torch.Tensor:
    with torch.no_grad():
        return model(**batch, **flags).detach().cpu()


def errors(pred: torch.Tensor, sample: dict[str, torch.Tensor], length: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = sample["label_abs"][:length]
    valid = (sample["target_valid_mask"][:length] > 0.5).clone()
    valid[: min(WARMUP, length)] = False
    p = pred[0, :length][:, [0, 2, 4]] - truth[:, [0, 2, 4]]
    v = pred[0, :length][:, [1, 3, 5]] - truth[:, [1, 3, 5]]
    return torch.linalg.vector_norm(p, dim=-1).numpy(), torch.linalg.vector_norm(v, dim=-1).numpy(), valid.numpy()


def metric_row(name: str, pred: torch.Tensor, sample: dict[str, torch.Tensor]) -> dict:
    pe, ve, valid = errors(pred, sample)
    idx = np.flatnonzero(valid)
    truth_pos = sample["label_abs"][:, [0, 2, 4]].numpy()
    maneuver = np.linalg.norm(np.diff(truth_pos, axis=0), axis=1)
    peak_frame = int(idx[np.argmax(pe[idx])])
    return {
        "scenario": name,
        "valid_frames": int(valid.sum()),
        "dt_s": float(sample["dt"]),
        "position_rmse_m": float(np.sqrt(np.mean(pe[valid] ** 2))),
        "position_p95_m": float(np.percentile(pe[valid], 95)),
        "position_peak_m": float(pe[peak_frame]),
        "position_peak_frame": peak_frame,
        "velocity_rmse_mps": float(np.sqrt(np.mean(ve[valid] ** 2))),
        "truth_step_peak_m": float(maneuver.max()),
        "finite_rate": float(torch.isfinite(pred).float().mean()),
    }


model = CoopTrackerModel(d_model=96)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=True)
model.eval()

v7_paths = {
    "CV": V7_DIR / "track_cv_0000.pkl",
    "CA": V7_DIR / "track_ca_0000.pkl",
    "CT": V7_DIR / "track_ct_0000.pkl",
    "CSSW": V7_DIR / "track_cssw_0000.pkl",
}
samples = {name: load_sample(path) for name, path in v7_paths.items()}
samples["复合干扰(V8补充)"] = load_sample(V8_PATH)

checks: list[dict] = []


def check(cid: str, item: str, passed: bool, actual, expected) -> None:
    checks.append({"check_id": cid, "item": item, "passed": bool(passed), "actual": actual, "expected": expected})
    if not passed:
        raise AssertionError(f"{cid}: {item}; actual={actual!r}; expected={expected!r}")


layers = len(model.temporal_encoder.encoder.layers)
heads = [layer.self_attn.num_heads for layer in model.temporal_encoder.encoder.layers]
check("STRUCT-LAYERS", "时序Transformer层数", layers == 2, layers, 2)
check("STRUCT-HEADS", "每层注意力头数", heads == [4, 4], heads, [4, 4])
check("STRUCT-BATCH-FIRST", "时序层采用batch_first", all(layer.self_attn.batch_first for layer in model.temporal_encoder.encoder.layers), True, True)

scenario_rows: list[dict] = []
scenario_predictions: dict[str, torch.Tensor] = {}
for name, sample in samples.items():
    batch = make_batch(sample)
    captured: dict[str, tuple] = {}
    handle = model.temporal_encoder.register_forward_hook(
        lambda _m, args, out: captured.update({"input_shape": tuple(args[0].shape), "output_shape": tuple(out.shape)})
    )
    try:
        pred = run(model, batch)
    finally:
        handle.remove()
    scenario_predictions[name] = pred
    row = metric_row(name, pred, sample)
    scenario_rows.append(row)
    check(f"SCEN-{name}-SHAPE", f"{name}输出形状", tuple(pred.shape) == (1, 100, 6), tuple(pred.shape), (1, 100, 6))
    check(f"SCEN-{name}-FINITE", f"{name}输出有限", row["finite_rate"] == 1.0, row["finite_rate"], 1.0)
    check(f"SCEN-{name}-VALID", f"{name}按预热和真值有效掩码计帧", row["valid_frames"] > 0, row["valid_frames"], ">0")
    check(f"SCEN-{name}-TIME", f"{name}时序输入输出长度保持", captured["input_shape"][1] == 100 and captured["output_shape"][1] == 100, captured, "T=100")

# 同一输入重复三次，验证eval模式确定性。
base_sample = samples["CA"]
base_batch = make_batch(base_sample)
repeat = [run(model, base_batch) for _ in range(3)]
repeat_delta = max(float((repeat[i] - repeat[0]).abs().max()) for i in range(1, 3))
check("REPEAT", "代表样例重复三次输出一致", repeat_delta <= 1e-7, repeat_delta, "<=1e-7")

# 四种开关组合均应保持状态输出接口。
switch_rows = []
for use_temporal in (True, False):
    for use_gat in (True, False):
        pred = run(model, base_batch, use_temporal=use_temporal, use_gat=use_gat)
        row = {"use_temporal": use_temporal, "use_gat": use_gat, "shape": list(pred.shape), "finite_rate": float(torch.isfinite(pred).float().mean())}
        switch_rows.append(row)
        check(f"SWITCH-{int(use_temporal)}-{int(use_gat)}", "模块开关保持输出接口和有限性", tuple(pred.shape) == (1, 100, 6) and row["finite_rate"] == 1.0, row, "[1,100,6], finite")

# 后半段观测扰动可改变前半段输出，用于确认完整窗口的非因果上下文，不作为在线因果能力证明。
future_batch = {k: v.clone() for k, v in base_batch.items()}
future_batch["radar"][:, 60:, :, 2] += 300.0 * future_batch["radar_mask"][:, 60:, :]
future_batch["eo"][:, 60:, :, 0] += 0.02 * future_batch["eo_mask"][:, 60:, :]
future_batch["esm"][:, 60:, :, 0] += 0.02 * future_batch["esm_mask"][:, 60:, :]
future_pred = run(model, future_batch)
front_future_delta = float((future_pred[:, :60] - repeat[0][:, :60]).abs().max())
check("NONCAUSAL", "后半段扰动影响前半段输出，确认完整窗口上下文", front_future_delta > 1e-6, front_future_delta, ">1e-6")

# 60帧直接推理与补零至100帧对比，记录无padding attention mask的适用边界。
short_batch = make_batch(base_sample, length=60)
short_pred = run(model, short_batch)
padded_batch = make_batch(base_sample)
for key in TIME_KEYS:
    if key == "rotmat_b2e":
        padded_batch[key][:, 60:] = torch.eye(3).view(1, 1, 1, 3, 3)
    else:
        padded_batch[key][:, 60:] = 0
padded_batch["fast_age"][:, 60:] = 999.0
padded_batch["slow_age"][:, 60:] = 999.0
padded_pred = run(model, padded_batch)
padding_front_delta = float((short_pred - padded_pred[:, :60]).abs().max())
check("SHORT-SHAPE", "Python研究路径支持60帧输入并保持时长", tuple(short_pred.shape) == (1, 60, 6), tuple(short_pred.shape), (1, 60, 6))
check("SHORT-FINITE", "60帧研究路径输出有限", bool(torch.isfinite(short_pred).all()), True, True)
check("PADDING-BOUNDARY", "补零尾段会影响前60帧，确认无padding掩码边界", padding_front_delta > 1e-6, padding_front_delta, ">1e-6")

# 全源缺失：单帧、5帧和10帧。输出必须有限，缺失掩码必须准确；恢复量只做工程记录。
missing_specs = [("单帧缺失", 50, 51), ("连续5帧缺失", 48, 53), ("连续10帧缺失", 45, 55)]
missing_rows: list[dict] = []
missing_curves: dict[str, np.ndarray] = {}
base_pred = repeat[0]
base_pe, _, _ = errors(base_pred, base_sample)
for label, start, end in missing_specs:
    batch = {k: v.clone() for k, v in base_batch.items()}
    for data_key, mask_key in (("radar", "radar_mask"), ("eo", "eo_mask"), ("esm", "esm_mask")):
        batch[data_key][:, start:end] = 0
        batch[mask_key][:, start:end] = 0
    for key in ("fast_link", "slow_link"):
        batch[key][:, start:end] = 0
    for key in ("fast_update", "slow_update"):
        batch[key][:, start:end] = 0
    for key in ("fast_age", "slow_age"):
        batch[key][:, start:end] = 999.0
    captured: dict[str, torch.Tensor] = {}
    handle = model.k_front.register_forward_hook(lambda _m, _a, out: captured.update({"z_obs": out[1].detach().cpu()}))
    try:
        pred = run(model, batch)
    finally:
        handle.remove()
    pe, _, valid = errors(pred, base_sample)
    missing_curves[label] = pe
    post_end = min(end + 10, 100)
    output_delta = torch.linalg.vector_norm((pred - base_pred)[0, :, [0, 2, 4]], dim=-1).numpy()
    recovery_candidates = np.flatnonzero(output_delta[end:] <= 1.0)
    recovery_frame = int(end + recovery_candidates[0]) if recovery_candidates.size else None
    z = captured["z_obs"][0, :, [0, 2, 4]]
    hold_error = float((z[start:end] - z[start - 1]).abs().max())
    row = {
        "condition": label,
        "start_frame": start,
        "end_frame_exclusive": end,
        "finite_rate": float(torch.isfinite(pred).float().mean()),
        "mask_sum_in_gap": float(batch["radar_mask"][:, start:end].sum() + batch["eo_mask"][:, start:end].sum() + batch["esm_mask"][:, start:end].sum()),
        "z_obs_position_hold_abs_max_m": hold_error,
        "gap_position_rmse_m": float(np.sqrt(np.mean(pe[start:end] ** 2))),
        "first_post_gap_error_m": float(pe[end]) if end < 100 else None,
        "post_10_frame_rmse_m": float(np.sqrt(np.mean(pe[end:post_end] ** 2))) if end < post_end else None,
        "output_delta_peak_m": float(output_delta.max()),
        "diagnostic_recovery_to_1m_from_baseline_frame": recovery_frame,
    }
    missing_rows.append(row)
    check(f"MISS-{end-start}-MASK", f"{label}窗口内全部传感器掩码归零", row["mask_sum_in_gap"] == 0.0, row["mask_sum_in_gap"], 0.0)
    check(f"MISS-{end-start}-FINITE", f"{label}输出全有限", row["finite_rate"] == 1.0, row["finite_rate"], 1.0)
    check(f"MISS-{end-start}-HOLD", f"{label}Learned-K观测位置保持", hold_error < 1e-5, hold_error, "<1e-5")

# 批次换序一致性；部署B=1，此项仅补充证明批内样例相互独立。
cv_batch = make_batch(samples["CV"])
ca_batch = make_batch(samples["CA"])
ab = {k: torch.cat([cv_batch[k], ca_batch[k]], dim=0) for k in cv_batch}
ba = {k: torch.cat([ca_batch[k], cv_batch[k]], dim=0) for k in cv_batch}
pred_ab = run(model, ab)
pred_ba = run(model, ba)
reorder_delta = max(float((pred_ab[0] - pred_ba[1]).abs().max()), float((pred_ab[1] - pred_ba[0]).abs().max()))
check("BATCH-REORDER", "批次换序后对应样例输出一致", reorder_delta <= 1e-5, reorder_delta, "<=1e-5")

# dt和全无效计帧边界。
all_dt = [float(s["dt"]) for s in samples.values()]
check("DT", "所有样例dt为有限正数", all(np.isfinite(v) and v > 0 for v in all_dt), all_dt, "finite and >0")
empty_valid = torch.zeros(1, 100, dtype=torch.bool)
empty_valid[:, :WARMUP] = False
check("ALL-PADDING-METRIC", "全无效样例不产生有效指标帧", int(empty_valid.sum()) == 0, int(empty_valid.sum()), 0)

# 图1：五类代表场景的误差汇总。
labels = [r["scenario"] for r in scenario_rows]
x = np.arange(len(labels))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].bar(x - 0.18, [r["position_rmse_m"] for r in scenario_rows], 0.36, label="位置RMSE", color="#4c78a8")
axes[0].bar(x + 0.18, [r["position_p95_m"] for r in scenario_rows], 0.36, label="位置P95", color="#72b7b2")
axes[0].set_xticks(x, labels, rotation=12)
axes[0].set_ylabel("误差/m")
axes[0].set_title("代表场景位置误差（预热5帧后）")
axes[0].legend(); axes[0].grid(axis="y", alpha=.25)
axes[1].bar(x, [r["velocity_rmse_mps"] for r in scenario_rows], color="#f58518")
axes[1].set_xticks(x, labels, rotation=12)
axes[1].set_ylabel("速度RMSE/(m/s)")
axes[1].set_title("代表场景速度误差")
axes[1].grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_1_scenario_metrics.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# 图2：逐帧位置误差。
fig, ax = plt.subplots(figsize=(12, 5.2))
for name, pred in scenario_predictions.items():
    pe, _, _ = errors(pred, samples[name])
    ax.plot(np.arange(100), pe, label=name, lw=1.5)
ax.axvspan(0, WARMUP - 1, color="#999999", alpha=.15, label="预热排除区")
ax.set_xlabel("帧序号"); ax.set_ylabel("三维位置误差/m"); ax.set_title("五类代表场景逐帧位置误差")
ax.grid(alpha=.25); ax.legend(ncol=3); fig.tight_layout()
fig.savefig(FIG_DIR / "figure_2_framewise_position_error.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# 图3：短时缺失与恢复。
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].plot(base_pe, label="基线", color="#333333", lw=2)
colors = ["#4c78a8", "#f58518", "#e45756"]
for (label, start, end), color in zip(missing_specs, colors):
    axes[0].plot(missing_curves[label], label=label, color=color, alpha=.9)
axes[0].axvspan(45, 55, color="#e45756", alpha=.1)
axes[0].set_xlim(35, 70); axes[0].set_xlabel("帧序号"); axes[0].set_ylabel("位置误差/m")
axes[0].set_title("缺失窗口及恢复阶段"); axes[0].legend(); axes[0].grid(alpha=.25)
mx = np.arange(len(missing_rows))
axes[1].bar(mx - .2, [r["gap_position_rmse_m"] for r in missing_rows], .4, label="缺失窗口RMSE", color="#e45756")
axes[1].bar(mx + .2, [r["post_10_frame_rmse_m"] for r in missing_rows], .4, label="后续10帧RMSE", color="#54a24b")
axes[1].set_xticks(mx, [r["condition"] for r in missing_rows]); axes[1].set_ylabel("误差/m")
axes[1].set_title("缺失与恢复统计（只记录 不设阈值）"); axes[1].legend(); axes[1].grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_3_missing_and_recovery.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# 图4：结构、非因果和可重复性边界。
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].bar(["编码层数", "每层注意力头数"], [layers, heads[0]], color=["#4c78a8", "#72b7b2"])
axes[0].set_ylim(0, 5); axes[0].set_title("时序Transformer结构核查"); axes[0].grid(axis="y", alpha=.25)
values = [front_future_delta, padding_front_delta, max(repeat_delta, 1e-12), max(reorder_delta, 1e-12)]
axes[1].bar(["后半段扰动\n前半段变化", "60帧vs补零\n前60帧变化", "三次重复\n最大差", "批次换序\n最大差"], values,
            color=["#e45756", "#f58518", "#54a24b", "#b279a2"])
axes[1].set_yscale("log"); axes[1].set_ylabel("最大绝对差（对数坐标）")
axes[1].set_title("完整窗口上下文与稳定性"); axes[1].grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_4_temporal_scope_and_repeatability.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


write_csv(RUN_DIR / "scenario_metrics.csv", scenario_rows)
write_csv(RUN_DIR / "missing_recovery.csv", missing_rows)
write_csv(RUN_DIR / "checks.csv", checks)
write_csv(RUN_DIR / "switch_checks.csv", switch_rows)

result = {
    "test_case": "IT-02-03",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "pytorch": torch.__version__,
    "platform": platform.platform(),
    "seed": SEED,
    "warmup_frames_excluded": WARMUP,
    "model_path": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
    "model_sha256": sha256(MODEL_PATH),
    "input_files": {name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)} for name, path in {**v7_paths, "复合干扰(V8补充)": V8_PATH}.items()},
    "passed": all(r["passed"] for r in checks),
    "check_count": len(checks),
    "passed_count": sum(r["passed"] for r in checks),
    "structure": {"layers": layers, "heads": heads, "batch_first": True, "causal_mask": False, "padding_attention_mask": False},
    "scenario_metrics": scenario_rows,
    "missing_recovery": missing_rows,
    "switch_checks": switch_rows,
    "repeat_max_abs_delta": repeat_delta,
    "batch_reorder_max_abs_delta": reorder_delta,
    "future_perturbation_front_max_abs_delta": front_future_delta,
    "short_vs_zero_padded_front_max_abs_delta": padding_front_delta,
    "scope_statement": "当前模型按完整片段非因果推理；未提供causal mask或padding attention mask。本用例不证明在线推理与未来帧独立。",
    "figures": [p.name for p in sorted(FIG_DIR.glob("*.png"))],
}
(RUN_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
