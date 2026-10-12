from __future__ import annotations

import csv
import hashlib
import json
import pickle
import platform
import sys
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


RUN_DIR = ROOT / "artifacts" / "reports" / "software_test" / "IT-02-02" / "20260923_run01"
FIG_DIR = RUN_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "results" / "runs_40off(v7)" / "run_20260728_103728" / "best_model.pt"
SAMPLE_PATH = sorted((ROOT / "data" / "val_data_v7_registered").glob("*.pkl"))[0]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(20260923)
np.random.seed(20260923)
torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

checks: list[dict] = []


def check(check_id: str, description: str, condition: bool, actual, expected) -> None:
    row = {
        "check_id": check_id,
        "description": description,
        "passed": bool(condition),
        "actual": actual,
        "expected": expected,
    }
    checks.append(row)
    if not condition:
        raise AssertionError(f"{check_id}: {description}; actual={actual!r}; expected={expected!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


with SAMPLE_PATH.open("rb") as handle:
    registered = pickle.load(handle)
sample = _parse_registered_pkl(registered, seq_len=100, num_uavs=5, basename=SAMPLE_PATH.name)[0]

steps, agents = 100, 5
extras = {
    "imu": torch.zeros(steps, agents, 6),
    "slow_link": torch.zeros(steps, agents, 3),
    "slow_update": torch.zeros(steps, agents),
    "fast_age": torch.zeros(steps, agents),
    "slow_age": torch.full((steps, agents), 999.0),
    "esm_mode": torch.zeros(steps, agents),
}
for name, value in extras.items():
    sample.setdefault(name, value)

model = CoopTrackerModel(d_model=96)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=True)
model.eval()

input_names = [
    "radar", "eo", "esm", "radar_mask", "eo_mask", "esm_mask", "ref_pos", "rotmat_b2e", "uav_pos",
    "fast_link", "slow_link", "fast_update", "slow_update", "imu", "fast_age", "slow_age", "esm_mode",
]


def make_batch(radar_on: int, eo_on: int, esm_on: int) -> dict[str, torch.Tensor]:
    batch = {name: sample[name].clone().unsqueeze(0) for name in input_names}
    batch["dt"] = sample["dt"].clone().view(1)
    for enabled, value_name, mask_name in (
        (radar_on, "radar", "radar_mask"),
        (eo_on, "eo", "eo_mask"),
        (esm_on, "esm", "esm_mask"),
    ):
        if not enabled:
            batch[value_name].zero_()
            batch[mask_name].zero_()
    return batch


def position_rmse(prediction: torch.Tensor) -> float:
    truth = sample["label_abs"][5:, [0, 2, 4]]
    estimate = prediction[0, 5:, [0, 2, 4]].cpu()
    valid = sample["target_valid_mask"][5:] > 0.5
    error = estimate[valid] - truth[valid]
    return float(torch.sqrt(torch.mean(torch.sum(error * error, dim=-1))).item())


def run_with_capture(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    modules = {
        "radar_encoded": model.radar_enc,
        "eo_encoded": model.eo_enc,
        "esm_encoded": model.esm_enc,
        "pose_projection": model.ego_pose_proj,
        "radar_context": model.cross_radar_to_passive,
        "passive_context": model.cross_passive_to_radar,
        "fusion_gate_logit": model.fusion_gate,
        "fusion_projection": model.fusion_proj,
    }
    handles = []
    for name, module in modules.items():
        handles.append(module.register_forward_hook(lambda _m, _a, out, key=name: captured.__setitem__(key, out.detach().cpu())))
    try:
        with torch.no_grad():
            prediction = model(**batch)
    finally:
        for handle in handles:
            handle.remove()
    return prediction.detach().cpu(), captured


mask_specs = [
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
]
mask_rows: list[dict] = []
captured_complete: dict[str, torch.Tensor] | None = None
complete_prediction: torch.Tensor | None = None

for radar_on, eo_on, esm_on in mask_specs:
    name = f"R{radar_on}E{eo_on}S{esm_on}"
    batch = make_batch(radar_on, eo_on, esm_on)
    prediction, captured = run_with_capture(batch)
    finite_rate = float(torch.isfinite(prediction).float().mean().item())
    check(f"MASK-{name}-SHAPE", f"{name}输出形状", tuple(prediction.shape) == (1, 100, 6), tuple(prediction.shape), (1, 100, 6))
    check(f"MASK-{name}-FINITE", f"{name}输出全有限", finite_rate == 1.0, finite_rate, 1.0)

    radar_mask_flat = batch["radar_mask"].permute(0, 2, 1).reshape(agents, steps)
    eo_mask_flat = batch["eo_mask"].permute(0, 2, 1).reshape(agents, steps)
    esm_mask_flat = batch["esm_mask"].permute(0, 2, 1).reshape(agents, steps)
    with torch.no_grad():
        passive, passive_valid = model._passive_fusion(
            captured["eo_encoded"], captured["esm_encoded"], eo_mask_flat, esm_mask_flat
        )
        eo_effective = torch.sigmoid(model.passive_conf_eo(captured["eo_encoded"])) * eo_mask_flat.unsqueeze(-1)
        esm_effective = torch.sigmoid(model.passive_conf_esm(captured["esm_encoded"])) * esm_mask_flat.unsqueeze(-1)
        denominator = eo_effective + esm_effective + 1e-6
        eo_weight = eo_effective / denominator
        esm_weight = esm_effective / denominator

    invalid_eo_max = float(eo_weight[eo_mask_flat == 0].abs().max().item()) if bool((eo_mask_flat == 0).any()) else 0.0
    invalid_esm_max = float(esm_weight[esm_mask_flat == 0].abs().max().item()) if bool((esm_mask_flat == 0).any()) else 0.0
    check(f"MASK-{name}-INVALID-WEIGHT", f"{name}无效被动源权重为零", invalid_eo_max == 0.0 and invalid_esm_max == 0.0, [invalid_eo_max, invalid_esm_max], [0.0, 0.0])

    no_passive = passive_valid == 0
    missing_error = 0.0
    if bool(no_passive.any()):
        missing = model.passive_missing.expand_as(passive)
        missing_error = float((passive[no_passive] - missing[no_passive]).abs().max().item())
        check(f"MASK-{name}-MISSING-TOKEN", f"{name}无EO/ESM位置使用被动缺失令牌", missing_error == 0.0, missing_error, 0.0)

    both_valid = (eo_mask_flat > 0.5) & (esm_mask_flat > 0.5)
    weight_sum_error = 0.0
    if bool(both_valid.any()):
        weight_sum_error = float(((eo_weight + esm_weight)[both_valid] - 1.0).abs().max().item())
        check(f"MASK-{name}-WEIGHT-SUM", f"{name}双被动源归一化权重和接近1", weight_sum_error < 5e-6, weight_sum_error, "<5e-6")

    mean_eo_weight = float(eo_weight[passive_valid > 0].mean().item()) if bool((passive_valid > 0).any()) else 0.0
    mean_esm_weight = float(esm_weight[passive_valid > 0].mean().item()) if bool((passive_valid > 0).any()) else 0.0
    mask_rows.append(
        {
            "combination": name,
            "radar_enabled": radar_on,
            "eo_enabled": eo_on,
            "esm_enabled": esm_on,
            "output_finite_rate": finite_rate,
            "stable_position_rmse_m": position_rmse(prediction),
            "eo_effective_weight_mean": mean_eo_weight,
            "esm_effective_weight_mean": mean_esm_weight,
            "passive_missing_ratio": float((passive_valid == 0).float().mean().item()),
            "invalid_eo_weight_abs_max": invalid_eo_max,
            "invalid_esm_weight_abs_max": invalid_esm_max,
            "missing_token_abs_error_max": missing_error,
            "weight_sum_abs_error_max": weight_sum_error,
        }
    )
    if (radar_on, eo_on, esm_on) == (1, 1, 1):
        complete_prediction = prediction
        captured_complete = captured

assert complete_prediction is not None and captured_complete is not None
for row in mask_rows:
    combo = next(spec for spec in mask_specs if f"R{spec[0]}E{spec[1]}S{spec[2]}" == row["combination"])
    prediction, _ = run_with_capture(make_batch(*combo))
    row["output_delta_vs_complete_rms"] = float(torch.sqrt(torch.mean((prediction - complete_prediction) ** 2)).item())

# Structural module snapshot and exact adaptive-alignment formula.
expected_shape = (agents, steps, 96)
module_rows: list[dict] = []
for name in ("radar_encoded", "eo_encoded", "esm_encoded", "pose_projection", "radar_context", "passive_context", "fusion_projection"):
    tensor = captured_complete[name]
    check(f"SHAPE-{name}", f"{name}张量形状", tuple(tensor.shape) == expected_shape, tuple(tensor.shape), expected_shape)
    check(f"FINITE-{name}", f"{name}张量全有限", bool(torch.isfinite(tensor).all()), True, True)
    module_rows.append({"module": name, "shape": str(tuple(tensor.shape)), "finite_rate": float(torch.isfinite(tensor).float().mean()), "mean": float(tensor.mean()), "std": float(tensor.std(unbiased=False))})

gate_logit = captured_complete["fusion_gate_logit"]
fusion_alpha = torch.sigmoid(gate_logit)
check("SHAPE-FUSION-GATE", "融合门张量形状", tuple(fusion_alpha.shape) == (agents, steps, 1), tuple(fusion_alpha.shape), (agents, steps, 1))
check("RANGE-FUSION-GATE", "融合门输出位于[0,1]且全有限", bool(torch.isfinite(fusion_alpha).all()) and float(fusion_alpha.min()) >= 0.0 and float(fusion_alpha.max()) <= 1.0, [float(fusion_alpha.min()), float(fusion_alpha.max())], "[0,1]")
module_rows.append({"module": "fusion_alpha", "shape": str(tuple(fusion_alpha.shape)), "finite_rate": float(torch.isfinite(fusion_alpha).float().mean()), "mean": float(fusion_alpha.mean()), "std": float(fusion_alpha.std(unbiased=False))})

aligned = model._adaptive_align(captured_complete["radar_encoded"])
feat = captured_complete["radar_encoded"]
manual_aligned = model.align_gamma.detach() * (feat - feat.mean(dim=1, keepdim=True)) / feat.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5) + model.align_beta.detach()
align_error = float((aligned - manual_aligned).abs().max().item())
check("ADAPTIVE-ALIGN", "自适应对齐与冻结公式逐元素一致", align_error == 0.0, align_error, 0.0)

check("ATTN-R2P-HEADS", "主动到被动交叉注意力为四头", model.cross_radar_to_passive.attn.num_heads == 4, model.cross_radar_to_passive.attn.num_heads, 4)
check("ATTN-P2R-HEADS", "被动到主动交叉注意力为四头", model.cross_passive_to_radar.attn.num_heads == 4, model.cross_passive_to_radar.attn.num_heads, 4)
check("ATTN-BATCH-FIRST", "双向交叉注意力采用batch_first张量排列", model.cross_radar_to_passive.attn.batch_first and model.cross_passive_to_radar.attn.batch_first, True, True)

# Encoder mask-token checks on the all-missing combination.
_, missing_capture = run_with_capture(make_batch(0, 0, 0))
for modality, module in (("radar", model.radar_enc), ("eo", model.eo_enc), ("esm", model.esm_enc)):
    encoded = missing_capture[f"{modality}_encoded"]
    expected = module.mask_token.detach().expand_as(encoded)
    error = float((encoded - expected).abs().max().item())
    check(f"MASK-TOKEN-{modality}", f"{modality}缺失通道由专用令牌替代", error == 0.0, error, 0.0)

# Passive-angle noise response. Direction is recorded, not prescribed.
noise_rows: list[dict] = []
generator = torch.Generator().manual_seed(20260923)
for sigma in (0.0, 0.002, 0.01, 0.05):
    batch = make_batch(1, 1, 1)
    if sigma:
        eo_noise = torch.randn(batch["eo"].shape, generator=generator) * sigma
        esm_noise = torch.randn(batch["esm"].shape, generator=generator) * sigma
        batch["eo"] += eo_noise * batch["eo_mask"].unsqueeze(-1)
        batch["esm"] += esm_noise * batch["esm_mask"].unsqueeze(-1)
    prediction, captured = run_with_capture(batch)
    alpha = torch.sigmoid(captured["fusion_gate_logit"])
    check(f"NOISE-{sigma}-FINITE", f"被动测角噪声σ={sigma} rad输出有限", bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(alpha).all()), True, True)
    noise_rows.append(
        {
            "category": "passive_noise",
            "case": f"sigma_{sigma}",
            "value": sigma,
            "stable_position_rmse_m": position_rmse(prediction),
            "fusion_alpha_mean": float(alpha.mean()),
            "fusion_alpha_min": float(alpha.min()),
            "fusion_alpha_max": float(alpha.max()),
            "output_delta_rms": float(torch.sqrt(torch.mean((prediction - complete_prediction) ** 2)).item()),
        }
    )

# Legal pose/IMU changes exercise the pose projection and sigmoid gate.
pose_rows: list[dict] = []
for case, amplitude in (("zero", 0.0), ("constant", 0.2), ("dynamic", 0.5)):
    batch = make_batch(1, 1, 1)
    if case == "constant":
        batch["imu"][..., :3] = amplitude
    elif case == "dynamic":
        time = torch.linspace(0.0, 2.0 * np.pi, steps).view(1, steps, 1)
        batch["imu"][..., 0] = amplitude * torch.sin(time)
        batch["imu"][..., 1] = amplitude * torch.cos(time)
        batch["imu"][..., 2] = 0.5 * amplitude * torch.sin(2.0 * time)
    prediction, captured = run_with_capture(batch)
    pose_gate = torch.sigmoid(captured["pose_projection"])
    check(f"POSE-{case}-SHAPE", f"{case}姿态门形状", tuple(pose_gate.shape) == expected_shape, tuple(pose_gate.shape), expected_shape)
    check(f"POSE-{case}-FINITE", f"{case}姿态门和输出有限", bool(torch.isfinite(pose_gate).all()) and bool(torch.isfinite(prediction).all()), True, True)
    check(f"POSE-{case}-RANGE", f"{case}姿态门位于[0,1]", float(pose_gate.min()) >= 0.0 and float(pose_gate.max()) <= 1.0, [float(pose_gate.min()), float(pose_gate.max())], "[0,1]")
    pose_rows.append(
        {
            "category": "pose",
            "case": case,
            "value": amplitude,
            "stable_position_rmse_m": position_rmse(prediction),
            "pose_gate_mean": float(pose_gate.mean()),
            "pose_gate_min": float(pose_gate.min()),
            "pose_gate_max": float(pose_gate.max()),
            "output_delta_rms": float(torch.sqrt(torch.mean((prediction - complete_prediction) ** 2)).item()),
        }
    )
check("POSE-RESPONSE", "姿态输入变化引起门控或输出响应", max(row["output_delta_rms"] for row in pose_rows[1:]) > 1e-6, [row["output_delta_rms"] for row in pose_rows], ">1e-6")

# ESM mode is a Learned-K raw feature. Record response without prescribing direction.
mode_rows: list[dict] = []
for mode in (0.0, 1.0):
    batch = make_batch(1, 1, 1)
    batch["esm_mode"].fill_(mode)
    prediction, captured = run_with_capture(batch)
    check(f"ESM-MODE-{int(mode)}-FINITE", f"ESM模式{int(mode)}输出有限", bool(torch.isfinite(prediction).all()), True, True)
    mode_rows.append(
        {
            "category": "esm_mode",
            "case": f"mode_{int(mode)}",
            "value": mode,
            "stable_position_rmse_m": position_rmse(prediction),
            "fusion_alpha_mean": float(torch.sigmoid(captured["fusion_gate_logit"]).mean()),
            "output_delta_rms": float(torch.sqrt(torch.mean((prediction - complete_prediction) ** 2)).item()),
        }
    )

# Figures.
labels = [row["combination"] for row in mask_rows]
x = np.arange(len(labels))
fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
axes[0].bar(x, [row["stable_position_rmse_m"] for row in mask_rows], color="#4c78a8")
axes[0].set_ylabel("稳定阶段三维位置RMSE / m")
axes[0].set_title("八种雷达 EO ESM mask组合的实际响应")
axes[0].grid(axis="y", alpha=0.25)
axes[1].bar(x, [row["output_delta_vs_complete_rms"] for row in mask_rows], color="#f58518", label="相对完整模态输出RMS差")
axes[1].plot(x, [100.0 * row["output_finite_rate"] for row in mask_rows], color="#54a24b", marker="o", label="有限值比例/%")
axes[1].set_ylabel("输出差异 / 有限比例")
axes[1].set_xticks(x, labels)
axes[1].set_xlabel("组合编码 R雷达 E光电 S电子侦察")
axes[1].grid(axis="y", alpha=0.25)
axes[1].legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "figure_1_eight_mask_combinations.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].bar(x - 0.18, [row["eo_effective_weight_mean"] for row in mask_rows], 0.36, label="EO有效权重", color="#4c78a8")
axes[0].bar(x + 0.18, [row["esm_effective_weight_mean"] for row in mask_rows], 0.36, label="ESM有效权重", color="#f58518")
axes[0].set_xticks(x, labels, rotation=35)
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("平均归一化权重")
axes[0].set_title("无效被动源权重归零")
axes[0].grid(axis="y", alpha=0.25)
axes[0].legend()
axes[1].bar(x, [100.0 * row["passive_missing_ratio"] for row in mask_rows], color="#b279a2")
axes[1].set_xticks(x, labels, rotation=35)
axes[1].set_ylabel("被动缺失令牌占比/%")
axes[1].set_title("EO与ESM同时无效时使用缺失令牌")
axes[1].grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "figure_2_passive_weights_and_missing_token.png", dpi=180, bbox_inches="tight")
plt.close(fig)

modules = [row["module"] for row in module_rows]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].barh(modules, [row["finite_rate"] * 100.0 for row in module_rows], color="#54a24b")
axes[0].set_xlim(0, 105)
axes[0].set_xlabel("有限值比例/%")
axes[0].set_title("模块级张量有限性")
axes[0].grid(axis="x", alpha=0.25)
axes[1].hist(fusion_alpha.numpy().ravel(), bins=20, color="#4c78a8", edgecolor="white")
axes[1].axvline(float(fusion_alpha.mean()), color="#e45756", ls="--", label=f"均值 {float(fusion_alpha.mean()):.4f}")
axes[1].set_xlim(0, 1)
axes[1].set_xlabel("融合门alpha")
axes[1].set_ylabel("样本数")
axes[1].set_title("完整模态下融合门分布")
axes[1].legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "figure_3_module_snapshot_and_fusion_gate.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
noise_x = [row["value"] for row in noise_rows]
axes[0].plot(noise_x, [row["stable_position_rmse_m"] for row in noise_rows], marker="o", color="#4c78a8", label="位置RMSE")
ax2 = axes[0].twinx()
ax2.plot(noise_x, [row["fusion_alpha_mean"] for row in noise_rows], marker="s", color="#f58518", label="融合门均值")
axes[0].set_xlabel("被动测角噪声标准差/rad")
axes[0].set_ylabel("位置RMSE/m", color="#4c78a8")
ax2.set_ylabel("融合门均值", color="#f58518")
axes[0].set_title("被动测角噪声响应（仅记录 不设单调判据）")
axes[0].grid(alpha=0.25)
pose_labels = [row["case"] for row in pose_rows]
px = np.arange(len(pose_labels))
axes[1].bar(px - 0.18, [row["pose_gate_mean"] for row in pose_rows], 0.36, color="#54a24b", label="姿态门均值")
axes[1].bar(px + 0.18, [row["output_delta_rms"] for row in pose_rows], 0.36, color="#b279a2", label="相对基线输出RMS差")
axes[1].set_xticks(px, pose_labels)
axes[1].set_title("姿态门控与输出响应")
axes[1].grid(axis="y", alpha=0.25)
axes[1].legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "figure_4_noise_and_pose_response.png", dpi=180, bbox_inches="tight")
plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


write_csv(RUN_DIR / "scenario_checks.csv", checks)
write_csv(RUN_DIR / "mask_combinations.csv", mask_rows)
write_csv(RUN_DIR / "module_snapshots.csv", module_rows)
write_csv(RUN_DIR / "noise_response.csv", noise_rows)
write_csv(RUN_DIR / "pose_response.csv", pose_rows)
write_csv(RUN_DIR / "esm_mode_response.csv", mode_rows)

result = {
    "test_case": "IT-02-02",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "pytorch": torch.__version__,
    "platform": platform.platform(),
    "seed": 20260923,
    "model_path": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
    "model_sha256": sha256(MODEL_PATH),
    "sample_path": str(SAMPLE_PATH.relative_to(ROOT)).replace("\\", "/"),
    "sample_sha256": sha256(SAMPLE_PATH),
    "sample_target_index": 0,
    "shape": {"batch": 1, "steps": steps, "agents": agents, "state": 6, "feature": 96},
    "passed": all(row["passed"] for row in checks),
    "check_count": len(checks),
    "passed_count": sum(row["passed"] for row in checks),
    "mask_combinations": {
        "count": len(mask_rows),
        "all_finite": all(row["output_finite_rate"] == 1.0 for row in mask_rows),
        "rmse_min_m": min(row["stable_position_rmse_m"] for row in mask_rows),
        "rmse_max_m": max(row["stable_position_rmse_m"] for row in mask_rows),
        "invalid_passive_weight_abs_max": max(max(row["invalid_eo_weight_abs_max"], row["invalid_esm_weight_abs_max"]) for row in mask_rows),
        "missing_token_abs_error_max": max(row["missing_token_abs_error_max"] for row in mask_rows),
    },
    "structure": {
        "cross_attention_heads": [model.cross_radar_to_passive.attn.num_heads, model.cross_passive_to_radar.attn.num_heads],
        "batch_first": bool(model.cross_radar_to_passive.attn.batch_first and model.cross_passive_to_radar.attn.batch_first),
        "adaptive_alignment_abs_error_max": align_error,
        "fusion_alpha_min": float(fusion_alpha.min()),
        "fusion_alpha_mean": float(fusion_alpha.mean()),
        "fusion_alpha_max": float(fusion_alpha.max()),
    },
    "noise_response": noise_rows,
    "pose_response": pose_rows,
    "esm_mode_response": mode_rows,
    "figures": [path.name for path in sorted(FIG_DIR.glob("*.png"))],
}
(RUN_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))

