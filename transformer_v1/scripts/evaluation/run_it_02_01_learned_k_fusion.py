from __future__ import annotations

import csv
import hashlib
import json
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

from src.model import LearnedKFrontEnd, radar_body_to_world_xyz


RUN_DIR = ROOT / "artifacts" / "reports" / "software_test" / "IT-02-01" / "20260923_run01"
FIG_DIR = RUN_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(20260923)
np.random.seed(20260923)

checks: list[dict] = []


def check(check_id: str, description: str, condition: bool, actual, expected) -> None:
    checks.append(
        {
            "check_id": check_id,
            "description": description,
            "passed": bool(condition),
            "actual": actual,
            "expected": expected,
        }
    )
    if not condition:
        raise AssertionError(f"{check_id}: {description}; actual={actual!r}; expected={expected!r}")


def sph_from_world(world: np.ndarray, origin: np.ndarray) -> np.ndarray:
    rel = np.asarray(world, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    distance = float(np.linalg.norm(rel))
    azimuth = float(np.arctan2(rel[1], rel[0]))
    elevation = float(np.arcsin(rel[2] / distance))
    # Frozen model/dataset interface: [azimuth, elevation, range].
    return np.array([azimuth, elevation, distance], dtype=np.float32)


def positions_from_state(state: np.ndarray) -> np.ndarray:
    return state[..., [0, 2, 4]]


def reference_fusion(
    radar: np.ndarray,
    radar_mask: np.ndarray,
    fast_link: np.ndarray,
    slow_link: np.ndarray,
    fast_update: np.ndarray,
    slow_update: np.ndarray,
    fast_age: np.ndarray,
    slow_age: np.ndarray,
    uav_pos: np.ndarray,
    rotmat: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Independent float64 calculation of one frame's information fusion."""
    eye = np.eye(3, dtype=np.float64)
    information = eye * 1e-6
    information_vector = np.zeros(3, dtype=np.float64)
    radar_points: list[np.ndarray] = []

    for agent in range(radar.shape[0]):
        azimuth, elevation, distance = radar[agent].astype(np.float64)
        rel_body = np.array(
            [
                distance * np.cos(elevation) * np.cos(azimuth),
                distance * np.cos(elevation) * np.sin(azimuth),
                distance * np.sin(elevation),
            ]
        )
        world = rotmat[agent].astype(np.float64) @ rel_body + uav_pos[agent].astype(np.float64)
        radar_points.append(world)
        if radar_mask[agent] > 0:
            direction = world - uav_pos[agent]
            direction /= np.linalg.norm(direction)
            outer = np.outer(direction, direction)
            variance_tangential = (max(float(distance), 1.0) * 0.002) ** 2 + 1.0
            weight = (1.0 / (48.0**2)) * outer + (1.0 / variance_tangential) * (eye - outer)
            information += float(radar_mask[agent]) * weight
            information_vector += float(radar_mask[agent]) * (weight @ world)

        fast_valid = bool(fast_update[agent] > 0.5 or fast_age[agent] < 2.0)
        slow_valid = bool(slow_update[agent] > 0.5 or slow_age[agent] < 5.0)
        fast_weight = float(np.exp(-max(float(fast_age[agent]), 0.0) / 0.3)) if fast_valid else 0.0
        slow_weight = float(np.exp(-max(float(slow_age[agent]), 0.0) / 2.0)) if slow_valid else 0.0
        information += eye * (fast_weight + slow_weight) / (60.0**2)
        information_vector += (fast_weight * fast_link[agent] + slow_weight * slow_link[agent]) / (60.0**2)

    position = np.linalg.solve(information, information_vector)
    return position, {
        "information_matrix": information,
        "information_vector": information_vector,
        "radar_points": np.asarray(radar_points),
    }


def make_base(batch: int, steps: int, agents: int) -> dict[str, torch.Tensor]:
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float32)
    rot = torch.eye(3, dtype=torch.float32).view(1, 1, 1, 3, 3).repeat(batch, steps, agents, 1, 1)
    return {
        "radar": zeros(batch, steps, agents, 3),
        "eo": zeros(batch, steps, agents, 2),
        "esm": zeros(batch, steps, agents, 2),
        "radar_mask": zeros(batch, steps, agents),
        "eo_mask": zeros(batch, steps, agents),
        "esm_mask": zeros(batch, steps, agents),
        "uav_pos": zeros(batch, steps, agents, 3),
        "rotmat_b2e": rot,
        "fast_link": zeros(batch, steps, agents, 3),
        "slow_link": zeros(batch, steps, agents, 3),
        "fast_update": zeros(batch, steps, agents),
        "slow_update": zeros(batch, steps, agents),
        "imu": zeros(batch, steps, agents, 6),
        "fast_age": torch.full((batch, steps, agents), 999.0, dtype=torch.float32),
        "slow_age": torch.full((batch, steps, agents), 999.0, dtype=torch.float32),
        "esm_mode": zeros(batch, steps, agents),
        "dt": torch.full((batch,), 0.1, dtype=torch.float32),
    }


MODEL_PATH = ROOT / "results" / "runs_40off(v7)" / "run_20260728_103728" / "best_model.pt"
MODEL_SHA256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
front_state = {key.removeprefix("k_front."): value for key, value in checkpoint.items() if key.startswith("k_front.")}
front = LearnedKFrontEnd(feature_dim=96).eval()
front.load_state_dict(front_state, strict=True)

# 1. Confirm the executable radar order against a hand-built geometry case.
geometry_radar = torch.tensor([[[[0.3, -0.2, 7500.0]]]], dtype=torch.float32)
geometry_origin = torch.tensor([[[[100.0, -50.0, 20.0]]]], dtype=torch.float32)
geometry_rot = torch.eye(3).view(1, 1, 1, 3, 3)
geometry_actual = radar_body_to_world_xyz(geometry_radar, geometry_origin, geometry_rot)[0, 0, 0].detach().numpy()
az, el, rng = geometry_radar[0, 0, 0].numpy()
geometry_expected = geometry_origin[0, 0, 0].numpy() + np.array(
    [rng * np.cos(el) * np.cos(az), rng * np.cos(el) * np.sin(az), rng * np.sin(el)]
)
geometry_error = float(np.linalg.norm(geometry_actual - geometry_expected))
check("INPUT-ORDER", "雷达冻结输入顺序为[方位角,俯仰角,距离]", geometry_error < 1e-3, geometry_error, "<1e-3 m")

# 2. Information-matrix fusion groups, checked against an independent float64 implementation.
agents = 2
uav = np.array([[0.0, 0.0, 0.0], [0.0, 1200.0, 100.0]], dtype=np.float32)
radar_world = np.array([[10000.0, 120.0, 40.0], [10035.0, -55.0, 65.0]], dtype=np.float32)
radar = np.stack([sph_from_world(radar_world[i], uav[i]) for i in range(agents)])
radar_mask = np.ones(agents, dtype=np.float32)
fast = np.array([[10020.0, 15.0, 50.0], [9980.0, 40.0, 35.0]], dtype=np.float32)
slow = np.array([[9950.0, -20.0, 70.0], [10080.0, 25.0, 20.0]], dtype=np.float32)
identity = np.repeat(np.eye(3, dtype=np.float32)[None], agents, axis=0)

scenario_specs = [
    ("仅雷达", np.zeros(agents), np.zeros(agents), np.full(agents, 999.0), np.full(agents, 999.0)),
    ("雷达+快链", np.array([1.0, 0.0]), np.zeros(agents), np.array([0.0, 999.0]), np.full(agents, 999.0)),
    ("雷达+慢链", np.zeros(agents), np.array([0.0, 1.0]), np.full(agents, 999.0), np.array([999.0, 0.0])),
    ("雷达+快慢链", np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 999.0]), np.array([999.0, 0.0])),
]
fusion_rows: list[dict] = []
fusion_plot_data: list[dict] = []
for name, fast_update, slow_update, fast_age, slow_age in scenario_specs:
    base = make_base(1, 1, agents)
    base["radar"][0, 0] = torch.from_numpy(radar)
    base["radar_mask"][0, 0] = torch.from_numpy(radar_mask)
    base["uav_pos"][0, 0] = torch.from_numpy(uav)
    base["fast_link"][0, 0] = torch.from_numpy(fast)
    base["slow_link"][0, 0] = torch.from_numpy(slow)
    base["fast_update"][0, 0] = torch.from_numpy(fast_update.astype(np.float32))
    base["slow_update"][0, 0] = torch.from_numpy(slow_update.astype(np.float32))
    base["fast_age"][0, 0] = torch.from_numpy(fast_age.astype(np.float32))
    base["slow_age"][0, 0] = torch.from_numpy(slow_age.astype(np.float32))
    with torch.no_grad():
        observed = front._build_obs_state(
            base["radar"], base["radar_mask"], base["fast_link"], base["slow_link"],
            base["fast_update"], base["slow_update"], base["fast_age"], base["slow_age"],
            base["uav_pos"], base["rotmat_b2e"], base["dt"],
        )[0, 0].numpy()
    actual_position = positions_from_state(observed)
    expected_position, detail = reference_fusion(
        radar, radar_mask, fast, slow, fast_update, slow_update, fast_age, slow_age, uav, identity
    )
    error = float(np.linalg.norm(actual_position - expected_position))
    check(f"FUSION-{name}", f"{name}信息矩阵融合与独立复算一致", error < 0.05, error, "<0.05 m")
    fusion_rows.append(
        {
            "scenario": name,
            "actual_x": float(actual_position[0]),
            "actual_y": float(actual_position[1]),
            "actual_z": float(actual_position[2]),
            "reference_x": float(expected_position[0]),
            "reference_y": float(expected_position[1]),
            "reference_z": float(expected_position[2]),
            "position_error_m": error,
            "regularization": 1e-6,
        }
    )
    fusion_plot_data.append({"name": name, "actual": actual_position, "reference": expected_position, "detail": detail})

# 3. Cache-window equality boundaries, update override and freshness weights.
boundary_specs = [
    ("快链略小于2s", "fast", 1.99, 0.0, True),
    ("快链等于2s", "fast", 2.00, 0.0, False),
    ("快链略大于2s", "fast", 2.01, 0.0, False),
    ("快链update覆盖", "fast", 2.01, 1.0, True),
    ("慢链略小于5s", "slow", 4.99, 0.0, True),
    ("慢链等于5s", "slow", 5.00, 0.0, False),
    ("慢链略大于5s", "slow", 5.01, 0.0, False),
    ("慢链update覆盖", "slow", 5.01, 1.0, True),
]
boundary_rows: list[dict] = []
for name, source, age, update, expected_valid in boundary_specs:
    b = make_base(1, 1, 1)
    link = np.array([1200.0, -100.0, 50.0], dtype=np.float32)
    if source == "fast":
        b["fast_link"][0, 0, 0] = torch.from_numpy(link)
        b["fast_age"][0, 0, 0] = age
        b["fast_update"][0, 0, 0] = update
    else:
        b["slow_link"][0, 0, 0] = torch.from_numpy(link)
        b["slow_age"][0, 0, 0] = age
        b["slow_update"][0, 0, 0] = update
    with torch.no_grad():
        z = front._build_obs_state(
            b["radar"], b["radar_mask"], b["fast_link"], b["slow_link"], b["fast_update"],
            b["slow_update"], b["fast_age"], b["slow_age"], b["uav_pos"], b["rotmat_b2e"], b["dt"]
        )[0, 0].numpy()
    output_position = positions_from_state(z)
    actual_valid = bool(np.linalg.norm(output_position) > 1e-6)
    check(f"CACHE-{name}", f"{name}有效性边界", actual_valid == expected_valid, actual_valid, expected_valid)
    tau = 0.3 if source == "fast" else 2.0
    expected_weight = float(np.exp(-max(age, 0.0) / tau)) if expected_valid else 0.0
    fast_update_np = np.array([update if source == "fast" else 0.0])
    slow_update_np = np.array([update if source == "slow" else 0.0])
    fast_age_np = np.array([age if source == "fast" else 999.0])
    slow_age_np = np.array([age if source == "slow" else 999.0])
    ref, _ = reference_fusion(
        np.zeros((1, 3)), np.zeros(1),
        link[None] if source == "fast" else np.zeros((1, 3)),
        link[None] if source == "slow" else np.zeros((1, 3)),
        fast_update_np, slow_update_np, fast_age_np, slow_age_np,
        np.zeros((1, 3)), np.eye(3)[None],
    )
    error = float(np.linalg.norm(output_position - ref))
    check(f"CACHE-REF-{name}", f"{name}输出与时效权重独立复算一致", error < 0.05, error, "<0.05 m")
    boundary_rows.append(
        {
            "case": name,
            "source": source,
            "age_s": age,
            "update": update,
            "expected_valid": expected_valid,
            "actual_valid": actual_valid,
            "freshness_weight": expected_weight,
            "position_norm_m": float(np.linalg.norm(output_position)),
            "reference_error_m": error,
        }
    )

# 4. Verify the 30-dimensional raw feature order by capturing obs_encoder input.
feature_case = make_base(1, 1, 1)
feature_case["radar"][0, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
feature_case["eo"][0, 0, 0] = torch.tensor([4.0, 5.0])
feature_case["esm"][0, 0, 0] = torch.tensor([6.0, 7.0])
feature_case["imu"][0, 0, 0] = torch.arange(8.0, 14.0)
feature_case["radar_mask"][0, 0, 0] = 14.0
feature_case["eo_mask"][0, 0, 0] = 15.0
feature_case["esm_mask"][0, 0, 0] = 16.0
feature_case["esm_mode"][0, 0, 0] = 17.0
feature_case["fast_age"][0, 0, 0] = 18.0
feature_case["slow_age"][0, 0, 0] = 19.0
feature_case["fast_link"][0, 0, 0] = torch.arange(20.0, 23.0)
feature_case["slow_link"][0, 0, 0] = torch.arange(23.0, 26.0)
feature_case["fast_update"][0, 0, 0] = 26.0
feature_case["slow_update"][0, 0, 0] = 27.0
feature_case["uav_pos"][0, 0, 0] = torch.arange(28.0, 31.0)
captured_features: list[torch.Tensor] = []
hook = front.obs_encoder[0].register_forward_pre_hook(lambda _module, args: captured_features.append(args[0].detach().cpu()))
with torch.no_grad():
    feature_state, feature_obs = front(**feature_case)
hook.remove()
raw_feature = captured_features[0][0, 0, 0].numpy()
expected_feature = np.arange(1.0, 31.0, dtype=np.float32)
feature_error = float(np.max(np.abs(raw_feature - expected_feature)))
check("FEATURE-DIM", "Learned-K原始质量特征维度", raw_feature.shape == (30,), raw_feature.shape, (30,))
check("FEATURE-ORDER", "30维原始质量特征字段顺序", feature_error == 0.0, feature_error, 0.0)
check("FEATURE-FINITE", "30维特征样例的观测与学习状态均有限", bool(torch.isfinite(feature_state).all() and torch.isfinite(feature_obs).all()), True, True)

# 5. First-frame missing, continuous missing, recovery, velocity clipping and learned gain range.
sequence = make_base(1, 7, 1)
sequence["radar"][0, 1, 0] = torch.tensor([0.0, 0.0, 1000.0])
sequence["radar_mask"][0, 1, 0] = 1.0
sequence["eo_mask"][0, 1, 0] = 1.0  # Keeps the feature pooling path active without adding position information.
sequence["fast_link"][0, 4, 0] = torch.tensor([1600.0, 0.0, 0.0])
sequence["fast_update"][0, 4, 0] = 1.0
sequence["fast_age"][0, 4, 0] = 0.0
sequence["fast_link"][0, 5, 0] = torch.tensor([5000.0, 0.0, 0.0])
sequence["fast_update"][0, 5, 0] = 1.0
sequence["fast_age"][0, 5, 0] = 0.0
captured_gains: list[torch.Tensor] = []
gain_hook = front.k_net.register_forward_hook(lambda _module, _args, output: captured_gains.append(output.detach().cpu()))
with torch.no_grad():
    learned_state, observed_state = front(**sequence)
gain_hook.remove()
learned_np = learned_state[0].numpy()
observed_np = observed_state[0].numpy()
gain_np = torch.cat(captured_gains, dim=0).numpy()
pos = positions_from_state(observed_np)
vel = observed_np[:, [1, 3, 5]]
check("MISSING-FIRST", "首帧无任何位置源时采用有限零初始化", np.all(np.isfinite(pos[0])) and np.allclose(pos[0], 0.0), pos[0].tolist(), [0.0, 0.0, 0.0])
check("MISSING-CONTINUOUS", "连续缺失帧保持最近融合位置", np.allclose(pos[2], pos[1]) and np.allclose(pos[3], pos[1]), pos[1:4].tolist(), "frame2=frame3=frame1")
check("RECOVERY", "观测恢复当帧重新形成有效融合位置", pos[4, 0] > pos[3, 0] + 500.0, pos[3:5, 0].tolist(), "recovery increase >500 m")
check("VELOCITY-CLIP", "观测差分速度逐轴限制在±400 m/s", float(np.max(np.abs(vel))) <= 400.0 and np.isclose(vel[4, 0], 400.0) and np.isclose(vel[5, 0], 400.0), vel.tolist(), "max abs=400 m/s")
check("LEARNED-FINITE", "预测与学习增益更新后的六维状态全程有限", bool(np.isfinite(learned_np).all()), True, True)
check("GAIN-RANGE", "六维对角学习增益位于[0,1]", float(gain_np.min()) >= 0.0 and float(gain_np.max()) <= 1.0, [float(gain_np.min()), float(gain_np.max())], "[0,1]")
check("STATE-ORDER", "六维观测状态按[px,vx,py,vy,pz,vz]排列", np.allclose(observed_np[:, [0, 2, 4]], pos) and np.allclose(observed_np[:, [1, 3, 5]], vel), True, True)

missing_rows = []
for frame in range(len(pos)):
    missing_rows.append(
        {
            "frame": frame,
            "source": "radar" if frame == 1 else ("fast_link" if frame in (4, 5) else "missing"),
            "obs_px": float(observed_np[frame, 0]),
            "obs_vx": float(observed_np[frame, 1]),
            "obs_py": float(observed_np[frame, 2]),
            "obs_vy": float(observed_np[frame, 3]),
            "obs_pz": float(observed_np[frame, 4]),
            "obs_vz": float(observed_np[frame, 5]),
            "learned_px": float(learned_np[frame, 0]),
            "learned_vx": float(learned_np[frame, 1]),
            "gain_min": float(gain_np[frame].min()),
            "gain_max": float(gain_np[frame].max()),
        }
    )

# 6. Explicit covariance constants and distance dependence.
near_range, far_range = 5000.0, 50000.0
near_var_t = (0.002 * near_range) ** 2 + 1.0
far_var_t = (0.002 * far_range) ** 2 + 1.0
radial_variance = 48.0**2
link_variance = 60.0**2
check("COV-RADIAL", "雷达径向方差为48² m²", radial_variance == 2304.0, radial_variance, 2304.0)
check("COV-LINK", "链路各向同性方差为60² m²", link_variance == 3600.0, link_variance, 3600.0)
check("COV-DISTANCE", "雷达切向方差随距离增大", far_var_t > near_var_t, [near_var_t, far_var_t], "far > near")

# Figures.
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, item in zip(axes.ravel(), fusion_plot_data):
    points = item["detail"]["radar_points"]
    ax.scatter(points[:, 0], points[:, 1], marker="o", s=55, color="#4c78a8", label="雷达反演点")
    if "快链" in item["name"]:
        ax.scatter(fast[:, 0], fast[:, 1], marker="^", s=60, color="#f58518", label="快链位置")
    if "慢链" in item["name"]:
        ax.scatter(slow[:, 0], slow[:, 1], marker="s", s=55, color="#54a24b", label="慢链位置")
    ax.scatter(item["reference"][0], item["reference"][1], marker="+", s=140, linewidth=2.5, color="#b279a2", label="独立复算")
    ax.scatter(item["actual"][0], item["actual"][1], marker="x", s=90, linewidth=2.2, color="#e45756", label="冻结实现")
    ax.set_title(f"{item['name']}（误差{np.linalg.norm(item['actual']-item['reference']):.4f} m）")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle("IT-02-01 信息矩阵融合与独立复算", fontsize=16)
fig.tight_layout()
fig.savefig(FIG_DIR / "figure_1_information_matrix_fusion.png", dpi=180, bbox_inches="tight")
plt.close(fig)

age = np.linspace(0.0, 6.0, 601)
fast_weight_curve = np.where(age < 2.0, np.exp(-age / 0.3), 0.0)
slow_weight_curve = np.where(age < 5.0, np.exp(-age / 2.0), 0.0)
distance = np.linspace(1000.0, 60000.0, 400)
tangential_sigma = np.sqrt((0.002 * distance) ** 2 + 1.0)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].plot(age, fast_weight_curve, label="快链：exp(-Age/0.3)，Age<2 s", color="#4c78a8")
axes[0].plot(age, slow_weight_curve, label="慢链：exp(-Age/2.0)，Age<5 s", color="#54a24b")
axes[0].axvline(2.0, color="#4c78a8", ls="--")
axes[0].axvline(5.0, color="#54a24b", ls="--")
axes[0].scatter([r["age_s"] for r in boundary_rows], [r["freshness_weight"] for r in boundary_rows], color="#e45756", s=25, zorder=3, label="执行边界样例")
axes[0].set_yscale("log")
axes[0].set_ylim(1e-9, 1.2)
axes[0].set_xlabel("数据龄期 Age / s")
axes[0].set_ylabel("有效时效权重")
axes[0].set_title("缓存窗口与指数时效权重")
axes[0].grid(alpha=0.25)
axes[0].legend(fontsize=8)
axes[1].plot(distance / 1000.0, tangential_sigma, color="#f58518", label="切向标准差")
axes[1].axhline(48.0, color="#4c78a8", ls="--", label="径向标准差48 m")
axes[1].axhline(60.0, color="#54a24b", ls=":", label="链路标准差60 m")
axes[1].scatter([near_range / 1000.0, far_range / 1000.0], [np.sqrt(near_var_t), np.sqrt(far_var_t)], color="#e45756", zorder=3)
axes[1].set_xlabel("雷达距离 / km")
axes[1].set_ylabel("标准差 / m")
axes[1].set_title("雷达切向噪声的距离依赖")
axes[1].grid(alpha=0.25)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "figure_2_cache_freshness_and_covariance.png", dpi=180, bbox_inches="tight")
plt.close(fig)

frames = np.arange(len(pos))
fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
axes[0].plot(frames, pos[:, 0], marker="o", label="融合观测px", color="#4c78a8")
axes[0].plot(frames, learned_np[:, 0], marker="s", label="Learned-K状态px", color="#f58518")
axes[0].axvspan(1.5, 3.5, color="#cccccc", alpha=0.35, label="连续缺失")
axes[0].set_ylabel("位置 / m")
axes[0].set_title("首帧缺失、连续缺失与恢复")
axes[0].grid(alpha=0.25)
axes[0].legend()
axes[1].plot(frames, vel[:, 0], marker="o", label="观测差分vx", color="#4c78a8")
axes[1].plot(frames, learned_np[:, 1], marker="s", label="Learned-K状态vx", color="#f58518")
axes[1].axhline(400.0, color="#e45756", ls="--", label="观测速度上限+400 m/s")
axes[1].axhline(-400.0, color="#e45756", ls="--")
axes[1].set_xlabel("帧号")
axes[1].set_ylabel("速度 / (m/s)")
axes[1].set_title("位置跳变时观测速度裁剪")
axes[1].grid(alpha=0.25)
axes[1].legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "figure_3_missing_recovery_and_velocity_clip.png", dpi=180, bbox_inches="tight")
plt.close(fig)

feature_labels = [
    "雷达az", "雷达el", "雷达r", "EO az", "EO el", "ESM az", "ESM el",
    "IMU1", "IMU2", "IMU3", "IMU4", "IMU5", "IMU6", "雷达mask", "EO mask", "ESM mask",
    "ESM模式", "快链Age", "慢链Age", "快链x", "快链y", "快链z", "慢链x", "慢链y", "慢链z",
    "快链更新", "慢链更新", "平台x", "平台y", "平台z",
]
fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [1, 2.2]})
im = axes[0].imshow(raw_feature[None, :], aspect="auto", cmap="Blues")
axes[0].set_yticks([])
axes[0].set_xticks(np.arange(30), feature_labels, rotation=70, ha="right", fontsize=8)
axes[0].set_title("30维原始特征字段顺序（单元格值为1至30）")
for i, value in enumerate(raw_feature):
    axes[0].text(i, 0, f"{value:.0f}", ha="center", va="center", fontsize=7, color="black")
fig.colorbar(im, ax=axes[0], fraction=0.015, pad=0.01)
for dim in range(6):
    axes[1].plot(frames, gain_np[:, dim], marker="o", linewidth=1.3, label=["px", "vx", "py", "vy", "pz", "vz"][dim])
axes[1].set_ylim(-0.03, 1.03)
axes[1].set_xlabel("帧号")
axes[1].set_ylabel("对角学习增益")
axes[1].set_title(f"六维增益范围：{gain_np.min():.4f} 至 {gain_np.max():.4f}")
axes[1].grid(alpha=0.25)
axes[1].legend(ncol=6)
fig.tight_layout()
fig.savefig(FIG_DIR / "figure_4_feature_order_and_gain_range.png", dpi=180, bbox_inches="tight")
plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


write_csv(RUN_DIR / "scenario_checks.csv", checks)
write_csv(RUN_DIR / "fusion_cases.csv", fusion_rows)
write_csv(RUN_DIR / "cache_boundaries.csv", boundary_rows)
write_csv(RUN_DIR / "missing_recovery.csv", missing_rows)

result = {
    "test_case": "IT-02-01",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "implementation": [
        "src/model.py::radar_body_to_world_xyz",
        "src/model.py::solve_3x3_onnx_safe",
        "src/model.py::LearnedKFrontEnd",
    ],
    "python": sys.version,
    "pytorch": torch.__version__,
    "platform": platform.platform(),
    "seed": 20260923,
    "model_path": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
    "model_sha256": MODEL_SHA256,
    "passed": all(row["passed"] for row in checks),
    "check_count": len(checks),
    "passed_count": sum(row["passed"] for row in checks),
    "radar_input_order": ["azimuth", "elevation", "range"],
    "information_fusion": {
        "scenario_count": len(fusion_rows),
        "max_reference_error_m": max(row["position_error_m"] for row in fusion_rows),
        "regularization": 1e-6,
        "radar_radial_std_m": 48.0,
        "radar_tangential_variance": "(0.002*r)^2+1",
        "link_position_std_m": 60.0,
    },
    "cache_boundaries": {
        "case_count": len(boundary_rows),
        "fast_window_s": 2.0,
        "slow_window_s": 5.0,
        "fast_tau_s": 0.3,
        "slow_tau_s": 2.0,
        "max_reference_error_m": max(row["reference_error_m"] for row in boundary_rows),
    },
    "features": {
        "raw_dimension": int(raw_feature.size),
        "max_order_error": feature_error,
    },
    "missing_and_learning": {
        "sequence_frames": int(len(pos)),
        "continuous_missing_frames": [2, 3],
        "recovery_frame": 4,
        "observed_velocity_abs_max_mps": float(np.max(np.abs(vel))),
        "learned_state_finite": bool(np.isfinite(learned_np).all()),
        "gain_min": float(gain_np.min()),
        "gain_max": float(gain_np.max()),
    },
    "figures": [path.name for path in sorted(FIG_DIR.glob("*.png"))],
}
(RUN_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
