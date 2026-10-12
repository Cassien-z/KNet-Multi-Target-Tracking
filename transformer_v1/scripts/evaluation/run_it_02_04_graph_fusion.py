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


RUN_DIR = ROOT / "artifacts" / "reports" / "software_test" / "IT-02-04" / "20260924_run01"
FIG_DIR = RUN_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ROOT / "results" / "runs_40off(v7)" / "run_20260728_103728" / "best_model.pt"
SAMPLE_PATH = ROOT / "data" / "val_data_v7_registered" / "track_cv_0000.pkl"
ABLATION_PATH = ROOT / "artifacts" / "metrics" / "acceptance" / "2ea4f9eb0f2845b599fcf4837d71a018" / "ablation_v7.json"
SEED = 20260924
WARMUP = 5

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


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


def construct_adj(positions: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the graph construction in CoopTrackerModel.forward for one frame."""
    distance = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0))[0]
    adjacency = valid[:, None] * valid[None, :]
    adjacency = adjacency * (distance <= 20000.0).to(adjacency.dtype)
    adjacency = torch.maximum(adjacency, torch.eye(positions.shape[0], dtype=adjacency.dtype))
    return adjacency, distance


model = CoopTrackerModel(d_model=96)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=True)
model.eval()
gat = model.coop_gat

# 19.9/20.0/20.1 km boundary and sensor-validity truth table.
boundary_specs = [
    ("19.9 km 双方有效", 19900.0, 1.0, 1.0, 1),
    ("20.0 km 双方有效", 20000.0, 1.0, 1.0, 1),
    ("20.1 km 双方有效", 20100.0, 1.0, 1.0, 0),
    ("19.9 km 节点0无效", 19900.0, 0.0, 1.0, 0),
    ("19.9 km 节点1无效", 19900.0, 1.0, 0.0, 0),
    ("19.9 km 双方无效", 19900.0, 0.0, 0.0, 0),
]
boundary_rows: list[dict] = []
for label, distance_m, valid0, valid1, expected in boundary_specs:
    positions = torch.tensor([[0.0, 0.0, 0.0], [distance_m, 0.0, 0.0], [50000.0, 0.0, 0.0], [80000.0, 0.0, 0.0], [110000.0, 0.0, 0.0]])
    valid = torch.tensor([valid0, valid1, 0.0, 0.0, 0.0])
    adj, dist = construct_adj(positions, valid)
    actual = int(adj[0, 1].item())
    row = {"case": label, "distance_m": float(dist[0, 1]), "valid_0": valid0, "valid_1": valid1, "cross_edge": actual, "expected_edge": expected, "self_loop_all": bool(torch.diag(adj).eq(1).all())}
    boundary_rows.append(row)
    check(f"BOUNDARY-{len(boundary_rows)}", label, actual == expected, actual, expected)
    check(f"SELF-{len(boundary_rows)}", f"{label}所有槽位保留自环", row["self_loop_all"], row["self_loop_all"], True)

# A five-node boundary snapshot: node 0 sees 19.9 km and 20.0 km, but not 20.1 km.
snapshot_positions = torch.tensor([
    [0.0, 0.0, 0.0], [19900.0, 0.0, 0.0], [20000.0, 0.0, 0.0],
    [20100.0, 0.0, 0.0], [50000.0, 0.0, 0.0],
])
all_valid = torch.ones(5)
snapshot_adj, snapshot_dist = construct_adj(snapshot_positions, all_valid)
check("SNAPSHOT-ROW0", "边界快照节点0邻接", snapshot_adj[0].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0], snapshot_adj[0].tolist(), [1, 1, 1, 0, 0])
invalid_valid = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
invalid_adj, _ = construct_adj(snapshot_positions, invalid_valid)
check("INVALID-CROSS", "无效节点2不与其他节点建立跨节点边", float(invalid_adj[2].sum()) == 1.0 and float(invalid_adj[:, 2].sum()) == 1.0, [float(invalid_adj[2].sum()), float(invalid_adj[:, 2].sum())], [1.0, 1.0])

# Graph-boundary feature isolation and valid-node pooling.
features = torch.randn(1, 5, 96)
graph_out = gat(features, adj_mask=invalid_adj.unsqueeze(0)).detach()
mutated = features.clone()
mutated[:, 2] = 10000.0 * torch.randn_like(mutated[:, 2])
mutated_out = gat(mutated, adj_mask=invalid_adj.unsqueeze(0)).detach()
valid_index = invalid_valid.bool()
valid_output_delta = float((graph_out[:, valid_index] - mutated_out[:, valid_index]).abs().max())
invalid_output_delta = float((graph_out[:, 2] - mutated_out[:, 2]).abs().max())
weights = invalid_valid.view(1, 5, 1)
pooled = (graph_out * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-6)
mutated_pooled = (mutated_out * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-6)
pooled_delta = float((pooled - mutated_pooled).abs().max())
check("INVALID-FEATURE-ISOLATION", "改变无效节点有限特征不影响有效节点", valid_output_delta <= 1e-6, valid_output_delta, "<=1e-6")
check("INVALID-POOL-ISOLATION", "改变无效节点有限特征不影响有效节点池化", pooled_delta <= 1e-6, pooled_delta, "<=1e-6")
check("INVALID-OUTPUT-RESPONDS", "无效节点自身输出可变化但不进入池化", invalid_output_delta > 1e-3, invalid_output_delta, ">1e-3")
check("POOL-WEIGHT", "无效节点池化权重为0", float(weights[0, 2, 0]) == 0.0, float(weights[0, 2, 0]), 0.0)

# Permutation equivariance at the GAT boundary and invariant pooled representation.
perm = torch.tensor([3, 0, 4, 1, 2])
inv_perm = torch.argsort(perm)
perm_features = features[:, perm]
perm_adj = invalid_adj[perm][:, perm].unsqueeze(0)
perm_weights = weights[:, perm]
perm_out = gat(perm_features, adj_mask=perm_adj).detach()
restored = perm_out[:, inv_perm]
permutation_node_delta = float((restored - graph_out).abs().max())
perm_pooled = (perm_out * perm_weights).sum(dim=1) / (perm_weights.sum(dim=1) + 1e-6)
permutation_pool_delta = float((perm_pooled - pooled).abs().max())
check("PERM-NODE", "同步置换图特征和邻接后节点输出等变", permutation_node_delta <= 1e-5, permutation_node_delta, "<=1e-5")
check("PERM-POOL", "同步置换池化权重后全局特征不变", permutation_pool_delta <= 1e-5, permutation_pool_delta, "<=1e-5")

# Single node, isolated nodes, zero-valid graph and 1..5 effective nodes.
node_rows: list[dict] = []
close_positions = torch.stack([torch.tensor([float(i * 1000), 0.0, 0.0]) for i in range(5)])
previous_pooled = None
for count in range(0, 6):
    valid = torch.zeros(5)
    valid[:count] = 1.0
    adj, _ = construct_adj(close_positions, valid)
    out = gat(features, adj_mask=adj.unsqueeze(0)).detach()
    weight = valid.view(1, 5, 1)
    denominator = float(weight.sum())
    pooled_value = (out * weight).sum(dim=1) / (weight.sum(dim=1) + 1e-6)
    delta_previous = 0.0 if previous_pooled is None else float(torch.linalg.vector_norm(pooled_value - previous_pooled))
    row = {
        "valid_nodes": count,
        "cross_edges": int(adj.sum().item() - 5),
        "self_loops": int(torch.diag(adj).sum().item()),
        "pool_denominator": denominator,
        "pooled_norm": float(torch.linalg.vector_norm(pooled_value)),
        "delta_vs_previous": delta_previous,
        "finite_rate": float(torch.isfinite(out).float().mean()),
    }
    node_rows.append(row)
    check(f"COUNT-{count}-SELF", f"{count}个有效节点时保留5个自环", row["self_loops"] == 5, row["self_loops"], 5)
    check(f"COUNT-{count}-FINITE", f"{count}个有效节点时图输出有限", row["finite_rate"] == 1.0, row["finite_rate"], 1.0)
    check(f"COUNT-{count}-DENOM", f"{count}个有效节点时池化分母正确", denominator == float(count), denominator, float(count))
    if count == 0:
        check("COUNT-0-POOL", "全无效图池化结果为零且无NaN", float(pooled_value.abs().max()) == 0.0 and bool(torch.isfinite(pooled_value).all()), float(pooled_value.abs().max()), 0.0)
    previous_pooled = pooled_value

isolated_positions = torch.tensor([[0.0, 0.0, 0.0], [25000.0, 0.0, 0.0], [60000.0, 0.0, 0.0], [90000.0, 0.0, 0.0], [120000.0, 0.0, 0.0]])
isolated_valid = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
isolated_adj, _ = construct_adj(isolated_positions, isolated_valid)
check("ISOLATED", "相距25 km的两个有效节点仅保留自环", isolated_adj[:2, :2].tolist() == [[1.0, 0.0], [0.0, 1.0]], isolated_adj[:2, :2].tolist(), [[1, 0], [0, 1]])

single_feature = features[:, :1]
single_output = gat(single_feature, adj_mask=torch.ones(1, 1, 1)).detach()
check("SINGLE", "单节点自环图输出有限且形状正确", tuple(single_output.shape) == (1, 1, 96) and bool(torch.isfinite(single_output).all()), tuple(single_output.shape), (1, 1, 96))

# Attention normalization snapshot using the same learned GAT parameters.
with torch.no_grad():
    q = gat.wq(features)
    k = gat.wk(features)
    score = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(features.shape[-1])
    score = score.masked_fill(invalid_adj.unsqueeze(0) <= 0, -1e9)
    attention = torch.softmax(score, dim=-1)[0]
attention_row_error = float((attention.sum(dim=-1) - 1.0).abs().max())
blocked_attention_max = float(attention[invalid_adj <= 0].abs().max())
check("ATTN-NORM", "每个节点的注意力权重和为1", attention_row_error <= 1e-6, attention_row_error, "<=1e-6")
check("ATTN-BLOCK", "非邻接边注意力为0", blocked_attention_max <= 1e-7, blocked_attention_max, "<=1e-7")

# Full-model switch check on one registered V7 sample.
with SAMPLE_PATH.open("rb") as handle:
    raw = pickle.load(handle)
sample = _parse_registered_pkl(raw, seq_len=100, num_uavs=5, basename=SAMPLE_PATH.name)[0]
extras = {
    "imu": torch.zeros(100, 5, 6), "slow_link": torch.zeros(100, 5, 3),
    "slow_update": torch.zeros(100, 5), "fast_age": torch.zeros(100, 5),
    "slow_age": torch.full((100, 5), 999.0), "esm_mode": torch.zeros(100, 5),
}
for name, value in extras.items():
    sample.setdefault(name, value)
input_names = [
    "radar", "eo", "esm", "radar_mask", "eo_mask", "esm_mask", "ref_pos", "rotmat_b2e", "uav_pos",
    "fast_link", "slow_link", "fast_update", "slow_update", "imu", "fast_age", "slow_age", "esm_mode",
]
batch = {name: sample[name].clone().unsqueeze(0) for name in input_names}
batch["dt"] = sample["dt"].clone().view(1)
with torch.no_grad():
    pred_gat = model(**batch, use_gat=True).detach().cpu()
    pred_no_gat = model(**batch, use_gat=False).detach().cpu()
valid_metric = sample["target_valid_mask"].bool().clone()
valid_metric[:WARMUP] = False
truth = sample["label_abs"][:, [0, 2, 4]]


def position_rmse(prediction: torch.Tensor) -> float:
    error = prediction[0, :, [0, 2, 4]][valid_metric] - truth[valid_metric]
    return float(torch.sqrt(torch.mean(torch.sum(error * error, dim=-1))))


sample_full_rmse = position_rmse(pred_gat)
sample_no_gat_rmse = position_rmse(pred_no_gat)
switch_delta = float((pred_gat - pred_no_gat).abs().max())
check("SWITCH-FULL", "完整模型开启GAT输出形状和有限性", tuple(pred_gat.shape) == (1, 100, 6) and bool(torch.isfinite(pred_gat).all()), tuple(pred_gat.shape), (1, 100, 6))
check("SWITCH-NOGAT", "完整模型关闭GAT输出形状和有限性", tuple(pred_no_gat.shape) == (1, 100, 6) and bool(torch.isfinite(pred_no_gat).all()), tuple(pred_no_gat.shape), (1, 100, 6))
check("SWITCH-EFFECT", "开启与关闭GAT产生可观测输出差异", switch_delta > 1e-6, switch_delta, ">1e-6")

ablation = json.loads(ABLATION_PATH.read_text(encoding="utf-8"))
check("ABLATION-MODEL", "V7完整集消融与当前冻结模型摘要一致", ablation["model_sha256"] == sha256(MODEL_PATH), ablation["model_sha256"], sha256(MODEL_PATH))
check("ABLATION-SCOPE", "消融说明为推理时屏蔽且未重训练", ablation["interpretation"] == "Inference-time component masking; no ablated model was retrained.", ablation["interpretation"], "Inference-time component masking; no ablated model was retrained.")
full_v7_rmse = float(ablation["results"]["full"]["position_rmse_m"])
no_gat_v7_rmse = float(ablation["results"]["dual_no_gat"]["position_rmse_m"])

# Figure 1: boundary adjacency snapshots.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
for ax, matrix, title in ((axes[0], snapshot_adj, "全部节点有效"), (axes[1], invalid_adj, "节点2无效")):
    image = ax.imshow(matrix.numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xlabel("目标节点"); ax.set_ylabel("源节点")
    ax.set_title(title)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="white" if matrix[i, j] > .5 else "#333")
fig.suptitle("20 km门限、传感器有效性与自环构图")
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_1_adjacency_boundary.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# Figure 2: learned attention and invalid-feature isolation.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
im = axes[0].imshow(attention.numpy(), cmap="YlGnBu", vmin=0, vmax=1)
axes[0].set_xticks(range(5)); axes[0].set_yticks(range(5)); axes[0].set_xlabel("被关注节点"); axes[0].set_ylabel("查询节点")
axes[0].set_title("学习型图注意力权重")
for i in range(5):
    for j in range(5):
        axes[0].text(j, i, f"{attention[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if attention[i,j] > .5 else "#222")
fig.colorbar(im, ax=axes[0], fraction=.046, pad=.04)
values = [max(valid_output_delta, 1e-12), max(pooled_delta, 1e-12), invalid_output_delta]
axes[1].bar(["有效节点输出差", "池化结果差", "无效节点自身输出差"], values, color=["#54a24b", "#4c78a8", "#e45756"])
axes[1].set_yscale("log"); axes[1].set_ylabel("最大绝对差（对数坐标）")
axes[1].set_title("无效节点有限填充值隔离")
axes[1].grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_2_attention_and_isolation.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# Figure 3: 0..5 effective nodes.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
counts = [r["valid_nodes"] for r in node_rows]
axes[0].plot(counts, [r["cross_edges"] for r in node_rows], marker="o", label="跨节点有向边数", color="#4c78a8")
axes[0].plot(counts, [r["self_loops"] for r in node_rows], marker="s", label="自环数", color="#f58518")
axes[0].set_xlabel("有效节点数"); axes[0].set_ylabel("边数"); axes[0].set_title("固定5槽位的邻接退化")
axes[0].set_xticks(counts); axes[0].legend(); axes[0].grid(alpha=.25)
axes[1].bar(counts, [r["pooled_norm"] for r in node_rows], color="#72b7b2")
axes[1].set_xlabel("有效节点数"); axes[1].set_ylabel("池化特征L2范数"); axes[1].set_title("有效节点归一化池化（仅记录）")
axes[1].set_xticks(counts); axes[1].grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_3_node_count_degradation.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# Figure 4: representative switch and complete-V7 ablation evidence.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
axes[0].bar(["开启GAT", "关闭GAT"], [sample_full_rmse, sample_no_gat_rmse], color=["#4c78a8", "#e45756"])
axes[0].set_ylabel("位置RMSE/m"); axes[0].set_title("代表CV样例推理开关")
axes[0].grid(axis="y", alpha=.25)
axes[1].bar(["完整V7模型", "关闭GAT"], [full_v7_rmse, no_gat_v7_rmse], color=["#4c78a8", "#e45756"])
axes[1].set_ylabel("位置RMSE/m"); axes[1].set_title("V7完整2000场景推理时消融")
axes[1].grid(axis="y", alpha=.25)
for ax in axes:
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .08, f"{bar.get_height():.3f}", ha="center")
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_4_gat_switch_and_v7_ablation.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


write_csv(RUN_DIR / "checks.csv", checks)
write_csv(RUN_DIR / "boundary_cases.csv", boundary_rows)
write_csv(RUN_DIR / "node_count_cases.csv", node_rows)

result = {
    "test_case": "IT-02-04",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "pytorch": torch.__version__,
    "platform": platform.platform(),
    "seed": SEED,
    "model_path": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
    "model_sha256": sha256(MODEL_PATH),
    "sample_path": str(SAMPLE_PATH.relative_to(ROOT)).replace("\\", "/"),
    "sample_sha256": sha256(SAMPLE_PATH),
    "ablation_path": str(ABLATION_PATH.relative_to(ROOT)).replace("\\", "/"),
    "passed": all(row["passed"] for row in checks),
    "check_count": len(checks),
    "passed_count": sum(row["passed"] for row in checks),
    "boundary_cases": boundary_rows,
    "adjacency_snapshot": snapshot_adj.tolist(),
    "invalid_node_snapshot": invalid_adj.tolist(),
    "attention_row_sum_abs_error_max": attention_row_error,
    "blocked_attention_abs_max": blocked_attention_max,
    "invalid_feature_valid_output_abs_delta_max": valid_output_delta,
    "invalid_feature_pool_abs_delta_max": pooled_delta,
    "invalid_feature_own_output_abs_delta_max": invalid_output_delta,
    "permutation_node_abs_delta_max": permutation_node_delta,
    "permutation_pool_abs_delta_max": permutation_pool_delta,
    "node_count_cases": node_rows,
    "representative_switch": {
        "sample_full_rmse_m": sample_full_rmse,
        "sample_no_gat_rmse_m": sample_no_gat_rmse,
        "output_abs_delta_max": switch_delta,
        "shape": list(pred_gat.shape),
    },
    "complete_v7_ablation": {
        "full_position_rmse_m": full_v7_rmse,
        "no_gat_position_rmse_m": no_gat_v7_rmse,
        "evaluated_points": int(ablation["results"]["full"]["evaluated_points"]),
        "interpretation": ablation["interpretation"],
    },
    "scope_statement": "图模块边界的置换等变和无效特征隔离不扩展为整个预处理系统无条件置换不变；关闭GAT为同一已训练权重的推理时消融，未重训练。",
    "figures": [path.name for path in sorted(FIG_DIR.glob("*.png"))],
}
(RUN_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
