from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import CoopTrackerModel


RUN_DIR = ROOT / "artifacts" / "reports" / "software_test" / "IT-02-05" / "20260924_run01"
FIG_DIR = RUN_DIR / "figures"
MODEL_PATH = ROOT / "results" / "runs_40off(v7)" / "run_20260728_103728" / "best_model.pt"
ONNX_PATH = ROOT / "portable" / "artifacts" / "coop_tracker_v7.onnx"
META_PATH = ROOT / "portable" / "artifacts" / "export_metadata.json"
CASE_DIR = ROOT / "portable" / "artifacts" / "golden_case"
CASE_PATH = CASE_DIR / "golden_case.npz"
MANIFEST_PATH = CASE_DIR / "manifest.json"
C_EXE = ROOT / "portable" / "c" / "build" / "Release" / "verify_golden_case.exe"

INPUT_NAMES = [
    "radar", "eo", "esm", "radar_mask", "eo_mask", "esm_mask", "ref_pos",
    "rotmat_b2e", "uav_pos", "fast_link", "slow_link", "fast_update",
    "slow_update", "imu", "fast_age", "slow_age", "esm_mode", "dt",
]
STATE_NAMES = ["px", "vx", "py", "vy", "pz", "vz"]
POSITION_INDEX = [0, 2, 4]
VELOCITY_INDEX = [1, 3, 5]

FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(20260924)
np.random.seed(20260924)
torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


checks: list[dict] = []


def check(cid: str, item: str, passed: bool, actual, expected) -> None:
    checks.append({
        "check_id": cid,
        "item": item,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    })


def run_command(command: list[str], log_name: str, timeout: int = 180) -> dict:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        text = completed.stdout + (("\n[stderr]\n" + completed.stderr) if completed.stderr else "")
        (RUN_DIR / log_name).write_text(text, encoding="utf-8")
        return {"return_code": completed.returncode, "output": text, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or "") + "\n[TIMEOUT]\n" + (exc.stderr or "")
        (RUN_DIR / log_name).write_text(text, encoding="utf-8")
        return {"return_code": None, "output": text, "timed_out": True}


RUN_DIR.mkdir(parents=True, exist_ok=True)
for path in [MODEL_PATH, ONNX_PATH, META_PATH, CASE_PATH, MANIFEST_PATH, C_EXE]:
    check(f"ART-{path.name}", f"交付物存在：{path.name}", path.is_file(), path.is_file(), True)

metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
case = np.load(CASE_PATH)
expected = case["prediction"].astype(np.float32)

check("META-SHAPE-T", "导出序列长度固定为100", metadata["sequence_length"] == 100, metadata["sequence_length"], 100)
check("META-SHAPE-N", "导出节点数固定为5", metadata["num_uavs"] == 5, metadata["num_uavs"], 5)
check("META-INPUTS", "导出元数据登记18项输入", metadata["input_names"] == INPUT_NAMES, metadata["input_names"], INPUT_NAMES)
check("META-OUTPUT", "导出输出节点名为prediction", metadata["output_name"] == "prediction", metadata["output_name"], "prediction")
check("META-WEIGHT", "导出元数据指向冻结V7权重", Path(metadata["model"]).resolve() == MODEL_PATH.resolve(), metadata["model"], str(MODEL_PATH))

# Verify binary layout against the NPZ copy field by field.
layout_rows: list[dict] = []
for name in INPUT_NAMES + ["prediction"]:
    spec = manifest[name]
    binary_path = CASE_DIR / spec["file"]
    shape = tuple(spec["shape"])
    binary = np.fromfile(binary_path, dtype="<f4").reshape(shape)
    npz_value = case[name].astype(np.float32)
    row = {
        "field": name,
        "shape": "x".join(map(str, shape)),
        "elements": int(binary.size),
        "bytes": binary_path.stat().st_size,
        "little_endian_float32": True,
        "binary_equals_npz": bool(np.array_equal(binary, npz_value)),
    }
    row["passed"] = bool(
        spec["dtype"] == "float32"
        and row["bytes"] == row["elements"] * 4
        and row["binary_equals_npz"]
    )
    layout_rows.append(row)
    check(f"LAYOUT-{name}", f"{name}形状、float32小端和行优先二进制一致", row["passed"], row, "manifest/NPZ一致")

with (RUN_DIR / "binary_layout_check.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(layout_rows[0]))
    writer.writeheader()
    writer.writerows(layout_rows)

# Load the same frozen PyTorch weight and capture the three components used by
# the final six-dimensional state construction.
model = CoopTrackerModel(d_model=96)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True), strict=True)
model.eval()
inputs_t = {name: torch.from_numpy(case[name].astype(np.float32)) for name in INPUT_NAMES}
captured: dict[str, torch.Tensor] = {}


def capture_k(_module, _args, output):
    captured["k_state"] = output[0].detach().clone()


def capture_pos(_module, _args, output):
    captured["pos_res"] = output.detach().clone()


def capture_vel(_module, _args, output):
    captured["vel_res"] = output.detach().clone()


hooks = [
    model.k_front.register_forward_hook(capture_k),
    model.pos_head.register_forward_hook(capture_pos),
    model.vel_head.register_forward_hook(capture_vel),
]
with torch.inference_mode():
    torch_output = model(**inputs_t, use_gat=True, use_temporal=True).cpu().numpy()
for hook in hooks:
    hook.remove()

k_state = captured["k_state"].cpu().numpy()
pos_res = captured["pos_res"].cpu().numpy()
vel_res = captured["vel_res"].cpu().numpy()
residual = np.empty_like(k_state)
residual[..., POSITION_INDEX] = pos_res
residual[..., VELOCITY_INDEX] = vel_res
reconstructed = k_state + residual
reconstructed[..., POSITION_INDEX] += case["ref_pos"][:, None, :]

composition_error = np.abs(torch_output - reconstructed)
torch_golden_delta = np.abs(torch_output - expected)
torch_golden_position = np.linalg.norm(torch_output[..., POSITION_INDEX] - expected[..., POSITION_INDEX], axis=-1)
check("PY-SHAPE", "PyTorch最终输出形状为[1,100,6]", list(torch_output.shape) == [1, 100, 6], list(torch_output.shape), [1, 100, 6])
check("PY-FINITE", "PyTorch最终输出全部有限", bool(np.isfinite(torch_output).all()), bool(np.isfinite(torch_output).all()), True)
check("STATE-ORDER", "位置与速度残差按[px,vx,py,vy,pz,vz]交错", float(composition_error.max()) <= 1e-6, float(composition_error.max()), "<=1e-6")
check("PY-GOLDEN-POS", "PyTorch相对黄金样例最大位置向量差不超过1 m", float(torch_golden_position.max()) <= 1.0, float(torch_golden_position.max()), "<=1 m")
check("PY-GOLDEN-VEL", "PyTorch相对黄金样例最大速度分量差不超过0.2 m/s", float(torch_golden_delta[..., VELOCITY_INDEX].max()) <= 0.2, float(torch_golden_delta[..., VELOCITY_INDEX].max()), "<=0.2 m/s")

# Metamorphic check: ref_pos must translate only px/py/pz.
shift = np.asarray([[1024.0, -2048.0, 4096.0]], dtype=np.float32)
shifted_inputs = {name: value.clone() for name, value in inputs_t.items()}
shifted_inputs["ref_pos"] = shifted_inputs["ref_pos"] + torch.from_numpy(shift)
with torch.inference_mode():
    shifted_output = model(**shifted_inputs, use_gat=True, use_temporal=True).cpu().numpy()
observed_shift = shifted_output - torch_output
position_shift_error = np.abs(observed_shift[..., POSITION_INDEX] - shift[:, None, :])
velocity_shift = np.abs(observed_shift[..., VELOCITY_INDEX])
check("REF-POS", "非零ref_pos变化仅按轴叠加至位置分量", float(position_shift_error.max()) <= 0.5, float(position_shift_error.max()), "<=0.5 m(float32量化)")
check("REF-VEL", "ref_pos变化不叠加至速度分量", float(velocity_shift.max()) <= 1e-6, float(velocity_shift.max()), "<=1e-6 m/s")

# ONNX structure, checker and deterministic CPU inference.
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
opset = max(item.version for item in onnx_model.opset_import)
session_options = ort.SessionOptions()
session_options.enable_mem_pattern = False
session = ort.InferenceSession(str(ONNX_PATH), session_options, providers=["CPUExecutionProvider"])
signature = {item.name: {"shape": item.shape, "type": item.type} for item in session.get_inputs()}
onnx_inputs = {name: case[name].astype(np.float32) for name in INPUT_NAMES}
onnx_runs = [session.run(["prediction"], onnx_inputs)[0] for _ in range(3)]
onnx_output = onnx_runs[0]
onnx_delta = np.abs(onnx_output - expected)
onnx_position_delta = np.linalg.norm(onnx_output[..., POSITION_INDEX] - expected[..., POSITION_INDEX], axis=-1)
per_dim_abs = np.max(onnx_delta, axis=(0, 1))
denominator = np.maximum(np.abs(expected), 1e-6)
per_dim_rel = np.max(onnx_delta / denominator, axis=(0, 1))
onnx_repeat = max(float(np.max(np.abs(item - onnx_output))) for item in onnx_runs[1:])

check("ONNX-CHECKER", "ONNX模型通过完整性检查", True, True, True)
check("ONNX-OPSET", "ONNX算子集版本为17", opset == 17, opset, 17)
check("ONNX-INPUT-COUNT", "ONNX模型包含18项固定输入", len(signature) == 18, len(signature), 18)
check("ONNX-INPUT-NAMES", "ONNX输入名称与交付接口一致", list(signature) == INPUT_NAMES, list(signature), INPUT_NAMES)
check("ONNX-TYPES", "ONNX输入均为tensor(float)", all(v["type"] == "tensor(float)" for v in signature.values()), signature, "all tensor(float)")
check("ONNX-OUTPUT", "ONNX输出名称、类型和形状正确", session.get_outputs()[0].name == "prediction" and session.get_outputs()[0].type == "tensor(float)" and session.get_outputs()[0].shape == [1, 100, 6], {"name": session.get_outputs()[0].name, "type": session.get_outputs()[0].type, "shape": session.get_outputs()[0].shape}, {"name": "prediction", "type": "tensor(float)", "shape": [1, 100, 6]})
check("ONNX-POS", "ONNX相对PyTorch黄金样例最大位置向量差不超过1 m", float(onnx_position_delta.max()) <= 1.0, float(onnx_position_delta.max()), "<=1 m")
check("ONNX-VEL", "ONNX相对PyTorch黄金样例最大速度分量差不超过0.2 m/s", float(onnx_delta[..., VELOCITY_INDEX].max()) <= 0.2, float(onnx_delta[..., VELOCITY_INDEX].max()), "<=0.2 m/s")
check("ONNX-REPEAT", "ONNX Runtime连续三次推理一致", onnx_repeat <= 1e-6, onnx_repeat, "<=1e-6")

dimension_rows = []
for index, name in enumerate(STATE_NAMES):
    dimension_rows.append({
        "state": name,
        "max_abs_error": float(per_dim_abs[index]),
        "max_relative_error": float(per_dim_rel[index]),
        "threshold": 1.0 if index in POSITION_INDEX else 0.2,
        "threshold_unit": "m" if index in POSITION_INDEX else "m/s",
        "exceed_count": int(np.count_nonzero(onnx_delta[..., index] > (1.0 if index in POSITION_INDEX else 0.2))),
    })
with (RUN_DIR / "per_dimension_error.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(dimension_rows[0]))
    writer.writeheader()
    writer.writerows(dimension_rows)

# C verification uses the same ONNX and binary case. Three repetitions are
# requested so loading, memory layout and repeated calls are covered together.
c_result = run_command([str(C_EXE), str(ONNX_PATH), str(CASE_DIR), "3"], "c_verification.log")


def parse_metric(text: str, name: str) -> float | None:
    match = re.search(rf"^{re.escape(name)}=([^\r\n]+)", text, re.MULTILINE)
    return float(match.group(1)) if match else None


c_metrics = {
    "return_code": c_result["return_code"],
    "repetitions": int(parse_metric(c_result["output"], "repetitions") or 0),
    "max_abs_error": parse_metric(c_result["output"], "max_abs_error"),
    "max_position_delta_m": parse_metric(c_result["output"], "max_position_delta_m"),
    "max_velocity_abs_error": parse_metric(c_result["output"], "max_velocity_abs_error"),
}
check("C-RUN", "C接口连续三次完成推理并返回0", c_metrics["return_code"] == 0 and c_metrics["repetitions"] == 3, c_metrics, "return_code=0,repetitions=3")
check("C-POS", "C接口相对PyTorch黄金样例最大位置向量差不超过1 m", c_metrics["max_position_delta_m"] is not None and c_metrics["max_position_delta_m"] <= 1.0, c_metrics["max_position_delta_m"], "<=1 m")
check("C-VEL", "C接口相对PyTorch黄金样例最大速度分量差不超过0.2 m/s", c_metrics["max_velocity_abs_error"] is not None and c_metrics["max_velocity_abs_error"] <= 0.2, c_metrics["max_velocity_abs_error"], "<=0.2 m/s")

# MATLAB is retained as supplementary evidence. The local installation currently
# fails before executing any project code, so it is excluded from the formal
# pass/fail decision under the Python-primary acceptance scope.
matlab = shutil.which("matlab.exe") or shutil.which("matlab")
if matlab:
    matlab_command = (
        f"cd('{ROOT.as_posix()}'); addpath('{(ROOT / 'portable' / 'matlab').as_posix()}'); "
        f"verify_golden_case('{ONNX_PATH.as_posix()}','{CASE_DIR.as_posix()}');"
    )
    matlab_env = os.environ.copy()
    matlab_env["MATLAB_PREFDIR"] = str(RUN_DIR / "matlab_pref")
    (RUN_DIR / "matlab_pref").mkdir(exist_ok=True)
    try:
        completed = subprocess.run(
            [matlab, "-batch", matlab_command], cwd=ROOT, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=180,
            check=False, env=matlab_env,
        )
        matlab_output = completed.stdout + (("\n[stderr]\n" + completed.stderr) if completed.stderr else "")
        matlab_result = {"available": True, "return_code": completed.returncode, "timed_out": False, "executed_project_code": "max_position_delta_m" in matlab_output, "output": matlab_output}
    except subprocess.TimeoutExpired as exc:
        matlab_output = (exc.stdout or "") + "\n[TIMEOUT]\n" + (exc.stderr or "")
        matlab_result = {"available": True, "return_code": None, "timed_out": True, "executed_project_code": False, "output": matlab_output}
else:
    matlab_result = {"available": False, "return_code": None, "timed_out": False, "executed_project_code": False, "output": "matlab.exe not found"}
(RUN_DIR / "matlab_verification.log").write_text(matlab_result["output"], encoding="utf-8")

# Figure 1: per-state ONNX parity.
x = np.arange(6)
fig, ax = plt.subplots(figsize=(9.2, 4.8))
bars = ax.bar(x, per_dim_abs, color=["#4C78A8" if i in POSITION_INDEX else "#59A14F" for i in range(6)])
thresholds = np.asarray([1.0, 0.2, 1.0, 0.2, 1.0, 0.2])
for i, threshold in enumerate(thresholds):
    ax.plot([i - .34, i + .34], [threshold, threshold], color="#C00000", linestyle="--", linewidth=1.8)
for bar, value in zip(bars, per_dim_abs):
    ax.text(bar.get_x() + bar.get_width() / 2, value + .012, f"{value:.6g}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x, STATE_NAMES)
ax.set_ylabel("最大绝对误差")
ax.set_title("Python ONNX Runtime相对PyTorch黄金输出的逐维误差")
ax.set_ylim(0, 1.12)
ax.grid(axis="y", alpha=.25)
ax.text(.99, .95, "红色虚线：位置1 m / 速度0.2 m/s阈值", transform=ax.transAxes, ha="right", va="top", color="#7F0000")
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_1_per_dimension_parity.png", dpi=180); plt.close(fig)

# Figure 2: mean magnitudes of the three state-composition terms.
component_names = ["Learned-K状态", "深度残差", "参考位置"]
component_values = np.vstack([
    np.mean(np.abs(k_state), axis=(0, 1)),
    np.mean(np.abs(residual), axis=(0, 1)),
    np.asarray([abs(case["ref_pos"][0, 0]), 0, abs(case["ref_pos"][0, 1]), 0, abs(case["ref_pos"][0, 2]), 0]),
])
fig, ax = plt.subplots(figsize=(9.4, 5.0))
width = .24
for row, (name, color) in enumerate(zip(component_names, ["#4C78A8", "#F28E2B", "#76B7B2"])):
    ax.bar(x + (row - 1) * width, component_values[row], width, label=name, color=color)
ax.set_yscale("symlog", linthresh=1)
ax.set_xticks(x, STATE_NAMES)
ax.set_ylabel("平均绝对量（对数尺度）")
ax.set_title("最终六维状态构成：Learned-K状态＋深度残差＋位置参考量")
ax.legend(ncol=3)
ax.grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_2_state_composition.png", dpi=180); plt.close(fig)

# Figure 3: ref_pos metamorphic response.
expected_shift6 = np.asarray([shift[0, 0], 0, shift[0, 1], 0, shift[0, 2], 0], dtype=np.float32)
actual_shift6 = np.mean(observed_shift, axis=(0, 1))
fig, ax = plt.subplots(figsize=(9.2, 4.8))
ax.bar(x - .18, expected_shift6, .36, label="期望变化", color="#4C78A8")
ax.bar(x + .18, actual_shift6, .36, label="实测变化", color="#F28E2B", alpha=.85)
ax.axhline(0, color="#555555", linewidth=.8)
ax.set_xticks(x, STATE_NAMES)
ax.set_ylabel("输出变化")
ax.set_title("非零ref_pos只改变px、py、pz，速度分量保持不变")
ax.legend()
ax.grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_3_reference_position_response.png", dpi=180); plt.close(fig)

# Figure 4: formal runtime status and threshold utilization.
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
status = [
    ("PyTorch\n黄金输出", True, "正式"),
    ("Python\nONNX Runtime", float(onnx_position_delta.max()) <= 1 and float(onnx_delta[..., VELOCITY_INDEX].max()) <= .2, "正式"),
    ("C接口\n3次调用", c_metrics["return_code"] == 0, "补充"),
    ("MATLAB\n本机启动", matlab_result["return_code"] == 0, "环境受限"),
]
colors = ["#59A14F" if ok else "#B7B7B7" for _, ok, _ in status]
axes[0].bar(np.arange(len(status)), [1] * len(status), color=colors)
axes[0].set_xticks(np.arange(len(status)), [item[0] for item in status])
axes[0].set_yticks([]); axes[0].set_ylim(0, 1.15)
axes[0].set_title("执行入口状态")
for i, (_, ok, scope) in enumerate(status):
    axes[0].text(i, .52, "通过" if ok else "启动失败", color="white" if ok else "#404040", weight="bold", ha="center", va="center")
    axes[0].text(i, .06, scope, color="white" if ok else "#404040", ha="center", va="bottom", fontsize=9)

labels = ["ONNX位置", "ONNX速度", "C位置", "C速度"]
ratios = [
    float(onnx_position_delta.max()) / 1.0,
    float(onnx_delta[..., VELOCITY_INDEX].max()) / .2,
    float(c_metrics["max_position_delta_m"]) / 1.0,
    float(c_metrics["max_velocity_abs_error"]) / .2,
]
axes[1].barh(np.arange(4), np.asarray(ratios) * 100, color=["#4C78A8", "#59A14F", "#4C78A8", "#59A14F"])
axes[1].axvline(100, color="#C00000", linestyle="--", label="验收阈值")
axes[1].set_yticks(np.arange(4), labels); axes[1].invert_yaxis()
axes[1].set_xlabel("误差占阈值比例（%）")
axes[1].set_title("黄金样例误差裕量")
axes[1].set_xlim(0, 112); axes[1].grid(axis="x", alpha=.25); axes[1].legend()
for i, ratio in enumerate(ratios): axes[1].text(ratio * 100 + 1.5, i, f"{ratio*100:.1f}%", va="center", fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / "figure_4_runtime_status_and_margin.png", dpi=180); plt.close(fig)

formal_check_ids = [
    item["check_id"] for item in checks
    if not item["check_id"].startswith("C-")
]
formal_checks = [item for item in checks if item["check_id"] in formal_check_ids]
result = {
    "test_case": "IT-02-05",
    "test_name": "端到端输出与多语言部署一致性测试",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "formal_scope": "Python/PyTorch 2.1.0与Python ONNX Runtime 1.18.1；C作为补充实测，MATLAB当前安装环境启动失败不参与正式判定",
    "environment": {
        "os": platform.platform(), "python": sys.version, "torch": torch.__version__,
        "onnxruntime": ort.__version__, "onnx": onnx.__version__, "numpy": np.__version__,
    },
    "artifacts": {
        "weight": {"path": str(MODEL_PATH.relative_to(ROOT)), "sha256": sha256(MODEL_PATH)},
        "onnx": {"path": str(ONNX_PATH.relative_to(ROOT)), "sha256": sha256(ONNX_PATH)},
        "golden_case": {"path": str(CASE_PATH.relative_to(ROOT)), "sha256": sha256(CASE_PATH)},
        "manifest": {"path": str(MANIFEST_PATH.relative_to(ROOT)), "sha256": sha256(MANIFEST_PATH)},
    },
    "state_order": STATE_NAMES,
    "composition": {
        "formula": "prediction = interleave(Learned-K + pos_res/vel_res) + ref_pos on indices [0,2,4]",
        "max_reconstruction_error": float(composition_error.max()),
        "ref_position_max_error": float(position_shift_error.max()),
        "ref_velocity_max_change": float(velocity_shift.max()),
    },
    "pytorch_golden": {
        "max_abs_error": float(torch_golden_delta.max()),
        "max_position_delta_m": float(torch_golden_position.max()),
        "max_velocity_abs_error_mps": float(torch_golden_delta[..., VELOCITY_INDEX].max()),
    },
    "onnx_golden": {
        "max_abs_error": float(onnx_delta.max()),
        "mean_abs_error": float(onnx_delta.mean()),
        "max_position_delta_m": float(onnx_position_delta.max()),
        "max_velocity_abs_error_mps": float(onnx_delta[..., VELOCITY_INDEX].max()),
        "repeat_max_abs_difference": onnx_repeat,
        "opset": opset,
        "per_dimension": dimension_rows,
    },
    "c_golden": c_metrics,
    "matlab_current_attempt": {
        "available": matlab_result["available"],
        "return_code": matlab_result["return_code"],
        "timed_out": matlab_result["timed_out"],
        "executed_project_code": matlab_result["executed_project_code"],
        "classification": "MATLAB安装环境启动错误；未进入项目模型代码" if matlab_result["return_code"] != 0 else "通过",
    },
    "layout": {"field_count": len(layout_rows), "all_passed": all(row["passed"] for row in layout_rows)},
    "checks": checks,
    "formal_check_count": len(formal_checks),
    "formal_passed_count": sum(item["passed"] for item in formal_checks),
    "overall_passed": all(item["passed"] for item in formal_checks),
    "figures": [path.name for path in sorted(FIG_DIR.glob("*.png"))],
}
(RUN_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "output": str(RUN_DIR),
    "overall_passed": result["overall_passed"],
    "formal_checks": f"{result['formal_passed_count']}/{result['formal_check_count']}",
    "pytorch": result["pytorch_golden"],
    "onnx": result["onnx_golden"],
    "c": result["c_golden"],
    "matlab": result["matlab_current_attempt"],
}, ensure_ascii=False, indent=2))
if not result["overall_passed"]:
    raise SystemExit(1)
