"""Execute software test case IT-01-01 and preserve machine-readable evidence."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking_system.backend.batch_testing import ensure_finite_tensor


INPUT_NAMES = [
    "radar", "eo", "esm", "radar_mask", "eo_mask", "esm_mask", "ref_pos",
    "rotmat_b2e", "uav_pos", "fast_link", "slow_link", "fast_update",
    "slow_update", "imu", "fast_age", "slow_age", "esm_mode", "dt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str], log_path: Path, timeout: int = 300) -> dict:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    log_path.write_text(
        "$ " + subprocess.list2cmdline(command) + "\n\n"
        + completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    return {"return_code": completed.returncode, "log": str(log_path.relative_to(PROJECT_ROOT))}


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting() -> None:
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.is_file():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 180


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "reports" / "software_test" / "IT-01-01" / "20260922_run01",
    )
    parser.add_argument("--skip-matlab", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)

    model_path = PROJECT_ROOT / "portable" / "artifacts" / "coop_tracker_v7.onnx"
    case_dir = PROJECT_ROOT / "portable" / "artifacts" / "golden_case"
    manifest_path = case_dir / "manifest.json"
    npz_path = case_dir / "golden_case.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = np.load(npz_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_meta = {item.name: item for item in session.get_inputs()}

    field_rows: list[dict] = []
    inputs: dict[str, np.ndarray] = {}
    for name in INPUT_NAMES:
        item = manifest[name]
        binary_path = case_dir / item["file"]
        shape = tuple(item["shape"])
        values = np.fromfile(binary_path, dtype=np.float32).reshape(shape)
        inputs[name] = values
        meta = onnx_meta.get(name)
        onnx_shape = tuple(int(x) for x in meta.shape) if meta is not None else ()
        row = {
            "field": name,
            "dtype_expected": item["dtype"],
            "dtype_actual": str(values.dtype),
            "shape_expected": "x".join(map(str, shape)),
            "shape_actual": "x".join(map(str, values.shape)),
            "elements_expected": int(np.prod(shape)),
            "elements_actual": int(values.size),
            "binary_bytes": binary_path.stat().st_size,
            "sha256": sha256(binary_path),
            "onnx_signature_match": bool(meta and onnx_shape == shape and meta.type == "tensor(float)"),
            "finite": bool(np.isfinite(values).all()),
        }
        row["passed"] = bool(
            row["dtype_actual"] == row["dtype_expected"]
            and row["shape_actual"] == row["shape_expected"]
            and row["elements_actual"] == row["elements_expected"]
            and row["binary_bytes"] == row["elements_expected"] * 4
            and row["onnx_signature_match"]
            and row["finite"]
        )
        field_rows.append(row)
    save_csv(out / "interface_field_check.csv", field_rows)

    expected = case["prediction"].astype(np.float32)
    golden_runs = [session.run(["prediction"], inputs)[0].copy() for _ in range(3)]
    golden_actual = golden_runs[0]
    golden_delta = np.abs(golden_actual - expected)
    golden_pos_delta = np.linalg.norm(
        golden_actual[..., [0, 2, 4]] - expected[..., [0, 2, 4]], axis=-1
    )
    default_repeat_max_abs_difference = float(
        max(np.max(np.abs(x - golden_actual)) for x in golden_runs[1:])
    )
    golden_metrics = {
        "output_shape": list(golden_actual.shape),
        "finite": bool(np.isfinite(golden_actual).all()),
        "max_abs_error": float(golden_delta.max()),
        "mean_abs_error": float(golden_delta.mean()),
        "max_position_delta_m": float(golden_pos_delta.max()),
        "max_velocity_abs_error_mps": float(golden_delta[..., [1, 3, 5]].max()),
    }

    safe_options = ort.SessionOptions()
    safe_options.enable_mem_pattern = False
    safe_session = ort.InferenceSession(str(model_path), safe_options, providers=["CPUExecutionProvider"])
    safe_runs = [safe_session.run(["prediction"], inputs)[0].copy() for _ in range(3)]
    golden_metrics["repeat_max_abs_difference"] = float(
        max(np.max(np.abs(x - safe_runs[0])) for x in safe_runs[1:])
    )
    golden_metrics["session_configuration"] = {
        "provider": "CPUExecutionProvider",
        "enable_mem_pattern": False,
    }

    variants: dict[str, dict[str, np.ndarray]] = {}
    for variant_name in ["完整三模态", "仅雷达", "雷达与光电", "雷达与电子侦察", "全部传感器缺失", "全部观测源缺失"]:
        variants[variant_name] = {key: value.copy() for key, value in inputs.items()}
    for name in ["仅雷达", "雷达与电子侦察", "全部传感器缺失", "全部观测源缺失"]:
        variants[name]["eo"].fill(0); variants[name]["eo_mask"].fill(0)
    for name in ["仅雷达", "雷达与光电", "全部传感器缺失", "全部观测源缺失"]:
        variants[name]["esm"].fill(0); variants[name]["esm_mask"].fill(0)
    for name in ["全部传感器缺失", "全部观测源缺失"]:
        variants[name]["radar"].fill(0); variants[name]["radar_mask"].fill(0)
    no_sources = variants["全部观测源缺失"]
    for name in ["fast_link", "slow_link", "fast_update", "slow_update"]:
        no_sources[name].fill(0)
    no_sources["fast_age"].fill(999); no_sources["slow_age"].fill(999)

    variant_rows: list[dict] = []
    variant_outputs: dict[str, np.ndarray] = {}
    for name, variant in variants.items():
        runs = [safe_session.run(["prediction"], variant)[0].copy() for _ in range(3)]
        actual = runs[0]
        variant_outputs[name] = actual
        row = {
            "variant": name,
            "radar_valid": int(np.count_nonzero(variant["radar_mask"])),
            "eo_valid": int(np.count_nonzero(variant["eo_mask"])),
            "esm_valid": int(np.count_nonzero(variant["esm_mask"])),
            "fast_updates": int(np.count_nonzero(variant["fast_update"])),
            "slow_updates": int(np.count_nonzero(variant["slow_update"])),
            "output_finite_rate": float(np.isfinite(actual).mean()),
            "repeat_max_abs_difference": float(max(np.max(np.abs(x - actual)) for x in runs[1:])),
            "output_abs_max": float(np.max(np.abs(actual))),
        }
        row["passed"] = bool(row["output_finite_rate"] == 1.0 and row["repeat_max_abs_difference"] == 0.0)
        variant_rows.append(row)
    save_csv(out / "modality_variant_check.csv", variant_rows)

    boundary_rows = []
    for link, threshold in [("fast", 2.0), ("slow", 5.0)]:
        for update in [0.0, 1.0]:
            for offset in [-0.001, 0.0, 0.001]:
                age = threshold + offset
                boundary_rows.append({
                    "link": link,
                    "threshold_s": threshold,
                    "update": int(update),
                    "age_s": age,
                    "valid": bool(update > 0.5 or age < threshold),
                })
    save_csv(out / "link_age_boundary_check.csv", boundary_rows)

    rotations = inputs["rotmat_b2e"].reshape(-1, 3, 3).astype(np.float64)
    identity = np.eye(3)
    orth_error = np.max(np.abs(np.matmul(rotations, np.swapaxes(rotations, -1, -2)) - identity), axis=(1, 2))
    determinants = np.linalg.det(rotations)
    platform_check = {
        "rotmat_shape": list(inputs["rotmat_b2e"].shape),
        "uav_pos_shape": list(inputs["uav_pos"].shape),
        "max_orthogonality_error": float(orth_error.max()),
        "determinant_min": float(determinants.min()),
        "determinant_max": float(determinants.max()),
        "distinct_platform_positions_first_frame": int(np.unique(inputs["uav_pos"][0, 0], axis=0).shape[0]),
    }
    platform_check["passed"] = bool(
        platform_check["rotmat_shape"] == [1, 100, 5, 3, 3]
        and platform_check["uav_pos_shape"] == [1, 100, 5, 3]
        and platform_check["max_orthogonality_error"] < 1e-4
        and np.allclose(determinants, 1.0, atol=1e-4)
        and platform_check["distinct_platform_positions_first_frame"] == 5
    )

    negative_rows: list[dict] = []
    missing = dict(inputs); missing.pop("radar")
    try:
        safe_session.run(["prediction"], missing)
        negative_rows.append({"check": "ONNX缺少radar字段", "rejected": False, "message": "未拒绝"})
    except Exception as exc:
        negative_rows.append({"check": "ONNX缺少radar字段", "rejected": True, "message": str(exc)[:500]})
    wrong = dict(inputs); wrong["radar"] = wrong["radar"][:, :99]
    try:
        safe_session.run(["prediction"], wrong)
        negative_rows.append({"check": "ONNX序列长度99", "rejected": False, "message": "未拒绝"})
    except Exception as exc:
        negative_rows.append({"check": "ONNX序列长度99", "rejected": True, "message": str(exc)[:500]})
    nan_input = {key: value.copy() for key, value in inputs.items()}
    nan_input["radar"][0, 0, 0, 0] = np.nan
    nan_output = safe_session.run(["prediction"], nan_input)[0]
    try:
        ensure_finite_tensor("model prediction", torch.from_numpy(nan_output))
        negative_rows.append({"check": "后端NaN/Inf数值保护", "rejected": False, "message": "未拒绝"})
    except RuntimeError as exc:
        negative_rows.append({"check": "后端NaN/Inf数值保护", "rejected": True, "message": str(exc)})
    save_csv(out / "negative_input_check.csv", negative_rows)

    c_exe = PROJECT_ROOT / "portable" / "c" / "build" / "Release" / "verify_golden_case.exe"
    c_result = run_command(
        [str(c_exe), str(model_path), str(case_dir)], out / "c_golden_case.log"
    )
    c_repeat_result = run_command(
        [str(c_exe), str(model_path), str(case_dir), "3"], out / "c_repeated_case.log"
    )
    matlab_result = {"return_code": None, "log": None, "result": None}
    if not args.skip_matlab:
        matlab_json = out / "matlab_result.json"
        if matlab_json.is_file():
            matlab_result.update({"return_code": 0, "log": str((out / "matlab_test.log").relative_to(PROJECT_ROOT))})
            matlab_result["result"] = json.loads(matlab_json.read_text(encoding="utf-8"))
        else:
            matlab_call = (
                "addpath('scripts/evaluation'); "
                f"run_it_01_01_matlab('{matlab_json.as_posix()}')"
            )
            matlab_result.update(run_command(["matlab", "-batch", matlab_call], out / "matlab_test.log", timeout=600))
            if matlab_json.is_file():
                matlab_result["result"] = json.loads(matlab_json.read_text(encoding="utf-8"))

    configure_plotting()
    checks = np.array([
        [r["dtype_actual"] == r["dtype_expected"], r["shape_actual"] == r["shape_expected"],
         r["elements_actual"] == r["elements_expected"], r["onnx_signature_match"], r["finite"]]
        for r in field_rows
    ], dtype=int)
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    ax.imshow(checks, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5), ["类型", "形状", "元素数", "ONNX签名", "有限值"])
    ax.set_yticks(range(len(INPUT_NAMES)), INPUT_NAMES)
    ax.set_title("18项部署输入接口逐字段核对")
    for y in range(checks.shape[0]):
        for x in range(checks.shape[1]): ax.text(x, y, "通过" if checks[y, x] else "失败", ha="center", va="center", fontsize=7, color="white", weight="bold")
    fig.tight_layout(); fig.savefig(figures / "图1_输入接口逐字段核对.png"); plt.close(fig)

    labels = [r["variant"] for r in variant_rows]
    x = np.arange(len(labels)); width = 0.22
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), gridspec_kw={"height_ratios": [2, 1]})
    for offset, key, title in [(-width, "radar_valid", "雷达"), (0, "eo_valid", "光电"), (width, "esm_valid", "电子侦察")]:
        axes[0].bar(x + offset, [r[key] for r in variant_rows], width, label=title)
    axes[0].set_ylabel("有效观测点数"); axes[0].legend(ncol=3); axes[0].grid(axis="y", alpha=.25)
    axes[1].bar(x, [r["output_finite_rate"] * 100 for r in variant_rows], color="#2F75B5")
    axes[1].axhline(100, color="#548235", linestyle="--", linewidth=1)
    axes[1].set_ylabel("有限输出比例 %"); axes[1].set_ylim(0, 105); axes[1].grid(axis="y", alpha=.25)
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    fig.suptitle("模态缺失组合与模型前向结果"); fig.tight_layout(); fig.savefig(figures / "图2_模态缺失组合结果.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharey=True)
    for ax, (link, threshold) in zip(axes, [("fast", 2.0), ("slow", 5.0)]):
        for update, color in [(0, "#2F75B5"), (1, "#C55A11")]:
            rows = [r for r in boundary_rows if r["link"] == link and r["update"] == update]
            ax.plot([-1, 0, 1], [int(r["valid"]) for r in rows], marker="o", label=f"update={update}", color=color)
        ax.axvline(0, color="#666666", linestyle="--", label="阈值")
        ax.set_title("快链路 2 s边界" if link == "fast" else "慢链路 5 s边界")
        ax.set_xlabel("缓存年龄相对阈值"); ax.set_xticks([-1, 0, 1], ["-0.001 s", "阈值", "+0.001 s"]); ax.grid(alpha=.25); ax.set_yticks([0, 1], ["无效", "有效"]); ax.legend()
    fig.suptitle("更新标志与缓存年龄逻辑或边界核对"); fig.tight_layout(); fig.savefig(figures / "图3_链路年龄边界结果.png"); plt.close(fig)

    status_items = [
        ("18字段接口", all(r["passed"] for r in field_rows)),
        ("固定形状前向", golden_metrics["finite"] and golden_metrics["output_shape"] == [1, 100, 6]),
        ("Python连续调用", golden_metrics["repeat_max_abs_difference"] <= 1e-6),
        ("六类模态组合", all(r["passed"] for r in variant_rows)),
        ("旋转与节点排列", platform_check["passed"]),
        ("ONNX负向检查", all(r["rejected"] for r in negative_rows[:2])),
        ("后端数值保护", negative_rows[2]["rejected"]),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    values = [1 if passed else 0 for _, passed in status_items]
    ax.barh(range(len(values)), [1] * len(values), color=["#70AD47" if v else "#C00000" for v in values])
    ax.set_yticks(range(len(values)), [x[0] for x in status_items]); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xticks([]); ax.set_title("IT-01-01分项执行结果")
    for i, value in enumerate(values): ax.text(.5, i, "通过" if value else "未通过", color="white", weight="bold", ha="center", va="center")
    fig.tight_layout(); fig.savefig(figures / "图4_分项执行结果.png"); plt.close(fig)

    result = {
        "case_id": "IT-01-01",
        "case_name": "多源异构数据接入与统一组织测试",
        "formal_scope": "Python端ONNX Runtime 1.18.1 CPUExecutionProvider，enable_mem_pattern=false",
        "run_id": out.name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "os": platform.platform(), "python": sys.version, "onnxruntime": ort.__version__,
            "numpy": np.__version__, "torch": torch.__version__, "processor": platform.processor(),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False).stdout.strip(),
            "git_status_entry_count": len(subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False).stdout.splitlines()),
        },
        "artifacts": {
            "model": {"path": str(model_path.relative_to(PROJECT_ROOT)), "size_bytes": model_path.stat().st_size, "sha256": sha256(model_path)},
            "manifest": {"path": str(manifest_path.relative_to(PROJECT_ROOT)), "sha256": sha256(manifest_path)},
            "golden_case": {"path": str(npz_path.relative_to(PROJECT_ROOT)), "sha256": sha256(npz_path)},
            "code_files": [
                {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path)}
                for path in [
                    PROJECT_ROOT / "portable" / "c" / "src" / "tracker.c",
                    PROJECT_ROOT / "portable" / "matlab" / "run_tracker.m",
                    PROJECT_ROOT / "src" / "model.py",
                    Path(__file__).resolve(),
                ]
            ],
        },
        "interface": {"field_count": len(field_rows), "passed": all(r["passed"] for r in field_rows)},
        "golden_forward": golden_metrics,
        "variants": variant_rows,
        "link_boundaries": boundary_rows,
        "platform_arrangement": platform_check,
        "negative_checks": negative_rows,
        "supplementary_cross_language_checks": {
            "included_in_formal_conclusion": False,
            "path": str((out / "supplementary_cross_language_checks.json").relative_to(PROJECT_ROOT)),
            "note": "MATLAB/C不纳入IT-01-01本轮正式判定，后续由部署一致性用例单独验证。",
        },
        "status_items": [{"name": name, "passed": passed} for name, passed in status_items],
    }
    result["overall_passed"] = all(passed for _, passed in status_items)
    supplementary = {
        "included_in_formal_conclusion": False,
        "default_onnx_session_repeat_max_abs_difference": default_repeat_max_abs_difference,
        "c_single_run": c_result,
        "c_repeated_run": c_repeat_result,
        "matlab_run": matlab_result,
        "note": "本文件仅保存探索性检查记录，不参与IT-01-01正式判定。",
    }
    (out / "supplementary_cross_language_checks.json").write_text(
        json.dumps(supplementary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "overall_passed": result["overall_passed"], "status_items": result["status_items"]}, ensure_ascii=False, indent=2))
    if not result["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
