"""Export the frozen cooperative tracker to a fixed-shape ONNX model.

The export deliberately fixes batch, sequence length, and UAV count.  This keeps
the MATLAB and C acceptance implementations small and deterministic.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import _parse_registered_pkl
from src.model import CoopTrackerModel


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "results" / "runs_40off(v7)" /
    "run_20260728_103728" / "best_model.pt"
)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "val_data_v7_registered"


INPUT_NAMES = [
    "radar",
    "eo",
    "esm",
    "radar_mask",
    "eo_mask",
    "esm_mask",
    "ref_pos",
    "rotmat_b2e",
    "uav_pos",
    "fast_link",
    "slow_link",
    "fast_update",
    "slow_update",
    "imu",
    "fast_age",
    "slow_age",
    "esm_mode",
    "dt",
]


class ExportableTracker(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        radar,
        eo,
        esm,
        radar_mask,
        eo_mask,
        esm_mask,
        ref_pos,
        rotmat_b2e,
        uav_pos,
        fast_link,
        slow_link,
        fast_update,
        slow_update,
        imu,
        fast_age,
        slow_age,
        esm_mode,
        dt,
    ):
        return self.model(
            radar=radar,
            eo=eo,
            esm=esm,
            radar_mask=radar_mask,
            eo_mask=eo_mask,
            esm_mask=esm_mask,
            ref_pos=ref_pos,
            rotmat_b2e=rotmat_b2e,
            uav_pos=uav_pos,
            fast_link=fast_link,
            slow_link=slow_link,
            fast_update=fast_update,
            slow_update=slow_update,
            imu=imu,
            fast_age=fast_age,
            slow_age=slow_age,
            esm_mode=esm_mode,
            dt=dt,
            use_gat=True,
            use_temporal=True,
        )


def _replace_matlab_incompatible_einsum(model: onnx.ModelProto) -> onnx.ModelProto:
    """Lower the two Kalman-filter Einsum patterns to basic ONNX operators.

    MATLAB R2026a does not import these Einsum equations, while ONNX Runtime
    does.  Replacing them with Unsqueeze/Mul and Unsqueeze/MatMul/Squeeze keeps
    one shared model for Python, MATLAB, and C without custom MATLAB layers.
    """
    inferred = onnx.shape_inference.infer_shapes(model)
    ranks: dict[str, int] = {}
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    for value in values:
        tensor_type = value.type.tensor_type
        if tensor_type.HasField("shape"):
            ranks[value.name] = len(tensor_type.shape.dim)

    replacement_nodes = []
    replacement_initializers = []
    replacements = 0
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Einsum":
            replacement_nodes.append(node)
            continue

        equation = next(
            (helper.get_attribute_value(attribute).decode("ascii")
             for attribute in node.attribute if attribute.name == "equation"),
            "",
        )
        prefix = node.name or f"einsum_{index}"
        input_a, input_b = node.input
        output = node.output[0]
        rank_a = ranks.get(input_a)
        rank_b = ranks.get(input_b)
        if rank_a is None or rank_b is None:
            raise RuntimeError(f"Cannot determine ranks for Einsum node {prefix}")

        if equation == "...i,...j->...ij":
            axes_a = f"{prefix}_axes_a"
            axes_b = f"{prefix}_axes_b"
            expanded_a = f"{prefix}_expanded_a"
            expanded_b = f"{prefix}_expanded_b"
            replacement_initializers.extend([
                numpy_helper.from_array(np.asarray([rank_a], dtype=np.int64), axes_a),
                numpy_helper.from_array(np.asarray([rank_b - 1], dtype=np.int64), axes_b),
            ])
            replacement_nodes.extend([
                helper.make_node("Unsqueeze", [input_a, axes_a], [expanded_a], name=f"{prefix}_UnsqueezeA"),
                helper.make_node("Unsqueeze", [input_b, axes_b], [expanded_b], name=f"{prefix}_UnsqueezeB"),
                helper.make_node("Mul", [expanded_a, expanded_b], [output], name=f"{prefix}_Mul"),
            ])
        elif equation == "...ij,...j->...i":
            axes = f"{prefix}_axes"
            expanded_b = f"{prefix}_expanded_b"
            product = f"{prefix}_product"
            replacement_initializers.append(
                numpy_helper.from_array(np.asarray([rank_b], dtype=np.int64), axes)
            )
            replacement_nodes.extend([
                helper.make_node("Unsqueeze", [input_b, axes], [expanded_b], name=f"{prefix}_Unsqueeze"),
                helper.make_node("MatMul", [input_a, expanded_b], [product], name=f"{prefix}_MatMul"),
                helper.make_node("Squeeze", [product, axes], [output], name=f"{prefix}_Squeeze"),
            ])
        else:
            raise RuntimeError(f"Unsupported Einsum equation for MATLAB lowering: {equation}")
        replacements += 1

    if replacements != 2:
        raise RuntimeError(f"Expected to replace 2 Einsum nodes, replaced {replacements}")
    del model.graph.node[:]
    model.graph.node.extend(replacement_nodes)
    model.graph.initializer.extend(replacement_initializers)
    return model


def _batched_inputs(sample_path: Path, seq_len: int, num_uavs: int) -> tuple[torch.Tensor, ...]:
    with sample_path.open("rb") as handle:
        raw = pickle.load(handle)
    samples = _parse_registered_pkl(raw, seq_len, num_uavs, sample_path.name)
    if not samples:
        raise ValueError(f"No registered targets in {sample_path.name}")
    sample = samples[0]
    steps = int(sample["radar"].shape[0])
    agents = int(sample["radar"].shape[1])
    defaults = {
        "imu": (steps, agents, 6),
        "fast_link": (steps, agents, 3),
        "slow_link": (steps, agents, 3),
        "fast_age": (steps, agents),
        "slow_age": (steps, agents),
        "fast_update": (steps, agents),
        "slow_update": (steps, agents),
        "esm_mode": (steps, agents),
    }
    for name, shape in defaults.items():
        sample.setdefault(name, torch.zeros(shape, dtype=torch.float32))

    tensors: list[torch.Tensor] = []
    for name in INPUT_NAMES:
        value = sample[name]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        value = value.detach().to(device="cpu", dtype=torch.float32)
        if name == "dt":
            value = value.reshape(1)
        else:
            value = value.unsqueeze(0)
        tensors.append(value.contiguous())
    return tuple(tensors)


def _save_golden_case(output_dir: Path, inputs: tuple[torch.Tensor, ...], prediction: np.ndarray) -> None:
    case_dir = output_dir / "golden_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: value.detach().cpu().numpy().astype(np.float32) for name, value in zip(INPUT_NAMES, inputs)}
    arrays["prediction"] = prediction.astype(np.float32)
    np.savez(case_dir / "golden_case.npz", **arrays)
    for name, value in arrays.items():
        value.tofile(case_dir / f"{name}.bin")
    manifest = {
        name: {"shape": list(value.shape), "dtype": "float32", "file": f"{name}.bin"}
        for name, value in arrays.items()
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--num-uavs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("portable/artifacts"))
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.sample or next(iter(sorted(args.data_dir.glob("*.pkl"))), None)
    if sample_path is None:
        raise FileNotFoundError(f"No PKL sample found in {args.data_dir}")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model = CoopTrackerModel(d_model=96)
    model.load_state_dict(checkpoint, strict=True)
    wrapper = ExportableTracker(model.eval()).eval()
    inputs = _batched_inputs(sample_path, args.seq_len, args.num_uavs)
    with torch.inference_mode():
        torch_prediction = wrapper(*inputs).detach().cpu().numpy()

    onnx_path = output_dir / "coop_tracker_v7.onnx"
    torch.onnx.export(
        wrapper,
        inputs,
        onnx_path,
        input_names=INPUT_NAMES,
        output_names=["prediction"],
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_model = _replace_matlab_incompatible_einsum(onnx.load(onnx_path))
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_path)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        name: value.detach().cpu().numpy().astype(np.float32)
        for name, value in zip(INPUT_NAMES, inputs)
    }
    ort_prediction = session.run(["prediction"], ort_inputs)[0]
    abs_error = np.abs(torch_prediction - ort_prediction)
    max_abs_error = float(abs_error.max())
    mean_abs_error = float(abs_error.mean())
    position_delta = np.linalg.norm(
        torch_prediction[..., [0, 2, 4]] - ort_prediction[..., [0, 2, 4]], axis=-1
    )
    velocity_error = abs_error[..., [1, 3, 5]]
    max_position_delta_m = float(position_delta.max())
    mean_position_delta_m = float(position_delta.mean())
    max_velocity_abs_error = float(velocity_error.max())
    mean_velocity_abs_error = float(velocity_error.mean())
    # Absolute ECEF positions are on the order of millions of metres.  ONNX
    # Runtime may therefore differ by one float32 ULP (commonly 0.25-0.5 m)
    # while remaining materially identical for the <=8 m acceptance metric.
    if max_position_delta_m > 1.0 or max_velocity_abs_error > 0.2:
        raise RuntimeError(
            f"ONNX parity failed: max_abs_error={max_abs_error:.6g}, "
            f"max_position_delta_m={max_position_delta_m:.6g}, "
            f"max_velocity_abs_error={max_velocity_abs_error:.6g}"
        )

    _save_golden_case(output_dir, inputs, torch_prediction)
    metadata = {
        "model": str(args.model.resolve()),
        "sample": str(sample_path.resolve()),
        "sequence_length": args.seq_len,
        "num_uavs": args.num_uavs,
        "input_names": INPUT_NAMES,
        "output_name": "prediction",
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_position_delta_m": max_position_delta_m,
        "mean_position_delta_m": mean_position_delta_m,
        "max_velocity_abs_error": max_velocity_abs_error,
        "mean_velocity_abs_error": mean_velocity_abs_error,
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
